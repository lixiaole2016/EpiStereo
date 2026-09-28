"""
Epipolar-Guided Stereo Cross-Attention (EG-SCA) and related modules.

Core innovations:
1. EG-SCA: Query-level cross-view attention with epipolar-constrained sampling
2. Stereo Geometric Consistency Loss (SGCL): Self-supervised stereo regularization
3. Confidence-Aware Depth Distillation (CADD): Reliable knowledge transfer from foundation models
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .ops.modules import MSDeformAttn


class StereoDepthRefiner(nn.Module):
    """Iterative Stereo-Depth Refinement (ISDR).

    After EG-SCA in each decoder layer, predicts a per-query disparity
    correction based on left/right feature comparison and current disparity.
    The refined disparity feeds into the next layer's EG-SCA, enabling
    progressive depth accuracy across decoder layers.

    Initialized to zero output (identity residual) so the model starts
    with unchanged behavior and gradually learns corrections.
    """

    def __init__(self, d_model=256):
        super().__init__()
        self.refine_net = nn.Sequential(
            nn.Linear(d_model * 2 + 1, d_model // 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model // 2, 1)
        )
        nn.init.zeros_(self.refine_net[-1].weight)
        nn.init.zeros_(self.refine_net[-1].bias)

    def forward(self, left_feat, right_feat, disparity_norm):
        """
        Args:
            left_feat: [B, N_q, C] left cross-attention features
            right_feat: [B, N_q, C] right cross-attention features (from EG-SCA)
            disparity_norm: [B, N_q] current normalized disparity
        Returns:
            refined_disparity: [B, N_q] updated normalized disparity
        """
        feat = torch.cat([left_feat, right_feat, disparity_norm.unsqueeze(-1)], dim=-1)
        delta = self.refine_net(feat).squeeze(-1)
        return (disparity_norm + delta).clamp(0.0, 0.5)


class ContextAwareOcclusionGate(nn.Module):
    """Context-Aware Occlusion Gate (CA-OG).

    Decomposes occlusion reasoning into two complementary branches:

    1. Geometric prior: the depth-ratio signal alpha*(1 - z_r/z_q) + tau,
       identical to OA-EG, providing a strong initial estimate.
    2. Semantic refinement: a Hadamard-product interaction between query
       and right-view feature projections, modulated by geometric context,
       that learns to correct the geometric estimate based on cross-view
       feature consistency.

    The two branches are combined additively inside a sigmoid, so at
    initialization (sem_score=0 due to zero-init) the gate behaves
    exactly like OA-EG, guaranteeing no performance regression.
    """

    def __init__(self, d_model=256, hidden_dim=64, geo_dim=3,
                 alpha=10.0, tau_init=-3.0):
        super().__init__()
        self.alpha = alpha
        self.tau = nn.Parameter(torch.tensor(tau_init))

        self.query_proj = nn.Linear(d_model, hidden_dim)
        self.right_proj = nn.Linear(d_model, hidden_dim)
        self.geo_modulator = nn.Sequential(
            nn.Linear(geo_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.sem_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 1),
        )
        nn.init.zeros_(self.sem_predictor[-1].weight)
        nn.init.zeros_(self.sem_predictor[-1].bias)

    def forward(self, query_feat, right_feat, depth_ratio, disparity_norm, query_depth):
        """
        Args:
            query_feat:    [B, N_q, C]
            right_feat:    [B, N_q, C]
            depth_ratio:   [B, N_q]     z_r / z_q
            disparity_norm:[B, N_q]
            query_depth:   [B, N_q]     metres
        Returns:
            occ_weight: [B, N_q, 1]  in [0, 1]
        """
        geo_score = self.alpha * (1.0 - depth_ratio) + self.tau

        q_emb = self.query_proj(query_feat.detach())
        r_emb = self.right_proj(right_feat.detach())
        mismatch = q_emb * r_emb

        geo_input = torch.stack([
            depth_ratio,
            disparity_norm,
            (query_depth / 80.0).clamp(0, 1),
        ], dim=-1)
        geo_mod = self.geo_modulator(geo_input)

        fused = mismatch * geo_mod
        sem_score = self.sem_predictor(fused).squeeze(-1)

        return torch.sigmoid((geo_score + sem_score).unsqueeze(-1))


class EpipolarStereoAttention(nn.Module):
    """Epipolar-Guided Stereo Cross-Attention.

    For each query, computes the corresponding right-view reference point
    using stereo geometry (depth -> disparity -> horizontal shift), then
    performs deformable cross-attention on right-view features.
    The left and right attended features are fused via a learned gating mechanism.
    """

    def __init__(self, d_model=256, n_levels=3, n_heads=8, n_points=4,
                 depth_adaptive_gate=True, use_occlusion_gate=False,
                 occlusion_gate_type='context_aware'):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.depth_adaptive_gate = depth_adaptive_gate
        self.use_occlusion_gate = use_occlusion_gate
        self.occlusion_gate_type = occlusion_gate_type

        self.right_cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.right_dropout = nn.Dropout(0.1)
        self.right_norm = nn.LayerNorm(d_model)

        self.stereo_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

        if depth_adaptive_gate:
            self.disparity_gate = nn.Sequential(
                nn.Linear(1, 32),
                nn.ReLU(inplace=True),
                nn.Linear(32, 1),
            )
            nn.init.constant_(self.disparity_gate[-1].bias, 1.0)

        if use_occlusion_gate:
            if occlusion_gate_type == 'legacy':
                self.occ_threshold = nn.Parameter(torch.tensor(-3.0))
            elif occlusion_gate_type == 'context_aware':
                self.ca_occ_gate = ContextAwareOcclusionGate(d_model)
            else:
                raise ValueError(f"Unsupported occlusion_gate_type: {occlusion_gate_type}")

    def compute_right_reference_points(
        self, left_ref_points, weighted_depth, focal_length, baseline, img_w
    ):
        """Compute right-view reference points from left-view refs using stereo geometry.

        Args:
            left_ref_points: [B, N_q, n_levels, D] where D is 2 or 6 or 8
            weighted_depth: [B, H_depth, W_depth] predicted depth map
            focal_length: [B] focal length in pixels
            baseline: float, stereo baseline in meters
            img_w: [B] effective image width in pixels

        Returns:
            right_ref_points: [B, N_q, n_levels, D] shifted reference points
        """
        cx = left_ref_points[:, :, 0, 0]
        cy = left_ref_points[:, :, 0, 1]

        sample_grid = torch.stack([cx * 2 - 1, cy * 2 - 1], dim=-1).unsqueeze(2)
        depth_sampled = F.grid_sample(
            weighted_depth.unsqueeze(1),
            sample_grid,
            mode='bilinear',
            align_corners=True,
            padding_mode='border'
        ).squeeze(1).squeeze(-1)

        depth_sampled = depth_sampled.clamp(min=1.0)
        disparity_pixels = focal_length.unsqueeze(1) * baseline / depth_sampled
        disparity_norm = disparity_pixels / img_w.unsqueeze(1)
        disparity_norm = disparity_norm.clamp(min=0.0, max=0.5)

        # Avoid inplace ops: compute shifted cx, then concatenate with remaining dims
        disp_broadcast = disparity_norm.unsqueeze(-1).expand_as(left_ref_points[:, :, :, 0])
        right_cx = (left_ref_points[:, :, :, 0] - disp_broadcast).clamp(min=0.001, max=0.999)
        right_ref = torch.cat([right_cx.unsqueeze(-1), left_ref_points[:, :, :, 1:]], dim=-1)

        return right_ref, disparity_norm

    def _shift_refs_by_disparity(self, left_ref_points, disparity_norm):
        """Shift reference points by a given per-query normalized disparity."""
        disp_broadcast = disparity_norm.unsqueeze(-1).expand_as(left_ref_points[:, :, :, 0])
        right_cx = (left_ref_points[:, :, :, 0] - disp_broadcast).clamp(min=0.001, max=0.999)
        return torch.cat([right_cx.unsqueeze(-1), left_ref_points[:, :, :, 1:]], dim=-1)

    def forward(
        self, tgt, query_pos,
        left_ref_points_input,
        src_right, right_spatial_shapes, right_level_start_index, right_padding_mask,
        right_valid_ratios,
        weighted_depth, focal_length, baseline, img_w,
        query_disparity=None
    ):
        """
        Args:
            tgt: [B, N_q, C] query features (after left cross-attention + norm)
            query_pos: [B, N_q, C] query positional embedding
            left_ref_points_input: [B, N_q, n_levels, D] left reference points (already scaled)
            src_right: [B, Σ(H_l*W_l), C] flattened right features
            right_spatial_shapes: [n_levels, 2]
            right_level_start_index: [n_levels]
            right_padding_mask: [B, Σ(H_l*W_l)]
            right_valid_ratios: [B, n_levels, 2]
            weighted_depth: [B, H, W]
            focal_length: [B]
            baseline: float
            img_w: [B]
            query_disparity: [B, N_q] optional per-query disparity from ISDR (overrides depth map sampling)

        Returns:
            tgt: [B, N_q, C] updated query features
            right_attn_feat: [B, N_q, C] raw right attention output (for consistency loss)
            disparity_norm: [B, N_q] per-query normalized disparity used
        """
        if query_disparity is not None:
            disparity_norm = query_disparity
            right_ref_input = self._shift_refs_by_disparity(left_ref_points_input, disparity_norm)
        else:
            right_ref_input, disparity_norm = self.compute_right_reference_points(
                left_ref_points_input, weighted_depth, focal_length, baseline, img_w
            )

        right_attn_feat = self.right_cross_attn(
            self.with_pos_embed(tgt, query_pos),
            right_ref_input,
            src_right,
            right_spatial_shapes,
            right_level_start_index,
            right_padding_mask
        )

        gate = self.stereo_gate(torch.cat([tgt, right_attn_feat], dim=-1))

        if self.depth_adaptive_gate:
            disp_input = disparity_norm.unsqueeze(-1)
            depth_gate = torch.sigmoid(self.disparity_gate(disp_input))
            gate = gate * depth_gate

        if self.use_occlusion_gate:
            cx = left_ref_points_input[:, :, 0, 0]
            cy = left_ref_points_input[:, :, 0, 1]
            query_grid = torch.stack([cx * 2 - 1, cy * 2 - 1], dim=-1).unsqueeze(2)
            query_depth = F.grid_sample(
                weighted_depth.unsqueeze(1), query_grid,
                mode='bilinear', align_corners=True, padding_mode='border'
            ).squeeze(1).squeeze(-1).clamp(min=1.0)

            right_cx = (cx - disparity_norm).clamp(0.001, 0.999)
            right_grid = torch.stack([right_cx * 2 - 1, cy * 2 - 1], dim=-1).unsqueeze(2)
            right_ref_depth = F.grid_sample(
                weighted_depth.unsqueeze(1), right_grid,
                mode='bilinear', align_corners=True, padding_mode='border'
            ).squeeze(1).squeeze(-1).clamp(min=1.0)

            depth_ratio = right_ref_depth / query_depth
            if self.occlusion_gate_type == 'legacy':
                occ_weight = torch.sigmoid(
                    10.0 * (1.0 - depth_ratio) + self.occ_threshold
                ).unsqueeze(-1)
            else:
                occ_weight = self.ca_occ_gate(
                    tgt, right_attn_feat, depth_ratio, disparity_norm, query_depth
                )
            gate = gate * (1.0 - occ_weight)

        tgt = tgt + self.right_dropout(gate * right_attn_feat)
        tgt = self.right_norm(tgt)

        return tgt, right_attn_feat, disparity_norm

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos


class StereoTriangulationAttention(nn.Module):
    """Stereo Triangulation Attention (STA).

    Each query sweeps K disparity hypotheses along the epipolar line in the
    right view. A single weight-shared MSDeformAttn processes all hypotheses
    in one batched call. Cosine similarity between left-attended and
    right-attended features scores each hypothesis, and temperature-scaled
    softmax aggregation produces the final stereo feature.

    This is equivalent to a per-query implicit cost volume with only a
    fusion gate and K learnable disparity scales as extra parameters.
    """

    def __init__(self, d_model=256, n_levels=3, n_heads=8, n_points=4,
                 n_hypotheses=4):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_hypotheses = n_hypotheses

        self.right_cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.right_dropout = nn.Dropout(0.1)
        self.right_norm = nn.LayerNorm(d_model)

        self.disparity_log_scales = nn.Parameter(torch.zeros(n_hypotheses))
        with torch.no_grad():
            self.disparity_log_scales.copy_(
                torch.linspace(math.log(0.7), math.log(1.4), n_hypotheses))

        self.log_temperature = nn.Parameter(torch.tensor(2.0))

        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def _sample_depth_at_queries(self, left_ref_points, weighted_depth):
        cx = left_ref_points[:, :, 0, 0]
        cy = left_ref_points[:, :, 0, 1]
        grid = torch.stack([cx * 2 - 1, cy * 2 - 1], dim=-1).unsqueeze(2)
        depth = F.grid_sample(
            weighted_depth.unsqueeze(1), grid,
            mode='bilinear', align_corners=True, padding_mode='border'
        ).squeeze(1).squeeze(-1)
        return depth.clamp(min=1.0)

    def compute_multi_hypothesis_refs(self, left_ref_points, weighted_depth,
                                      focal_length, baseline, img_w):
        B, N_q, n_levels, D = left_ref_points.shape
        K = self.n_hypotheses

        depth_sampled = self._sample_depth_at_queries(left_ref_points, weighted_depth)
        base_disp_norm = (
            focal_length.unsqueeze(1) * baseline / depth_sampled / img_w.unsqueeze(1)
        ).clamp(0.0, 0.5)

        scale_factors = torch.exp(self.disparity_log_scales)
        multi_disp = (base_disp_norm.unsqueeze(-1) * scale_factors).clamp(0.0, 0.5)

        left_exp = left_ref_points.unsqueeze(2).expand(B, N_q, K, n_levels, D)
        disp_exp = multi_disp.unsqueeze(-1).unsqueeze(-1).expand(B, N_q, K, n_levels, 1)

        right_cx = (left_exp[..., 0:1] - disp_exp).clamp(0.001, 0.999)
        right_refs = torch.cat([right_cx, left_exp[..., 1:]], dim=-1)
        return right_refs.reshape(B, N_q * K, n_levels, D), multi_disp

    def forward(
        self, tgt, query_pos,
        left_ref_points_input,
        src_right, right_spatial_shapes, right_level_start_index,
        right_padding_mask, right_valid_ratios,
        weighted_depth, focal_length, baseline, img_w
    ):
        B, N_q, C = tgt.shape
        K = self.n_hypotheses

        right_refs, multi_disp = self.compute_multi_hypothesis_refs(
            left_ref_points_input, weighted_depth, focal_length, baseline, img_w)

        q = self.with_pos_embed(tgt, query_pos)
        q_exp = q.unsqueeze(2).expand(B, N_q, K, C).reshape(B, N_q * K, C)

        right_feats = self.right_cross_attn(
            q_exp, right_refs,
            src_right, right_spatial_shapes,
            right_level_start_index, right_padding_mask
        ).reshape(B, N_q, K, C)

        left_exp = tgt.unsqueeze(2).expand_as(right_feats)
        cos_sim = F.cosine_similarity(left_exp, right_feats, dim=-1)
        temperature = torch.exp(self.log_temperature).clamp(1.0, 100.0)
        matching_weights = F.softmax(cos_sim * temperature, dim=-1)

        right_agg = (right_feats * matching_weights.unsqueeze(-1)).sum(dim=2)

        gate = self.fusion_gate(torch.cat([tgt, right_agg], dim=-1))
        tgt = tgt + self.right_dropout(gate * right_agg)
        tgt = self.right_norm(tgt)

        return tgt, right_agg, matching_weights, multi_disp

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos


class StereoGeometricConsistencyLoss(nn.Module):
    """Contrastive Stereo Geometric Consistency Loss (C-SGCL).

    Extends basic stereo consistency with an InfoNCE contrastive objective:
      - Positive pairs: left[i] and right[i] for the same query i
      - Negative pairs: left[i] and right[j] for different queries j≠i
    This learns discriminative stereo representations where geometrically
    corresponding features are pulled together while non-corresponding
    features are pushed apart.
    """

    def __init__(self, loss_weight=1.0, margin=0.1,
                 use_contrastive=False, contrastive_temperature=0.07):
        super().__init__()
        self.loss_weight = loss_weight
        self.margin = margin
        self.use_contrastive = use_contrastive
        self.contrastive_temperature = contrastive_temperature

    def forward(self, left_attn_feats, right_attn_feats, disparity_norms=None):
        """
        Args:
            left_attn_feats: list of [B, N_q, C] from each decoder layer
            right_attn_feats: list of [B, N_q, C] from each decoder layer

        Returns:
            loss: scalar
        """
        total_loss = torch.tensor(0.0, device=left_attn_feats[0].device)
        count = 0

        for left_f, right_f in zip(left_attn_feats, right_attn_feats):
            left_norm = F.normalize(left_f, dim=-1)
            right_norm = F.normalize(right_f, dim=-1)

            cosine_sim = (left_norm * right_norm).sum(dim=-1)
            align_loss = F.relu(self.margin - cosine_sim).mean()

            if self.use_contrastive:
                B, N_q, _ = left_norm.shape
                sim_matrix = torch.bmm(left_norm, right_norm.transpose(1, 2))
                sim_matrix = sim_matrix / self.contrastive_temperature
                labels = torch.arange(N_q, device=left_norm.device).unsqueeze(0).expand(B, -1)
                contrastive_loss = F.cross_entropy(
                    sim_matrix.reshape(B * N_q, N_q),
                    labels.reshape(B * N_q))
                layer_loss = align_loss + contrastive_loss
            else:
                layer_loss = align_loss

            total_loss = total_loss + layer_loss
            count += 1

        if count > 0:
            total_loss = total_loss / count

        return total_loss * self.loss_weight


def compute_depth_gradient(depth_map):
    """Compute gradient magnitude of a depth map using Sobel filters.

    Used to identify depth discontinuities (object boundaries) where stereo
    matching uncertainty is high and hypothesis spacing should be wider.

    Args:
        depth_map: [B, H, W] monocular depth map (e.g., from Depth Anything V2)
    Returns:
        grad_magnitude: [B, H, W]
    """
    depth = depth_map.unsqueeze(1)
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
                           dtype=depth.dtype, device=depth.device).reshape(1, 1, 3, 3) / 4.0
    sobel_y = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
                           dtype=depth.dtype, device=depth.device).reshape(1, 1, 3, 3) / 4.0
    grad_x = F.conv2d(depth, sobel_x, padding=1)
    grad_y = F.conv2d(depth, sobel_y, padding=1)
    return torch.sqrt(grad_x ** 2 + grad_y ** 2 + 1e-8).squeeze(1)


class MonocularGuidedMHEA(nn.Module):
    """Monocular-Guided Multi-Hypothesis Epipolar Attention (MG-MHEA).

    Core innovation: uses monocular depth prior (Depth Anything V2) to
    adaptively control the hypothesis spacing in multi-hypothesis stereo
    matching within the DETR decoder.

    DA V2's depth gradient indicates local depth uncertainty:
      - High gradient (object boundaries) → wider hypothesis range
      - Low gradient (flat surfaces) → narrower hypothesis range

    This leverages DA V2's strong structural understanding without requiring
    scale-aligned distillation, avoiding the fundamental scale mismatch problem.
    """

    def __init__(self, d_model=256, n_levels=3, n_heads=8, n_points=4,
                 n_hypotheses=5):
        super().__init__()
        self.d_model = d_model
        self.n_levels = n_levels
        self.n_hypotheses = n_hypotheses

        self.right_cross_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.right_dropout = nn.Dropout(0.1)
        self.right_norm = nn.LayerNorm(d_model)

        self.disparity_log_offsets = nn.Parameter(torch.zeros(n_hypotheses))
        with torch.no_grad():
            self.disparity_log_offsets.copy_(
                torch.linspace(-0.35, 0.35, n_hypotheses))

        self.range_predictor = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(inplace=True),
            nn.Linear(16, 1),
            nn.Softplus()
        )
        nn.init.constant_(self.range_predictor[2].bias, 0.5)

        self.log_temperature = nn.Parameter(torch.tensor(2.0))

        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.Sigmoid()
        )

    def _sample_at_queries(self, left_ref_points, feature_map):
        """Sample a 2D feature map at query reference point positions."""
        cx = left_ref_points[:, :, 0, 0]
        cy = left_ref_points[:, :, 0, 1]
        grid = torch.stack([cx * 2 - 1, cy * 2 - 1], dim=-1).unsqueeze(2)
        sampled = F.grid_sample(
            feature_map.unsqueeze(1), grid,
            mode='bilinear', align_corners=True, padding_mode='border'
        ).squeeze(1).squeeze(-1)
        return sampled

    def compute_mono_guided_refs(self, left_ref_points, weighted_depth,
                                 focal_length, baseline, img_w,
                                 mono_depth_grad):
        B, N_q, n_levels, D = left_ref_points.shape
        K = self.n_hypotheses

        depth_sampled = self._sample_at_queries(
            left_ref_points, weighted_depth).clamp(min=1.0)
        base_disp_norm = (
            focal_length.unsqueeze(1) * baseline / depth_sampled / img_w.unsqueeze(1)
        ).clamp(0.0, 0.5)

        if mono_depth_grad is not None:
            grad_sampled = self._sample_at_queries(left_ref_points, mono_depth_grad)
            grad_max = grad_sampled.max(dim=1, keepdim=True)[0].clamp(min=1e-6)
            grad_norm = (grad_sampled / grad_max).unsqueeze(-1)
            range_scale = self.range_predictor(grad_norm).squeeze(-1)
        else:
            range_scale = torch.ones(B, N_q, device=left_ref_points.device)

        offsets = self.disparity_log_offsets
        scaled_offsets = offsets.unsqueeze(0).unsqueeze(0) * range_scale.unsqueeze(-1)
        scale_factors = torch.exp(scaled_offsets)

        multi_disp = (base_disp_norm.unsqueeze(-1) * scale_factors).clamp(0.0, 0.5)

        left_exp = left_ref_points.unsqueeze(2).expand(B, N_q, K, n_levels, D)
        disp_exp = multi_disp.unsqueeze(-1).unsqueeze(-1).expand(B, N_q, K, n_levels, 1)

        right_cx = (left_exp[..., 0:1] - disp_exp).clamp(0.001, 0.999)
        right_refs = torch.cat([right_cx, left_exp[..., 1:]], dim=-1)

        return right_refs.reshape(B, N_q * K, n_levels, D), multi_disp

    def forward(
        self, tgt, query_pos, left_ref_points_input,
        src_right, right_spatial_shapes, right_level_start_index,
        right_padding_mask, right_valid_ratios,
        weighted_depth, focal_length, baseline, img_w,
        mono_depth_grad=None
    ):
        B, N_q, C = tgt.shape
        K = self.n_hypotheses

        right_refs, multi_disp = self.compute_mono_guided_refs(
            left_ref_points_input, weighted_depth, focal_length, baseline, img_w,
            mono_depth_grad)

        q = self.with_pos_embed(tgt, query_pos)
        q_exp = q.unsqueeze(2).expand(B, N_q, K, C).reshape(B, N_q * K, C)

        right_feats = self.right_cross_attn(
            q_exp, right_refs,
            src_right, right_spatial_shapes,
            right_level_start_index, right_padding_mask
        ).reshape(B, N_q, K, C)

        left_exp = tgt.unsqueeze(2).expand_as(right_feats)
        cos_sim = F.cosine_similarity(left_exp, right_feats, dim=-1)
        temperature = torch.exp(self.log_temperature).clamp(1.0, 100.0)
        matching_weights = F.softmax(cos_sim * temperature, dim=-1)

        right_agg = (right_feats * matching_weights.unsqueeze(-1)).sum(dim=2)

        gate = self.fusion_gate(torch.cat([tgt, right_agg], dim=-1))
        tgt = tgt + self.right_dropout(gate * right_agg)
        tgt = self.right_norm(tgt)

        return tgt, right_agg, matching_weights, multi_disp

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos


class ConfidenceAwareDistillation(nn.Module):
    """Confidence-Aware Depth Distillation from foundation models.

    Learns a per-pixel confidence map that predicts where the teacher depth
    (Depth Anything V2) is reliable. The distillation loss is weighted by
    this confidence, and a regularization term prevents the confidence from
    collapsing to zero.
    """

    def __init__(self, in_channels=256, loss_weight=0.1, reg_weight=0.5):
        super().__init__()
        self.loss_weight = loss_weight
        self.reg_weight = reg_weight

        self.confidence_net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
        )

        self.scale_predictor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(in_channels, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 2),
        )

    def forward(self, pred_depth, teacher_depth, depth_features, valid_mask=None):
        """
        Args:
            pred_depth: [B, H_p, W_p] model predicted metric depth
            teacher_depth: [B, H_t, W_t] DA V2 relative depth (per-image normalized)
            depth_features: [B, C, H_f, W_f] features from depth predictor
            valid_mask: [B, H, W] mask for valid teacher depth regions

        Returns:
            loss: scalar distillation loss
        """
        H_f, W_f = depth_features.shape[2:]

        if pred_depth.shape[1:] != (H_f, W_f):
            pred_depth = F.interpolate(
                pred_depth.unsqueeze(1), size=(H_f, W_f), mode='bilinear', align_corners=True
            ).squeeze(1)
        if teacher_depth.shape[1:] != (H_f, W_f):
            teacher_depth = F.interpolate(
                teacher_depth.unsqueeze(1), size=(H_f, W_f), mode='bilinear', align_corners=True
            ).squeeze(1)

        if valid_mask is None:
            valid_mask = teacher_depth > 0.5
        elif valid_mask.shape[1:] != (H_f, W_f):
            valid_mask = F.interpolate(
                valid_mask.unsqueeze(1).float(), size=(H_f, W_f), mode='nearest'
            ).squeeze(1).bool()

        if valid_mask.sum() < 10:
            return torch.tensor(0.0, device=pred_depth.device)

        confidence = self.confidence_net(depth_features).squeeze(1)
        scale_shift = self.scale_predictor(depth_features)
        scale = F.softplus(scale_shift[:, 0:1]).unsqueeze(-1) + 0.1
        shift = scale_shift[:, 1:2].unsqueeze(-1)

        aligned_teacher = (teacher_depth * scale + shift).detach()

        pixel_loss = F.smooth_l1_loss(pred_depth, aligned_teacher, reduction='none', beta=2.0)
        weighted_loss = (confidence * pixel_loss * valid_mask.float()).sum() / (
            confidence * valid_mask.float()).sum().clamp(min=1.0)

        conf_reg = -torch.log(confidence[valid_mask] + 1e-6).mean()

        loss = weighted_loss + self.reg_weight * conf_reg
        return loss * self.loss_weight
