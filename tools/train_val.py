import warnings
warnings.filterwarnings("ignore")

import os
import sys
import torch

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import yaml
import argparse
import datetime

from lib.helpers.model_helper import build_model
from lib.helpers.dataloader_helper import build_dataloader
from lib.helpers.optimizer_helper import build_optimizer
from lib.helpers.scheduler_helper import build_lr_scheduler
from lib.helpers.trainer_helper import Trainer
from lib.helpers.tester_helper import Tester
from lib.helpers.utils_helper import create_logger
from lib.helpers.utils_helper import set_random_seed


parser = argparse.ArgumentParser(description='EpiStereo stereo 3D object detection')
parser.add_argument('--config', default='config.yaml', help='model and training settings in YAML format')
parser.add_argument('-e', '--evaluate_only', action='store_true', help='run inference instead of training')
parser.add_argument('--test-set', action='store_true', help='run inference on the KITTI test split')
parser.add_argument('--checkpoint', help='checkpoint path for inference')
parser.add_argument('--epochs', type=int, help='override the number of training epochs')
parser.add_argument('--batch-size', type=int, help='override the training and evaluation batch size')
parser.add_argument('--output-dir', help='relative output directory for this run')
args = parser.parse_args()


def main():
    if args.test_set and not args.evaluate_only:
        parser.error('--test-set requires --evaluate_only')
    if args.checkpoint and not args.evaluate_only:
        parser.error('--checkpoint requires --evaluate_only')
    if args.epochs is not None and args.epochs < 1:
        parser.error('--epochs must be at least 1')
    if args.batch_size is not None and args.batch_size < 1:
        parser.error('--batch-size must be at least 1')
    if args.output_dir and os.path.isabs(args.output_dir):
        parser.error('--output-dir must be relative to the project directory')
    print("args.config>>>",  args.config)
    assert os.path.exists(args.config), args.config
    with open(args.config, 'r') as config_file:
        cfg = yaml.safe_load(config_file)
    if args.epochs is not None:
        cfg['trainer']['max_epoch'] = args.epochs
    if args.batch_size is not None:
        cfg['dataset']['batch_size'] = args.batch_size
    if args.output_dir:
        cfg['trainer']['save_path'] = args.output_dir
    if args.evaluate_only:
        # The full checkpoint replaces all backbone weights; no download is needed.
        cfg['model']['pretrained_backbone'] = False
        if args.checkpoint:
            cfg['tester']['checkpoint_path'] = args.checkpoint
    if args.test_set:
        cfg['dataset']['root_dir_eval'] = 'data/KITTI/testing'
        cfg['dataset']['eval_txt'] = 'data/KITTI/ImageSets/test.txt'
        cfg['dataset']['test_split'] = 'test'
        cfg['trainer']['save_path'] = 'outputs/test/'
        cfg['tester']['checkpoint_path'] = args.checkpoint or 'weights/checkpoint.pth'
    set_random_seed(cfg.get('random_seed', 444))

    model_name = cfg['model_name']
    output_path = os.path.join('./' + cfg["trainer"]['save_path'], model_name)
    os.makedirs(output_path, exist_ok=True)

    log_file = os.path.join(output_path, 'train.log.%s' % datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
    logger = create_logger(log_file)

    # build dataloader
    train_loader, test_loader = build_dataloader(cfg['dataset'], include_train=not args.evaluate_only)

    # build model
    model, loss = build_model(cfg['model'])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_ids = list(map(int, cfg['trainer']['gpu_ids'].split(',')))

    if len(gpu_ids) == 1:
        model = model.to(device)
    else:
        model = torch.nn.DataParallel(model, device_ids=gpu_ids).to(device)

    if args.evaluate_only:
        logger.info('###################  Evaluation Only  ##################')
        tester = Tester(cfg=cfg['tester'],
                        model=model,
                        dataloader=test_loader,
                        logger=logger,
                        train_cfg=cfg['trainer'],
                        model_name=model_name)
        tester.test()
        return
    #ipdb.set_trace()
    #  build optimizer
    optimizer = build_optimizer(cfg['optimizer'], model)
    # build lr scheduler
    lr_scheduler, warmup_lr_scheduler = build_lr_scheduler(cfg['lr_scheduler'], optimizer, last_epoch=-1)

    trainer = Trainer(cfg=cfg['trainer'],
                      model=model,
                      optimizer=optimizer,
                      train_loader=train_loader,
                      test_loader=test_loader,
                      lr_scheduler=lr_scheduler,
                      warmup_lr_scheduler=warmup_lr_scheduler,
                      logger=logger,
                      loss=loss,
                      model_name=model_name)

    tester = Tester(cfg=cfg['tester'],
                    model=trainer.model,
                    dataloader=test_loader,
                    logger=logger,
                    train_cfg=cfg['trainer'],
                    model_name=model_name)
    if cfg['dataset']['test_split'] != 'test':
        trainer.tester = tester

    logger.info('###################  Training  ##################')
    logger.info('Batch Size: %d' % (cfg['dataset']['batch_size']))
    logger.info('Learning Rate: %f' % (cfg['optimizer']['lr']))

    trainer.train()

    if cfg['dataset']['test_split'] == 'test':
        return

    logger.info('###################  Testing  ##################')
    logger.info('Batch Size: %d' % (cfg['dataset']['batch_size']))
    logger.info('Split: %s' % (cfg['dataset']['test_split']))

    tester.test()


if __name__ == '__main__':
    main()
