import logging
import os
import os.path as osp
import datetime
import random
import argparse
from torch.utils.tensorboard import SummaryWriter
from preprocess import preprocess
from utils import seed_torch, set_logger, Cfg

import sys

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--yaml_file', help='The configuration file.', required=True)
    parser.add_argument('--multi_run_mode', help='Run multiple experiments with the same config.', action='store_true')
    args = parser.parse_args()
    conf_file = args.yaml_file

    cfg = Cfg(conf_file)

    sizes = [int(i) for i in cfg.model_args.sizes.split('-')]
    cfg.model_args.sizes = sizes

    # cuda setting
    if int(cfg.run_args.gpu) >= 0:
        device = 'cuda:' + str(cfg.run_args.gpu)
    else:
        device = 'cpu'
    cfg.run_args.device = device

    # for multiple runs, seed is replaced with random value
    if args.multi_run_mode:
        cfg.run_args.seed = None
    if cfg.run_args.seed is None:
        seed = random.randint(0, 100000000)
    else:
        seed = int(cfg.run_args.seed)

    seed_torch(seed)

    current_time = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    cfg.run_args.save_path = f'tensorboard/{current_time}/{cfg.dataset_args.dataset_name}'
    cfg.run_args.log_path = f'log/{current_time}/{cfg.dataset_args.dataset_name}'

    if not osp.isdir(cfg.run_args.save_path):
        os.makedirs(cfg.run_args.save_path)
    if not osp.isdir(cfg.run_args.log_path):
        os.makedirs(cfg.run_args.log_path)

    set_logger(cfg.run_args)
    summary_writer = SummaryWriter(log_dir=cfg.run_args.save_path)

    hparam_dict = {}
    for group, hparam in cfg.__dict__.items():
        hparam_dict.update(hparam.__dict__)
    hparam_dict['seed'] = seed
    hparam_dict['sizes'] = '-'.join([str(item) for item in cfg.model_args.sizes])

    # Preprocess data
    preprocess(cfg)
    sys.exit()
