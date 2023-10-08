import argparse
import logging
import os
import random
import shutil

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict

from src.models import build_model
from src.trainers import build_trainer

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str, required=False, help='Configuration file path to start training from scratch')
parser.add_argument('-cp', '--continue_path', type=str, required=False, help='Path to the existing training run to continue interrupted training')
opt = parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    with open(args.config_path or os.path.join(args.continue_path, 'hparams.yaml')) as fid:
        opt = yaml.safe_load(fid)
        opt = edict(opt)
    seed_everything(opt.seed)

    model = build_model(opt.task_name, opt=opt.model, img_size=opt.dataset.img_size)
    trainer = build_trainer(opt.task_name, opt, model, args.continue_path)

    if args.continue_path is None:
        shutil.copy2(args.config_path, trainer.logging_dir / 'hparams.yaml')
    logging.info('Started training.')
    trainer.fit()
    logging.info('Finished training.')


if __name__ == '__main__':
    main(parser.parse_args())
