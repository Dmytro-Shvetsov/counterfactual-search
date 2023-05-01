import argparse
import logging
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from easydict import EasyDict as edict

logging.basicConfig(level=logging.INFO)

from src.dataset import get_dataloaders, get_transforms
from src.cgan.vanila_cgan import VanilaCGAN
from src.cgan.lungs_cgan import LungsCGAN
from src.trainer import Trainer

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str, help='Configuration file path')
opt = parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main(args):
    with open(args.config_path) as fid:
        opt = yaml.safe_load(fid)
        opt = edict(opt)
    seed_everything(opt.seed)

    transforms = get_transforms(opt.dataset)
    dataloaders = get_dataloaders(transforms, opt.dataset)

    # model = VanilaCGAN(opt.dataset.img_size, opt.model)
    model = LungsCGAN(opt.dataset.img_size, opt.model)
    
    if torch.cuda.is_available():
        model.cuda()
    logging.info(f'Using model: {model.__class__.__name__}')

    trainer = Trainer(opt, model)
    logging.info('Started training.')
    trainer.fit(dataloaders)
    logging.info('Finished training.')


if __name__ == '__main__':
    main(parser.parse_args())
