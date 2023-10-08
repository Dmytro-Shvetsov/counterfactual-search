import argparse
import logging
import os
import random

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict

from datasets.lungs import get_dataloaders, get_transforms
from src.models.cgan import CounterfactualLungsCGAN, build_gan
from trainers.counterfactual import CounterfactualTrainer

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('-cp', '--continue_path', type=str, required=True, help='Path to the existing training run to continue interrupted training')
parser.add_argument('-t', '--tau', type=float, required=False, default=0.8, help='Theshold for the counterfactual score metric')
opt = parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def main(args):
    with open(os.path.join(args.continue_path, 'hparams.yaml')) as fid:
        opt = yaml.safe_load(fid)
        opt = edict(opt)
    seed_everything(opt.seed)

    model: CounterfactualLungsCGAN = build_gan(opt.model, img_size=opt.dataset.img_size)

    transforms = get_transforms(opt.dataset)
    _, test_loader = get_dataloaders(transforms, opt.dataset, sampler_labels=None)
    trainer = CounterfactualTrainer(opt, model, args.continue_path)
    trainer.evaluate_counterfactual(test_loader, args.tau)


if __name__ == '__main__':
    main(parser.parse_args())
