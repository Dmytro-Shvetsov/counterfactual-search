import argparse
import logging
import os
import shutil
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from easydict import EasyDict as edict

logging.basicConfig(level=logging.INFO)

from src.dataset import get_dataloaders, get_transforms
from src.cgan import build_gan
from src.classifier import compute_sampler_condition_labels, predict_probs
from src.trainer import Trainer

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

    model_kind = opt.model.get('kind')
    model = build_gan(opt.model, img_size=opt.dataset.img_size)

    transforms = get_transforms(opt.dataset)
    if model_kind.startswith('counterfactual'):
        # compute sampler labels to create batches with uniformly distributed labels 
        params = edict(opt.dataset, use_sampler=False, shuffle_test=False)
        # GAN's train data is expected to be classifier's validation data
        train_loader, _ = get_dataloaders({'train': transforms['val'], 'val': transforms['train']}, params)
        posterior_probs, _ = predict_probs(train_loader, model.classifier_f)
        sampler_labels = compute_sampler_condition_labels(posterior_probs, model.explain_class_idx, model.num_bins)
    else:
        sampler_labels = None

    dataloaders = get_dataloaders(transforms, opt.dataset, sampler_labels=sampler_labels)

    trainer = Trainer(opt, model, args.continue_path)
    if args.continue_path is None:
        shutil.copy2(args.config_path, trainer.logging_dir / 'hparams.yaml')
    logging.info('Started training.')
    trainer.fit(dataloaders)
    logging.info('Finished training.')


if __name__ == '__main__':
    main(parser.parse_args())
