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
from src.cgan import build_gan
from src.classifier import compute_sampler_condition_labels, predict_probs
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

    # model = VanilaCGAN(opt.dataset.img_size, opt.model)
    model_kind = opt.model.get('kind')
    model = build_gan(opt.model, img_size=opt.dataset.img_size)

    transforms = get_transforms(opt.dataset)
    if model_kind.startswith('counterfactual'):
        # compute sampler labels to create batches with uniformly distributed labels 
        params = edict(opt.dataset, use_sampler=False, shuffle_test=False)
        # GAN's train loader is expected to be classifier's validation data
        train_loader, _ = get_dataloaders({'train': transforms['val'], 'val': transforms['train']}, params)
        posterior_probs, _ = predict_probs(train_loader, model.classifier_f)
        sampler_labels = compute_sampler_condition_labels(posterior_probs, model.explain_class_idx, model.num_bins)
    else:
        sampler_labels = None

    dataloaders = get_dataloaders(transforms, opt.dataset, sampler_labels=sampler_labels)

    if torch.cuda.is_available():
        model.cuda()
    logging.info(f'Using model: {model.__class__.__name__}')

    trainer = Trainer(opt, model)
    logging.info('Started training.')
    trainer.fit(dataloaders)
    logging.info('Finished training.')


if __name__ == '__main__':
    main(parser.parse_args())
