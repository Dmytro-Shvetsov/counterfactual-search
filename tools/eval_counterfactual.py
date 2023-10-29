import argparse
import logging
import os

import yaml
from easydict import EasyDict as edict

from src.models import build_model
from src.models.cgan.counterfactual_cgan import CounterfactualCGAN
from src.trainers import build_trainer
from src.trainers.counterfactual import CounterfactualTrainer
from src.utils.generic_utils import seed_everything

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('-cp', '--continue_path', type=str, required=True, help='Path to the existing training run to continue interrupted training')
parser.add_argument('-t', '--tau', type=float, required=False, default=0.8, help='Theshold for the counterfactual score metric')
opt = parser.parse_args()


def main(args):
    with open(os.path.join(args.continue_path, 'hparams.yaml')) as fid:
        opt = yaml.safe_load(fid)
        opt = edict(opt)
    seed_everything(opt.seed)

    model: CounterfactualCGAN = build_model(opt.task_name, opt=opt.model, img_size=opt.dataset.img_size)
    trainer: CounterfactualTrainer = build_trainer(opt.task_name, opt, model, args.continue_path)
    _, test_loader = trainer.get_dataloaders(skip_cf_sampler=True)
    trainer.evaluate_counterfactual(test_loader, args.tau)


if __name__ == '__main__':
    main(parser.parse_args())
