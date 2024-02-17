import argparse
import logging
import os
import shutil
import torch

import yaml
from easydict import EasyDict as edict

from src.models import build_model
from src.trainers import build_trainer
from src.utils.generic_utils import seed_everything

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str, required=False, help='Configuration file path to start training from scratch')
parser.add_argument('-cp', '--continue_path', type=str, required=False, help='Path to the existing training run to continue interrupted training')
opt = parser.parse_args()


def main(args):
    with open(args.config_path or os.path.join(args.continue_path, 'hparams.yaml')) as fid:
        opt = yaml.safe_load(fid)
        opt = edict(opt)
    seed_everything(opt.seed)

    model = build_model(opt.task_name, opt=opt.model, img_size=opt.dataset.img_size)
    trainer = build_trainer(opt.task_name, opt, model, args.continue_path)

    
    # train_loader, val_loader = trainer.get_dataloaders()

    # from PIL import Image
    # for i, batch in enumerate(train_loader):
    #     inputs, labels = batch['image'], batch['label']
    #     print(i, labels)
    #     img = (((inputs[0, 0] + 1) / 2)*255).clamp(0, 255).byte()
    #     Image.fromarray(img.cpu().numpy()).save(f'tmp/b/{i}.png')
    #     if i == 30:
    #         break
    # if args.continue_path is None:
    #     shutil.copy2(args.config_path, trainer.logging_dir / 'hparams.yaml')
    # logging.info('Started training.')
    trainer.fit()
    torch.save(model.state_dict(), 'sd1.pt')
    # logging.info('Finished training.')


if __name__ == '__main__':
    main(parser.parse_args())
