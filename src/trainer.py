import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import yaml
from easydict import EasyDict as edict
import logging
from tqdm import tqdm
from torchvision.utils import save_image

from src.utils.avg_meter import AvgMeter


class Trainer:
    def __init__(self, opt, model:nn.Module) -> None:
        self.opt = opt
        self.model = model
        self.current_epoch = 0

    def training_epoch(self, loader:torch.utils.data.DataLoader) -> None:
        self.model.train()

        stats = AvgMeter()
        for i, batch in tqdm(enumerate(loader), desc=f'Training epoch: {self.current_epoch}', leave=False, total=len(loader)):
            outs = self.model(batch, training=True)
            stats.update(outs['loss'])

            self.batches_done = self.current_epoch * len(loader) + i
            if self.batches_done % self.opt.sample_interval == 0:
                save_image(outs['gen_imgs'][:16].data, "images/%d_train_%d.png" % (self.current_epoch, i), nrow=4, normalize=True)
                print()
                logging.info("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (self.current_epoch, self.opt.n_epochs, i, len(loader), outs['loss']['d_loss'], outs['loss']['g_loss'])
                )
        epoch_stats = stats.average()
        return epoch_stats

    @torch.no_grad()
    def validation_epoch(self, loader:torch.utils.data.DataLoader) -> None:
        self.model.eval()

        stats = AvgMeter()
        for i, batch in tqdm(enumerate(loader), desc=f'Validation epoch: {self.current_epoch}', leave=False, total=len(loader)):
            outs = self.model(batch, training=False)
            stats.update(outs['loss'])

            # self.batches_done = self.current_epoch * len(loader) + i
            if i % self.opt.sample_interval == 0:
                save_image(outs['gen_imgs'][:16].data, "images/%d_val_%d.png" % (self.batches_done, i), nrow=4, normalize=True)

                # logging.info("[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                #     % (self.current_epoch, self.opt.n_epochs, i, len(loader), outs['loss']['d_loss'], outs['loss']['g_loss'])
                # )
                # sample_image(n_row=10, batches_done=batches_done)
        epoch_stats = stats.average()
        return epoch_stats

    def fit(self, data_loaders):
        train_loader, val_loader = data_loaders

        for i in range(self.opt.n_epochs):
            epoch_stats = self.training_epoch(train_loader)
            logging.info("[Finished training epoch %d/%d] [Epoch D loss: %f] [Epoch G loss: %f]"
                % (self.current_epoch, self.opt.n_epochs, epoch_stats['d_loss'], epoch_stats['g_loss'])
            )
            epoch_stats = self.validation_epoch(val_loader)
            logging.info("[Finished validation epoch %d/%d] [Epoch D loss: %f] [Epoch G loss: %f]"
                % (self.current_epoch, self.opt.n_epochs, epoch_stats['d_loss'], epoch_stats['g_loss'])
            )
            self.current_epoch += 1
