import argparse
import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from easydict import EasyDict as edict
from torchvision.utils import save_image
from tqdm import tqdm

from src.utils.avg_meter import AvgMeter
from src.cgan import CounterfactualLungsCGAN
from src.utils.generic_utils import get_experiment_folder_path, save_model

from .logger import Logger


class Trainer:
    def __init__(self, opt:edict, model:nn.Module, continue_path:Optional[str]=None) -> None:
        self.opt = opt
        self.model = model
        if torch.cuda.is_available():
            self.model.cuda()
        self.compute_norms = self.opt.get('compute_norms', False)
        self.batches_done = 0
        self.current_epoch = 0

        self.logging_dir = Path(continue_path or get_experiment_folder_path(opt.logging_dir, opt.model.kind))
        self.vis_dir = self.logging_dir / 'visualizations'
        self.ckpt_dir = self.logging_dir / 'checkpoints'
        self.logger = Logger(self.logging_dir)
        if continue_path is not None:
            assert self.logging_dir.exists(), f'Unable to find model directory {continue_path}'
            self._restore_state()
        else: 
            self.vis_dir.mkdir(exist_ok=True)
            self.ckpt_dir.mkdir(exist_ok=True)
            self.logger.reset()
        logging.info(f'Using model: {self.model.__class__.__name__}')

    def _restore_state(self):
        latest_ckpt = max(self.ckpt_dir.glob('*.pth'), key=lambda p: len(p.name))
        state = torch.load(latest_ckpt)
        self.batches_done = state['step']
        self.current_epoch = state['epoch']
        self.model.load_state_dict(state['model'], strict=True)
        self.model.optimizer_G.load_state_dict(state['optimizers'][0])
        self.model.optimizer_D.load_state_dict(state['optimizers'][1])
        self.logger.info(f"Restored checkpoint {latest_ckpt} ({state['date']})")

    def training_epoch(self, loader:torch.utils.data.DataLoader) -> None:
        self.model.train()

        stats = AvgMeter()
        with tqdm(enumerate(loader), desc=f'Training epoch: {self.current_epoch}', leave=False, total=len(loader)) as prog:
            for i, batch in prog:
                self.batches_done = self.current_epoch * len(loader) + i
                sample_step = self.batches_done % self.opt.sample_interval == 0
                outs = self.model(batch, training=True, compute_norms=sample_step and self.compute_norms)
                stats.update(outs['loss'])

                if sample_step:
                    save_image(outs['gen_imgs'][:16].data, self.vis_dir / ("%d_train_%d.png" % (self.current_epoch, i)), nrow=4, normalize=True)
                    postf = '[Batch %d/%d] [D loss: %f] [G loss: %f]' % (
                        i, len(loader), outs['loss']['d_loss'], outs['loss']['g_loss']
                    )
                    prog.set_postfix_str(postf, refresh=True)
                    if self.compute_norms:
                        for model_name, norms in self.model.norms.items():
                            self.logger.log(norms, self.batches_done, f'{model_name}_gradients_norm')
        epoch_stats = stats.average()
        self.logger.log(epoch_stats, self.current_epoch, 'train')
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
                save_image(outs['gen_imgs'][:16].data, self.vis_dir / ("%d_val_%d.png" % (self.batches_done, i)), nrow=4, normalize=True)
        epoch_stats = stats.average()
        self.logger.log(epoch_stats, self.current_epoch, 'val')
        return epoch_stats

    def fit(self, data_loaders):
        train_loader, val_loader = data_loaders
        for i in range(self.current_epoch, self.opt.n_epochs):
            epoch_stats = self.training_epoch(train_loader)
            self.logger.info("[Finished training epoch %d/%d] [Epoch D loss: %f] [Epoch G loss: %f]"
                % (self.current_epoch, self.opt.n_epochs, epoch_stats['d_loss'], epoch_stats['g_loss'])
            )
            epoch_stats = self.validation_epoch(val_loader)
            self.logger.info("[Finished validation epoch %d/%d] [Epoch D loss: %f] [Epoch G loss: %f]"
                % (self.current_epoch, self.opt.n_epochs, epoch_stats['d_loss'], epoch_stats['g_loss'])
            )
            if self.current_epoch % self.opt.checkpoint_freq == 0:
                ckpt_path = save_model(self.opt, self.model, (self.model.optimizer_G, self.model.optimizer_D), 
                                    self.batches_done, self.current_epoch, self.ckpt_dir)
                self.logger.info(f'Saved checkpoint parameters at epoch {self.current_epoch}: {ckpt_path}')
            self.current_epoch += 1
        ckpt_path = save_model(self.opt, self.model, (self.model.optimizer_G, self.model.optimizer_D), 
                               self.batches_done, self.current_epoch, self.ckpt_dir)
        self.logger.info(f'Saved checkpoint parameters at epoch {self.current_epoch}: {ckpt_path}')

    @torch.no_grad()
    def evaluate_counterfactual(self, loader, tau=0.8):
        self.model.eval()

        classes = []
        y_true, y_pred = [], []
        posterior_true, posterior_pred = [], []
        out_dir = self.logging_dir / 'counterfactuals'
        out_dir.mkdir(exist_ok=True)
        for i, batch in tqdm(enumerate(loader), desc=f'Validating counterfactuals:', leave=False, total=len(loader)):
            real_imgs, labels, masks = batch['image'].cuda(non_blocking=True), batch['label'], batch['mask']
            classes.extend(labels.cpu().squeeze().numpy())
            self.model:CounterfactualLungsCGAN
            real_f_x, real_f_x_discrete, real_f_x_desired, real_f_x_desired_discrete = self.model.posterior_prob(real_imgs)
            # our ground truth is the `flipped` labels
            y_true.extend(real_f_x_desired_discrete.cpu().squeeze().numpy())
            posterior_true.extend(real_f_x.cpu().squeeze().numpy())

            # computes I_f(x, c)
            gen_imgs = self.model.explanation_function(real_imgs, real_f_x_desired_discrete)

            gen_f_x, gen_f_x_discrete, gen_f_x_desired, gen_f_x_desired_discrete = self.model.posterior_prob(gen_imgs)
            # our prediction is the classifier's label for the generated images given the desired posterior probability
            y_pred.extend(gen_f_x_discrete.cpu().squeeze().numpy())
            posterior_pred.extend(gen_f_x.cpu().squeeze().numpy())

            real_imgs, gen_imgs = real_imgs[0], gen_imgs[0]
            # difference map 
            diff = (real_imgs - gen_imgs).abs()
            vis = torch.stack((real_imgs, gen_imgs, diff), dim=0)
            save_image(
                vis.data, 
                out_dir / ("counterfactual_%d_label_%d_true_%d_pred_%d.png" % (i, labels[0], real_f_x_desired_discrete[0][0], gen_f_x_discrete[0][0])), 
                nrow=1, 
                normalize=True
            )

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        cacc = np.mean(y_true == y_pred)
        self.logger.info(f'Counterfactual accuracy = {cacc}')

        posterior_true, posterior_pred = np.array(posterior_true), np.array(posterior_pred)
        # Counterfactual Validity score 
        cv_score = np.mean((posterior_true - posterior_pred) > tau)
        self.logger.info(f'CV(X, Xc) = {cv_score:.3f} (τ={tau})')

        with open(out_dir / 'probs.txt', 'w') as fid:
            fid.write('i,label,bin_true,bin_pred,posterior_real,posterior_gen\n')
            for i in range(y_true.shape[0]):
                fid.write(','.join(
                    map(str, [i, classes[i], y_true[i], y_pred[i], round(posterior_true[i], 3), round(posterior_pred[i])])
                ) + '\n')
        return cv_score
