from copy import deepcopy
from unittest import loader

import lightning as L
import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from torchmetrics.image.fid import FrechetInceptionDistance
from torchvision.utils import save_image
from tqdm import tqdm

from src.datasets import get_dataloaders
from src.datasets.augmentations import get_transforms
from src.models.cgan import CounterfactualCGAN
from src.models.classifier import compute_sampler_condition_labels, predict_probs
from src.trainers.trainer import BaseTrainer
from src.utils.avg_meter import AvgMeter
from src.utils.generic_utils import save_model


class CounterfactualTrainer(BaseTrainer):
    def __init__(self, opt: edict, model: CounterfactualCGAN, continue_path: str = None) -> None:
        super().__init__(opt, model, continue_path)
        self.cf_vis_dir = self.logging_dir / 'counterfactuals'
        self.cf_vis_dir.mkdir(exist_ok=True)
        
        self.compute_norms = self.opt.get('compute_norms', False)
        # 64, 192, 768, 2048
        self.fid_features = opt.get('fid_features', 768)
        self.val_fid = FrechetInceptionDistance(self.fid_features, normalize=True).to(self.device)

    def restore_state(self):
        latest_ckpt = max(self.ckpt_dir.glob('*.pth'), key=lambda p: int(p.name.replace('.pth', '').split('_')[1]))
        state = torch.load(latest_ckpt)
        self.batches_done = state['step']
        self.current_epoch = state['epoch']
        self.model.load_state_dict(state['model'], strict=True)
        self.model.optimizer_G.load_state_dict(state['optimizers'][0])
        self.model.optimizer_D.load_state_dict(state['optimizers'][1])
        self.logger.info(f"Restored checkpoint {latest_ckpt} ({state['date']})")

    def save_state(self) -> str:
        return save_model(self.opt, self.model, (self.model.optimizer_G, self.model.optimizer_D), self.batches_done, self.current_epoch, self.ckpt_dir)

    def get_dataloaders(self, ret_cf_labels:bool=False, skip_cf_sampler=False) -> tuple:
        transforms = get_transforms(self.opt.dataset)
        if self.opt.task_name == 'counterfactual' and not skip_cf_sampler: # NB
            # compute sampler labels to create batches with uniformly distributed labels
            params = edict(deepcopy(self.opt.dataset), use_sampler=False, shuffle_test=False)
            # GAN's train data is expected to be classifier's validation data
            train_loader, _ = get_dataloaders(params, {'train': transforms['val'], 'val': transforms['train']})
            posterior_probs, _ = predict_probs(train_loader, self.model.classifier_f, task='binary' if self.model.n_classes == 1 else 'multiclass')
            # sampler labels are just digitized classifier's posterior probabilities on input images
            # i.e we sample into batches images where classifier predicts p<0.5 in half of the cases and p>0.5 otherwise
            sampler_labels = compute_sampler_condition_labels(posterior_probs, self.model.explain_class_idx, self.model.num_bins)
            self.logger.info(f'Precomputed condition labels for sampling. Num positive conditions: {sampler_labels.sum()}')
        else:
            sampler_labels = None
        loaders = get_dataloaders(self.opt.dataset, transforms, sampler_labels=sampler_labels)
        return (loaders, sampler_labels) if ret_cf_labels else loaders

    def training_epoch(self, loader: torch.utils.data.DataLoader) -> None:
        self.model.train()

        stats = AvgMeter()
        epoch_steps = self.opt.get('epoch_steps')
        with tqdm(enumerate(loader), desc=f'Training epoch {self.current_epoch}', leave=False, total=epoch_steps or len(loader)) as prog:
            for i, batch in prog:
                if i == epoch_steps:
                    break
                self.batches_done = self.current_epoch * len(loader) + i
                sample_step = self.batches_done % self.opt.sample_interval == 0
                outs = self.model(batch, training=True, compute_norms=sample_step and self.compute_norms, global_step=self.batches_done)
                stats.update(outs['loss'])
                if sample_step:
                    save_image(outs['gen_imgs'][:16].data, self.vis_dir / ('%d_train_%d.jpg' % (self.current_epoch, i)), nrow=4, normalize=True)
                    postf = '[Batch %d/%d] [D loss: %f] [G loss: %f]' % (i, len(loader), outs['loss']['d_loss'], outs['loss']['g_loss'])
                    prog.set_postfix_str(postf, refresh=True)
                    if self.compute_norms:
                        for model_name, norms in self.model.norms.items():
                            self.logger.log(norms, self.batches_done, f'{model_name}_gradients_norm')
        epoch_stats = stats.average()
        self.logger.log(epoch_stats, self.current_epoch, 'train')
        self.logger.info(
            '[Finished training epoch %d/%d] [Epoch D loss: %f] [Epoch G loss: %f]'
            % (self.current_epoch, self.opt.n_epochs, epoch_stats['d_loss'], epoch_stats['g_loss'])
        )
        return epoch_stats

    @torch.no_grad()
    def validation_epoch(self, loader: torch.utils.data.DataLoader) -> None:
        self.model.eval()

        stats = AvgMeter()
        avg_pos_to_neg_ratio = 0.0
        for i, batch in tqdm(enumerate(loader), desc=f'Validation epoch {self.current_epoch}', leave=False, total=len(loader)):
            avg_pos_to_neg_ratio += batch['label'].sum() / batch['label'].shape[0]
            outs = self.model(batch, validation=True)
            stats.update(outs['loss'])
            if i % self.opt.sample_interval == 0 and self.opt.get('vis_gen', True):
                save_image(outs['gen_imgs'][:16].data, self.vis_dir / ('%d_val_%d.jpg' % (self.batches_done, i)), nrow=4, normalize=True)
        self.logger.info('[Average positives/negatives ratio in batch: %f]' % round(avg_pos_to_neg_ratio.item() / len(loader), 3))
        epoch_stats = stats.average()
        if self.current_epoch % self.opt.eval_counter_freq == 0:
            epoch_stats['counter_acc'], epoch_stats['cv_80'], epoch_stats['fid'] = self.evaluate_counterfactual(loader)
        self.logger.log(epoch_stats, self.current_epoch, 'val')
        self.logger.info(
            '[Finished validation epoch %d/%d] [Epoch D loss: %f] [Epoch G loss: %f]'
            % (self.current_epoch, self.opt.n_epochs, epoch_stats['d_loss'], epoch_stats['g_loss'])
        )
        return epoch_stats

    @torch.no_grad()
    def evaluate_counterfactual(self, loader, tau=0.8):
        self.model.eval()

        classes = []
        cv_y_true, cv_y_pred = [], []
        posterior_true, posterior_pred = [], []

        for i, batch in tqdm(enumerate(loader), desc='Validating counterfactuals', leave=False, total=len(loader)):
            # Evaluate Counterfactual Validity Metric
            real_imgs, labels = batch['image'].cuda(non_blocking=True), batch['label']
            self.model: CounterfactualCGAN
            real_f_x, real_f_x_discrete, real_f_x_desired, real_f_x_desired_discrete = self.model.posterior_prob(real_imgs)

            real_neg_pos_group = ((real_f_x < 0.2) | (real_f_x > 0.8)).view(-1)
            if not real_neg_pos_group.any():
                continue

            # filter out samples not belonging to either real negative or positive groups
            real_imgs, labels = real_imgs[real_neg_pos_group], labels[real_neg_pos_group.cpu()]
            real_f_x, real_f_x_discrete, real_f_x_desired, real_f_x_desired_discrete = (
                real_f_x[real_neg_pos_group],
                real_f_x_discrete[real_neg_pos_group],
                real_f_x_desired[real_neg_pos_group],
                real_f_x_desired_discrete[real_neg_pos_group],
            )

            # our ground truth is the `flipped` labels
            cv_y_true.extend(real_f_x_desired_discrete.cpu().squeeze(1).numpy())
            posterior_true.extend(real_f_x.cpu().squeeze(1).numpy())
            classes.extend(labels.cpu().numpy())

            # computes I_f(x, c)
            gen_imgs = self.model.explanation_function(real_imgs, real_f_x_desired_discrete)

            gen_f_x, gen_f_x_discrete, gen_f_x_desired, gen_f_x_desired_discrete = self.model.posterior_prob(gen_imgs)
            # our prediction is the classifier's label for the generated images given the desired posterior probability
            cv_y_pred.extend(gen_f_x_discrete.cpu().squeeze(1).numpy())
            posterior_pred.extend(gen_f_x.cpu().squeeze(1).numpy())

            # denorm values from [-1; 1] to [0, 1] range, B x 1 x H x W
            real_imgs.add_(1).div_(2)
            gen_imgs.add_(1).div_(2)

            # take the first example, compute difference map for it and save the image
            # difference map
            diff = (real_imgs[0] - gen_imgs[0]).abs()
            vis = torch.stack((real_imgs[0], gen_imgs[0], diff), dim=1)
            save_image(
                vis.data,
                self.cf_vis_dir / (f'epoch_%d_counterfactual_%d_label_%d_true_%d_pred_%d.jpg' % (
                    self.current_epoch, i, labels[0], real_f_x_desired_discrete[0][0], gen_f_x_discrete[0][0])
                ),
                nrow=3,
                normalize=False,
                # value_range=(-1, 1),
            )
            
            # Evaluate Frechet Inception Distance (FID)
            # upsample to InceptionV3's resolution and convert to RGB
            real_imgs = nn.functional.interpolate(real_imgs, size=(299, 299), mode='bilinear', align_corners=False)
            real_imgs = real_imgs.repeat_interleave(repeats=3, dim=1)
            self.val_fid.update(real_imgs, real=True)
            
            # upsample to InceptionV3's resolution and convert to RGB
            gen_imgs = nn.functional.interpolate(gen_imgs, size=(299, 299), mode='bilinear', align_corners=False)
            gen_imgs = gen_imgs.repeat_interleave(repeats=3, dim=1)
            self.val_fid.update(gen_imgs, real=False)

        num_samples = len(posterior_true)
        self.logger.info(f'Finished evaluating counterfactual results for epoch: {self.current_epoch}')

        # Counterfactual Accuracy (flip rate) Score
        cv_y_true, cv_y_pred = np.array(cv_y_true), np.array(cv_y_pred)
        cacc = np.mean(cv_y_true == cv_y_pred)
        self.logger.info(f'Counterfactual accuracy = {cacc} (num_samples={len(cv_y_pred)})')

        posterior_true, posterior_pred = np.array(posterior_true), np.array(posterior_pred)
        # Counterfactual Validity Score
        cv_score = np.mean(np.abs(posterior_true - posterior_pred) > tau)
        self.logger.info(f'CV(X, Xc) = {cv_score:.3f} (τ={tau}, num_samples={num_samples})')

        # Frechet Inception Distance (FID) Score
        fid_score = self.val_fid.compute().item()
        self.logger.info(f'FID(X, Xc) = {fid_score:.3f} (num_samples={num_samples}, features={self.fid_features})')
        self.val_fid.reset()
        return cacc, cv_score, fid_score
