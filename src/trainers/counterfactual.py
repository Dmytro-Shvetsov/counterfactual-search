from copy import deepcopy
from unittest import loader

import numpy as np
import torch
import torch.nn as nn
from easydict import EasyDict as edict
from torchvision.utils import save_image
from tqdm import tqdm

# from src.datasets.lungs import get_covid_dataloaders
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
        self.compute_norms = self.opt.get('compute_norms', False)

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

    def get_dataloaders(self, ret_counterfactual_labels:bool=False) -> tuple:
        transforms = get_transforms(self.opt.dataset)
        if self.opt.task_name == 'counterfactual': # NB
            # compute sampler labels to create batches with uniformly distributed labels
            params = edict(deepcopy(self.opt.dataset), use_sampler=False, shuffle_test=False)
            # if params.get('scan_params', {}).get('synth_params'):
            #     params['scan_params']['synth_params']['p'] = 1.0
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
        return (loaders, sampler_labels) if ret_counterfactual_labels else loaders

    def training_epoch(self, loader: torch.utils.data.DataLoader) -> None:
        self.model.train()

        stats = AvgMeter()
        epoch_steps = self.opt.get('epoch_steps')
        with tqdm(enumerate(loader), desc=f'Training epoch: {self.current_epoch}', leave=False, total=epoch_steps or len(loader)) as prog:
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
        for i, batch in tqdm(enumerate(loader), desc=f'Validation epoch: {self.current_epoch}', leave=False, total=len(loader)):
            avg_pos_to_neg_ratio += batch['label'].sum() / batch['label'].shape[0]
            outs = self.model(batch, training=False)
            stats.update(outs['loss'])
            # self.batches_done = self.current_epoch * len(loader) + i
            if i % self.opt.sample_interval == 0:
                save_image(outs['gen_imgs'][:16].data, self.vis_dir / ('%d_val_%d.jpg' % (self.batches_done, i)), nrow=4, normalize=True)
        self.logger.info('[Average positives/negatives ratio in batch: %f]' % round(avg_pos_to_neg_ratio.item() / len(loader), 3))
        epoch_stats = stats.average()
        if self.current_epoch % self.opt.eval_counter_freq == 0:
            epoch_stats['counter_acc'], epoch_stats['cv_80'] = self.evaluate_counterfactual(loader)
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
        y_true, y_pred = [], []
        posterior_true, posterior_pred = [], []
        out_dir = self.logging_dir / 'counterfactuals'
        out_dir.mkdir(exist_ok=True)
        for i, batch in tqdm(enumerate(loader), desc='Validating counterfactuals:', leave=False, total=len(loader)):
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
            y_true.extend(real_f_x_desired_discrete.cpu().squeeze(1).numpy())
            posterior_true.extend(real_f_x.cpu().squeeze(1).numpy())
            classes.extend(labels.cpu().numpy())

            # computes I_f(x, c)
            gen_imgs = self.model.explanation_function(real_imgs, real_f_x_desired_discrete)

            gen_f_x, gen_f_x_discrete, gen_f_x_desired, gen_f_x_desired_discrete = self.model.posterior_prob(gen_imgs)
            # our prediction is the classifier's label for the generated images given the desired posterior probability
            y_pred.extend(gen_f_x_discrete.cpu().squeeze(1).numpy())
            posterior_pred.extend(gen_f_x.cpu().squeeze(1).numpy())

            real_imgs, gen_imgs = real_imgs[0], gen_imgs[0]
            # difference map
            diff = (real_imgs - gen_imgs).abs()
            vis = torch.stack((real_imgs, gen_imgs, diff), dim=0)
            save_image(
                vis.data,
                out_dir / (f'epoch_%d_counterfactual_%d_label_%d_true_%d_pred_%d.jpg' % (self.current_epoch, i, labels[0], real_f_x_desired_discrete[0][0], gen_f_x_discrete[0][0])),
                nrow=1,
                normalize=True,
            )

        y_true, y_pred = np.array(y_true), np.array(y_pred)
        cacc = np.mean(y_true == y_pred)
        self.logger.info(f'Counterfactual accuracy = {cacc} (num_samples={len(y_pred)})')

        posterior_true, posterior_pred = np.array(posterior_true), np.array(posterior_pred)
        # Counterfactual Validity score
        cv_score = np.mean(np.abs(posterior_true - posterior_pred) > tau)
        self.logger.info(f'CV(X, Xc) = {cv_score:.3f} (τ={tau}, num_samples={len(posterior_true)})')

        # with open(out_dir / 'probs.txt', 'w') as fid:
        #     fid.write('i,label,bin_true,bin_pred,posterior_real,posterior_gen\n')
        #     for i in range(y_true.shape[0]):
        #         fid.write(','.join(map(str, [i, classes[i], y_true[i], y_pred[i], round(posterior_true[i], 3), round(posterior_pred[i])])) + '\n')
        return cacc, cv_score
