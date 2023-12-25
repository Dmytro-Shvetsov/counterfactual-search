import torch
from easydict import EasyDict as edict
from torchmetrics import Dice, F1Score, MetricCollection, Precision, Recall
from tqdm import tqdm

from src.datasets import get_dataloaders
from src.datasets.augmentations import get_transforms
from src.models.seg_aux import SegmentationAuxModel
from src.trainers.trainer import BaseTrainer
from src.utils.avg_meter import AvgMeter
from src.utils.generic_utils import save_model
from src.visualizations import visualize_seg_predictions


class SegmentationTrainer(BaseTrainer):
    def __init__(self, opt: edict, model: SegmentationAuxModel, continue_path: str = None) -> None:
        super().__init__(opt, model, continue_path)

        self.task = 'binary' if opt.model.n_classes == 1 else 'multiclass'

        self.train_metrics_cls = MetricCollection([
            Precision(num_classes=opt.model.n_classes, task=self.task, average='macro'),
            Recall(num_classes=opt.model.n_classes, task=self.task, average='macro'),
            F1Score(num_classes=opt.model.n_classes, task=self.task, average='macro'),
        ]).to(self.device)
        self.val_metrics_cls = self.train_metrics_cls.clone()

        self.train_metrics_seg = MetricCollection([
            Dice(num_classes=self.model.n_masks, average='macro'),
        ]).to(self.device)
        self.val_metrics_seg = self.train_metrics_seg.clone()
        
        if self.opt.dataset.kind == 'merged':
            self.mask_classes = self.opt.dataset.datasets[0].scan_params.load_masks
        else:
            raise NotImplementedError
            # self.mask_classes = self.opt.dataset.scan_params.classes

    def get_dataloaders(self) -> tuple:
        transforms = get_transforms(self.opt.dataset)
        return get_dataloaders(self.opt.dataset, transforms)

    def restore_state(self):
        latest_ckpt = max(self.ckpt_dir.glob('*.pth'), key=lambda p: int(p.name.replace('.pth', '').split('_')[1]))
        load_ckpt = latest_ckpt if self.ckpt_name is None else (self.ckpt_dir / self.ckpt_name)
        state = torch.load(load_ckpt)
        self.batches_done = state['step']
        self.current_epoch = state['epoch']
        self.model.load_state_dict(state['model'], strict=True)
        self.model.optimizer.load_state_dict(state['optimizers'][0])
        self.logger.info(f"Restored checkpoint {latest_ckpt} ({state['date']})")

    def save_state(self) -> str:
        return save_model(self.opt, self.model, (self.model.optimizer,), self.batches_done, self.current_epoch, self.ckpt_dir)

    def training_epoch(self, loader: torch.utils.data.DataLoader) -> None:
        self.model.train()
        self.train_metrics_cls.reset()
        
        stats = AvgMeter()
        epoch_steps = self.opt.get('epoch_steps')
        epoch_length = epoch_steps or len(loader)
        avg_pos_to_neg_ratio = 0.0
        with tqdm(enumerate(loader), desc=f'Training epoch: {self.current_epoch}', leave=False, total=epoch_length) as prog:
            for i, batch in prog:
                if i == epoch_steps:
                    break
                batch = {k: batch[k].to(self.device) for k in {'image', 'masks', 'label'}}
                outs = self.model(batch, training=True, global_step=self.batches_done)

                # segmentation metrics
                if self.model.n_masks == 1:
                    self.train_metrics_seg.update(outs['mask_preds'].view(-1), batch['masks'][:, self.model.segment_mask_ids].view(-1))
                else:
                    self.train_metrics_seg.update(outs['mask_preds'].argmax(1).view(-1), batch['masks'][:, self.model.segment_mask_ids].argmax(1).view(-1))

                # classification metrics
                if self.model.aux_output:
                    self.train_metrics_cls.update(outs['cls_preds'], batch['label'])

                stats.update(outs['loss'])

                avg_pos_to_neg_ratio += batch['label'].sum() / batch['label'].shape[0]
                self.batches_done = self.current_epoch * len(loader) + i

                sample_step = self.batches_done % self.opt.sample_interval == 0
                if sample_step:
                    prog.set_postfix_str('training_loss={:.5f}'.format(outs['loss']['loss_total'].item()), refresh=True)
                    visualize_seg_predictions(
                        batch['image'],
                        batch['masks'],
                        outs['mask_preds'],
                        batch['label'],
                        outs['cls_preds'],
                        # agg_order=(1, 3, 2),
                        classes=self.mask_classes,
                        out_file_path=self.vis_dir / '{0:05d}_train_{1:05d}.jpg'.format(self.current_epoch, i)
                    )
        self.logger.info('[Average positives/negatives ratio in batch: %f]' % round(avg_pos_to_neg_ratio.item() / epoch_length, 3))
        epoch_stats = stats.average()
        epoch_stats.update(self.train_metrics_seg.compute())

        if self.model.aux_output:
            epoch_stats.update(self.train_metrics_cls.compute())

        self.logger.log(epoch_stats, self.current_epoch, 'train')
        self.logger.info(
            '[Finished training epoch %d/%d] [Epoch loss: %f] [Epoch F1 score: %f]'
            % (self.current_epoch, self.opt.n_epochs, epoch_stats['loss_total'], epoch_stats[f'{self.task.title()}F1Score'])
        )
        return epoch_stats

    @torch.no_grad()
    def validation_epoch(self, loader: torch.utils.data.DataLoader) -> None:
        self.model.eval()
        self.val_metrics_cls.reset()
        stats = AvgMeter()
        for i, batch in tqdm(enumerate(loader), desc=f'Validation epoch: {self.current_epoch}', leave=False, total=len(loader)):
            batch = {k: batch[k].to(self.device) for k in {'image', 'masks', 'label'}}
            # print('val', batch['label'].sum())
            outs = self.model(batch, training=False)
            # segmentation metrics
            if self.model.n_masks == 1:
                self.val_metrics_seg.update(outs['mask_preds'].view(-1), batch['masks'][:, self.model.segment_mask_ids].view(-1))
            else:
                self.val_metrics_seg.update(outs['mask_preds'].argmax(1).view(-1), batch['masks'][:, self.model.segment_mask_ids].argmax(1).view(-1))
            
            # classification metrics
            if self.model.aux_output:
                self.val_metrics_cls.update(outs['cls_preds'], batch['label'])
            stats.update(outs['loss'])
            sample_step = self.batches_done 
            if i % self.opt.sample_interval == 0:
                visualize_seg_predictions(
                    batch['image'],
                    batch['masks'],
                    outs['mask_preds'],
                    batch['label'],
                    outs['cls_preds'],
                    # agg_order=(1, 3, 2),
                    classes=self.mask_classes,
                    out_file_path=self.vis_dir / '{0:05d}_val_{1:05d}.jpg'.format(self.current_epoch, i)
                )
        
        epoch_stats = stats.average()
        epoch_stats.update(self.val_metrics_seg.compute())
        if self.model.aux_output:
            epoch_stats.update(self.val_metrics_cls.compute())

        self.logger.log(epoch_stats, self.current_epoch, 'val')
        self.logger.info(
            '[Finished validation epoch %d/%d] [Epoch loss: %f] [Epoch F1 score: %f]'
            % (self.current_epoch, self.opt.n_epochs, epoch_stats['loss_total'], epoch_stats[f'{self.task.title()}F1Score'])
        )
        return epoch_stats
