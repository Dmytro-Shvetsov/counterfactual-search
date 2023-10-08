import torch
from easydict import EasyDict as edict
from torchmetrics import F1Score, MetricCollection, Precision, Recall
from tqdm import tqdm

from src.datasets.augmentations import get_transforms
from src.datasets.tsm_synth_dataset import get_totalsegmentor_dataloaders
from src.models.classifier import ClassificationModel
from src.trainers.trainer import BaseTrainer
from src.utils.generic_utils import save_model


class ClassificationTrainer(BaseTrainer):
    def __init__(self, opt: edict, model: ClassificationModel, continue_path: str | None = None) -> None:
        super().__init__(opt, model, continue_path)

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        self.task = 'binary' if opt.model.n_classes == 1 else 'multiclass'

        metric_list = [
            Precision(num_classes=opt.model.n_classes, task=self.task, average='macro'),
            Recall(num_classes=opt.model.n_classes, task=self.task, average='macro'),
            F1Score(num_classes=opt.model.n_classes, task=self.task, average='macro'),
        ]
        self.train_metrics = MetricCollection(metric_list).to(self.device)
        self.val_metrics = self.train_metrics.clone()

    def get_dataloaders(self) -> tuple:
        transforms = get_transforms(self.opt.dataset, max_pixel_value=1.0)
        return get_totalsegmentor_dataloaders(transforms, self.opt.dataset)

    def restore_state(self):
        latest_ckpt = max(self.ckpt_dir.glob('*.pth'), key=lambda p: int(p.name.replace('.pth', '').split('_')[1]))
        state = torch.load(latest_ckpt)
        self.batches_done = state['step']
        self.current_epoch = state['epoch']
        self.model.load_state_dict(state['model'], strict=True)
        self.model.optimizer.load_state_dict(state['optimizers'][0])
        self.logger.info(f"Restored checkpoint {latest_ckpt} ({state['date']})")

    def save_state(self) -> str:
        return save_model(self.opt, self.model, (self.model.optimizer,), self.batches_done, self.current_epoch, self.ckpt_dir)

    def training_epoch(self, loader: torch.utils.data.DataLoader) -> None:
        self.model.train()
        self.train_metrics.reset()
        losses = []

        epoch_steps = self.opt.get('epoch_steps')
        with tqdm(enumerate(loader), desc=f'Training epoch: {self.current_epoch}', leave=False, total=epoch_steps or len(loader)) as prog:
            for i, batch in prog:
                if i == epoch_steps:
                    break
                batch = {k: v.to(self.device) for k, v in batch.items()}
                print(batch['label'].sum())
                self.batches_done = self.current_epoch * len(loader) + i
                sample_step = self.batches_done % self.opt.sample_interval == 0
                outs = self.model(batch, training=True, global_step=self.batches_done)

                if not self.opt.dataset.get('use_sampler', False):
                    self.train_metrics.update(outs['preds'], batch['label'])
                losses.append(outs['loss'])

                if sample_step:
                    prog.set_postfix_str('training_loss_{:.5f}'.format(outs['loss'].item()), refresh=True)
        epoch_stats = {'loss': torch.mean(torch.tensor(losses))}
        if not epoch_steps:
            epoch_stats.update(self.train_metrics.compute())
        else:
            epoch_stats[f'{self.task.title()}F1Score'] = 0.0
        self.logger.log(epoch_stats, self.current_epoch, 'train')
        self.logger.info(
            '[Finished training epoch %d/%d] [Epoch loss: %f] [Epoch F1 score: %f]'
            % (self.current_epoch, self.opt.n_epochs, epoch_stats['loss'], epoch_stats[f'{self.task.title()}F1Score'])
        )
        return epoch_stats

    @torch.no_grad()
    def validation_epoch(self, loader: torch.utils.data.DataLoader) -> None:
        self.model.eval()
        self.val_metrics.reset()
        losses = []
        for _, batch in tqdm(enumerate(loader), desc=f'Validation epoch: {self.current_epoch}', leave=False, total=len(loader)):
            batch = {k: v.to(self.device) for k, v in batch.items()}
            print('val', batch['label'].sum())
            outs = self.model(batch, training=False)

            self.val_metrics.update(outs['preds'], batch['label'])
            losses.append(outs['loss'])
        epoch_stats = {'loss': torch.mean(torch.tensor(losses)), **self.val_metrics.compute()}
        self.logger.log(epoch_stats, self.current_epoch, 'val')
        self.logger.info(
            '[Finished validation epoch %d/%d] [Epoch loss: %f] [Epoch F1 score: %f]'
            % (self.current_epoch, self.opt.n_epochs, epoch_stats['loss'], epoch_stats[f'{self.task.title()}F1Score'])
        )
        return epoch_stats
