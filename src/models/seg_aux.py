import segmentation_models_pytorch as smp
import torch
from pytorch_toolbelt.losses import BinaryFocalLoss, CrossEntropyFocalLoss, JaccardLoss, JointLoss, WeightedLoss


class SegmentationAuxModel(torch.nn.Module):
    def __init__(self, img_size, opt, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_classes = opt.n_classes
        self.in_channels = opt.in_channels

        self.aux_output = opt.get('aux_output', False)

        self.segment_mask_ids = opt.segment_mask_ids
        self.n_masks = len(self.segment_mask_ids)
        self.aux_params = opt.get('aux_params', dict(
            pooling='max', # one of 'avg', 'max'
            dropout=0.3, # dropout ratio, default is None
            activation=None, # activation function
            classes=opt.n_classes, # define number of output labels
        ))

        self.model = smp.create_model(
            opt.kind, 
            encoder_name=opt.encoder_name,
            encoder_weights=opt.encoder_weights,
            in_channels=opt.in_channels,
            classes=self.n_masks,
            aux_params=self.aux_params if self.aux_output else None,
        )
        if opt.n_classes == 1:
            if self.n_masks == 1:
                self.seg_loss = {
                    'loss_seg_focal': WeightedLoss(BinaryFocalLoss(), 0.9),
                    'loss_seg_iou': WeightedLoss(JaccardLoss(mode='binary', from_logits=True), 0.1),
                }
            else:
                self.seg_loss = {
                    'loss_seg_focal': WeightedLoss(CrossEntropyFocalLoss(), 0.9),
                    'loss_seg_iou': WeightedLoss(JaccardLoss(mode='multiclass', from_logits=True), 0.1),
                }
            self.cls_loss = torch.nn.BCEWithLogitsLoss()
        else:
            raise NotImplementedError
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2), weight_decay=opt.get('weight_decay', 0.0))

    def forward(self, batch, training=False, *args, **kwargs):
        inputs, masks, labels = batch['image'], batch['masks'], batch['label']
        bs = inputs.shape[0]

        # select only specific classes for segmentation
        masks = masks[:, self.segment_mask_ids]

        if self.n_masks > 1:
            masks = masks.argmax(1)
        
        if training:
            self.optimizer.zero_grad()

        # classification logits -  # B x n_cls_classes
        # segmentation logits -  # B x n_masks x H x W
        outputs = self.model(inputs)  # B x n_classes
        
        loss = 0
        loss_logs = {}
        
        if self.aux_output:
            mask_preds, cls_preds = outputs
            # classification loss
            loss_logs['loss_cls'] = cls_loss = self.cls_loss(cls_preds, labels.float().unsqueeze(1))
            loss += cls_loss
        else:
            mask_preds, cls_preds = outputs, None

        # segmentation loss
        for k, seg_loss in self.seg_loss.items():
            loss_logs[k] = loss_term = seg_loss(mask_preds, masks)
            loss += loss_term

        if training:
            loss.backward()
            self.optimizer.step()

        loss_logs['loss_total'] = loss
        return {'loss': loss_logs, 'mask_preds': mask_preds, 'cls_preds': cls_preds.squeeze(1) if cls_preds is not None else []}
