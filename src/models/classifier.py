from typing import List

import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm


def build_classifier(kind, n_classes, pretrained=True, restore_ckpt=None, in_channels=1):
    if kind == 'resnet18':
        model = models.resnet18(pretrained=pretrained)
    elif kind == 'resnet34':
        model = models.resnet34(pretrained=pretrained)
    elif kind == 'resnet50':
        model = models.resnet50(pretrained=pretrained)
    else:
        raise NotImplementedError(kind)
    in_conv = model.conv1
    # reset input layer to accept grayscale images
    model.conv1 = torch.nn.Conv2d(
        in_channels,
        in_conv.out_channels,
        in_conv.kernel_size,
        in_conv.stride,
        in_conv.padding,
        in_conv.dilation,
        in_conv.groups,
        in_conv.bias,
        in_conv.padding_mode,
    )
    model.conv1.weight.data.copy_(in_conv.weight.mean(dim=1, keepdims=True))
    if in_conv.bias is not None:
        model.conv1.bias.data.copy_(in_conv.bias)

    # reset the classification layer
    model.fc = nn.Linear(model.fc.in_features, n_classes)
    if restore_ckpt is not None:
        state = torch.load(restore_ckpt)
        if 'model' in state:
            state = {k.replace('model.', ''):v for k, v in state['model'].items()}
        model.load_state_dict(state)
        print(f'Restored the classifier from: {restore_ckpt}')
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return model


@torch.no_grad()
def predict_probs(loader: torch.utils.data.DataLoader, model: torch.nn.Module, task: str = 'multiclass') -> List[float]:
    model.eval()
    probs = []
    classes = []
    device = next(iter(model.parameters())).device
    for batch in tqdm(loader, total=len(loader), desc='Precomputing posterior probabilities'):
        inputs, labels = batch['image'].to(device), batch['label'].to(device)
        outputs = model(inputs)
        preds = outputs.softmax(1) if task == 'multiclass' else outputs.sigmoid()
        probs.extend(preds.cpu().numpy())
        classes.extend(labels.cpu().numpy())
    return probs, classes


def compute_sampler_condition_labels(probs: List[float], explain_class_idx: int, num_bins: int) -> List[int]:
    """Converts continious probabilities into discrete bin indices distributed on the 0-1 range."""
    pbs = [item[explain_class_idx] for item in probs]
    bin_step = 1 / num_bins
    bins = np.arange(0, 1, bin_step)
    bin_ids = np.digitize(pbs, bins=bins) - 1
    return bin_ids


class ClassificationModel(torch.nn.Module):
    def __init__(self, img_size, opt, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.n_classes = opt.n_classes
        self.in_channels = opt.in_channels
        self.model = build_classifier(opt.kind, opt.n_classes, pretrained=False)
        self.loss = torch.nn.BCEWithLogitsLoss() if opt.n_classes == 1 else torch.nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

    def forward(self, batch, training=False, *args, **kwargs):
        # batch_size = imgs.shape[0]

        inputs, labels = batch['image'], batch['label']

        if training:
            self.optimizer.zero_grad()
        outputs = self.model(inputs)  # B x n_classes
        loss = self.loss(outputs, labels.float().unsqueeze(1))

        if training:
            loss.backward()
            self.optimizer.step()

        outs = {'loss': loss, 'preds': outputs.squeeze()}
        return outs
