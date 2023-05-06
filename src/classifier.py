from typing import List
import numpy as np
import torch
import torch.nn as nn
from torchvision import models
from tqdm import tqdm


def build_classifier(num_classes, pretrained=True, restore_ckpt=None):
    model = models.resnet18(pretrained=pretrained) # Returns Defined Densenet model with weights trained on ImageNet
    in_conv = model.conv1
    # reset input layer to accept grayscale images
    model.conv1 = torch.nn.Conv2d(1, in_conv.out_channels, in_conv.kernel_size, in_conv.stride, 
                                  in_conv.padding, in_conv.dilation, in_conv.groups, in_conv.bias, in_conv.padding_mode)
    model.conv1.weight.data.copy_(in_conv.weight.mean(dim=1, keepdims=True))
    if in_conv.bias is not None:
        model.conv1.bias.data.copy_(in_conv.bias )

    # reset the classification layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    if restore_ckpt is not None:
        model.load_state_dict(torch.load(restore_ckpt))
        print(f'Restored the classifier from: {restore_ckpt}')
    model.to('cuda' if torch.cuda.is_available() else 'cpu')
    return model


@torch.no_grad()
def predict_probs(loader:torch.utils.data.DataLoader, model:torch.nn.Module) -> List[float]:
    model.eval()
    probs = []
    classes = []
    device = next(iter(model.parameters())).device
    for i, batch in tqdm(enumerate(loader), total=len(loader), desc='Precomputing posterior probabilities'):
        inputs, labels = batch['image'].to(device), batch['label'].to(device)
        outputs = model(inputs)
        preds = outputs.softmax(1)
        probs.extend(preds.cpu().numpy())
        classes.extend(labels.cpu().numpy())
    return probs, classes


def compute_sampler_condition_labels(probs:List[float], explain_class_idx:int, num_bins:int) -> List[int]:
    pbs = [item[explain_class_idx] for item in probs]
    bin_step = 1 / num_bins
    bins = np.arange(0, 1, bin_step)
    bin_ids = np.digitize(pbs, bins=bins) - 1
    return bin_ids
