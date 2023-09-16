from pathlib import Path

import albumentations as albu
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchsampler import ImbalancedDatasetSampler


class LungsDataset(torch.utils.data.Dataset):
    CLASSES = 'normal', 'lung_opacity', 'viral_pneumonia', 'covid'

    def __init__(self, ann_path, transforms=None):
        self.labels, self.image_paths, self.mask_paths = self._load_anns(ann_path)
        self.labels = [self.CLASSES.index(lb) for lb in self.labels]
        self.transforms = transforms

    def _load_anns(self, ann_path):
        with open(ann_path) as fid:
            return zip(*(line.rstrip().split(',') for line in fid))

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        label, img_path, mask_path = self.labels[index], self.image_paths[index], self.mask_paths[index]
        img = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        if len(mask.shape) == 3:
            mask = cv2.cvtColor(mask, cv2.COLOR_RGB2GRAY)

        sample = {
            'image': img,
            'mask': mask,
        }
        if self.transforms:
            sample = self.transforms(**sample)
        sample['label'] = label
        sample['image'] = sample['image'][np.newaxis]
        sample['mask'] = (sample['mask'].clip(0, 1))[np.newaxis]
        return sample

    def get_labels(self):
        # required by https://github.com/ufoym/imbalanced-dataset-sampler/blob/master/torchsampler/imbalanced.py
        return self.labels


def get_transforms(opt):
    if opt.imagenet_norm:
        imagenet_mean = 0.485, 0.456, 0.406
        imagenet_std = 0.229, 0.224, 0.225
        grayscale_coefs = 0.2989, 0.587, 0.114

        grayscale_mean = sum(m * c for m, c in zip(imagenet_mean, grayscale_coefs))
        grayscale_std = sum(m * c for m, c in zip(imagenet_std, grayscale_coefs))

        # imagenet norm stats converted to grayscale
        mean = [grayscale_mean]
        std = [grayscale_std]
    else:
        mean = [0.5]
        std = [0.5]

    train_ops = []
    if 'hflip' in opt.augs:
        train_ops.append(albu.HorizontalFlip(p=0.5))
    if 'vflip' in opt.augs:
        train_ops.append(albu.VerticalFlip(p=0.1))
    if 'shift_scale_rotate' in opt.augs:
        train_ops.append(albu.ShiftScaleRotate(scale_limit=0.1, rotate_limit=10, shift_limit=0.07, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=0))

    data_transforms = {
        'train': albu.Compose(
            [
                *train_ops,
                albu.Resize(*opt.img_size, cv2.INTER_CUBIC),
                albu.Normalize(mean, std),
            ]
        ),
        'val': albu.Compose(
            [
                albu.Resize(*opt.img_size, cv2.INTER_CUBIC),
                albu.Normalize(mean, std),
            ]
        ),
    }
    return data_transforms


def get_dataloaders(data_transforms, params, sampler_labels=None):
    train_data = LungsDataset(Path(params.root_dir, params.annotations.train), data_transforms['train'])
    train_sampler = ImbalancedDatasetSampler(train_data, labels=sampler_labels) if params.get('use_sampler', True) else None

    test_data = LungsDataset(Path(params.root_dir, params.annotations.val), data_transforms['val'])
    train_loader = torch.utils.data.DataLoader(
        train_data,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        batch_size=params.batch_size,
        pin_memory=True,
        num_workers=params.num_workers,
    )
    test_loader = torch.utils.data.DataLoader(
        test_data,
        batch_size=params.batch_size,
        shuffle=params.get('shuffle_test', False),
        pin_memory=True,
        num_workers=params.num_workers,
    )
    return train_loader, test_loader


if __name__ == '__main__':
    DATA_PATH = Path('../COVID-19_Radiography_Dataset_v2/')
    train_dst = LungsDataset(DATA_PATH / 'train.csv')
    sample = train_dst[0]
    plt.imshow(sample['image'][0], cmap='gray')
    plt.title(train_dst.CLASSES[sample['label']])
