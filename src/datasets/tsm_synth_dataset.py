from itertools import chain
from pathlib import Path

import numpy
import torch
from torch.utils.data import ConcatDataset
from torchsampler import ImbalancedDatasetSampler

from src.datasets.tsm_scan import CTScan

slicing_dims = {
    'sagittal': 0,  # side view
    'coronal': 1,  # front view
    'axial': 2,  # top down view
}


class TSMSyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, scans_dir: Path, labels_dir: Path, **scan_kwargs):
        self.scan_paths = sorted(scans_dir.glob('**/*.gz'))
        self.label_paths = [labels_dir / p.name for p in self.scan_paths]
        self.scans = [CTScan(sp, lp, **scan_kwargs) for sp, lp in zip(self.scan_paths, self.label_paths)]

        assert self.scans, f'No scans found from the directory: {scans_dir}'

        self.scans_dataset = ConcatDataset(self.scans)

    def get_sampling_labels(self):
        return list(chain.from_iterable(scan.get_sampling_labels() for scan in self.scans))

    def __len__(self):
        return len(self.scans_dataset)

    def __getitem__(self, index):
        return self.scans_dataset[index]


def get_totalsegmentor_dataloaders(data_transforms, params, sampler_labels=None):
    scans_train_dir = Path(params.root_dir, 'train').resolve()
    scans_val_dir = Path(params.root_dir, 'test').resolve()
    labels_dir = Path(params.root_dir, 'nnUNet_predictions').resolve()

    assert scans_train_dir.exists() and scans_val_dir.exists() and labels_dir.exists()

    train_data = TSMSyntheticDataset(scans_train_dir, labels_dir, transforms=data_transforms['train'], **params.scan_params)

    sampler_labels_cache = Path(scans_train_dir / 'sampler_labels.npy')
    use_sampler = params.get('use_sampler', True)
    if use_sampler:
        if sampler_labels_cache.exists() and not params.get('reset_sampler', True):
            sampler_labels = numpy.load(sampler_labels_cache)
        else:
            sampler_labels = train_data.get_sampling_labels() if sampler_labels is None else sampler_labels
            numpy.save(sampler_labels_cache, sampler_labels)
            print('Cached training sampler labels at:', sampler_labels_cache)

    train_sampler = ImbalancedDatasetSampler(train_data, labels=sampler_labels) if use_sampler else None
    print('Instantiated training dataset for number of slices:', len(train_data))

    test_data = TSMSyntheticDataset(scans_val_dir, labels_dir, transforms=data_transforms['val'], **params.scan_params)
    print('Instantiated validation dataset for number of slices:', len(test_data))

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
