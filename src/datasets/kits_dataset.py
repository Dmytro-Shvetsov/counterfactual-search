from itertools import chain
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset

from src.datasets.tsm_scan import CTScan as _CTScan


class CTScan(_CTScan):
    slicing_dims = {
        'axial': 0,  # top down view
        'sagittal': 1,  # side view
        'coronal': 2,  # front view
    }
    
    def __getitem__(self, index):
        s = super().__getitem__(index)
        s['image'] = s['image'].transpose(2, 1).flip(2)
        if s['masks'].shape[0] != 0:
            s['masks'] = s['masks'].transpose(2, 1).flip(2)
        return s
        

class KITSDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir:str, split:str, split_dir:str='splits', limit_scans:int = 99999, **scan_params):
        self.root_dir = Path(root_dir)
        self.split = split
        
        self.ann_path = self.root_dir / split_dir / f'{split}_scans.csv'
        assert self.ann_path.exists()
        with open(self.ann_path, 'r') as fid:
            scan_names = fid.read().splitlines()
        assert scan_names, f'No scans found in split: {self.ann_path}'
        
        self.scans = []
        for i, sn in enumerate(scan_names):
            if i > limit_scans:
                break
            self.scans.append(CTScan(self.root_dir / sn / 'imaging.nii.gz', self.root_dir / sn / 'segmentation.nii.gz', **scan_params))

        self.scans_dataset = ConcatDataset(self.scans)
        self.classes = self.scans[0].classes

    def get_sampling_labels(self):
        lbs = list(chain.from_iterable(scan.get_sampling_labels() for scan in self.scans))
        print(f'[KiTS dataset] Number of slices with positive sampling label:', sum(lbs))
        return lbs

    def __len__(self):
        return len(self.scans_dataset)

    def __getitem__(self, index):
        return self.scans_dataset[index]


# def get_kits_dataloaders(data_transforms, params, sampler_labels=None):
#     train_data = KITSDataset(params.root_dir, 'train', params.split_dir, transforms=data_transforms['train'], **params.scan_params)

#     sampler_labels_cache = Path(params.root_dir, params.split_dir, 'sampler_labels.npy')
#     use_sampler = params.get('use_sampler', True)
#     if use_sampler:
#         if sampler_labels_cache.exists() and not params.get('reset_sampler', True):
#             sampler_labels = numpy.load(sampler_labels_cache)
#         else:
#             sampler_labels = train_data.get_sampling_labels() if sampler_labels is None else sampler_labels
#             numpy.save(sampler_labels_cache, sampler_labels)
#             print('Cached training sampler labels at:', sampler_labels_cache)

#     train_sampler = ImbalancedDatasetSampler(train_data, labels=sampler_labels) if use_sampler else None
#     print('Instantiated training dataset for number of slices:', len(train_data))

#     test_data = KITSDataset(params.root_dir, 'test', params.split_dir, transforms=data_transforms['val'], **params.scan_params)
#     print('Instantiated validation dataset for number of slices:', len(test_data))

#     train_loader = torch.utils.data.DataLoader(
#         train_data,
#         sampler=train_sampler,
#         shuffle=train_sampler is None,
#         batch_size=params.batch_size,
#         pin_memory=True,
#         num_workers=params.num_workers,
#     )
#     test_loader = torch.utils.data.DataLoader(
#         test_data,
#         batch_size=params.batch_size,
#         shuffle=params.get('shuffle_test', False),
#         pin_memory=True,
#         num_workers=params.num_workers,
#     )
#     return train_loader, test_loader
