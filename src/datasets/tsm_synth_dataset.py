from itertools import chain
from pathlib import Path

import torch
from torch.utils.data import ConcatDataset

from src.datasets.tsm_scan import CTScan

slicing_dims = {
    'sagittal': 0,  # side view
    'coronal': 1,  # front view
    'axial': 2,  # top down view
}


class TSMSyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir:str, split:str, labels_dir:str='nnUNet_predictions', limit_scans: int = 99999, **scan_params):
        self.root_dir = root_dir
        self.split = split
        scans_dir = Path(root_dir, split).resolve()
        labels_dir = Path(root_dir, labels_dir).resolve()
        assert scans_dir.exists() and labels_dir.exists()
        self.scan_paths = sorted(scans_dir.glob('**/*.nii.gz'))
        self.label_paths = [labels_dir / p.name for p in self.scan_paths]

        self.scans = []
        for i, (sp, lp) in enumerate(zip(self.scan_paths, self.label_paths)):
            if i > limit_scans:
                break
            self.scans.append(CTScan(sp, lp, **scan_params))
        assert self.scans, f'No scans found from the directory: {scans_dir}'
        self.scans_dataset = ConcatDataset(self.scans)
        self.classes = self.scans[0].classes

    def get_sampling_labels(self):
        lbs = list(chain.from_iterable(scan.get_sampling_labels() for scan in self.scans))
        print(f'[Totalsegmentor dataset] Number of slices with positive sampling label:', sum(lbs))
        return lbs

    def __len__(self):
        return len(self.scans_dataset)

    def __getitem__(self, index):
        return self.scans_dataset[index]
