from pathlib import Path

import albumentations as albu
import nibabel as nib
import numpy as np
import torch
from numpy.lib.format import open_memmap

slicing_dims = {
    'sagittal': 0,  # side view
    'coronal': 1,  # front view
    'axial': 2,  # top down view
}


class CTScan(torch.utils.data.Dataset):
    def __init__(
        self,
        scan_path: Path,
        labels_path: Path,
        transforms: albu.Compose = None,
        min_max_normalization: bool = True,
        slicing_direction: str = 'axial',
        classes: list[str] = ('empty', 'kidney'),
        sampling_class: str = 'kidney',
    ):
        super().__init__()
        self.min_max_norm = min_max_normalization
        self.transforms = transforms
        self.slicing_dim = slicing_dims[slicing_direction]

        self.scan = self.load_volume(scan_path)
        self.labels = self.load_volume(labels_path)

        assert self.scan.shape == self.labels.shape, 'Shapes of provided scan and labels volumes do not match'

        self.sampling_class = sampling_class
        self.classes = classes  # TODO: check out if nnUnet has tumor label as 2 or 3
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

    def load_volume(self, scan_path: Path) -> np.ndarray:
        mmap_path = scan_path.with_name(scan_path.stem + '.npy')
        if scan_path.suffix == '.gz' and not mmap_path.exists():
            ct_scan = nib.load(scan_path)  # loads the file. this usually comes with lots of metadata
            vol = ct_scan.get_fdata()
            vol_mmap = open_memmap(mmap_path, 'w+', np.int32, shape=vol.shape)
            vol_mmap[:] = vol[:].astype(np.int32)
            print(f'Created memory mapped object for: {scan_path}')
            return vol_mmap
        # print(f'Loaded memory mapped object from: {mmap_path}')
        return np.load(mmap_path, mmap_mode='r')

    def get_sampling_labels(self):
        class_id = self.class_to_idx[self.sampling_class]
        dims = tuple(i for i in range(len(self.scan.shape)) if i != self.slicing_dim)
        return (self.labels == class_id).any(axis=dims).astype(np.uint8)

    def _get_slicer(self, index: int) -> np.ndarray:
        if self.slicing_dim == 0:
            return np.s_[index]
        elif self.slicing_dim == 1:
            return np.s_[:, index]
        elif self.slicing_dim == 2:
            return np.s_[:, :, index]
        else:
            raise RuntimeError(f'Unable to slice dimension: {self.slicing_dim}')

    def get_ith_slice(self, volume: np.ndarray, index: int) -> np.ndarray:
        return volume[self._get_slicer(index)]

    def __len__(self):
        return self.scan.shape[self.slicing_dim]

    def __getitem__(self, index):
        scan_slice = self.get_ith_slice(self.scan, index)
        label_slice = self.get_ith_slice(self.labels, index)

        clip_range = np.percentile(scan_slice, q=0.05), np.percentile(scan_slice, q=99.5)
        scan_slice = np.clip(scan_slice, *clip_range)  # normalization

        # normalize image
        smin, smax = scan_slice.min(), scan_slice.max()
        #         print(smin, smax)
        if self.min_max_norm:
            scan_slice = (scan_slice - smin) / max((smax - smin), 1.0)
        else:
            scan_slice = (scan_slice - scan_slice.mean()) / scan_slice.std()
        #         print(scan_slice.min(), scan_slice.max())

        # prepare masks
        if len(self.classes) == 2:
            # binary setting
            masks = [(label_slice == self.class_to_idx[self.classes[-1]]).astype(np.uint8)]
        else:
            # multiclass/multilabel setting
            return NotImplemented
            masks = [(label_slice == self.class_to_idx[cname]).astype(np.uint8) for cname in self.classes]

        sample = {'image': scan_slice, 'masks': masks}
        if self.transforms:
            sample = self.transforms(**sample)

        sample['image'] = torch.from_numpy(sample['image'][np.newaxis])
        sample['masks'] = torch.from_numpy(np.stack(sample['masks']))
        sample['label'] = masks[0].any().astype(np.uint8)  # FIXME
        return sample
