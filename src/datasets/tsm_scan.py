from math import ceil
from pathlib import Path

import albumentations as albu
import cv2
import nibabel as nib
import numpy as np
import torch
from numpy.lib.format import open_memmap

slicing_dims = {
    'sagittal': 0,  # side view
    'coronal': 1,  # front view
    'axial': 2,  # top down view
}


def gaussian_blob(size, mu=0, sigma=0.5):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    d = np.sqrt(x**2 + y**2)
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma**2)))
    return g


def sample_random_mask_point(mask, margin_pct=0.1, cnt=None, rect=None):
    if cnt is None:
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if not cnts:
            return None, None
        cnt = max(cnts, key=cv2.contourArea)
        rect = cv2.boundingRect(cnt)
    x, y, w, h = rect
    offset_x, offset_y = int(w * margin_pct / 2), int(h * margin_pct / 2)
    cx = np.random.randint(x + offset_x, x + w - offset_x)
    cy = np.random.randint(y + offset_y, y + h - offset_y)

    if cv2.pointPolygonTest(cnt, (cx, cy), False) != 1:
        return sample_random_mask_point(mask, margin_pct, cnt, (x, y, w, h))
    return (cx, cy), rect


def paste_image(src, target, center_pt, blend_fn=None):
    (cx, cy), (h, w) = center_pt, src.shape[:2]
    midh, midw = h / 2, w / 2
    top, bottom, left, right = cy - int(midh), cy + ceil(midh), cx - int(midw), cx + ceil(midw)

    if blend_fn is not None:
        target[top:bottom, left:right] = blend_fn(target[top:bottom, left:right], src)
    else:
        target[top:bottom, left:right] |= src
    return target


def increase_brightness(target, brightness_map, center_pt, alpha=0.9, max_value=1.0):
    (cx, cy), (h, w) = center_pt, brightness_map.shape[:2]
    midh, midw = h / 2, w / 2
    top, bottom, left, right = cy - int(midh), cy + ceil(midh), cx - int(midw), cx + ceil(midw)

    brightness_map = brightness_map.astype(np.float32) * alpha
    crop = target[top:bottom, left:right].astype(np.float32)
    crop += brightness_map
    target[top:bottom, left:right] = np.clip(crop, 0, max_value)
    return target


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
        synth_params: dict = None,
        load_masks: bool = False,
    ):
        super().__init__()
        self.min_max_norm = min_max_normalization
        self.transforms = transforms
        self.slicing_dim = slicing_dims[slicing_direction]

        self.name = scan_path.name
        self.scan = self.load_volume(scan_path)
        self.labels = self.load_volume(labels_path)

        assert self.scan.shape == self.labels.shape, 'Shapes of provided scan and labels volumes do not match'

        self.sampling_class = sampling_class
        self.classes = classes  # TODO: check out if nnUnet has tumor label as 2 or 3
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.synth_params = synth_params
        self.anomaly_template = gaussian_blob(size=synth_params['size'], sigma=synth_params['sigma']) if synth_params else None
        self.anomaly_transforms = albu.Compose(
            [
                albu.RandomScale(scale_limit=(-0.5, 0.1), p=1.0),
                # albu.ElasticTransform(p=1.0, sigma=10, alpha_affine=20, border_mode=cv2.BORDER_CONSTANT),
                albu.GridDistortion(num_steps=1, distort_limit=(-0.4, 0.3), border_mode=cv2.BORDER_CONSTANT, value=0),
                albu.Rotate(limit=(-90, 90), border_mode=cv2.BORDER_CONSTANT),
            ]
        )
        self.load_masks = load_masks

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
        if self.min_max_norm:
            scan_slice = (scan_slice - smin) / max((smax - smin), 1.0)
        else:
            # scan_slice = (scan_slice - scan_slice.mean()) / scan_slice.std()
            raise NotImplementedError

        target_mask = (label_slice == self.class_to_idx[self.sampling_class]).astype(np.uint8)

        label = 0  # no anomaly by default

        augment_anomaly = (
            self.synth_params
            and np.random.rand() < self.synth_params['p']
            and target_mask.sum() > (self.anomaly_template.shape[0] * self.anomaly_template.shape[1])
        )

        if augment_anomaly:
            try:
                cpt, rect = sample_random_mask_point(target_mask)
                if rect is not None and (rect[-1] * rect[-2] > (self.anomaly_template.shape[0] * self.anomaly_template.shape[1])):
                    anomaly_blob = self.anomaly_transforms(image=self.anomaly_template)['image']
                    scan_slice = increase_brightness(scan_slice, anomaly_blob, cpt, alpha=self.synth_params.get('alpha', 0.9))
                    # print('augmented')
                    label = 1  # has anomaly
            except Exception:
                print(f'Unable to augment synthetic anomalies for scan {self}. Index - {index}')

        # prepare masks
        masks = []

        if self.load_masks:
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
        sample['masks'] = torch.from_numpy(np.stack(sample['masks'])) if self.load_masks else torch.tensor([])
        # sample['label'] = masks[0].any().astype(np.uint8)  # FIXME
        sample['label'] = torch.tensor(label).long()
        return sample

    def __str__(self) -> str:
        return self.name
