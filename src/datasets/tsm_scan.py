from math import ceil
from pathlib import Path
from random import choice
from typing import Union

import albumentations as albu
import cv2
import nibabel as nib
import numpy as np
import torch
from numpy.lib.format import open_memmap

from src.datasets.normalizers import ImageNormalization, get_normalization_scheme


def gaussian_blob(size, mu=0, sigma=0.5):
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    d = np.sqrt(x**2 + y**2)
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma**2)))
    return g


def sample_random_mask_point(mask, margin_pct=0.1, cnt=None, rect=None, num_comps_choose_from=2):
    if cnt is None:
        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        if not cnts:
            return None, None
        cnt = choice(sorted(cnts, key=cv2.contourArea)[-num_comps_choose_from:])
        rect = cv2.boundingRect(cnt)
    x, y, w, h = rect
    offset_x, offset_y = int(w * margin_pct / 2), int(h * margin_pct / 2)
    cx = np.random.randint(x + offset_x, x + w - offset_x)
    cy = np.random.randint(y + offset_y, y + h - offset_y)

    if cv2.pointPolygonTest(cnt, (cx, cy), False) != 1:
        return sample_random_mask_point(mask, margin_pct, cnt, rect, num_comps_choose_from)
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
    top, bottom = max(cy - int(midh), 0), min(cy + ceil(midh), target.shape[0])
    left, right = max(cx - int(midw), 0), min(cx + ceil(midw), target.shape[1])

    brightness_map = brightness_map.astype(np.float32) * alpha
    crop = target[top:bottom, left:right].astype(np.float32)
    crop += brightness_map
    target[top:bottom, left:right] = np.clip(crop, 0, max_value)
    return target


def paste_mask(target, mask, center_pt):
    (cx, cy), (h, w) = center_pt, mask.shape[:2]
    midh, midw = h / 2, w / 2
    top, bottom = max(cy - int(midh), 0), min(cy + ceil(midh), target.shape[0])
    left, right = max(cx - int(midw), 0), min(cx + ceil(midw), target.shape[1])

    target[top:bottom, left:right] = np.maximum(target[top:bottom, left:right], mask)
    return target


class CTScan(torch.utils.data.Dataset):
    slicing_dims = {
        'sagittal': 0,  # side view
        'coronal': 1,  # front view
        'axial': 2,  # top down view
    }

    def __init__(
        self,
        scan_path: Path,
        labels_path: Path,
        norm_scheme: dict = {'kind': 'minmax'},
        transforms: albu.Compose = None,
        min_max_normalization: bool = True,
        slicing_direction: str = 'axial',
        classes: list[str] = ('empty', 'kidney'),
        sampling_class: str = None,
        classify_labels: str = None,
        classify_labels_thresh: int = 32,
        filter_class_slices: str = None,
        filter_class_slices_thresh: int = 32,
        synth_params: dict = None,
        load_masks: Union[bool, list] = False,
        default_label: int = None,
        blur_background_sigma: int = None,
    ):
        super().__init__()
        self.min_max_norm = min_max_normalization
        self.transforms = transforms
        self.slicing_dim = self.slicing_dims[slicing_direction]

        self.name = scan_path.stem
        self.scan_path = scan_path
        self.scan = self.load_volume(scan_path, np.int16)

        self.labels_path = labels_path
        self.segm = self.load_volume(labels_path, np.uint8)

        assert self.scan.shape == self.segm.shape, 'Shapes of provided scan and labels volumes do not match'

        self.sampling_class = sampling_class
        
        self.classify_labels = classify_labels
        self.classify_labels_thresh = classify_labels_thresh
        self.filter_class_slices = filter_class_slices
        self.filter_class_slices_thresh = filter_class_slices_thresh
        self.classes = classes
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}

        self.synth_params = synth_params
        self.anomaly_template, self.anomaly_transforms = None, None
        if synth_params:
            self.anomaly_template = gaussian_blob(size=synth_params['size'], sigma=synth_params['sigma'])
            self.anomaly_template_mask = (self.anomaly_template > synth_params.get('mask_thresh', 0.7)).astype(np.uint8)
            self.anomaly_transforms = albu.Compose(
                [
                    albu.RandomScale(scale_limit=(-0.25, 0), p=1.0),
                    # albu.ElasticTransform(p=1.0, sigma=10, alpha_affine=20, border_mode=cv2.BORDER_CONSTANT),
                    albu.GridDistortion(num_steps=1, distort_limit=(-0.2, 0.2), border_mode=cv2.BORDER_CONSTANT, value=0),
                    albu.Rotate(limit=(-90, 90), border_mode=cv2.BORDER_CONSTANT),
                ]
            )
            self.classes = self.classes.copy() + ['gaussian']
            self.class_to_idx['gaussian'] = len(self.classes) - 1
        self.load_masks = load_masks

        self.slice_indices = None
        if self.filter_class_slices is not None:
            # TODO: add caching
            self.slice_indices = self.get_slice_indices(filter_class_slices)

        self.labels = None
        if self.classify_labels:
            self.labels = self._get_slices_with_classes_mask(classify_labels, classify_labels_thresh).astype(np.uint8)
            # if self.slice_indices.shape[0]:
                # print(self.labels.shape, self.scan.shape, self.slice_indices.min(), self.slice_indices.max())
            if self.slice_indices is not None:
                self.labels = self.labels[self.slice_indices]
        self.default_label = default_label
        self.blur_background_sigma = blur_background_sigma
        self.norm = get_normalization_scheme(**norm_scheme)

    def load_volume(self, scan_path: Path, dtype=np.int32) -> np.ndarray:
        mmap_path = scan_path.with_name(scan_path.stem + '.npy')
        if scan_path.suffix == '.gz' and not mmap_path.exists():
            ct_scan = nib.load(scan_path, mmap=True)
            vol_mmap = open_memmap(mmap_path, 'w+', dtype, shape=ct_scan.shape)
            vol_mmap[:] = ct_scan.get_fdata(dtype=np.float32, caching='unchanged').astype(dtype)
            print(f'Created memory mapped object for: {scan_path}')
            return vol_mmap
        # print(f'Loaded memory mapped object from: {mmap_path}')
        return np.load(mmap_path, mmap_mode='r')

    def _get_slices_with_classes_mask(self, class_names, th=0):
        labels_cache = self.labels_path.with_name(self.name + '_' + '_'.join(class_names) + f'_thresh_{th}.npy')
        # print(labels_cache)
        if labels_cache.exists():
            return np.load(labels_cache)
        
        dims = tuple(i for i in range(len(self.scan.shape)) if i != self.slicing_dim)
        filter_mask = None
        for c in class_names:
            cm = (self.segm == self.class_to_idx[c]).sum(axis=dims) > th
            filter_mask = cm if filter_mask is None else (filter_mask | cm)
        
        # array of booleans
        np.save(labels_cache, filter_mask)
        return filter_mask

    def get_sampling_labels(self):
        if self.labels is not None:
            return self.labels
        
        if self.default_label is not None:
            # all slices have default label
            return np.zeros(len(self), np.uint8)

        assert self.sampling_class, f'Unable to determine sampling labels as sampling class is not set'
        filter_mask = self._get_slices_with_classes_mask(self.sampling_class)

        sampling_labels = filter_mask.astype(np.uint8)
        if self.slice_indices is not None:
            sampling_labels = sampling_labels[self.slice_indices]
        return sampling_labels

    def get_slice_indices(self, filter_classes:list[str]) -> list[int]:
        filter_mask = self._get_slices_with_classes_mask(filter_classes, self.filter_class_slices_thresh)
        indices = np.nonzero(filter_mask)[0]
        if indices.shape[0] == 0:
            print(f'{self} has no slices for classes: {filter_classes}')
        return indices

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
        return len(self.slice_indices) if self.slice_indices is not None else self.scan.shape[self.slicing_dim]

    def __getitem__(self, index):
        slice_index = self.slice_indices[index] if self.slice_indices is not None else index
        try:
            scan_slice = self.get_ith_slice(self.scan, slice_index)
            label_slice = self.get_ith_slice(self.segm, slice_index)
        except IndexError:
            print(f'Unable to load {slice_index} slice - {self} ({index} dataset item)')
            exit()

        # prepare masks
        masks = []
        if self.load_masks:
            # TODO: fix reconstruction loss
            # multiclass/multilabel setting
            classes = self.load_masks if isinstance(self.load_masks, list) else self.classes
            masks = [(label_slice == self.class_to_idx[c]).astype(np.uint8) if c != 'zero_mask' else np.zeros_like(label_slice, np.uint8) 
                     for c in classes if c != 'gaussian']

        # normalize image
        scan_slice = self.norm(scan_slice, label_slice)

        # label determines whether an anomaly is present in the slice or not
        if self.synth_params:
            scan_slice = (scan_slice + 1) / 2
            target_mask = masks[-1] if masks else label_slice.astype(np.uint8)
            label, scan_slice, anomaly_mask = self._augment_synthetic_anomaly(index, scan_slice, target_mask, bool(masks))
            if anomaly_mask is not None:
                masks.append(anomaly_mask)
            scan_slice = 2*scan_slice  - 1
        else:
            if self.labels is None and self.default_label is not None:
                label = self.default_label
            else:
                assert self.classify_labels is not None, 'Unable to determine the anomaly label for the slice. Set synthetic augmentation of classify_labels parameters'
                # print(len(self), self.labels.shape, index)
                label = self.labels[index]
        
        if self.blur_background_sigma is not None:
            assert self.load_masks
            bg_mask = masks[0]
            bg_mask = cv2.erode(bg_mask, np.ones((30, 30), np.uint8), iterations=1, borderValue=0, borderType=cv2.BORDER_CONSTANT).astype(bool)
            bg_blur = cv2.GaussianBlur(scan_slice, (5, 5), self.blur_background_sigma)
            scan_slice[bg_mask] = bg_blur[bg_mask]
        
        sample = {'image': scan_slice, 'masks': masks}

        if self.transforms:
            sample = self.transforms(**sample)

        sample['image'] = torch.from_numpy(sample['image'][np.newaxis])
        sample['masks'] = torch.from_numpy(np.stack(sample['masks'])) if self.load_masks else torch.tensor([])
        sample['label'] = torch.tensor(label).long()
        sample['scan_name'] = self.name
        sample['slice_index'] = slice_index
        return sample

    def _augment_synthetic_anomaly(self, index:int, scan_slice, target_mask:np.ndarray, ret_seg:bool = False):
        assert len(self.classes) == 3
        label = 0  # no anomaly by default
        anomaly_mask = np.zeros_like(target_mask) if ret_seg else None
        if np.random.rand() < self.synth_params['p']:
            try:
                # randomly sample a point in one of the kidneys
                cpt, rect = sample_random_mask_point(target_mask, num_comps_choose_from=2)
                if rect is not None and (rect[-1] * rect[-2] > (self.anomaly_template.shape[0] * self.anomaly_template.shape[1])):
                    anomaly_blob = self.anomaly_transforms(image=self.anomaly_template, mask=self.anomaly_template_mask)
                    blob_img, blob_mask = anomaly_blob['image'], anomaly_blob['mask']
                    scan_slice = increase_brightness(scan_slice, blob_img, cpt, alpha=self.synth_params.get('alpha', 0.9))
                    if anomaly_mask is not None:
                        anomaly_mask = paste_mask(anomaly_mask, blob_mask, cpt)
                    label = 1 # has anomaly
            except (RecursionError, ValueError):
                pass
            except Exception as exc:
                print(f'Unable to augment synthetic anomalies for scan {self}. Index - {index}')
                raise exc
        return label, scan_slice, anomaly_mask

    def __str__(self) -> str:
        return self.name
