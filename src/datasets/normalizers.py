from abc import ABC, abstractmethod
from json import load
from typing import Type

import numpy as np
from numpy import number


class ImageNormalization(ABC):
    def __init__(self, use_mask_for_norm: bool = None, fingerprint_path: str = None,
                 target_dtype: Type[number] = np.float32):
        assert use_mask_for_norm is None or isinstance(use_mask_for_norm, bool)
        self.use_mask_for_norm = use_mask_for_norm
        
        if fingerprint_path is not None:
            with open(fingerprint_path) as fid:
                self.intensity_properties = load(fid)
        else:
            self.intensity_properties = None
        self.target_dtype = target_dtype

    @abstractmethod
    def __call__(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        """
        Image and seg must have the same shape. Seg is not always used
        """
        pass


class CTNormalization(ImageNormalization):
    def __init__(self, rescale=False, use_mask_for_norm: bool = None, fingerprint_path: str = None, target_dtype: Type[number] = np.float32):
        super().__init__(use_mask_for_norm, fingerprint_path, target_dtype)
        self.rescale = rescale
    
    @staticmethod
    def scale_array(unscaled, to_min, to_max, from_min, from_max):
        return (to_max - to_min) * (unscaled - from_min) / (from_max - from_min) + to_min
    
    def __call__(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert self.intensity_properties is not None, "CTNormalization requires intensity properties"
        image = image.astype(self.target_dtype)
        mean_intensity = self.intensity_properties['mean']
        std_intensity = self.intensity_properties['std']
        lower_bound = self.intensity_properties['percentile_00_5']
        upper_bound = self.intensity_properties['percentile_99_5']
        image = np.clip(image, lower_bound, upper_bound)
        image = (image - mean_intensity) / max(std_intensity, 1e-8)
        
        if self.rescale:
            current_min = (lower_bound - mean_intensity) / std_intensity
            current_max = (upper_bound - mean_intensity) / std_intensity
            # [-1; 1] range
            image = self.scale_array(image, -1.0, 1.0, current_min, current_max)
        return image


class CTWindowNormalization(ImageNormalization):
    def __init__(self, rescale=False, use_mask_for_norm: bool = None, fingerprint_path: str = None, target_dtype: Type[number] = np.float32):
        super().__init__(use_mask_for_norm, fingerprint_path, target_dtype)
        self.rescale = rescale
    
    @staticmethod
    def scale_array(unscaled, to_min, to_max, from_min, from_max):
        return (to_max - to_min) * (unscaled - from_min) / (from_max - from_min) + to_min
    
    def __call__(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        assert self.intensity_properties is not None, "CTNormalization requires intensity properties"
        image = image.astype(self.target_dtype)
        lower_bound = self.intensity_properties['percentile_00_5']
        upper_bound = self.intensity_properties['percentile_99_5']
        image = np.clip(image, lower_bound, upper_bound)
        
        if self.rescale:
            # [-1; 1] range
            image = self.scale_array(image, -1.0, 1.0, lower_bound, upper_bound)
        return image


class MinMaxNormalization(ImageNormalization):
    def __call__(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        clip_range = np.percentile(image, q=0.05), np.percentile(image, q=99.5)
        image = np.clip(image, *clip_range)  # normalization
        smin, smax = image.min(), image.max()
        image = (image - smin) / max((smax - smin), 1.0)
        return image


class NoNormalization(ImageNormalization):
    def __call__(self, image: np.ndarray, seg: np.ndarray = None) -> np.ndarray:
        return image


def get_normalization_scheme(kind:str, *args, **kwargs):
    if kind == 'minmax':
        return MinMaxNormalization(*args, **kwargs)
    elif kind == 'ct':
        return CTNormalization(*args, **kwargs)
    elif kind == 'ct-window':
        return CTWindowNormalization(*args, **kwargs)
    elif kind == 'identity':
        return NoNormalization(*args, **kwargs)
    else:
        raise ValueError(f'Unsupported normalization scheme: {kind}')
