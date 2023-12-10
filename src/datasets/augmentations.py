import albumentations as albu
import cv2
import numpy as np
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.transforms.abstract_transforms import AbstractTransform, Compose
from batchgenerators.transforms.color_transforms import BrightnessMultiplicativeTransform, ContrastAugmentationTransform, GammaTransform
from batchgenerators.transforms.noise_transforms import GaussianBlurTransform, GaussianNoiseTransform
from batchgenerators.transforms.resample_transforms import SimulateLowResolutionTransform
from batchgenerators.transforms.spatial_transforms import MirrorTransform, SpatialTransform
from batchgenerators.transforms.utility_transforms import NumpyToTensor, RemoveLabelTransform, RenameTransform


def get_transforms(opt):
    max_pixel_value = 255 if opt.kind.endswith('lungs') else 1.0
    if opt.imagenet_norm:
        imagenet_mean = 0.485, 0.456, 0.406
        imagenet_std = 0.229, 0.224, 0.225
        # grayscale_coefs = 0.2989, 0.587, 0.114
        grayscale_coefs = 1, 1, 1

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
                albu.Resize(*opt.img_size, cv2.INTER_LINEAR),
                # albu.Normalize(mean, std, max_pixel_value=max_pixel_value),
            ]
        ),
        'val': albu.Compose(
            [
                albu.Resize(*opt.img_size, cv2.INTER_LINEAR),
                # albu.Normalize(mean, std, max_pixel_value=max_pixel_value),
            ]
        ),
    }
    return data_transforms


def nnunet_transforms(opt):
    tr_transforms = []

    patch_size_spatial = opt.img_size
    rotation_for_DA = {
        'x': (-np.pi, np.pi),
        'y': 0,
        'z': 0
    }
    order_resampling_data = 3
    # border_val_seg = -1
    border_val_seg = 0
    order_resampling_seg = 1
    ignore_axes=None

    mirror_axes=(0, 1)
    data_key, seg_key = 'image', 'masks'

    tr_transforms.append(SpatialTransform(
        patch_size_spatial, patch_center_dist_from_border=None,
        do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
        do_rotation=False, angle_x=0, angle_y=0, angle_z=0,
        p_rot_per_axis=0,
        do_scale=False,
        border_mode_data="constant", border_cval_data=0, order_data=order_resampling_data,
        border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=order_resampling_seg,
        random_crop=False, 
        p_el_per_sample=0, p_scale_per_sample=0, p_rot_per_sample=0,
        independent_scale_for_each_axis=False,
        data_key=data_key, seg_key=seg_key,
    ))

    tr_transforms.append(GaussianNoiseTransform(p_per_sample=0.1, data_key=data_key, seg_key=seg_key))
    tr_transforms.append(GaussianBlurTransform((0.5, 1.), different_sigma_per_channel=True, p_per_sample=0.2,
                                               p_per_channel=0.5, data_key=data_key, seg_key=seg_key))
    tr_transforms.append(BrightnessMultiplicativeTransform(multiplier_range=(0.75, 1.25), p_per_sample=0.15, data_key=data_key, seg_key=seg_key))
    tr_transforms.append(ContrastAugmentationTransform(p_per_sample=0.15, data_key=data_key, seg_key=seg_key))
    tr_transforms.append(SimulateLowResolutionTransform(zoom_range=(0.5, 1), per_channel=True,
                                                        p_per_channel=0.5,
                                                        order_downsample=0, order_upsample=3, p_per_sample=0.25,
                                                        ignore_axes=ignore_axes,
                                                        data_key=data_key, seg_key=seg_key))
    tr_transforms.append(GammaTransform((0.7, 1.5), True, True, retain_stats=True, p_per_sample=0.1, data_key=data_key, seg_key=seg_key))
    tr_transforms.append(GammaTransform((0.7, 1.5), False, True, retain_stats=True, p_per_sample=0.3, data_key=data_key, seg_key=seg_key))

    if mirror_axes is not None and len(mirror_axes) > 0:
        tr_transforms.append(MirrorTransform(mirror_axes, data_key=data_key, seg_key=seg_key))

    # tr_transforms.append(RemoveLabelTransform(-1, 0))

    # tr_transforms.append(RenameTransform('seg', 'target', True))

    # if regions is not None:
    #     # the ignore label must also be converted
    #     tr_transforms.append(ConvertSegmentationToRegionsTransform(list(regions) + [ignore_label]
    #                                                                if ignore_label is not None else regions,
    #                                                                'target', 'target'))

    # if deep_supervision_scales is not None:
    #     tr_transforms.append(DownsampleSegForDSTransform2(deep_supervision_scales, 0, input_key='target',
    #                                                       output_key='target'))
    # tr_transforms.append(NumpyToTensor(['data', 'target'], 'float'))
    
    val_transforms = [SpatialTransform(
        patch_size_spatial, patch_center_dist_from_border=None,
        do_elastic_deform=False, alpha=(0, 0), sigma=(0, 0),
        do_rotation=True, angle_x=rotation_for_DA['x'], angle_y=rotation_for_DA['y'], angle_z=rotation_for_DA['z'],
        p_rot_per_axis=1,  # todo experiment with this
        do_scale=True, scale=(0.7, 1.4),
        border_mode_data="constant", border_cval_data=0, order_data=order_resampling_data,
        border_mode_seg="constant", border_cval_seg=border_val_seg, order_seg=order_resampling_seg,
        random_crop=False,  # random cropping is part of our dataloaders
        p_el_per_sample=0, p_scale_per_sample=0.2, p_rot_per_sample=0.2,
        independent_scale_for_each_axis=False,  # todo experiment with this
        data_key=data_key, seg_key=seg_key,
    )]
    return {
        'train': Compose(tr_transforms),
        'val': Compose(val_transforms),
    }
