import argparse
import logging
from json import dump
from pathlib import Path
from pprint import pprint

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from tqdm import tqdm

from src.datasets import get_dataloaders
from src.datasets.augmentations import get_transforms
from src.datasets.tsm_scan import CTScan
from src.utils.generic_utils import seed_everything

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str, required=True, help='Configuration file path to start training from scratch')
parser.add_argument('-o', '--output_json_path', type=str, default='dataset_fingerprint.json', help='Output JSON path to save the results')
parser.add_argument('-fg', '--foreground_labels', type=int, nargs='+', default=[], help='Labels to be considered a foreground for computing the stats')
opt = parser.parse_args()


# def create_nonzero_mask(data:np.ndarray, dim:np.ndarray=0):
#     """
#     :return: the mask is True where the data is nonzero
#     """
#     from scipy.ndimage import binary_fill_holes
#     assert data.ndim == 3, "data must have shape (X, Y, Z)"
#     nonzero_mask = (data != 0).any(axis=dim)
#     nonzero_mask = binary_fill_holes(nonzero_mask)
#     return np.expand_dims(nonzero_mask, dim)


def collect_foreground_intensities(image: np.ndarray, segmentation: np.ndarray, num_samples:int) -> list[float]:
    assert not np.any(np.isnan(image)), "Images contains NaN values. grrrr.... :-("
    assert not np.any(np.isnan(segmentation)), "Segmentation contains NaN values. grrrr.... :-("

    foreground_pixels = image[segmentation > 0]
    num_fg = len(foreground_pixels)
    if num_fg == 0:
        logging.warning('Encountered scan with no foreground pixels')
        return []

    rs = np.random.RandomState(1234)
    intensities_per_channel = rs.choice(foreground_pixels, num_samples, replace=True) if num_fg > 0 else []
    return intensities_per_channel


def analyze_case(image_data: np.ndarray, segmentation_data:np.ndarray, slicing_dim:int, num_samples):
    # TODO: think about cropping intencities within zero-signal areas of the image_data for other datasets
    # data_cropped, seg_cropped, bbox = crop_to_nonzero(image_data, segmentation_data, slicing_dim)
    # spacing = properties_images['spacing']
    return collect_foreground_intensities(image_data, segmentation_data, num_samples)

            
def extract_fingerprint(datasets:list[torch.utils.data.Dataset], output_json_path:str, foreground_labels:list[int] = None) -> None:
    intencities_per_channel = []
    num_scans = sum(len(d.scans) for d in datasets)
    num_fg_samples_per_case = int(10e7 // num_scans)
    logging.info(f'Number of foreground pixels sampled per scan: {num_fg_samples_per_case} (num_scans={num_scans})')
    for dataset in datasets:
        assert hasattr(dataset, 'scans')
        scans = list(dataset.scans)
        s: CTScan
        for s in tqdm(scans):
            # clipping is done to filter out possible false positives produced by nnUNet in totalsegmentor dataset
            segm = s.segm.clip(0, len(s.classes) - 1)
            if foreground_labels:
                remove_labels = [i for i in range(len(s.classes) - 1) if i not in foreground_labels]
                for lb in remove_labels:
                    segm[segm == lb] = 0
            intencities = analyze_case(s.scan, segm, s.slicing_dim, num_fg_samples_per_case)
            intencities_per_channel.extend(intencities)

    fingerprint = {
        'mean': float(np.mean(intencities_per_channel)),
        'median': float(np.median(intencities_per_channel)),
        'std': float(np.std(intencities_per_channel)),
        'min': float(np.min(intencities_per_channel)),
        'max': float(np.max(intencities_per_channel)),
        'percentile_99_5': float(np.percentile(intencities_per_channel, 99.5)),
        'percentile_00_5': float(np.percentile(intencities_per_channel, 0.5)),
    }

    pprint(fingerprint)
    
    output_json_path = Path(output_json_path)
    if output_json_path.suffix:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
    else:
        output_json_path /= 'dataset_fingerprint.json'

    with open(output_json_path, 'w') as fid:
        dump(fingerprint, fid, indent=4)
    return output_json_path


def main(args):
    with open(args.config_path) as fid:
        opt = yaml.safe_load(fid)
        opt = edict(opt)
    seed_everything(opt.seed)

    logging.info('Creating data loaders to be analyzed...')
    transforms = get_transforms(opt.dataset)
    params = edict(opt.dataset, use_sampler=False, reset_sampler=False, shuffle_test=False)
    train_loader, test_loader = get_dataloaders(params, transforms)

    logging.info(f'Total number of samples: {len(train_loader.dataset) + len(test_loader.dataset)}')
    logging.info('Extracting fingerprint...')
    json_path = extract_fingerprint([train_loader.dataset, test_loader.dataset], args.output_json_path, args.foreground_labels)
    logging.info(f'Saved the dataset fingerprint at: {json_path}')


if __name__ == '__main__':
    main(parser.parse_args())
