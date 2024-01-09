import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from matplotlib import pyplot as plt
from tqdm import tqdm

from src.datasets import get_dataloaders
from src.datasets.augmentations import get_transforms
from src.utils.generic_utils import seed_everything
from src.visualizations import CLASS_COLORS, DEFAULT_COLOR

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('-c', '--config_path', type=str, required=True, help='Configuration file path to start training from scratch')
parser.add_argument('-od', '--output_dir', type=str, required=True, help='Output directory where to write the visualizations')
parser.add_argument('-st', '--step_size', type=int, default=10, help='Every nth record to be visualized')
parser.add_argument('-s', '--sampling_only', action='store_true', default=False, help='Visualize only dataset examples where the sampler ')
parser.add_argument('-na', '--no_aug', action='store_true', default=False, help='Disable augmentations for visualization')
parser.add_argument('-la', '--label_agg_order', nargs='+', type=int, help='The order at which to overlay the classes masks.')
opt = parser.parse_args()

def visualize_dataset(dataset:torch.utils.data.Dataset, out_dir:str, step:int=10, alpha=0.7, sampling_only:bool = False, 
                      agg_order:list[int]=None) -> None:
    if hasattr(dataset, 'scans'):
        for s in dataset.scans:
            s.load_masks = True
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    inds = list(range(0, len(dataset), step)) if not sampling_only else np.nonzero(dataset.get_sampling_labels())[0][::step]
    
    for i in tqdm(inds):
        s = dataset[i]
        assert s['masks'].shape[0] > 0, f'No masks are loaded in the dataset: {dataset}'
        
        name = dataset.split + '_' + s.get('scan_name', 'sample').split('.')[0] + '_label_' + str(s['label'].item())

        img = ((s['image'][0].numpy() + 1) / 2) * 255
        img = np.stack([img.astype(np.uint8)]*3, -1)
        # print(img.shape, s['scan_name'], s['slice_index'])
        
        vis_mask = np.zeros((*s['masks'].shape[1:], 3), dtype=np.uint8)
        
        for j in (agg_order or range(1, s['masks'].shape[0])):
            # print(s['masks'].shape, dataset.classes)
            class_name = dataset.classes[j]
            color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)
            cl_mask = np.stack([s['masks'][j].numpy()]*3, -1).astype(np.uint8) * np.array(color).reshape(1, 1, 3)
            vis_mask = vis_mask + cl_mask
        vis_mask = vis_mask.astype(np.uint8)

        fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(12, 7), sharex=True, sharey=True)
        ax1.imshow(np.flip(img.transpose(1, 0, 2), (0, 1)))
        ax1.set_axis_off()
        ax1.set_title(name)
        
        vis_img = (img*alpha).astype(np.uint8) + ((1-alpha)*vis_mask).astype(np.uint8)
        ax2.imshow(np.flip(vis_img.transpose(1, 0, 2), (0, 1)))
        ax2.set_axis_off()
        ax2.set_title(name)
        
        fig.tight_layout()
        
        name = f'{name}_{str(i).zfill(5)}'
        plt.savefig(out_dir / f'{name}.jpg')
        
        # fig.title
        plt.close()
        plt.cla()
        plt.clf()
    

def main(args):
    with open(args.config_path) as fid:
        opt = yaml.safe_load(fid)
        opt = edict(opt)
    seed_everything(opt.seed)

    print('Creating data loaders to be visualized...')
    transforms = get_transforms(opt.dataset)
    if args.no_aug:
        transforms['train'] = transforms['val']
    params = edict(opt.dataset, use_sampler=False, reset_sampler=False, shuffle_test=False)
    train_loader, test_loader = get_dataloaders(params, transforms)
    print(f'Total number of samples: {len(train_loader.dataset) + len(test_loader.dataset)}')
    print('Visualizing training set...')
    visualize_dataset(train_loader.dataset, args.output_dir, step=args.step_size, 
                      sampling_only=args.sampling_only, agg_order=args.label_agg_order)
    
    print('Visualizing test set...')
    visualize_dataset(test_loader.dataset, args.output_dir, step=args.step_size, 
                      sampling_only=args.sampling_only, agg_order=args.label_agg_order)
    

if __name__ == '__main__':
    main(parser.parse_args())
