import argparse
from copy import deepcopy
import logging
from pathlib import Path

import torch
import yaml
from easydict import EasyDict as edict
from tqdm import tqdm

from src.models import build_model
from src.models.cgan.counterfactual_cgan import CounterfactualCGAN
from src.trainers import build_trainer
from src.trainers.counterfactual import CounterfactualTrainer
from src.utils.generic_utils import seed_everything
from src.attributors import get_attributor, ATTRIBUTORS
from torchmetrics.classification import BinaryJaccardIndex
from torchvision.utils import save_image
from src.visualizations import confmat_vis_img

logging.basicConfig(level=logging.INFO)

parser = argparse.ArgumentParser()
parser.add_argument('-cp', '--continue_path', type=str, required=True, help='Path to the existing training run to continue interrupted training')
parser.add_argument('-t', '--tau', type=float, required=False, default=0.8, help='Theshold for the counterfactual score metric')
parser.add_argument('-st', '--sal_threshold', type=float, required=False, default=0.975, help='Theshold to be used for thresholding saliency maps into segmentation masks')
parser.add_argument('-sf', '--skip_fid', action='store_true', default=False, help='Whether to skip FID metric calculation')
parser.add_argument('-at', '--attributors', nargs='+', required=False, default=[], help='Attribution methods to be additionally evaluated')
opt = parser.parse_args()


def eval_attributors(attrib_kinds:list[str], cf_trainer: CounterfactualTrainer, loader:torch.utils.data.DataLoader, sal_threshold:float=0.95):
    cf_explainer = cf_trainer.model
    ious = {k: BinaryJaccardIndex(sal_threshold).to(cf_trainer.device) for k in attrib_kinds}
    attribs = {k: get_attributor(k, deepcopy(cf_explainer.classifier_f), cf_explainer.classifier_kind, cf_explainer.img_size) for k in attrib_kinds}

    num_abnormal_samples = 0
    for i, batch in tqdm(enumerate(loader), desc=f'Validating {list(attrib_kinds)}', leave=False, total=len(loader)):
        abnormal_mask = batch['label'].bool()
        if not abnormal_mask.any():
            continue
        imgs = batch['image'][abnormal_mask].cuda(non_blocking=True)
        masks = batch['masks'][abnormal_mask][:, cf_trainer.cf_gt_seg_mask_idx].cuda(non_blocking=True)
        num_abnormal_samples += abnormal_mask.sum()
        B = imgs.shape[0]

        for k, attrib in attribs.items():
            vis_dir = cf_trainer.logging_dir / k
            vis_dir.mkdir(parents=True, exist_ok=True)
    
            sal_maps = torch.stack([attrib.attribute(img) for img in imgs])
            sal_masks = (sal_maps > sal_threshold).byte()
            ious[k].update(sal_masks.view(B, -1), masks.view(B, -1))
            
            vis_img = imgs[0].clone().add_(1).div_(2)
            vis = torch.stack((
                vis_img, (vis_img*0.3 + sal_maps[0]*0.7),
                torch.zeros_like(vis_img), torch.zeros_like(vis_img), 
            ), dim=0).permute(0, 2, 3, 1)
            vis = torch.cat((vis, vis, vis), 3)
            vis_confmat = confmat_vis_img(masks[0].unsqueeze(0).unsqueeze(0), sal_masks[0].unsqueeze(0), normalized=True)[0]
            vis[2] = 0.3*vis[0] + 0.7 * vis_confmat
            vis[3] = vis_confmat
            vis = vis.permute(0, 3, 1, 2)
            # save first example for visualization
            vis_path = vis_dir / (f'item_%d.jpg' % i)
            save_image(vis.data, vis_path, nrow=2, normalize=False)

    for k, iou in ious.items():
        sal_iou = iou.compute().item()
        cf_trainer.logger.info(f'[{k}] IoU(S, Sc) = {sal_iou:.3f} (sal_thresh={sal_threshold}, num_samples={num_abnormal_samples}, mask={cf_trainer.cf_gt_seg_mask_idx})')


def main(args):
    continue_path = Path(args.continue_path)
    continue_path = continue_path.parent.parent if continue_path.suffix == '.pth' else continue_path
    with open(continue_path.joinpath('hparams.yaml')) as fid:
        opt = yaml.safe_load(fid)
        opt = edict(opt)
    seed_everything(opt.seed)

    model: CounterfactualCGAN = build_model(opt.task_name, opt=opt.model, img_size=opt.dataset.img_size)
    trainer: CounterfactualTrainer = build_trainer(opt.task_name, opt, model, args.continue_path)
    _, test_loader = trainer.get_dataloaders(skip_cf_sampler=True)
    trainer.evaluate_counterfactual(test_loader, args.tau, args.skip_fid)

    attributors = ATTRIBUTORS.keys() if args.attributors == ['all'] else args.attributors
    if attributors:
        eval_attributors(attributors, trainer, test_loader, args.sal_threshold)


if __name__ == '__main__':
    main(parser.parse_args())
