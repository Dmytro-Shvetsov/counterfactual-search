import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec

# orange
DEFAULT_COLOR = (255, 156, 18)

CLASS_COLORS = {
    # red
    'kidney_cyst_left': (235, 64, 52),
    'kidney_cyst_right': (235, 64, 52),
    'kidney_cyst': (235, 64, 52),
    'cyst': (235, 64, 52),
    
    # green
    'kidney_left': (50, 168, 82),
    'kidney_right': (50, 168, 82),
    'kidney': (50, 168, 82),
    
    # purple
    'tumor': (173, 66, 245),
    
    # blue
    'gaussian': (18, 140, 255),
}

def overlay_masks(img:np.ndarray, masks:np.ndarray, classes:list[str], alpha:float=0.7, agg_order:list[int]=None, onehot_masks=True):
    vis_mask = np.zeros((*masks.shape[1:], 3), dtype=np.uint8)
    
    for j in (agg_order or range(1, len(classes))):
        class_name = classes[j]
        color = CLASS_COLORS.get(class_name, DEFAULT_COLOR)
        cl_mask = masks[j] if onehot_masks else (masks == j)
        cl_mask = np.stack([cl_mask]*3, -1, dtype=np.uint8) * np.array(color).reshape(1, 1, 3) # RGB 0-255 colored mask
        vis_mask = vis_mask + cl_mask
    vis_mask = vis_mask.astype(np.uint8)
    vis_img = img*alpha + (1-alpha)*vis_mask
    return vis_img.astype(np.uint8)


def visualize_seg_predictions(
    batch_img:torch.Tensor, 
    batch_masks_gt:torch.Tensor, 
    batch_masks_pred:torch.Tensor,
    batch_labels_gt:torch.Tensor,  
    batch_labels_pred:torch.Tensor, 
    classes:list[str], 
    alpha:float=.7,
    out_file_path:str = None,
    agg_order:list[int] = None, 
    num_vis: int = 2,
):

    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    gs = GridSpec(ncols=3, nrows=num_vis, figure=fig, wspace=0.01, hspace=0.01)

    inds = np.random.randint(0, batch_img.shape[0] - 1, num_vis)

    is_multiclass = batch_masks_pred.shape[1] > 1

    batch_masks_gt = batch_masks_gt[:, 1:] if not is_multiclass else batch_masks_gt
    for i in range(num_vis):
        
        ax1, ax2, ax3 = [fig.add_subplot(gs[i, j] ) for j in range(3)]
        
        idx = inds[i]
        img = (((batch_img[idx][0] + 1) / 2) * 255).byte().cpu().numpy().astype(np.uint8)
        img = np.stack([img]*3, -1, dtype=np.uint8) # RGB 0-255 image
        
        gt_label = batch_labels_gt[idx].item()
        pred_label = (batch_labels_pred[idx] > 0).byte().item()
        
        # classes and gts are expected to always have background class as first index
        gt_masks = batch_masks_gt[idx]
        pred_masks = batch_masks_pred[idx].argmax(0).byte() if is_multiclass else (batch_masks_pred[idx] > 0).byte()
        
        # print(gt_masks.shape, pred_masks.shape, classes)
        ax1.imshow(img)
        ax1.set_axis_off()
        ax1.set_title('Input Slice')

        vis_gt = overlay_masks(img, gt_masks.cpu().numpy(), classes, alpha, onehot_masks=True, agg_order=agg_order)
        ax2.imshow(vis_gt)
        ax2.set_axis_off()
        ax2.set_title(f'y_true={gt_label}')

        vis_pred = overlay_masks(img, pred_masks.cpu().numpy(), classes, alpha, onehot_masks=not is_multiclass, agg_order=agg_order)
        ax3.imshow(vis_pred)
        ax3.set_axis_off()
        ax3.set_title(f'y_pred={pred_label}')
 
    if out_file_path:
        plt.savefig(out_file_path)
        plt.close()
        plt.cla()
        plt.clf()


@torch.no_grad()
def _tp_fp_fn(
    output: torch.LongTensor,
    target: torch.LongTensor,
) -> tuple[torch.FloatTensor]:
    output = output.bool()
    target = target.bool()
    tp = output & target
    fp = output & (~target)
    fn = (~output) & target
    return tp, fp, fn


def confmat_vis_img(
    batch_masks_gt:torch.Tensor, 
    batch_masks_pred:torch.Tensor,
    normalized:bool=False
):
    B, _, H, W = batch_masks_gt.shape
    tp, fp, fn = _tp_fp_fn(batch_masks_pred.squeeze(1), batch_masks_gt.squeeze(1))
    vis = torch.zeros((B, H, W, 3)).byte()
    for i in range(B):
        vis[i][tp[i]] = torch.tensor((50, 168, 82)).byte().reshape(1, 1, 3)  # green
        vis[i][fp[i]] = torch.tensor((235, 64, 52)).byte().reshape(1, 1, 3)  # red
        vis[i][fn[i]] = torch.tensor((207, 207, 31)).byte().reshape(1, 1, 3) # yellow
    vis = vis.to(batch_masks_gt.device)
    return vis if not normalized else vis.div(255)


def visualize_tp_fp_fn_predictions(
    batch_img:torch.Tensor, 
    batch_masks_gt:torch.Tensor, 
    batch_masks_pred:torch.Tensor,
    batch_labels_gt:torch.Tensor,  
    batch_labels_pred:torch.Tensor, 
    alpha:float=.7,
    out_file_path:str = None,
    num_vis: int = 2,
):
    assert batch_masks_gt.shape[1] == 1
    assert batch_masks_pred.shape[1] == 1
    fig = plt.figure(figsize=(12, 7), constrained_layout=True)
    gs = GridSpec(ncols=3, nrows=num_vis, figure=fig, wspace=0.01, hspace=0.01)

    inds = np.random.randint(0, batch_img.shape[0] - 1, num_vis)

    batch_img, batch_masks_gt, batch_masks_pred, batch_labels_gt, batch_labels_pred = [
        t[inds] for t in (batch_img, batch_masks_gt, batch_masks_pred, batch_labels_gt, batch_labels_pred)
    ]
    batch_img = (((batch_img + 1) / 2) * 255).byte()
    batch_img = torch.cat((batch_img, batch_img, batch_img), 1).permute(0, 2, 3, 1) # RGB 0-255 image (channels last)
    vis_confmat = confmat_vis_img(batch_masks_gt, batch_masks_pred) # RGB 0-255 image (channels last)
    vis_overlay = (alpha*batch_img + (1-alpha)*vis_confmat).byte()
    
    batch_img, vis_overlay, vis_confmat = batch_img.cpu(), vis_overlay.cpu(), vis_confmat.cpu()
    
    for i in range(num_vis):
        ax1, ax2, ax3 = [fig.add_subplot(gs[i, j] ) for j in range(3)]
        gt_label = batch_labels_gt[i].item()
        pred_label = (batch_labels_pred[i] > 0).byte().item()

        # print(gt_masks.shape, pred_masks.shape, classes)
        ax1.imshow(batch_img[i])
        ax1.set_axis_off()
        ax1.set_title(f'y_true={gt_label}')

        ax2.imshow(vis_overlay[i])
        ax2.set_axis_off()
        ax2.set_title(f'y_pred={pred_label}')

        ax3.imshow(vis_confmat[i])
        ax3.set_axis_off()
        ax3.set_title(f'y_pred={pred_label}')
 
    if out_file_path:
        plt.savefig(out_file_path)
        plt.close()
        plt.cla()
        plt.clf()
