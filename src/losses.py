import torch
import torch.nn.functional as F


def CARL(x:torch.Tensor, x_prime:torch.Tensor, masks:torch.Tensor) -> torch.Tensor:
    """
    CARL loss from https://arxiv.org/pdf/2101.04230v3.pdf (formula 8).
    This loss implements only the semantic segmentation term for the simplicity.

    Args:
        x (torch.Tensor): input image
        x_prime (torch.Tensor): explanation image
        masks (torch.Tensor): binary masks with the segmentation regions for enforcing local consistency

    Returns:
        torch.Tensor: calculated loss
    """
    B = x.shape[0]
    x, x_prime, masks = x.view(B, -1), x_prime.view(B, -1), masks.view(B, -1).bool()
    l_rec = F.l1_loss(x, x_prime, reduction='none')
    l_rec *= masks
    # calculate reconstruction loss sample-wise
    l_rec = l_rec.mean(dim=1)
    # average across batches
    return l_rec.mean(dim=0)
