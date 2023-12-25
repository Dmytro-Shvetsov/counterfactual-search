import torch
import torch.nn.functional as F


def _CARL(x:torch.Tensor, x_prime:torch.Tensor, masks:torch.Tensor, eps:float=1e-8) -> torch.Tensor:
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
    # print(masks.shape, x_prime.shape, x.shape)
    # exit()
    x, x_prime, masks = x.view(B, -1), x_prime.view(B, -1), masks.view(B, -1)
    l_rec = (F.l1_loss(x, x_prime, reduction='none') * masks).sum(dim=1, keepdim=True) / (masks.sum(dim=1, keepdim=True) + eps)
    # l_rec *= masks
    # calculate reconstruction loss sample-wise
    # l_rec = l_rec.mean(dim=1)
    # average across batches
    return l_rec.mean(dim=0)


def CARL(x:torch.Tensor, x_prime:torch.Tensor, masks:torch.Tensor) -> torch.Tensor:
    if masks.shape[1] > 2:
        masks = masks[:, :2]
    return sum(_CARL(x, x_prime, masks[:, i]) for i in range(masks.shape[1]))


def kl_divergence(y_pred, y_true, eps=1e-5):
    """
    Computes KL(Q||P) where P is the distribution of the observations and Q denotes the model for the binary classification case.
    """
    # Clip probabilities to avoid log(0) and log(1)
    y_pred = torch.clamp(y_pred, eps, 1.0 - eps)
    y_true = torch.clamp(y_true, eps, 1.0 - eps)

    # Compute KL divergence
    # kl = y_pred * torch.log(y_pred / y_true) + (1 - y_pred) * torch.log((1 - y_pred) / (1 - y_true))
    kl = y_true * torch.log(y_true / y_pred) + (1 - y_true) * torch.log((1 - y_true) / (1 - y_pred))
    kl = torch.mean(kl)

    return kl


# Hinge Loss
def loss_hinge_dis(dis_fake, dis_real):
    d_real = F.relu(1. - dis_real).mean()
    d_fake = F.relu(1. + dis_fake).mean()
    return d_real, d_fake


def loss_hinge_gen(dis_fake):
    return -dis_fake.mean()


def tv_loss(img):
    """
    Compute total variation loss.

    Inputs:
    - img: PyTorch Variable of shape (B, C, H, W) holding an input image.

    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    w_variance = torch.mean(torch.pow(img[:, :, :, :-1] - img[:, :, :, 1:], 2), dim=(1, 2, 3))
    h_variance = torch.mean(torch.pow(img[:, :, :-1, :] - img[:, :, 1:, :], 2), dim=(1, 2, 3))
    return torch.mean(h_variance + w_variance)
