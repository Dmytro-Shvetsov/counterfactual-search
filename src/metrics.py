from typing import Any

import torch
from segmentation_models_pytorch.losses.tversky import soft_tversky_score


class TverskyScore(torch.nn.Module):
    """Implementation of Tversky score for image segmentation task. 
    Where TP and FP is weighted by alpha and beta params.
    With alpha == beta == 0.5, this score becomes equal DiceScore.

    Args:
        alpha: Weight constant that penalize model for FPs (False Positives)
        beta: Weight constant that penalize model for FNs (False Negatives)
        gamma: Constant that squares the error function. Defaults to ``1.0``
    """

    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        gamma: float = 1.0,
    ):
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute_score(self, output, target, smooth=1e-15, eps=1e-7, dims=None) -> torch.Tensor:
        return soft_tversky_score(output, target, self.alpha, self.beta, smooth, eps, dims)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.compute_score(*args, **kwargs)


class IoU(torch.nn.Module):
    def __init__(self, eps=1e-15):
        super().__init__()
        self.eps = eps

    def __call__(self, output, target):
        intersection = (target * output).sum()
        union = target.sum() + output.sum() - intersection
        result = (intersection + self.eps) / (union + self.eps)
        return result


class Precision(torch.nn.Module):
    def __init__(self, eps=1e-15):
        super().__init__()
        self.eps = eps

    def __call__(self, output, target):
        tp = (output * target).sum()
        fp = (output * (1. - target)).sum()
        result = (tp + self.eps) / (tp + fp + self.eps)
        return result
