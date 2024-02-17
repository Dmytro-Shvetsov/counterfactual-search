import torch
from abc import ABC, abstractmethod

class BaseAttributor(ABC):
    def __init__(self, classifier:torch.nn.Module, classifier_kind:str, img_size:tuple[int]) -> None:
        self.classifier = classifier.eval().cuda()
        self.classifier_kind = classifier_kind
        self.img_size = img_size
    
    @abstractmethod
    def attribute(self, img:torch.Tensor) -> torch.Tensor:
        """
        Runs the attribution method on a single image and returns the saliency map in [0; 1] range.

        Args:
            img (torch.Tensor): pre-processed image to be fed into the classifier

        Returns:
            torch.Tensor: resulting saliency map
        """
