import torch
import torch.nn.functional as F
from .base import BaseAttributor
from .cam_methods.scorecam import ScoreCAM as _ScoreCAM


class ScoreCAM(BaseAttributor):
    def __init__(self, classifier:torch.nn.Module, classifier_kind:str, img_size:tuple[int]):
        super().__init__(classifier, classifier_kind, img_size)
        
        if classifier_kind.startswith('resnet'):
            self.model_dict = dict(type='resnet', arch=classifier, layer_name='layer4', input_size=self.img_size)
        else:
            raise NotImplementedError()
        
        self._scorecam = _ScoreCAM(self.model_dict)

    def attribute(self, img: torch.Tensor) -> torch.Tensor:
        for p in self.classifier.parameters():
            p.requires_grad = True
        scorecam_map = self._scorecam(img.unsqueeze(0))
        return scorecam_map.squeeze(0)
