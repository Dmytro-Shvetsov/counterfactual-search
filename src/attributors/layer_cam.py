import torch
import torch.nn.functional as F
from .base import BaseAttributor
from .cam_methods.layercam import LayerCAM as _LayerCam


class LayerCAM(BaseAttributor):
    def __init__(self, classifier:torch.nn.Module, classifier_kind:str, img_size:tuple[int]):
        super().__init__(classifier, classifier_kind, img_size)
        
        if classifier_kind.startswith('resnet'):
            self.model_dict = dict(type='resnet', arch=classifier, layer_name='layer4', input_size=self.img_size)
        elif classifier_kind.startswith('efficientnet_v2'):
            self.model_dict = dict(type='efficientnet_v2', arch=classifier, layer_name='features_7', input_size=self.img_size)
        else:
            raise NotImplementedError()
        
        self._layercam = _LayerCam(self.model_dict)

    def attribute(self, img: torch.Tensor) -> torch.Tensor:
        for p in self.classifier.parameters():
            p.requires_grad = True
        layercam_map = self._layercam(img.unsqueeze(0))
        return layercam_map.squeeze(0)
