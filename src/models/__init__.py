from src.models.cgan import build_gan
from src.models.classifier import ClassificationModel
from src.models.seg_aux import SegmentationAuxModel


def build_model(task_name: str, *args, **kwargs):
    if task_name.startswith('counterfactual'):
        return build_gan(*args, **kwargs)
    elif task_name == 'classification':
        return ClassificationModel(*args, **kwargs)
    elif task_name == 'segmentation':
        return SegmentationAuxModel(*args, **kwargs)
    else:
        raise ValueError(f'Unsupported task provided: {task_name}')


__all__ = ['build_model']
