from src.models.cgan import build_gan
from src.models.classifier import ClassificationModel


def build_model(task_name: str, *args, **kwargs):
    if task_name == 'counterfactual':
        return build_gan(*args, **kwargs)
    elif task_name == 'classification':
        return ClassificationModel(*args, **kwargs)
    else:
        raise ValueError(f'Unsupported task provided: {task_name}')


__all__ = ['build_model']
