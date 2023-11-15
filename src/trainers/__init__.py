from src.trainers.classification import ClassificationTrainer
from src.trainers.counterfactual import CounterfactualTrainer
from src.trainers.segmentation import SegmentationTrainer
from src.trainers.trainer import BaseTrainer


def build_trainer(task_name: str, *args, **kwargs) -> BaseTrainer:
    if task_name == 'classification':
        return ClassificationTrainer(*args, **kwargs)
    elif task_name == 'counterfactual':
        return CounterfactualTrainer(*args, **kwargs)
    elif task_name == 'segmentation':
        return SegmentationTrainer(*args, **kwargs)
    else:
        raise ValueError(f'Unsupported task provided: {task_name}')
