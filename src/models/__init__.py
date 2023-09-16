from src.models.cgan import build_gan


def build_model(task_name: str, *args, **kwargs):
    if task_name == 'counterfactual':
        return build_gan(*args, **kwargs)
    else:
        raise ValueError(f'Unsupported task provided: {task_name}')


__all__ = ['build_model']
