from .base import BaseAttributor
from .score_cam import ScoreCAM
from .rise import RISE
from .layer_cam import LayerCAM

ATTRIBUTORS = {
    'score_cam': ScoreCAM,
    'layer_cam': LayerCAM,
    'rise': RISE,
}

def get_attributor(kind:str, *args, **kwargs) -> BaseAttributor:
    if kind in ATTRIBUTORS:
        return ATTRIBUTORS[kind](*args, **kwargs)
    else:
        raise ValueError(f'Unsupported attributor kind provided: {kind}')
