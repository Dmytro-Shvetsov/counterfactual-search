import datetime
import logging
import os
import random
import subprocess
from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch


def seed_everything(seed: int):
    import random, os
    import numpy as np
    import torch
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    random.seed(seed)
    np.random.seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_commit_hash():
    """https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script"""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "0000000"
    return commit


def get_experiment_folder_path(root_path, model_name, experiment_name='exp'):
    """Get an experiment folder path with the current date and time"""
    date_str = datetime.datetime.now().strftime("%B-%d-%Y_%I+%M%p")
    commit_hash = get_commit_hash()
    output_folder = os.path.join(root_path, model_name + "-" + date_str + "-" + commit_hash + '-' + experiment_name)
    os.makedirs(output_folder, exist_ok=True)
    return output_folder


def count_parameters(model):
    r"""Count number of trainable parameters in a network"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_logger(name, log_file=None, message_format='[%(asctime)s|%(levelname)s] - %(message)s', level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)

    formatter = logging.Formatter(message_format, datefmt=r'%Y-%m-%d %H:%M:%S')
    
    if log_file:
        fh = logging.FileHandler(log_file, mode='w', encoding='utf8')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    logger.propagate = False
    return logger


def save_model(config:dict, model:torch.nn.Module, optimizers:List[torch.nn.Module], 
               current_step:int, epoch:int, checkpoint_dir:Path, **kwargs) -> Path:
    state = {
        'config': config,
        'model': model.state_dict(),
        'optimizers': [opt.state_dict() for opt in optimizers],
        'step': current_step,
        'epoch': epoch,
        'date': datetime.date.today().strftime('%B %d, %Y'),
        **kwargs,
    }
    checkpoint_path = checkpoint_dir / f'checkpoint_{epoch}.pth'
    torch.save(state, checkpoint_path)
    return checkpoint_path
