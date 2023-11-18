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
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


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


def setup_logger(name, log_file=None):
    logging.basicConfig(
        level=logging.INFO, 
        format= r'[%(asctime)s|%(levelname)s] - %(message)s',
        datefmt=r'%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler()] + ([logging.FileHandler(log_file, 'a+')] if log_file is not None else [])
    )
    return logging.getLogger(name)


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
