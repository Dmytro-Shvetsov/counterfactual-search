from pathlib import Path
from typing import Tuple, List
import torch
import datetime
import os
import subprocess
import logging


def get_commit_hash():
    """https://stackoverflow.com/questions/14989858/get-the-current-git-hash-in-a-python-script"""
    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"]).decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        commit = "0000000"
    return commit


def get_experiment_folder_path(root_path, model_name):
    """Get an experiment folder path with the current date and time"""
    date_str = datetime.datetime.now().strftime("%B-%d-%Y_%I+%M%p")
    commit_hash = get_commit_hash()
    output_folder = os.path.join(root_path, model_name + "-" + date_str + "-" + commit_hash)
    os.makedirs(output_folder)
    return output_folder


def count_parameters(model):
    r"""Count number of trainable parameters in a network"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def setup_logger(name, log_file=None):
    logging.basicConfig(
        level=logging.INFO, 
        format= r'[%(asctime)s|%(levelname)s] - %(message)s',
        datefmt=r'%Y-%m-%d %H:%M:%S',
        handlers=[logging.StreamHandler()] + ([logging.FileHandler(log_file, 'a')] if log_file is not None else [])
    )
    return logging.getLogger(__name__)


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
    checkpoint_path = checkpoint_dir / f'checkpoint_{current_step}.pth'
    torch.save(state, checkpoint_path)
    return checkpoint_path
