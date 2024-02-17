import datetime
import logging
import time
from collections import OrderedDict
from pathlib import Path

import torch
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter

from src.utils.generic_utils import setup_logger


class Logger:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        self.writer = SummaryWriter(log_dir)
        self.logger = setup_logger(__name__, self.log_dir / 'logs.log')
        self.logger.info(f"================ Session ({time.strftime('%c')}) ================")
        self.logger.info(f'Logging directory: {self.log_dir}')

    def log(self, data, step, phase='train'):
        for key, value in data.items():
            self.writer.add_scalar(f'{phase}/{key}', value, step)

    def info(self, msg, *args, **kwags):
        return self.logger.info(msg, *args, **kwags)

    def reset(self):
        for p in self.log_dir.glob('events*'):
            p.unlink(True)

    def __del__(self):
        self.writer.flush()
