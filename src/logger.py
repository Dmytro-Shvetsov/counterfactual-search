from collections import OrderedDict
import datetime
import logging

from pathlib import Path
import time
import torch
from torch.nn import functional as F

from torch.utils.tensorboard import SummaryWriter


class Logger:
    def __init__(self, log_dir):
        self.log_dir = Path(log_dir)
        self.writer = SummaryWriter(log_dir)
        self.log_file = self.log_dir / 'logs.log'
        logging.basicConfig(
            level=logging.INFO, 
            format= r'[%(asctime)s|%(levelname)s] - %(message)s',
            datefmt=r'%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(self.log_file, 'a')]
        )
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"================ Session ({time.strftime('%c')}) ================")

    def log(self, data, step, phase='train'):
        for key, value in data.items():
            self.writer.add_scalar(f'{phase}/{key}', value, step)

    def reset(self):
        for p in self.log_dir.glob('events*'):
            p.unlink(True)

    def __del__(self):
        self.writer.flush()
