from pathlib import Path
import torch
import torch.nn as nn
import numpy as np
from skimage.transform import resize
from tqdm import tqdm

from .base import BaseAttributor


class RISE(BaseAttributor):
    def __init__(self, classifier:torch.nn.Module, classifier_kind:str, img_size:tuple[int], masks_path:str=Path('./masks.npy')):
        classifier_act = nn.Sequential(classifier, nn.Sigmoid())
        super().__init__(classifier_act, classifier_kind, img_size)

        self._rise = RISEModule(self.classifier, self.img_size, 256)
        if not masks_path.is_file():
            self._rise.generate_masks(N=6000, s=8, p1=0.1, savepath=masks_path)
        else:
            self._rise.load_masks(masks_path)
            self._rise.p1 = 0.1

    def attribute(self, img: torch.Tensor) -> torch.Tensor:
        for p in self.classifier.parameters():
            p.requires_grad = False
        rise_map = self._rise(img.unsqueeze(0))
        mn, mx = rise_map.min(), rise_map.max()
        rise_map = (rise_map - mn) / (mx - mn)
        return rise_map


class RISEModule(nn.Module):
    def __init__(self, model, input_size, gpu_batch=100):
        super().__init__()
        self.model = model
        self.input_size = input_size
        self.gpu_batch = gpu_batch

    def generate_masks(self, N, s, p1, savepath='masks.npy'):
        cell_size = np.ceil(np.array(self.input_size) / s)
        up_size = (s + 1) * cell_size

        grid = np.random.rand(N, s, s) < p1
        grid = grid.astype('float32')

        self.masks = np.empty((N, *self.input_size))

        for i in tqdm(range(N), desc='Generating filters'):
            # Random shifts
            x = np.random.randint(0, cell_size[0])
            y = np.random.randint(0, cell_size[1])
            # Linear upsampling and cropping
            self.masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                         anti_aliasing=False)[x:x + self.input_size[0], y:y + self.input_size[1]]
        self.masks = self.masks.reshape(-1, 1, *self.input_size)
        np.save(savepath, self.masks)
        self.masks = torch.from_numpy(self.masks).float()
        self.masks = self.masks.cuda()
        self.N = N
        self.p1 = p1

    def load_masks(self, filepath):
        self.masks = np.load(filepath)
        self.masks = torch.from_numpy(self.masks).float().cuda()
        self.N = self.masks.shape[0]

    def forward(self, x):
        N = self.N
        _, _, H, W = x.size()
        # Apply array of filters to the image
        stack = torch.mul(self.masks, x.data)

        # p = nn.Softmax(dim=1)(model(stack)) processed in batches
        p = []
        for i in range(0, N, self.gpu_batch):
            p.append(self.model(stack[i:min(i + self.gpu_batch, N)]))
        p = torch.cat(p)
        # Number of classes
        CL = p.size(1)
        sal = torch.matmul(p.data.transpose(0, 1), self.masks.view(N, H * W))
        sal = sal.view((CL, H, W))
        sal = sal / N / self.p1
        return sal
    
