"""CIFAR-10 loading and GPU-side augmentation.

The dataset is small enough to keep on the GPU, which removes the CPU data
loader bottleneck and makes epoch times comparable across runs.
"""
from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import torchvision
from torchvision.datasets import CIFAR10

CIFAR_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR_STD = (0.2023, 0.1994, 0.2010)


class GPUCIFAR10:
    def __init__(self, root: str, device: torch.device, pad: int = 4):
        cache = Path(root) / "cifar10_tensors.pt"
        if cache.exists():
            d = torch.load(cache, weights_only=False)
            self.x_train = d["x_train"].to(device)
            self.y_train = d["y_train"].to(device)
            self.x_test = d["x_test"].to(device)
            self.y_test = d["y_test"].to(device)
        else:
            train = CIFAR10(root=root, train=True, download=False)
            test = CIFAR10(root=root, train=False, download=False)
            self.x_train = torch.from_numpy(train.data).permute(0, 3, 1, 2).contiguous().to(device)
            self.y_train = torch.tensor(train.targets, dtype=torch.long, device=device)
            self.x_test = torch.from_numpy(test.data).permute(0, 3, 1, 2).contiguous().to(device)
            self.y_test = torch.tensor(test.targets, dtype=torch.long, device=device)
        self.mean = torch.tensor(CIFAR_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(CIFAR_STD, device=device).view(1, 3, 1, 1)
        self.pad = pad
        self.device = device
        self.n_train = self.x_train.shape[0]

    def normalize(self, x_uint8: torch.Tensor) -> torch.Tensor:
        return (x_uint8.float().div_(255.0) - self.mean) / self.std

    def train_batch(self, idx: torch.Tensor, augment: bool = True) -> torch.Tensor:
        """Random crop + horizontal flip + normalization for a batch of uint8 images."""
        x = self.x_train[idx].float().div_(255.0)
        if augment:
            x = F.pad(x, (self.pad,) * 4)
            b = x.shape[0]
            offs = torch.randint(0, 2 * self.pad + 1, (b, 2), device=x.device)
            rows = offs[:, 0].view(b, 1, 1, 1) + torch.arange(32, device=x.device).view(1, 1, 32, 1)
            cols = offs[:, 1].view(b, 1, 1, 1) + torch.arange(32, device=x.device).view(1, 1, 1, 32)
            bidx = torch.arange(b, device=x.device).view(b, 1, 1, 1)
            cidx = torch.arange(3, device=x.device).view(1, 3, 1, 1)
            x = x[bidx, cidx, rows, cols]
            flip = torch.rand(b, device=x.device) < 0.5
            x = torch.where(flip.view(b, 1, 1, 1), torch.flip(x, dims=[3]), x)
        return (x - self.mean) / self.std

    def test_tensor(self) -> torch.Tensor:
        return self.normalize(self.x_test)

    def corrupted_test(self, kind: str, severity: int, seed: int = 0) -> torch.Tensor:
        """Deterministic CIFAR-10-C style corruption on the uint8 test images."""
        g = torch.Generator(device=self.device).manual_seed(seed + 1000 * severity)
        x = self.x_test.float().div_(255.0)
        if kind == "noise":
            sigma = [0.04, 0.08, 0.12][severity - 1]
            x = (x + sigma * torch.randn(x.shape, generator=g, device=x.device)).clamp(0, 1)
        elif kind == "blur":
            k = [3, 5, 7][severity - 1]
            x = F.avg_pool2d(F.pad(x, (k // 2,) * 4, mode="reflect"), k, 1)
        elif kind == "contrast":
            factor = [0.7, 0.5, 0.3][severity - 1]
            mean = x.mean(dim=(1, 2, 3), keepdim=True)
            x = ((x - mean) * factor + mean).clamp(0, 1)
        elif kind == "brightness":
            delta = [0.1, 0.2, 0.3][severity - 1]
            x = (x + delta).clamp(0, 1)
        else:
            raise ValueError(kind)
        return (x - self.mean) / self.std
