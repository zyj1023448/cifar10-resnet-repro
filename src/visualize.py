"""Qualitative visualisations: Grad-CAM and PCA of penultimate features."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, layer: torch.nn.Module):
        self.model = model
        self.acts: torch.Tensor | None = None
        self.grads: torch.Tensor | None = None
        layer.register_forward_hook(self._forward)
        layer.register_full_backward_hook(self._backward)

    def _forward(self, module, inp, out):
        self.acts = out.detach()

    def _backward(self, module, grad_in, grad_out):
        self.grads = grad_out[0].detach()

    def __call__(self, x: torch.Tensor, class_idx: torch.Tensor | None = None) -> np.ndarray:
        self.model.eval()
        x = x.clone().requires_grad_(True)
        logits = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(1)
        score = logits.gather(1, class_idx.view(-1, 1)).sum()
        self.model.zero_grad(set_to_none=True)
        score.backward()
        w = self.grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((w * self.acts).sum(dim=1))
        cam = cam - cam.amin(dim=(1, 2), keepdim=True)
        cam = cam / (cam.amax(dim=(1, 2), keepdim=True) + 1e-8)
        return cam.detach().cpu().numpy()


def pca_2d(feats: np.ndarray) -> np.ndarray:
    feats = feats.astype(np.float64)
    feats = feats - feats.mean(axis=0, keepdims=True)
    u, s, vt = np.linalg.svd(feats, full_matrices=False)
    return (u[:, :2] * s[:2]).astype(np.float32)
