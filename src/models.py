"""CIFAR-10 networks: plain VGG-style, residual (post-/pre-activation) and SE-ResNet.

All models share the same 3-stage CIFAR stem so that comparisons isolate the
effect of residual connections and channel attention.
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock(nn.Module):
    """Squeeze-and-Excitation channel recalibration (Hu et al., CVPR 2018)."""

    def __init__(self, channels: int, reduction: int = 8) -> None:
        super().__init__()
        hidden = max(channels // reduction, 4)
        self.fc1 = nn.Conv2d(channels, hidden, kernel_size=1)
        self.fc2 = nn.Conv2d(hidden, channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        s = F.adaptive_avg_pool2d(x, 1)
        s = F.relu(self.fc1(s), inplace=True)
        s = torch.sigmoid(self.fc2(s))
        return x * s


class BasicBlock(nn.Module):
    """2-conv residual block; supports post-activation (v1) and pre-activation (v2)."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        stride: int = 1,
        use_se: bool = False,
        se_reduction: int = 8,
        preact: bool = False,
    ) -> None:
        super().__init__()
        self.preact = preact
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        # Pre-activation blocks normalise their input, post-activation blocks
        # normalise the convolution output.
        self.bn1 = nn.BatchNorm2d(in_ch if preact else out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.se = SEBlock(out_ch, se_reduction) if use_se else None

        self.downsample: nn.Module | None = None
        if stride != 1 or in_ch != out_ch:
            if preact:
                # Pre-activation ResNet uses a plain 1x1 projection without BN.
                self.downsample = nn.Conv2d(in_ch, out_ch, 1, stride, bias=False)
            else:
                self.downsample = nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 1, stride, bias=False),
                    nn.BatchNorm2d(out_ch),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x if self.downsample is None else self.downsample(x)
        if self.preact:
            out = self.conv1(F.relu(self.bn1(x), inplace=True))
            out = self.conv2(F.relu(self.bn2(out), inplace=True))
            out = out + identity
            if self.se is not None:
                out = self.se(out)
        else:
            out = F.relu(self.bn1(self.conv1(x)), inplace=True)
            out = self.bn2(self.conv2(out))
            out = out + identity
            if self.se is not None:
                out = self.se(out)
            out = F.relu(out, inplace=True)
        return out


class PlainBlock(nn.Module):
    """Same convolution stack as BasicBlock but without the identity path."""

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = F.relu(self.bn2(self.conv2(out)), inplace=True)
        return out


class CIFARNet(nn.Module):
    def __init__(
        self,
        block: type[nn.Module],
        num_blocks: Iterable[int],
        num_classes: int = 10,
        use_se: bool = False,
        se_reduction: int = 8,
        se_scope: str = "all",
        preact: bool = False,
    ) -> None:
        super().__init__()
        self.preact = preact
        self.in_ch = 16
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], 1, use_se, se_reduction, se_scope, 0, preact)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], 2, use_se, se_reduction, se_scope, 1, preact)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], 2, use_se, se_reduction, se_scope, 2, preact)
        self.bn_final = nn.BatchNorm2d(64) if preact else nn.Identity()
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(64, num_classes)
        self._init_weights()

    def _use_se_here(self, scope: str, stage: int, block_idx: int, n_blocks: int) -> bool:
        if scope == "all":
            return True
        if scope == "stage3":
            return stage == 2
        if scope == "last":
            return block_idx == n_blocks - 1
        if scope == "none":
            return False
        raise ValueError(f"unknown se_scope: {scope}")

    def _make_layer(
        self,
        block: type[nn.Module],
        out_ch: int,
        n_blocks: int,
        stride: int,
        use_se: bool,
        se_reduction: int,
        se_scope: str,
        stage: int,
        preact: bool,
    ) -> nn.Sequential:
        layers = []
        for i in range(n_blocks):
            s = stride if i == 0 else 1
            if block is BasicBlock:
                layers.append(
                    block(
                        self.in_ch,
                        out_ch,
                        s,
                        use_se=use_se and self._use_se_here(se_scope, stage, i, n_blocks),
                        se_reduction=se_reduction,
                        preact=preact,
                    )
                )
            else:
                layers.append(block(self.in_ch, out_ch, s))
            self.in_ch = out_ch
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.preact:
            x = self.conv1(x)
            x = self.bn1(x)
        else:
            x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.preact:
            x = F.relu(self.bn_final(x), inplace=True)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.fc(x)

    def features(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate 64-d pooled representation (for feature visualisation)."""
        if self.preact:
            x = self.conv1(x)
            x = self.bn1(x)
        else:
            x = F.relu(self.bn1(self.conv1(x)), inplace=True)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.preact:
            x = F.relu(self.bn_final(x), inplace=True)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


_SPECS = {
    # name: (block, blocks per stage, use_se, preact)
    "plain20": (PlainBlock, (3, 3, 3), False, False),
    "plain56": (PlainBlock, (9, 9, 9), False, False),
    "resnet20": (BasicBlock, (3, 3, 3), False, False),
    "resnet56": (BasicBlock, (9, 9, 9), False, False),
    "se_resnet20": (BasicBlock, (3, 3, 3), True, False),
    "se_resnet56": (BasicBlock, (9, 9, 9), True, False),
    "preact20": (BasicBlock, (3, 3, 3), False, True),
}


def build_model(
    name: str,
    num_classes: int = 10,
    se_reduction: int = 8,
    se_scope: str = "all",
) -> CIFARNet:
    key = name.lower()
    if key not in _SPECS:
        raise KeyError(f"unknown model {name}; choose from {sorted(_SPECS)}")
    block, blocks, use_se, preact = _SPECS[key]
    return CIFARNet(
        block,
        blocks,
        num_classes=num_classes,
        use_se=use_se,
        se_reduction=se_reduction,
        se_scope=se_scope,
        preact=preact,
    )


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def count_macs(model: nn.Module, input_size: tuple[int, int, int, int] = (1, 3, 32, 32)) -> int:
    """Multiply-accumulate operations for one forward pass (conv + linear)."""
    macs = 0
    handles = []

    def conv_hook(module: nn.Conv2d, inp, out):
        nonlocal macs
        out_elems = out.shape[2] * out.shape[3]
        macs += out_elems * module.in_channels * module.out_channels * module.kernel_size[0] * module.kernel_size[1]

    def linear_hook(module: nn.Linear, inp, out):
        nonlocal macs
        macs += module.in_features * module.out_features

    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(conv_hook))
        elif isinstance(m, nn.Linear):
            handles.append(m.register_forward_hook(linear_hook))
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    model(torch.zeros(*input_size, device=device))
    for h in handles:
        h.remove()
    if was_training:
        model.train()
    return macs


if __name__ == "__main__":
    for name in _SPECS:
        m = build_model(name)
        print(f"{name:14s} params={count_params(m) / 1e6:.3f}M macs={count_macs(m) / 1e6:.1f}M")
