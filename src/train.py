"""Train one CIFAR-10 model and persist weights, logits and metrics.

Usage:
    python -m src.train --model se_resnet20 --seed 42 --epochs 100 --tag se20_s42
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from . import metrics, models
from .data import GPUCIFAR10

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_at(epoch: int, epochs: int, base_lr: float, warmup: int) -> float:
    if epoch < warmup:
        return base_lr * (epoch + 1) / warmup
    p = (epoch - warmup) / max(1, epochs - warmup)
    return 0.5 * base_lr * (1.0 + math.cos(math.pi * p))


def param_groups(model: nn.Module, wd: float) -> list[dict]:
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if p.ndim == 1 or name.endswith(".bias"):
            no_decay.append(p)
        else:
            decay.append(p)
    return [{"params": decay, "weight_decay": wd}, {"params": no_decay, "weight_decay": 0.0}]


@torch.no_grad()
def evaluate(model: nn.Module, x: torch.Tensor, y: torch.Tensor, amp: str, batch_size: int = 500):
    model.eval()
    logits = []
    for i in range(0, x.shape[0], batch_size):
        xb = x[i:i + batch_size]
        if amp == "bf16":
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(xb)
        else:
            out = model(xb)
        logits.append(out.float())
    logits = torch.cat(logits)
    loss = F.cross_entropy(logits, y).item()
    return logits, loss


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="resnet20")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--wd", type=float, default=5e-4)
    ap.add_argument("--mixup", type=float, default=0.0)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--se-reduction", type=int, default=8)
    ap.add_argument("--se-scope", default="all")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "fp32"])
    ap.add_argument("--tag", default=None)
    ap.add_argument("--limit-train", type=int, default=0, help="debug: use only N training images")
    ap.add_argument("--benchmark-only", type=int, default=0, help="run N epochs and exit (no save)")
    args = ap.parse_args()

    run_id = args.tag or f"{args.model}_s{args.seed}"
    out_dir = ROOT / "results"
    for d in ("runs", "preds"):
        (out_dir / d).mkdir(parents=True, exist_ok=True)
    (ROOT / "weights").mkdir(exist_ok=True)

    set_seed(args.seed)
    device = torch.device("cuda")
    torch.backends.cudnn.benchmark = True

    data = GPUCIFAR10(str(DATA), device)
    x_train, y_train = data.x_train, data.y_train
    if args.limit_train:
        x_train, y_train = x_train[: args.limit_train], y_train[: args.limit_train]
    x_test = data.test_tensor()

    model = models.build_model(
        args.model, se_reduction=args.se_reduction, se_scope=args.se_scope
    ).to(device)
    n_params = models.count_params(model)
    macs = models.count_macs(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.SGD(
        param_groups(model, args.wd), lr=args.lr, momentum=0.9, nesterov=True
    )

    n = x_train.shape[0]
    steps = math.ceil(n / args.batch_size)
    history = []
    t_start = time.time()
    best_acc = 0.0

    print(
        f"[{run_id}] model={args.model} params={n_params} macs={macs / 1e6:.1f}M "
        f"epochs={args.epochs} bs={args.batch_size} amp={args.amp} seed={args.seed}",
        flush=True,
    )

    for epoch in range(args.epochs):
        lr = lr_at(epoch, args.epochs, args.lr, args.warmup)
        for g in optimizer.param_groups:
            g["lr"] = lr
        model.train()
        perm = torch.randperm(n, device=device)
        t0 = time.time()
        total_loss, total_correct = 0.0, 0
        for step in range(steps):
            idx = perm[step * args.batch_size:(step + 1) * args.batch_size]
            if idx.numel() == 0:
                continue
            xb = data.train_batch(idx, augment=True)
            yb = y_train[idx]
            if args.mixup > 0:
                lam = float(np.random.beta(args.mixup, args.mixup))
                perm2 = torch.randperm(xb.shape[0], device=device)
                xb = lam * xb + (1.0 - lam) * xb[perm2]
                yb2 = yb[perm2]
            optimizer.zero_grad(set_to_none=True)
            if args.amp == "bf16":
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    out = model(xb)
                    if args.mixup > 0:
                        loss = lam * criterion(out, yb) + (1 - lam) * criterion(out, yb2)
                    else:
                        loss = criterion(out, yb)
            else:
                out = model(xb)
                if args.mixup > 0:
                    loss = lam * criterion(out, yb) + (1 - lam) * criterion(out, yb2)
                else:
                    loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * idx.numel()
            total_correct += (out.detach().argmax(1) == yb).sum().item()
        train_loss = total_loss / n
        train_acc = total_correct / n
        logits, test_loss = evaluate(model, x_test, data.y_test, args.amp)
        test_acc = metrics.accuracy(logits.cpu().numpy(), data.y_test.cpu().numpy())
        best_acc = max(best_acc, test_acc)
        dt = time.time() - t0
        history.append(
            {
                "epoch": epoch,
                "lr": lr,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "test_loss": test_loss,
                "test_acc": test_acc,
                "time_s": dt,
            }
        )
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(
                f"  epoch {epoch:3d}/{args.epochs} lr={lr:.4f} train_loss={train_loss:.3f} "
                f"train_acc={train_acc * 100:.2f} test_loss={test_loss:.3f} "
                f"test_acc={test_acc * 100:.2f} ({dt:.1f}s)",
                flush=True,
            )
        if args.benchmark_only and epoch + 1 >= args.benchmark_only:
            print(f"benchmark: {dt:.2f}s/epoch, {n / dt:.0f} img/s", flush=True)
            return

    total_time = time.time() - t_start
    logits, test_loss = evaluate(model, x_test, data.y_test, args.amp)
    logits_np = logits.cpu().numpy()
    labels_np = data.y_test.cpu().numpy()
    final = {
        "run_id": run_id,
        "model": args.model,
        "seed": args.seed,
        "epochs": args.epochs,
        "params": n_params,
        "macs": macs,
        "test_acc": metrics.accuracy(logits_np, labels_np),
        "test_loss": test_loss,
        "macro_f1": metrics.macro_f1(logits_np, labels_np),
        "ece": metrics.ece(logits_np, labels_np),
        "nll": metrics.nll(logits_np, labels_np),
        "brier": metrics.brier(logits_np, labels_np),
        "best_test_acc": best_acc,
        "train_time_s": total_time,
        "ms_per_epoch": 1000 * total_time / args.epochs,
        "images_per_s": n * args.epochs / total_time,
        "peak_mem_mb": torch.cuda.max_memory_allocated() / 1024 ** 2,
        "config": vars(args),
    }

    torch.save(
        {
            "state_dict": model.state_dict(),
            "config": vars(args),
            "metrics": final,
            "model_name": args.model,
        },
        ROOT / "weights" / f"{run_id}.pt",
    )
    np.savez_compressed(
        out_dir / "preds" / f"{run_id}.npz", logits=logits_np.astype(np.float32), labels=labels_np
    )
    with open(out_dir / "runs" / f"{run_id}.json", "w", encoding="utf8") as f:
        json.dump({"final": final, "history": history}, f, indent=2)

    print(
        f"[{run_id}] FINAL acc={final['test_acc'] * 100:.2f}% f1={final['macro_f1'] * 100:.2f} "
        f"ece={final['ece']:.4f} time={total_time / 60:.1f}min "
        f"({final['ms_per_epoch'] / 1000:.1f}s/epoch)",
        flush=True,
    )


if __name__ == "__main__":
    main()
