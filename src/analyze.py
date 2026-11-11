"""Aggregate experiment runs into tables, figures and a machine-readable summary."""
from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from . import metrics, models
from .data import CIFAR_MEAN, CIFAR_STD, GPUCIFAR10
from .visualize import GradCAM, pca_2d

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIG = RESULTS / "figures"
TAB = RESULTS / "tables"
WEIGHTS = ROOT / "weights"
DATA = ROOT / "data"

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 10,
        "axes.labelsize": 9,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)
C_PLAIN, C_RES, C_SE = "#7f7f7f", "#1f77b4", "#d62728"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def load_runs() -> dict:
    runs = {}
    for p in sorted((RESULTS / "runs").glob("*.json")):
        with open(p, encoding="utf8") as f:
            runs[p.stem] = json.load(f)
    return runs


def variant_name(cfg: dict) -> str:
    base = cfg["model"]
    pretty = {
        "plain20": "Plain-20",
        "plain56": "Plain-56",
        "resnet20": "ResNet-20",
        "resnet56": "ResNet-56",
        "se_resnet20": "SE-ResNet-20",
        "se_resnet56": "SE-ResNet-56",
        "preact20": "PreAct-ResNet-20",
    }[base]
    tags = []
    if base.startswith("se_") and cfg.get("se_scope", "all") != "all":
        tags.append({"stage3": "stage3-only", "last": "last-block-only"}[cfg["se_scope"]])
    if base.startswith("se_") and cfg.get("se_reduction", 8) != 8:
        tags.append(f"r={cfg['se_reduction']}")
    if cfg.get("mixup", 0):
        tags.append(f"Mixup({cfg['mixup']})")
    if cfg.get("label_smoothing", 0):
        tags.append(f"LS({cfg['label_smoothing']})")
    return pretty + (" + " + " + ".join(tags) if tags else "")


def group(runs: dict) -> dict:
    g = defaultdict(list)
    for rid, rec in runs.items():
        g[variant_name(rec["final"]["config"])].append((rid, rec))
    return dict(g)


def mean_std(vals):
    a = np.asarray(vals, dtype=np.float64)
    return float(a.mean()), float(a.std(ddof=1)) if len(a) > 1 else 0.0


def summarize(g: dict) -> dict:
    out = {}
    for name, items in g.items():
        accs = [r["final"]["test_acc"] for _, r in items]
        f1s = [r["final"]["macro_f1"] for _, r in items]
        eces = [r["final"]["ece"] for _, r in items]
        nlls = [r["final"]["nll"] for _, r in items]
        times = [r["final"]["train_time_s"] for _, r in items]
        m, s = mean_std(accs)
        out[name] = {
            "runs": [rid for rid, _ in items],
            "n": len(items),
            "acc_mean": 100 * m,
            "acc_std": 100 * s,
            "acc_per_seed": [100 * a for a in accs],
            "f1_mean": 100 * mean_std(f1s)[0],
            "f1_std": 100 * mean_std(f1s)[1],
            "ece_mean": mean_std(eces)[0],
            "ece_std": mean_std(eces)[1],
            "nll_mean": mean_std(nlls)[0],
            "nll_std": mean_std(nlls)[1],
            "params": items[0][1]["final"]["params"],
            "macs": items[0][1]["final"]["macs"],
            "time_mean_s": mean_std(times)[0],
            "best_acc": 100 * max(accs),
        }
    return out


def write_csv(name: str, header: list[str], rows: list[list]) -> None:
    TAB.mkdir(parents=True, exist_ok=True)
    with open(TAB / name, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.writer(f)
        w.writerow(header)
        w.writerows(rows)


def load_model(run_id: str) -> tuple[torch.nn.Module, dict]:
    ckpt = torch.load(WEIGHTS / f"{run_id}.pt", map_location="cuda", weights_only=False)
    cfg = ckpt["config"]
    model = models.build_model(
        cfg["model"], se_reduction=cfg.get("se_reduction", 8), se_scope=cfg.get("se_scope", "all")
    ).cuda()
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, cfg


@torch.no_grad()
def predict(model, x, batch_size: int = 500) -> np.ndarray:
    outs = []
    for i in range(0, x.shape[0], batch_size):
        outs.append(model(x[i:i + batch_size]).float())
    return torch.cat(outs).cpu().numpy()


def pick(g: dict, name: str) -> tuple[str, dict]:
    items = g[name]
    return sorted(items, key=lambda kv: kv[0])[0]


def plot_curves(g: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.8))
    for name, color in [("Plain-20", C_PLAIN), ("ResNet-20", C_RES), ("SE-ResNet-20", C_SE)]:
        items = g[name]
        hist = np.array([[h["test_acc"] for h in rec["history"]] for _, rec in items])
        loss = np.array([[h["test_loss"] for h in rec["history"]] for _, rec in items])
        e = np.arange(hist.shape[1])
        axes[0].plot(e, 100 * (1 - hist.mean(0)), color=color, label=name, lw=1.2)
        axes[0].fill_between(e, 100 * (1 - hist.max(0)), 100 * (1 - hist.min(0)), color=color, alpha=0.15)
        axes[1].plot(e, loss.mean(0), color=color, label=name, lw=1.2)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Test error (%)")
    axes[0].set_title("(a) Test error")
    axes[0].legend(frameon=False)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Test cross-entropy")
    axes[1].set_title("(b) Test loss")
    fig.tight_layout()
    fig.savefig(FIG / "fig_curves.png")
    plt.close(fig)


def plot_depth(g: dict, summ: dict) -> None:
    fig, ax = plt.subplots(figsize=(3.8, 2.8))
    series = [
        (["Plain-20", "Plain-56"], C_PLAIN, "Plain"),
        (["ResNet-20", "ResNet-56"], C_RES, "Residual"),
    ]
    if "SE-ResNet-56" in summ:
        series.append((["SE-ResNet-20", "SE-ResNet-56"], C_SE, "Residual + SE"))
    for names, color, label in series:
        if any(n not in summ for n in names):
            continue
        xs = [20, 56]
        ys = [summ[n]["acc_mean"] for n in names]
        es = [summ[n]["acc_std"] for n in names]
        ax.errorbar(xs, ys, yerr=es, marker="o", color=color, label=label, capsize=3, lw=1.4)
    ax.set_xticks([20, 56])
    ax.set_xlabel("Network depth")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Depth vs. accuracy")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "fig_depth.png")
    plt.close(fig)


def plot_ablation(g: dict, summ: dict, names: list[str]) -> None:
    names = [n for n in names if n in summ]
    if not names:
        return
    vals = [summ[n]["acc_mean"] for n in names]
    errs = [summ[n]["acc_std"] for n in names]
    fig, ax = plt.subplots(figsize=(5.4, 0.36 * len(names) + 1.2))
    y = np.arange(len(names))
    colors = [C_SE if "SE-ResNet-20" == n else "#f0a202" for n in names]
    ax.barh(y, vals, xerr=errs, color=colors, alpha=0.85, capsize=3, height=0.62)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    lo = min(vals) - 0.6
    ax.set_xlim(lo, max(vals) + 0.5)
    ax.set_xlabel("Test accuracy (%)")
    for yi, v, e in zip(y, vals, errs):
        ax.text(v + e + 0.04, yi, f"{v:.2f}", va="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_ablation.png")
    plt.close(fig)


def plot_efficiency(summ: dict) -> None:
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    offsets = {
        "Plain-20": (6, -10),
        "ResNet-20": (6, 6),
        "SE-ResNet-20": (6, -14),
        "Plain-56": (-12, 8),
        "ResNet-56": (6, 5),
        "SE-ResNet-56": (6, -12),
    }
    for name, s in summ.items():
        if name not in offsets:
            continue
        x = s["params"] / 1e6
        y = s["acc_mean"]
        color = C_SE if name.startswith("SE") else C_RES if name.startswith("ResNet") else C_PLAIN
        ax.scatter(x, y, s=34, color=color, zorder=3)
        ax.annotate(name, (x, y), textcoords="offset points", xytext=offsets[name], fontsize=7)
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Test accuracy (%)")
    ax.set_title("Accuracy vs. model size")
    fig.tight_layout()
    fig.savefig(FIG / "fig_efficiency.png")
    plt.close(fig)


def plot_confusion(run_ids: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 3.0))
    for ax, (name, rid) in zip(axes, run_ids.items()):
        d = np.load(RESULTS / "preds" / f"{rid}.npz")
        cm = metrics.confusion_matrix(d["logits"], d["labels"]).astype(float)
        acc = (d["logits"].argmax(1) == d["labels"]).mean()
        cm /= cm.sum(1, keepdims=True)
        err = np.ma.masked_array(cm, mask=np.eye(10, dtype=bool))
        im = ax.imshow(err, cmap="Reds", vmin=0, vmax=0.22)
        for i in range(10):
            for j in range(10):
                if i != j and cm[i, j] >= 0.04:
                    ax.text(j, i, f"{cm[i, j] * 100:.0f}", ha="center", va="center", fontsize=5.5)
        ax.set_title(f"{name} (acc={100 * acc:.2f}%)", fontsize=8.5)
        ax.set_xticks(range(10))
        ax.set_yticks(range(10))
        ax.set_xticklabels(CLASSES, rotation=90, fontsize=5)
        ax.set_yticklabels(CLASSES, fontsize=5)
        ax.set_xlabel("Predicted")
    axes[0].set_ylabel("True")
    fig.colorbar(im, ax=axes, fraction=0.02, pad=0.02, label="error rate")
    fig.savefig(FIG / "fig_confusion.png", bbox_inches="tight")
    plt.close(fig)


def plot_reliability(run_ids: dict) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 3.0))
    ax.plot([0, 1], [0, 1], "--", color="k", lw=0.8)
    for name, rid, color in [
        ("Plain-20", run_ids["Plain-20"], C_PLAIN),
        ("ResNet-20", run_ids["ResNet-20"], C_RES),
        ("SE-ResNet-20", run_ids["SE-ResNet-20"], C_SE),
    ]:
        d = np.load(RESULTS / "preds" / f"{rid}.npz")
        p = metrics.softmax(d["logits"])
        conf, pred = p.max(1), p.argmax(1)
        correct = pred == d["labels"]
        bins = np.linspace(0, 1, 11)
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (conf > lo) & (conf <= hi)
            if m.sum() > 30:
                xs.append(conf[m].mean())
                ys.append(correct[m].mean())
        ax.plot(xs, ys, marker="o", ms=2.5, color=color, lw=1.2,
                label=f"{name} (ECE={metrics.ece(d['logits'], d['labels']):.3f})")
    ax.set_xlabel("Confidence")
    ax.set_ylabel("Accuracy")
    ax.set_title("Reliability diagram")
    ax.legend(frameon=False, fontsize=7, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / "fig_reliability.png")
    plt.close(fig)


@torch.no_grad()
def feature_pca(run_ids: dict) -> None:
    data = GPUCIFAR10(str(DATA), torch.device("cuda"))
    idx = torch.randperm(10000, device="cuda")[:2000]
    x = data.test_tensor()[idx]
    y = data.y_test[idx].cpu().numpy()
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.8))
    for ax, (name, rid) in zip(axes, run_ids.items()):
        model, _ = load_model(rid)
        feats = model.features(x).float().cpu().numpy()
        z = pca_2d(feats)
        for c in range(10):
            m = y == c
            ax.scatter(z[m, 0], z[m, 1], s=2, alpha=0.6, label=CLASSES[c])
        ax.set_title(name, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
    axes[-1].legend(markerscale=4, fontsize=5, frameon=False, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    fig.savefig(FIG / "fig_pca.png")
    plt.close(fig)


def plot_gradcam(run_ids: dict) -> None:
    data = GPUCIFAR10(str(DATA), torch.device("cuda"))
    g = torch.Generator(device="cuda").manual_seed(7)
    idx = torch.randperm(10000, device="cuda", generator=g)[:6]
    x = data.test_tensor()[idx]
    y = data.y_test[idx]
    imgs = data.x_test[idx].float().cpu().numpy() / 255.0
    fig, axes = plt.subplots(3, 6, figsize=(7.4, 4.0))
    for row, (name, rid) in enumerate(run_ids.items()):
        model, _ = load_model(rid)
        cam_fn = GradCAM(model, model.layer3[-1])
        cams = cam_fn(x)
        pred = model(x).argmax(1)
        for col in range(6):
            ax = axes[row, col]
            ax.imshow(np.transpose(imgs[col], (1, 2, 0)), interpolation="bilinear")
            cam = torch.from_numpy(cams[col])[None, None]
            cam = torch.nn.functional.interpolate(cam, size=(32, 32), mode="bilinear",
                                                  align_corners=False)[0, 0].numpy()
            ax.imshow(cam, cmap="jet", alpha=0.45, interpolation="bilinear")
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"T:{CLASSES[int(y[col])]}\nP:{CLASSES[int(pred[col])]}", fontsize=6)
            if col == 0:
                ax.set_ylabel(name, fontsize=7)
    fig.tight_layout()
    fig.savefig(FIG / "fig_gradcam.png")
    plt.close(fig)


@torch.no_grad()
def robustness(run_ids: dict) -> dict:
    data = GPUCIFAR10(str(DATA), torch.device("cuda"))
    kinds = ["noise", "blur", "contrast", "brightness"]
    out = {}
    for name, rid in run_ids.items():
        model, _ = load_model(rid)
        out[name] = {}
        for kind in kinds:
            out[name][kind] = {}
            for sev in (1, 2, 3):
                x = data.corrupted_test(kind, sev, seed=0)
                acc = metrics.accuracy(predict(model, x), data.y_test.cpu().numpy())
                out[name][kind][str(sev)] = 100 * acc
    with open(RESULTS / "robustness.json", "w", encoding="utf8") as f:
        json.dump(out, f, indent=2)
    return out


def plot_robustness(rob: dict) -> None:
    kinds = ["noise", "blur", "contrast", "brightness"]
    titles = ["Gaussian noise", "Gaussian blur", "Contrast", "Brightness"]
    fig, axes = plt.subplots(1, 4, figsize=(7.6, 2.2))
    for ax, kind, title in zip(axes, kinds, titles):
        for name, color in [("Plain-20", C_PLAIN), ("ResNet-20", C_RES), ("SE-ResNet-20", C_SE)]:
            ys = [rob[name][kind][str(s)] for s in (1, 2, 3)]
            ax.plot([1, 2, 3], ys, marker="o", ms=3, color=color, label=name, lw=1.2)
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("Severity")
        ax.set_xticks([1, 2, 3])
    axes[0].set_ylabel("Accuracy (%)")
    axes[-1].legend(frameon=False, fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "fig_robustness.png")
    plt.close(fig)


def plot_per_class(run_ids: dict) -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.6))
    width = 0.27
    xs = np.arange(10)
    for k, (name, rid, color) in enumerate([
        ("Plain-20", run_ids["Plain-20"], C_PLAIN),
        ("ResNet-20", run_ids["ResNet-20"], C_RES),
        ("SE-ResNet-20", run_ids["SE-ResNet-20"], C_SE),
    ]):
        d = np.load(RESULTS / "preds" / f"{rid}.npz")
        acc = metrics.per_class_accuracy(d["logits"], d["labels"])
        ax.bar(xs + (k - 1) * width, 100 * np.array(acc), width, color=color, label=name)
    ax.set_xticks(xs)
    ax.set_xticklabels(CLASSES, rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("Per-class accuracy (%)")
    ax.set_ylim(70, 100)
    ax.legend(frameon=False, ncol=3, fontsize=6, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    fig.savefig(FIG / "fig_per_class.png")
    plt.close(fig)


def bootstrap_diff(logits_a: np.ndarray, logits_b: np.ndarray, labels: np.ndarray, n_boot: int = 2000):
    ca = (logits_a.argmax(1) == labels).astype(np.float64)
    cb = (logits_b.argmax(1) == labels).astype(np.float64)
    rng = np.random.default_rng(0)
    diffs = []
    n = len(labels)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs.append(cb[idx].mean() - ca[idx].mean())
    diffs = np.array(diffs)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    TAB.mkdir(parents=True, exist_ok=True)
    runs = load_runs()
    if not runs:
        raise SystemExit("no runs found")
    g = group(runs)
    summ = summarize(g)
    print(f"{len(runs)} runs in {len(g)} groups")

    # main table
    main_names = ["Plain-20", "ResNet-20", "SE-ResNet-20"]
    rows = []
    for n in main_names:
        s = summ[n]
        rows.append([n, f"{s['acc_mean']:.2f} ± {s['acc_std']:.2f}", f"{s['f1_mean']:.2f}",
                     f"{s['ece_mean']:.4f}", f"{s['nll_mean']:.4f}", f"{s['params']/1e6:.3f}",
                     f"{s['macs']/1e6:.1f}", f"{s['time_mean_s']/60:.1f}", s["n"]])
    write_csv("main.csv", ["method", "acc", "macro_f1", "ece", "nll", "params_M", "macs_M", "train_min", "n"], rows)

    # depth table
    rows = []
    for n in ["Plain-20", "Plain-56", "ResNet-20", "ResNet-56", "SE-ResNet-56"]:
        if n in summ:
            s = summ[n]
            rows.append([n, f"{s['acc_mean']:.2f} ± {s['acc_std']:.2f}", f"{s['params']/1e6:.3f}", s["n"]])
    write_csv("depth.csv", ["method", "acc", "params_M", "n"], rows)

    # ablation table
    abl_names = [n for n in summ if n.startswith("SE-ResNet-20") or n.startswith("PreAct")]
    rows = []
    for n in abl_names:
        s = summ[n]
        rows.append([n, f"{s['acc_mean']:.2f} ± {s['acc_std']:.2f}", f"{s['ece_mean']:.4f}", s["n"]])
    write_csv("ablation.csv", ["variant", "acc", "ece", "n"], rows)

    run_ids = {}
    for n in main_names:
        run_ids[n] = pick(g, n)[0]
    # prefer the lowest seed for qualitative figures
    run_ids = {n: sorted(g[n], key=lambda kv: kv[1]["final"]["seed"])[0][0] for n in main_names}

    plot_curves(g)
    plot_depth(g, summ)
    plot_ablation(g, summ, [n for n in abl_names])
    plot_efficiency(summ)
    plot_confusion(run_ids)
    plot_reliability(run_ids)
    feature_pca(run_ids)
    plot_gradcam(run_ids)
    rob = robustness(run_ids)
    plot_robustness(rob)
    plot_per_class(run_ids)

    # per-class and robustness tables
    rows = []
    for n, rid in run_ids.items():
        d = np.load(RESULTS / "preds" / f"{rid}.npz")
        acc = metrics.per_class_accuracy(d["logits"], d["labels"])
        rows.append([n] + [f"{100*a:.2f}" for a in acc])
    write_csv("per_class.csv", ["method"] + CLASSES, rows)
    rows = []
    for n in run_ids:
        vals = [rob[n][k][str(s)] for k in ["noise", "blur", "contrast", "brightness"] for s in (1, 2, 3)]
        rows.append([n, f"{np.mean(vals):.2f}"] + [f"{v:.2f}" for v in vals])
    write_csv("robustness.csv", ["method", "mean"] + [f"{k}_{s}" for k in ["noise", "blur", "contrast", "brightness"] for s in (1, 2, 3)], rows)

    # paired statistics on the first seed
    stats = {}
    for a, b in [("Plain-20", "ResNet-20"), ("ResNet-20", "SE-ResNet-20")]:
        da = np.load(RESULTS / "preds" / f"{run_ids[a]}.npz")
        db = np.load(RESULTS / "preds" / f"{run_ids[b]}.npz")
        mc = metrics.mcnemar(da["logits"], db["logits"], da["labels"])
        lo, hi = bootstrap_diff(da["logits"], db["logits"], da["labels"])
        stats[f"{b}_vs_{a}"] = {
            "mcnemar": mc,
            "bootstrap_diff_pp": [100 * lo, 100 * hi],
            "delta_pp": 100 * (metrics.accuracy(db["logits"], db["labels"]) - metrics.accuracy(da["logits"], da["labels"])),
        }

    summary = {
        "groups": summ,
        "paired": stats,
        "robustness": rob,
        "run_ids": run_ids,
        "config": {
            "dataset": "CIFAR-10",
            "epochs": runs[list(runs)[0]]["final"]["epochs"],
        },
    }
    with open(RESULTS / "summary.json", "w", encoding="utf8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("wrote", RESULTS / "summary.json")
    for k, v in stats.items():
        print(k, v)


if __name__ == "__main__":
    main()
