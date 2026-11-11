"""Assemble a self-contained submission folder (paper + code/weights + notes)."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT.parent / "提交材料_ResNet"
SUMMARY = json.loads((ROOT / "results" / "summary.json").read_text(encoding="utf8"))
G = SUMMARY["groups"]
REPO = "https://github.com/zyj1023448/cifar10-resnet-repro"

if OUT.exists():
    shutil.rmtree(OUT)
(OUT / "01_论文").mkdir(parents=True)
(OUT / "02_代码与权重").mkdir(parents=True)

for name in ["CIFAR10_ResNet_SE_复现与实证分析.docx", "CIFAR10_ResNet_SE_复现与实证分析.pdf"]:
    shutil.copy2(ROOT / "paper" / name, OUT / "01_论文" / name)

ignore = shutil.ignore_patterns(".git", "data", ".tmp", "__pycache__", "*.pyc", ".venv*")
for item in ROOT.iterdir():
    if item.name in {".git", "data", ".tmp", "__pycache__"}:
        continue
    dst = OUT / "02_代码与权重" / item.name
    if item.is_dir():
        shutil.copytree(item, dst, ignore=ignore)
    else:
        shutil.copy2(item, dst)

main_rows = "\n".join(
    f"| {n} | {G[n]['acc_mean']:.2f} ± {G[n]['acc_std']:.2f} | {G[n]['ece_mean']:.4f} | {G[n]['n']} |"
    for n in ["Plain-20", "ResNet-20", "SE-ResNet-20"]
)
notes = f"""# 论文提交材料

## 1 论文

- `01_论文/CIFAR10_ResNet_SE_复现与实证分析.docx`：Word 版，含中英文题目、作者、单位、摘要、
  关键词、引言、相关工作、方法、实验、结论、26 条参考文献、10 幅图、6 张表。
- `01_论文/CIFAR10_ResNet_SE_复现与实证分析.pdf`：PDF 版（15 页）。

## 2 代码与权重

- GitHub 仓库：{REPO}
- 本地副本：`02_代码与权重/`（含 `src/`、`scripts/`、`weights/`、`results/`、`paper/`）。
- 共 {len(list((ROOT / 'weights').glob('*.pt')))} 组训练权重：
  - 基线 Plain-20：`weights/plain20_s42.pt`（另有 s43、s44）。
  - 复现的 ResNet-20：`weights/resnet20_s42.pt`。
  - 提出方法 SE-ResNet-20：`weights/se20_s42.pt`（另有 s43、s44）。
  - 其余为 56 层、SE 设计与 Mixup/标签平滑消融的权重。
- 每个 `.pt` 同时保存 `state_dict`、训练配置和最终指标，可直接
  `torch.load(..., weights_only=False)` 读取。
- 权重校验和：`weights/SHA256SUMS.txt`；汇总指标：`results/summary.json`；
  逐样本预测：`results/preds/*.npz`。

## 3 主要结果（CIFAR-10 测试集，均值 ± 标准差）

| 方法 | 准确率/% | ECE | 种子数 |
|---|---:|---:|---:|
{main_rows}

- 残差连接相对普通网络：+{G['ResNet-20']['acc_mean'] - G['Plain-20']['acc_mean']:.2f} 个百分点。
- 56 层：Plain-56 为 {G['Plain-56']['acc_mean']:.2f}%，ResNet-56 为 {G['ResNet-56']['acc_mean']:.2f}%，
  深度对普通网络是负收益，对残差网络是正收益。
- SE 的精度增益仅 {G['SE-ResNet-20']['acc_mean'] - G['ResNet-20']['acc_mean']:+.2f} 个百分点，但
  ECE 从 {G['ResNet-20']['ece_mean']:.4f} 降到 {G['SE-ResNet-20']['ece_mean']:.4f}，四类损坏下平均
  准确率高约 2 个百分点。
- Mixup 在 SE 基础上再提升 {G['SE-ResNet-20 + Mixup(0.2)']['acc_mean'] - G['SE-ResNet-20']['acc_mean']:+.2f}
  个百分点；标签平滑反而降到 {G['SE-ResNet-20 + LS(0.1)']['acc_mean']:.2f}% 且 ECE 变差。

## 4 复现

```powershell
python -m venv .venv
.venv\\Scripts\\python.exe -m pip install -r requirements.txt
python scripts/prepare_cifar.py   # 下载/缓存 CIFAR-10
python scripts/run_grid.py        # 26 组训练，单卡 RTX 5060 约 3 小时
python -m src.analyze             # 汇总、检验、全部图表
python paper/build_paper.py       # 由 summary.json 重新生成论文 DOCX
```

环境：RTX 5060 Laptop 8 GB，CUDA 12.8，PyTorch 2.11.0+cu128，Python 3.13。
"""
(OUT / "提交说明.md").write_text(notes, encoding="utf8")

shutil.make_archive(str(OUT), "zip", root_dir=OUT.parent, base_dir=OUT.name)
print("wrote", OUT)
print("zip", OUT.with_suffix(".zip"), (OUT.with_suffix(".zip")).stat().st_size / 1e6, "MB")
