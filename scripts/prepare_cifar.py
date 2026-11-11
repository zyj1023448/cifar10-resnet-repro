"""Fetch CIFAR-10 from the fast.ai S3 mirror and cache it as tensors.

The official Toronto mirror is frequently unreachable from this network, so the
archive is fetched as parallel byte ranges from a public S3 mirror, decoded from
the tar stream, and stored as a single uint8 tensor file used by src/data.py.
"""
from __future__ import annotations

import io
import os
import tarfile
import threading
import time
import urllib.request

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
URL = "https://s3.amazonaws.com/fast-ai-imageclas/cifar10.tgz"
TGZ = os.path.join(DATA, "cifar10.tgz")
CACHE = os.path.join(DATA, "cifar10_tensors.pt")
N_THREADS = 16
CHUNK = 4 * 1024 * 1024
CLASSES = ["airplane", "automobile", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]


def download() -> int:
    os.makedirs(DATA, exist_ok=True)
    if os.path.exists(CACHE):
        print("cache exists:", CACHE, flush=True)
        return 0
    req = urllib.request.Request(URL, method="HEAD")
    total = int(urllib.request.urlopen(req, timeout=30).headers["Content-Length"])
    spans = [(s, min(s + CHUNK - 1, total - 1)) for s in range(0, total, CHUNK)]
    parts = [os.path.join(DATA, f".cifar_{i:03d}") for i in range(len(spans))]
    lock = threading.Lock()

    def fetch(i: int) -> None:
        s, e = spans[i]
        path = parts[i]
        for _ in range(20):
            have = os.path.getsize(path) if os.path.exists(path) else 0
            if have >= e - s + 1:
                return
            try:
                req = urllib.request.Request(URL, headers={"Range": f"bytes={s + have}-{e}"})
                with urllib.request.urlopen(req, timeout=30) as r, open(path, "ab") as f:
                    while True:
                        b = r.read(512 * 1024)
                        if not b:
                            break
                        f.write(b)
            except Exception:  # noqa: BLE001 - transient
                time.sleep(0.5)

    threads = [threading.Thread(target=fetch, args=(i,)) for i in range(len(spans))]
    t0 = time.time()
    for t in threads:
        t.start()
    while any(t.is_alive() for t in threads):
        done = sum(os.path.getsize(p) for p in parts if os.path.exists(p))
        print(f"  {done / 1e6:6.1f}/{total / 1e6:.1f} MB "
              f"{done / 1e6 / max(time.time() - t0, 1e-6):.2f} MB/s", flush=True)
        time.sleep(10)
    for t in threads:
        t.join()
    tmp = TGZ + ".part"
    with open(tmp, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                while True:
                    b = f.read(8 * 1024 * 1024)
                    if not b:
                        break
                    out.write(b)
    os.replace(tmp, TGZ)
    for p in parts:
        os.remove(p)
    print("downloaded", os.path.getsize(TGZ), flush=True)
    return total


def parse() -> None:
    if os.path.exists(CACHE):
        print("cache exists:", CACHE, flush=True)
        return
    xs = {"train": [], "test": []}
    ys = {"train": [], "test": []}
    with tarfile.open(TGZ, "r:gz") as tf:
        for m in tf:
            if not m.isfile() or not m.name.lower().endswith(".png"):
                continue
            parts = m.name.replace("\\", "/").split("/")
            if len(parts) < 4:
                continue
            split, cls = parts[-3], parts[-2]
            if split not in xs or cls not in CLASSES:
                continue
            with tf.extractfile(m) as f:
                arr = np.asarray(Image.open(io.BytesIO(f.read())).convert("RGB"), dtype=np.uint8)
            if arr.shape != (32, 32, 3):
                continue
            xs[split].append(arr)
            ys[split].append(CLASSES.index(cls))
    out = {
        "x_train": torch.from_numpy(np.stack(xs["train"])).permute(0, 3, 1, 2).contiguous(),
        "y_train": torch.tensor(ys["train"], dtype=torch.long),
        "x_test": torch.from_numpy(np.stack(xs["test"])).permute(0, 3, 1, 2).contiguous(),
        "y_test": torch.tensor(ys["test"], dtype=torch.long),
    }
    print({k: tuple(v.shape) for k, v in out.items()}, flush=True)
    assert out["x_train"].shape[0] == 50000 and out["x_test"].shape[0] == 10000
    torch.save(out, CACHE)
    print("saved", CACHE, os.path.getsize(CACHE) / 1e6, "MB", flush=True)


if __name__ == "__main__":
    download()
    parse()
