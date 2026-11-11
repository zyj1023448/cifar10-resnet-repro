"""Download PyTorch CUDA wheels into .tmp/wheels (pip's PEP 658 metadata
download is blocked in this sandbox, so wheels are fetched directly)."""
import os
import sys
import time
import urllib.request

WHEELS = {
    "torch-2.11.0+cu128-cp313-cp313-win_amd64.whl":
        "https://download-r2.pytorch.org/whl/cu128/torch-2.11.0%2Bcu128-cp313-cp313-win_amd64.whl",
    "torchvision-0.26.0+cu128-cp313-cp313-win_amd64.whl":
        "https://download-r2.pytorch.org/whl/cu128/torchvision-0.26.0%2Bcu128-cp313-cp313-win_amd64.whl",
}

out_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".tmp", "wheels")
os.makedirs(out_dir, exist_ok=True)


def human(n):
    return f"{n / 1024 ** 3:.2f} GiB"


for name, url in WHEELS.items():
    dest = os.path.join(out_dir, name)
    if os.path.exists(dest) and os.path.getsize(dest) > 100 * 1024 * 1024:
        print("exists", name, human(os.path.getsize(dest)), flush=True)
        continue
    tmp = dest + ".part"
    print("downloading", name, flush=True)
    t0 = time.time()
    with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
        total = int(r.headers.get("Content-Length", 0))
        done = 0
        last = 0
        while True:
            chunk = r.read(8 * 1024 * 1024)
            if not chunk:
                break
            f.write(chunk)
            done += len(chunk)
            if time.time() - last > 5:
                last = time.time()
                pct = 100 * done / total if total else 0
                print(f"  {pct:5.1f}% {human(done)} / {human(total)} "
                      f"{done / 1e6 / max(time.time() - t0, 1e-6):.1f} MB/s", flush=True)
    os.replace(tmp, dest)
    print("done", name, human(os.path.getsize(dest)), f"{time.time() - t0:.0f}s", flush=True)
