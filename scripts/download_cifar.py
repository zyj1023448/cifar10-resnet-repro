"""Parallel, resumable CIFAR-10 download.

The public Toronto mirror throttles single connections, so the archive is fetched
as independent byte ranges by several threads and then concatenated.
"""
from __future__ import annotations

import os
import tarfile
import threading
import time
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
URL = "http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
DEST = os.path.join(DATA, "cifar-10-python.tar.gz")
EXPECTED = 170498071
N_THREADS = 8
CHUNK = 2 * 1024 * 1024


def fetch_range(start: int, end: int, path: str, tid: int) -> None:
    for attempt in range(30):
        have = os.path.getsize(path) if os.path.exists(path) else 0
        want = end - start + 1
        if have >= want:
            return
        headers = {"Range": f"bytes={start + have}-{end}"}
        try:
            with urllib.request.urlopen(urllib.request.Request(URL, headers=headers), timeout=25) as r, open(path, "ab") as f:
                while True:
                    chunk = r.read(512 * 1024)
                    if not chunk:
                        break
                    f.write(chunk)
        except Exception as e:  # noqa: BLE001
            print(f"  thread {tid} attempt {attempt}: {type(e).__name__}", flush=True)
            time.sleep(1)


def main() -> None:
    os.makedirs(DATA, exist_ok=True)
    # Probe total size.
    req = urllib.request.Request(URL, headers={"Range": "bytes=0-0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        total = int(r.headers["Content-Range"].split("/")[1])
    assert total == EXPECTED, total
    spans = [(s, min(s + CHUNK - 1, total - 1)) for s in range(0, total, CHUNK)]
    parts = [os.path.join(DATA, f".chunk_{i:03d}") for i in range(len(spans))]
    print(f"{total} bytes in {len(spans)} chunks", flush=True)
    threads = []
    t0 = time.time()
    for i, ((s, e), p) in enumerate(zip(spans, parts)):
        t = threading.Thread(target=fetch_range, args=(s, e, p, i), daemon=True)
        t.start()
        threads.append(t)
    while any(t.is_alive() for t in threads):
        done = sum(os.path.getsize(p) for p in parts if os.path.exists(p))
        print(f"  {done / 1e6:6.1f} MB  {done / 1e6 / (time.time() - t0):5.2f} MB/s", flush=True)
        time.sleep(10)
    for t in threads:
        t.join()
    tmp = DEST + ".part"
    with open(tmp, "wb") as out:
        for p in parts:
            with open(p, "rb") as f:
                while True:
                    b = f.read(8 * 1024 * 1024)
                    if not b:
                        break
                    out.write(b)
    assert os.path.getsize(tmp) == total
    os.replace(tmp, DEST)
    for p in parts:
        os.remove(p)
    with tarfile.open(DEST) as t:
        t.extractall(DATA)
    print("extracted", sorted(os.listdir(os.path.join(DATA, "cifar-10-batches-py"))), flush=True)


if __name__ == "__main__":
    main()
