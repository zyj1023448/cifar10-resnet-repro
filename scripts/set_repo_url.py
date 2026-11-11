"""Bind the public repository URL into README, CITATION.cff and the paper."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("url")
    args = ap.parse_args()
    url = args.url.rstrip("/")
    (ROOT / "REPO_URL.txt").write_text(url + "\n", encoding="utf8")

    cff = ROOT / "CITATION.cff"
    text = cff.read_text(encoding="utf8")
    text = re.sub(r"repository-code:.*", f'repository-code: "{url}"', text)
    cff.write_text(text, encoding="utf8")

    readme = ROOT / "README.md"
    if readme.exists():
        text = readme.read_text(encoding="utf8")
        text = re.sub(r"\*\*Code and weights:\*\*.*", f"**Code and weights:** {url}", text)
        readme.write_text(text, encoding="utf8")
    print("bound", url)


if __name__ == "__main__":
    main()
