#!/usr/bin/env python
"""Download ONLY the leather category of MVTec AD from a public Hugging Face mirror, then import it.

    python scripts/download_mvtec_leather.py            # ~525 MB, 463 files -> data/leather
    python scripts/download_mvtec_leather.py --force    # replace whatever is in data/leather

Source: https://huggingface.co/datasets/foersben/mvtec-ad (a mirror of MVTec AD, CC BY-NC-SA 4.0, non-commercial).
Files land in a cache folder first (resumable: files already there with the right size are skipped), then
scripts/import_mvtec_leather.py copies them into data/leather and runs the data checks.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import sys
import urllib.parse
import urllib.request
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REPO = "foersben/mvtec-ad"
CATEGORY = "leather"
LIST_URL = f"https://huggingface.co/api/datasets/{REPO}/tree/main/{CATEGORY}?recursive=true"
FILE_URL = f"https://huggingface.co/datasets/{REPO}/resolve/main/"
DEFAULT_CACHE = ROOT / ".cache" / "mvtec_hf"


def safe_entries(listing: list[dict]) -> list[tuple[str, int]]:
    """(path, size) for files under leather/, rejecting anything that could escape the cache folder."""
    out = []
    for item in listing:
        if item.get("type") != "file":
            continue
        path = str(item.get("path", ""))
        parts = PurePosixPath(path).parts
        if not parts or parts[0] != CATEGORY or ".." in parts or path.startswith("/"):
            continue
        out.append((path, int(item.get("size") or 0)))
    return sorted(out)


def list_files(url: str = LIST_URL) -> list[tuple[str, int]]:
    with urllib.request.urlopen(url, timeout=60) as resp:
        return safe_entries(json.load(resp))


def download_one(path: str, size: int, cache: Path) -> str | None:
    target = cache.joinpath(*PurePosixPath(path).parts)
    if target.exists() and (size == 0 or target.stat().st_size == size):
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".part")
    err: Exception | None = None
    for _ in range(3):
        try:
            urllib.request.urlretrieve(FILE_URL + urllib.parse.quote(path), tmp)
            tmp.replace(target)
            return None
        except Exception as exc:  # noqa: BLE001 - retried, then reported
            err = exc
    return f"{path}: {err}"


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cache", default=str(DEFAULT_CACHE), help="where raw downloads go (git-ignored)")
    p.add_argument("--out", default=str(ROOT / "data" / CATEGORY))
    p.add_argument("--force", action="store_true", help="replace whatever is in --out")
    p.add_argument("--workers", type=int, default=16)
    a = p.parse_args(argv)
    cache = Path(a.cache)
    files = list_files()
    total_mb = sum(s for _, s in files) / 1e6
    print(f"{len(files)} files, {total_mb:.0f} MB from huggingface.co/datasets/{REPO} -> {cache}")
    with cf.ThreadPoolExecutor(a.workers) as ex:
        fails = [r for r in ex.map(lambda f: download_one(f[0], f[1], cache), files) if r]
    if fails:
        print(f"{len(fails)} downloads failed (re-run to retry):", *fails[:5], sep="\n  ")
        return 1
    from scripts import import_mvtec_leather

    return import_mvtec_leather.main([str(cache / CATEGORY), "--out", a.out] + (["--force"] if a.force else []))


if __name__ == "__main__":
    raise SystemExit(main())
