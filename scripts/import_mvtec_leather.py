#!/usr/bin/env python
"""Copy ONLY the leather category of the real MVTec AD dataset into data/leather.

You download the data yourself (MVTec asks you to fill a short form and accept the licence):
    https://www.mvtec.com/company/research/datasets/mvtec-ad   (Downloads section)
Then point this script at whatever you got:

    python scripts/import_mvtec_leather.py ~/Downloads/mvtec_anomaly_detection.tar.xz
    python scripts/import_mvtec_leather.py ~/Downloads/leather.tar.xz          # if the site offers per-category files
    python scripts/import_mvtec_leather.py ~/Downloads/mvtec_anomaly_detection # an already-unpacked folder
    python scripts/import_mvtec_leather.py ~/Downloads/archive.zip

Accepted: .tar, .tar.xz, .tar.gz, .tgz, .zip, or a folder. Everything except the leather folder is ignored,
so the huge full archive is fine (reading it end to end still takes a few minutes).

Licence: MVTec AD is CC BY-NC-SA 4.0, non-commercial. Fine for practice; do not build a product on it.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CATEGORY = "leather"
KEEP_SUFFIXES = {".png", ".jpg", ".jpeg", ".txt", ".md"}  # images, masks, and the dataset's own readme/licence text


def relative_in_category(name: str) -> PurePosixPath | None:
    """'x/mvtec_anomaly_detection/leather/test/cut/000.png' -> 'test/cut/000.png'. None if not leather or unsafe."""
    parts = PurePosixPath(name.replace("\\", "/")).parts
    if not parts or parts[0] == "/" or ".." in parts or any(":" in p for p in parts[:1]):
        return None
    if CATEGORY not in parts:
        return None
    rest = parts[parts.index(CATEGORY) + 1 :]
    if not rest:
        return None
    rel = PurePosixPath(*rest)
    if rel.suffix.lower() not in KEEP_SUFFIXES:
        return None
    return rel


def _write(dest_root: Path, rel: PurePosixPath, data_stream) -> None:
    target = dest_root.joinpath(*rel.parts)
    target.parent.mkdir(parents=True, exist_ok=True)
    with open(target, "wb") as out:
        shutil.copyfileobj(data_stream, out)


def import_from_tar(src: Path, dest: Path) -> int:
    count = 0
    with tarfile.open(src, "r:*") as tar:
        for member in tar:
            if not member.isfile():
                continue
            rel = relative_in_category(member.name)
            if rel is None:
                continue
            stream = tar.extractfile(member)
            if stream is None:
                continue
            _write(dest, rel, stream)
            count += 1
    return count


def import_from_zip(src: Path, dest: Path) -> int:
    count = 0
    with zipfile.ZipFile(src) as zf:
        for info in zf.infolist():
            if info.is_dir():
                continue
            rel = relative_in_category(info.filename)
            if rel is None:
                continue
            with zf.open(info) as stream:
                _write(dest, rel, stream)
            count += 1
    return count


def find_leather_dir(src: Path) -> Path | None:
    if (src / "train").is_dir() and (src / "test").is_dir():
        return src
    for candidate in [src / CATEGORY, *src.rglob(CATEGORY)]:
        if candidate.is_dir() and (candidate / "train").is_dir() and (candidate / "test").is_dir():
            return candidate
    return None


def import_from_dir(src: Path, dest: Path) -> int:
    leather = find_leather_dir(src)
    if leather is None:
        return 0
    count = 0
    for path in leather.rglob("*"):
        if not path.is_file():
            continue
        rel = PurePosixPath(*path.relative_to(leather).parts)
        if rel.suffix.lower() not in KEEP_SUFFIXES:
            continue
        with open(path, "rb") as stream:
            _write(dest, rel, stream)
        count += 1
    return count


def import_leather(src: Path, dest: Path, force: bool = False) -> int:
    src, dest = Path(src).expanduser(), Path(dest)
    if not src.exists():
        raise FileNotFoundError(f"{src} does not exist")
    if dest.exists() and any(dest.iterdir()):
        if not force:
            raise FileExistsError(f"{dest} is not empty. Pass --force to replace it (fake data is replaced; real data too).")
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)
    name = src.name.lower()
    if src.is_dir():
        count = import_from_dir(src, dest)
    elif name.endswith(".zip"):
        count = import_from_zip(src, dest)
    elif name.endswith((".tar", ".tar.xz", ".tar.gz", ".tgz", ".txz")):
        count = import_from_tar(src, dest)
    else:
        raise ValueError(f"Don't know how to read '{src.name}'. Use .tar.xz, .tar.gz, .tar, .zip or a folder.")
    if count == 0:
        shutil.rmtree(dest, ignore_errors=True)
        raise ValueError(f"No '{CATEGORY}' images found inside {src}. Is this the right file?")
    return count


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", help="archive or folder you downloaded from MVTec")
    p.add_argument("--out", default=str(ROOT / "data" / "leather"))
    p.add_argument("--force", action="store_true", help="replace whatever is in --out")
    a = p.parse_args(argv)
    try:
        count = import_leather(Path(a.source), Path(a.out), a.force)
    except (FileNotFoundError, FileExistsError, ValueError, tarfile.TarError, zipfile.BadZipFile) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Imported {count} leather files into {a.out}.\n")
    from scripts import check_setup

    checks = check_setup.check_data(Path(a.out))
    print(check_setup.render(checks))
    return 1 if any(c.status == "fail" for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
