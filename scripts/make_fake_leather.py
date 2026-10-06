#!/usr/bin/env python
"""Generate a FAKE leather dataset in the same folder layout as MVTec AD, so you can build
and test the whole inspector before (or without) downloading the real data.

    python scripts/make_fake_leather.py                       # -> data/leather, 120 images
    python scripts/make_fake_leather.py --out data/leather --seed 1 --size 512 --force

Layout (same as MVTec AD, one category):
    <out>/train/good/000.png ...
    <out>/test/good/000.png ...
    <out>/test/<defect>/000.png ...
    <out>/ground_truth/<defect>/000_mask.png ...      (white = defective pixels)
    <out>/FAKE_DATA.txt                               marker so fake data is never mistaken for real
    <out>/_fake_manifest.csv                          severity/area per image (ground truth, for analysis only)

Defects: color, cut, fold, glue, poke. These are the leather defect names as I remember them
from MVTec AD; the images here are procedural look-alikes, NOT real photos. Real MVTec results
will differ, so treat anything measured on this data as a plumbing test, not a result.

The agent must never read `_fake_manifest.csv` or `ground_truth/`; only the eval harness may.
Needs numpy and pillow only.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

DEFECTS = ("color", "cut", "fold", "glue", "poke")
MARKER = "FAKE_DATA.txt"
MANIFEST = "_fake_manifest.csv"
BASE_COLOR = np.array([112.0, 72.0, 46.0])


# --------------------------------------------------------------------------- helpers


def blur(a: np.ndarray, sigma: float) -> np.ndarray:
    """Gaussian blur of a 2-D array via FFT (wraps at the edges, fine for texture)."""
    h, w = a.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    transfer = np.exp(-2.0 * (np.pi * sigma) ** 2 * (fx**2 + fy**2))
    return np.fft.irfft2(np.fft.rfft2(a) * transfer, s=(h, w))


def unit(a: np.ndarray) -> np.ndarray:
    return (a - a.mean()) / (a.std() + 1e-9)


def stroke(size: int, pts: np.ndarray, width: float, edge: float = 1.0) -> np.ndarray:
    """Alpha map (0..1) of a thick polyline: solid core of `width`, `edge` px soft falloff."""
    alpha = np.zeros((size, size))
    radius = width / 2.0
    reach = int(np.ceil(radius + edge)) + 1
    for x, y in pts:
        x0, x1 = max(0, int(x) - reach), min(size, int(x) + reach + 1)
        y0, y1 = max(0, int(y) - reach), min(size, int(y) + reach + 1)
        if x0 >= x1 or y0 >= y1:
            continue
        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist = np.hypot(xx - x, yy - y)
        local = np.clip(1.0 - (dist - radius) / edge, 0.0, 1.0)
        alpha[y0:y1, x0:x1] = np.maximum(alpha[y0:y1, x0:x1], local)
    return alpha


def curve(rng: np.random.Generator, size: int, length: float, bend: float = 0.004) -> tuple[np.ndarray, np.ndarray]:
    """A gently bending path fully inside the image. Returns (points, unit normal at the start)."""
    margin = size * 0.1
    for _ in range(50):
        x, y = rng.uniform(margin, size - margin, 2)
        angle = rng.uniform(0, 2 * np.pi)
        turn = rng.normal(0, bend)
        pts = []
        for _ in range(int(length)):
            pts.append((x, y))
            angle += turn + rng.normal(0, bend / 2)
            x += np.cos(angle)
            y += np.sin(angle)
        arr = np.array(pts)
        if arr.min() >= margin / 2 and arr.max() <= size - margin / 2:
            break
    start_angle = np.arctan2(arr[min(5, len(arr) - 1), 1] - arr[0, 1], arr[min(5, len(arr) - 1), 0] - arr[0, 0])
    normal = np.array([-np.sin(start_angle), np.cos(start_angle)])
    return arr, normal


def blob(rng: np.random.Generator, size: int, rx: float, ry: float) -> np.ndarray:
    """Soft irregular blob. Returns a field that is > 0 inside; clip(field / 0.25, 0, 1) is its alpha."""
    margin = max(rx, ry) * 1.1
    cx, cy = rng.uniform(margin, size - margin, 2) if size > 2 * margin else (size / 2, size / 2)
    yy, xx = np.mgrid[0:size, 0:size]
    angle = rng.uniform(0, np.pi)
    dx, dy = xx - cx, yy - cy
    u = dx * np.cos(angle) + dy * np.sin(angle)
    v = -dx * np.sin(angle) + dy * np.cos(angle)
    dist = np.sqrt((u / rx) ** 2 + (v / ry) ** 2)
    wobble = unit(blur(rng.standard_normal((size, size)), max(rx, ry) / 3.0))
    return 1.0 - dist + 0.18 * wobble


# --------------------------------------------------------------------------- base texture


def base_texture(seed: int, size: int) -> np.ndarray:
    """Float RGB (0..255) 'leather': pebbly grain with dark creases between pebbles."""
    rng = np.random.default_rng([seed, 1])
    s = size / 512.0
    fine = unit(blur(rng.standard_normal((size, size)), 1.2 * s))
    mid = unit(blur(rng.standard_normal((size, size)), 3.5 * s))
    broad = unit(blur(rng.standard_normal((size, size)), 14 * s))
    band = unit(blur(rng.standard_normal((size, size)), 5 * s))
    creases = np.exp(-((band / 0.28) ** 2))  # lines where the band-passed noise crosses zero
    yy, xx = np.mgrid[0:size, 0:size] / size
    gx, gy = rng.uniform(-0.05, 0.05, 2)
    lighting = 1.0 + gx * (xx - 0.5) * 2 + gy * (yy - 0.5) * 2  # slight lighting gradient

    lum = (1.0 + 0.09 * fine + 0.11 * mid + 0.06 * broad - 0.20 * creases) * lighting
    tint = 1.0 + rng.normal(0, 0.03, 3)  # per-image colour jitter
    img = BASE_COLOR[None, None, :] * tint[None, None, :] * lum[..., None]
    img += rng.normal(0, 2.0, img.shape)  # sensor noise
    return np.clip(img, 0, 255)


# --------------------------------------------------------------------------- defects


def _defect_rng(seed: int) -> np.random.Generator:
    return np.random.default_rng([seed, 2])  # separate stream: the base texture is unchanged by defects


def apply_color(img: np.ndarray, rng: np.random.Generator, sev: float) -> tuple[np.ndarray, np.ndarray]:
    size = img.shape[0]
    tints = [(0.78, 0.98, 0.82), (0.7, 0.7, 0.72), (1.25, 0.92, 0.72), (0.62, 0.52, 0.52)]
    tint = np.array(tints[rng.integers(len(tints))])
    field = blob(rng, size, rng.uniform(0.05, 0.14) * size, rng.uniform(0.04, 0.10) * size)
    alpha = np.clip(field / 0.25, 0, 1)
    effective = 1.0 + (tint - 1.0) * sev
    out = img * (1 - alpha[..., None]) + img * effective[None, None, :] * alpha[..., None]
    return out, alpha > 0.5


def apply_cut(img: np.ndarray, rng: np.random.Generator, sev: float) -> tuple[np.ndarray, np.ndarray]:
    size = img.shape[0]
    s = size / 512.0
    pts, _ = curve(rng, size, rng.uniform(0.12, 0.35) * size, bend=0.003)
    width = (1.6 + 2.4 * sev) * s
    core = stroke(size, pts, width)
    wide = stroke(size, pts, width + 5 * s, edge=2.0 * s + 0.5)
    out = img * (1 - 0.85 * sev * core[..., None])
    out = out * (1 + 0.18 * sev * np.clip(wide - core, 0, 1))[..., None]
    mask = stroke(size, pts, width + 2 * s) > 0.5
    return out, mask


def apply_fold(img: np.ndarray, rng: np.random.Generator, sev: float) -> tuple[np.ndarray, np.ndarray]:
    size = img.shape[0]
    s = size / 512.0
    pts, normal = curve(rng, size, rng.uniform(0.4, 0.8) * size, bend=0.0015)
    w = rng.uniform(14, 24) * s
    dark = blur(stroke(size, pts - normal * w / 3, w * 0.6), 2.5 * s)
    light = blur(stroke(size, pts + normal * w / 3, w * 0.6), 2.5 * s)
    shade = 1 - 0.32 * sev * np.clip(dark * 1.6, 0, 1) + 0.24 * sev * np.clip(light * 1.6, 0, 1)
    mask = stroke(size, pts, w) > 0.5
    return img * shade[..., None], mask


def apply_glue(img: np.ndarray, rng: np.random.Generator, sev: float) -> tuple[np.ndarray, np.ndarray]:
    size = img.shape[0]
    s = size / 512.0
    field = blob(rng, size, rng.uniform(0.04, 0.10) * size, rng.uniform(0.03, 0.08) * size)
    alpha = np.clip(field / 0.25, 0, 1)
    smooth = np.stack([blur(img[..., c], 5 * s) for c in range(3)], axis=-1)  # glue hides the grain
    mixed = img * (1 - 0.75 * sev * alpha[..., None]) + smooth * (0.75 * sev * alpha[..., None])
    mixed = mixed + (22 * sev * alpha)[..., None] + (45 * sev * np.clip(field, 0, 1) ** 2)[..., None]  # sheen
    return mixed, alpha > 0.5


def apply_poke(img: np.ndarray, rng: np.random.Generator, sev: float) -> tuple[np.ndarray, np.ndarray]:
    size = img.shape[0]
    s = size / 512.0
    radius = rng.uniform(5, 11) * s * (0.7 + 0.3 * sev)
    margin = radius * 3
    cx, cy = rng.uniform(margin, size - margin, 2)
    yy, xx = np.mgrid[0:size, 0:size]
    dist = np.hypot(xx - cx, yy - cy)
    hole = np.clip((radius - dist) / 1.5, 0, 1)
    rim = np.exp(-(((dist - radius * 1.6) / (radius * 0.6)) ** 2))
    out = img * (1 - (0.5 + 0.45 * sev) * hole[..., None])
    out = out * (1 + 0.14 * sev * rim)[..., None]
    return out, dist <= radius + 1.0


APPLY = {"color": apply_color, "cut": apply_cut, "fold": apply_fold, "glue": apply_glue, "poke": apply_poke}


def render(seed: int, size: int = 512, defect: str | None = None, severity: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    """One image. The same `seed` always gives the same base leather, with or without a defect.

    Returns (uint8 RGB array, bool mask of defective pixels; all False for good images).
    """
    img = base_texture(seed, size)
    if defect is None:
        return img.round().astype(np.uint8), np.zeros((size, size), dtype=bool)
    if defect not in APPLY:
        raise ValueError(f"unknown defect '{defect}', expected one of {DEFECTS}")
    out, mask = APPLY[defect](img, _defect_rng(seed), float(severity))
    return np.clip(out, 0, 255).round().astype(np.uint8), mask


# --------------------------------------------------------------------------- dataset writer


def make_dataset(
    out: Path,
    seed: int = 0,
    size: int = 512,
    n_train: int = 40,
    n_test_good: int = 20,
    per_defect: int = 12,
    force: bool = False,
) -> list[dict]:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        if not force:
            raise FileExistsError(f"{out} is not empty. Pass --force to replace it.")
        if not (out / MARKER).exists():
            raise FileExistsError(f"{out} has no {MARKER}, so it may be REAL data. Refusing to delete it.")
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / MARKER).write_text(
        "This folder holds FAKE, procedurally generated leather images (scripts/make_fake_leather.py).\n"
        "Do not report results from it as real. Delete it and use the real MVTec AD 'leather' folder instead.\n"
    )

    rows: list[dict] = []
    rng = np.random.default_rng([seed, 0])

    def save(img: np.ndarray, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(img).save(path)

    next_seed = int(rng.integers(1, 10**6)) * 1000  # unique seed per image, reproducible from `seed`

    def take_seed() -> int:
        nonlocal next_seed
        next_seed += 1
        return next_seed

    for i in range(n_train):
        img, _ = render(take_seed(), size)
        save(img, out / "train" / "good" / f"{i:03d}.png")
    for i in range(n_test_good):
        s = take_seed()
        img, _ = render(s, size)
        save(img, out / "test" / "good" / f"{i:03d}.png")
        rows.append({"split": "test", "defect": "good", "file": f"test/good/{i:03d}.png", "severity": "", "area_px": 0, "seed": s})
    for defect in DEFECTS:
        for i in range(per_defect):
            s = take_seed()
            severity = float(np.random.default_rng([s, 3]).uniform(0.35, 1.0))
            img, mask = render(s, size, defect, severity)
            save(img, out / "test" / defect / f"{i:03d}.png")
            save((mask * 255).astype(np.uint8), out / "ground_truth" / defect / f"{i:03d}_mask.png")
            rows.append(
                {"split": "test", "defect": defect, "file": f"test/{defect}/{i:03d}.png", "severity": round(severity, 3), "area_px": int(mask.sum()), "seed": s}
            )

    with open(out / MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["split", "defect", "file", "severity", "area_px", "seed"])
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/leather")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", type=int, default=512, help="image side in pixels (real MVTec leather is larger)")
    p.add_argument("--train", type=int, default=40)
    p.add_argument("--test-good", type=int, default=20)
    p.add_argument("--per-defect", type=int, default=12)
    p.add_argument("--force", action="store_true", help="replace an existing FAKE dataset")
    a = p.parse_args(argv)
    try:
        rows = make_dataset(Path(a.out), a.seed, a.size, a.train, a.test_good, a.per_defect, a.force)
    except FileExistsError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(f"Wrote fake leather data to {a.out}: {a.train} train/good, {a.test_good} test/good, {a.per_defect} x {len(DEFECTS)} defects ({len(rows)} test images).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
