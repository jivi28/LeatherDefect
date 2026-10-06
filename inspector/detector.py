"""No-LLM anomaly detector for texture images, fitted only on known-good images.

Recipe: downscale to 256 px -> per-pixel maps (luminance, two opponent colours, gradients, Laplacian)
-> patch features at two scales (16 and 32 px, stride 8) -> Mahalanobis distance to the pooled
distribution of good patches -> normalised so 1.0 = the 99th percentile of held-out good patches
-> painted back into a 64x64 anomaly map. Image score = max of the map.

Texture is position-invariant, so one Gaussian is fitted over all patch positions.
Only `train/good` is ever read here; labels and masks are not.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from .data import ROOT

WORK_SIZE = 256
MAP_SIZE = 64
SCALES: tuple[tuple[int, int], ...] = ((16, 8), (32, 8))  # (patch, stride) at WORK_SIZE
FEATURES = ("lum", "red_green", "yellow_blue", "contrast", "edge", "coherence", "laplacian")
CACHE_DIR = ROOT / "results" / "cache"
EPS = 1e-6


def load_rgb(path: str | Path, size: int = WORK_SIZE) -> np.ndarray:
    """RGB float32 array in 0..1, resized to size x size."""
    with Image.open(path) as im:
        im = im.convert("RGB").resize((size, size), Image.BILINEAR)
        return np.asarray(im, dtype=np.float32) / 255.0


def _box_mean(x: np.ndarray, patch: int, stride: int) -> np.ndarray:
    """Mean over every patch x patch window at the given stride (integral image)."""
    c = np.pad(x, ((1, 0), (1, 0))).cumsum(0).cumsum(1)
    n = (x.shape[0] - patch) // stride + 1
    idx = np.arange(n) * stride
    s = c[idx[:, None] + patch, idx[None, :] + patch] - c[idx[:, None], idx[None, :] + patch]
    s -= c[idx[:, None] + patch, idx[None, :]] - c[idx[:, None], idx[None, :]]
    return s / (patch * patch)


def patch_features(rgb: np.ndarray, patch: int, stride: int) -> np.ndarray:
    """(grid, grid, len(FEATURES)) feature array for one image at one scale."""
    r, g, b = rgb[..., 0], rgb[..., 1], rgb[..., 2]
    lum = 0.299 * r + 0.587 * g + 0.114 * b
    rg = r - g
    yb = 0.5 * (r + g) - b
    gy, gx = np.gradient(lum)
    mag = np.sqrt(gx**2 + gy**2)
    lap = np.abs(np.roll(lum, 1, 0) + np.roll(lum, -1, 0) + np.roll(lum, 1, 1) + np.roll(lum, -1, 1) - 4 * lum)

    m = lambda a: _box_mean(a, patch, stride)  # noqa: E731
    mean_l = m(lum)
    contrast = np.sqrt(np.maximum(m(lum**2) - mean_l**2, 0))
    jxx, jyy, jxy = m(gx * gx), m(gy * gy), m(gx * gy)
    coherence = np.sqrt((jxx - jyy) ** 2 + 4 * jxy**2) / (jxx + jyy + EPS)
    return np.stack([mean_l, m(rg), m(yb), contrast, m(mag), coherence, m(lap)], axis=-1)


@dataclass
class ScaleModel:
    patch: int
    stride: int
    mean: np.ndarray
    inv_cov: np.ndarray
    median: float  # distance of a typical good patch
    q99: float  # 99th percentile distance of held-out good patches

    def distances(self, feats: np.ndarray) -> np.ndarray:
        d = feats - self.mean
        return np.sqrt(np.maximum(np.einsum("...i,ij,...j->...", d, self.inv_cov, d), 0))

    def normalised(self, feats: np.ndarray) -> np.ndarray:
        return (self.distances(feats) - self.median) / max(self.q99 - self.median, EPS)


@dataclass
class Region:
    box: tuple[float, float, float, float]  # normalised x0, y0, x1, y1
    score: float  # peak normalised anomaly score inside the region
    area: float  # share of the image above half the peak inside the region


@dataclass
class Detection:
    score: float  # image-level score = max of the map
    anomaly_map: np.ndarray  # MAP_SIZE x MAP_SIZE, 1.0 ~ 99th pct of good patches
    regions: list[Region] = field(default_factory=list)


def _paint(scores: np.ndarray, patch: int, stride: int, size: int = WORK_SIZE) -> np.ndarray:
    """Spread patch scores back to pixels (mean of covering patches), then shrink to MAP_SIZE."""
    acc = np.zeros((size, size), np.float32)
    cnt = np.zeros((size, size), np.float32)
    for i in range(scores.shape[0]):
        for j in range(scores.shape[1]):
            y, x = i * stride, j * stride
            acc[y : y + patch, x : x + patch] += scores[i, j]
            cnt[y : y + patch, x : x + patch] += 1
    full = acc / np.maximum(cnt, 1)
    k = size // MAP_SIZE
    return full.reshape(MAP_SIZE, k, MAP_SIZE, k).mean(axis=(1, 3))


def _flood(mask: np.ndarray, seed: tuple[int, int]) -> np.ndarray:
    comp = np.zeros_like(mask)
    stack = [seed]
    h, w = mask.shape
    while stack:
        y, x = stack.pop()
        if 0 <= y < h and 0 <= x < w and mask[y, x] and not comp[y, x]:
            comp[y, x] = True
            stack.extend(((y + 1, x), (y - 1, x), (y, x + 1), (y, x - 1)))
    return comp


def top_regions(anomaly_map: np.ndarray, k: int = 3, min_size: float = 0.08) -> list[Region]:
    """Greedy peak picking: take the highest point, grow its blob above half the peak, box it, suppress, repeat."""
    work = anomaly_map.astype(np.float64).copy()
    h, w = work.shape
    out: list[Region] = []
    for _ in range(k):
        y, x = np.unravel_index(int(np.argmax(work)), work.shape)
        peak = float(work[y, x])
        if not np.isfinite(peak):
            break
        level = peak - 0.5 * max(peak - float(np.median(anomaly_map)), EPS)
        blob = _flood(work >= level, (int(y), int(x)))
        ys, xs = np.nonzero(blob)
        x0, x1, y0, y1 = float(xs.min()) / w, float(xs.max() + 1) / w, float(ys.min()) / h, float(ys.max() + 1) / h
        cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
        bw, bh = max(x1 - x0, min_size), max(y1 - y0, min_size)
        box = (max(0.0, cx - bw / 2), max(0.0, cy - bh / 2), min(1.0, cx + bw / 2), min(1.0, cy + bh / 2))
        out.append(Region(tuple(round(float(v), 3) for v in box), round(peak, 3), round(float(blob.mean()), 4)))  # type: ignore[arg-type]
        # suppress the blob plus a small margin so the next peak is somewhere else
        pad = 2
        ys0, ys1 = max(0, ys.min() - pad), min(h, ys.max() + 1 + pad)
        xs0, xs1 = max(0, xs.min() - pad), min(w, xs.max() + 1 + pad)
        work[ys0:ys1, xs0:xs1] = -np.inf
    return out


class Detector:
    def __init__(self, scales: list[ScaleModel], threshold: float = 1.0, fitted_on: int = 0) -> None:
        self.scales = scales
        self.threshold = threshold  # image score above this = anomalous (set from good hold-out, retune on dev)
        self.fitted_on = fitted_on

    @classmethod
    def fit(cls, good_paths: list[Path], holdout_every: int = 5, ridge: float = 1e-3) -> "Detector":
        """Fit on good images; every `holdout_every`-th image is held out to set the normalisation and threshold."""
        good_paths = sorted(good_paths)
        if len(good_paths) < 4:
            raise ValueError("need at least 4 good images to fit the detector")
        hold = good_paths[::holdout_every]
        fit = [p for p in good_paths if p not in hold]
        fit_imgs = [load_rgb(p) for p in fit]
        hold_imgs = [load_rgb(p) for p in hold]
        scales: list[ScaleModel] = []
        for patch, stride in SCALES:
            x = np.concatenate([patch_features(im, patch, stride).reshape(-1, len(FEATURES)) for im in fit_imgs])
            mean = x.mean(0)
            cov = np.cov(x, rowvar=False)
            cov += ridge * np.trace(cov) / len(FEATURES) * np.eye(len(FEATURES))
            model = ScaleModel(patch, stride, mean, np.linalg.inv(cov), 0.0, 1.0)
            hd = np.concatenate([model.distances(patch_features(im, patch, stride)).ravel() for im in hold_imgs])
            model.median, model.q99 = float(np.median(hd)), float(np.quantile(hd, 0.99))
            scales.append(model)
        det = cls(scales, fitted_on=len(fit))
        hold_scores = [det.anomaly_map(im).max() for im in hold_imgs]
        # label-free default: a bit above the worst held-out good image
        det.threshold = float(np.max(hold_scores) * 1.05)
        return det

    def anomaly_map(self, rgb: np.ndarray) -> np.ndarray:
        maps = [_paint(s.normalised(patch_features(rgb, s.patch, s.stride)), s.patch, s.stride) for s in self.scales]
        return np.maximum.reduce(maps)

    def detect(self, image: str | Path | np.ndarray, top_k: int = 3) -> Detection:
        rgb = load_rgb(image) if not isinstance(image, np.ndarray) else image
        amap = self.anomaly_map(rgb)
        return Detection(round(float(amap.max()), 4), amap, top_regions(amap, top_k))

    def margin(self, score: float) -> float:
        """Signed distance from the threshold in threshold units. Near 0 = uncertain."""
        return (score - self.threshold) / max(self.threshold, EPS)

    # ---- persistence ----
    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: dict[str, np.ndarray] = {"threshold": np.array(self.threshold), "fitted_on": np.array(self.fitted_on)}
        for i, s in enumerate(self.scales):
            arrays[f"s{i}_meta"] = np.array([s.patch, s.stride, s.median, s.q99])
            arrays[f"s{i}_mean"], arrays[f"s{i}_inv"] = s.mean, s.inv_cov
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: Path) -> "Detector":
        z = np.load(path)
        scales = []
        i = 0
        while f"s{i}_meta" in z:
            patch, stride, median, q99 = z[f"s{i}_meta"]
            scales.append(ScaleModel(int(patch), int(stride), z[f"s{i}_mean"], z[f"s{i}_inv"], float(median), float(q99)))
            i += 1
        return cls(scales, float(z["threshold"]), int(z["fitted_on"]))


def _cache_key(paths: list[Path]) -> str:
    h = hashlib.sha1()
    for p in sorted(paths):
        h.update(str(p).encode())
        h.update(str(p.stat().st_size).encode())
    h.update(repr((WORK_SIZE, MAP_SIZE, SCALES, FEATURES)).encode())
    return h.hexdigest()[:12]


def get_detector(good_paths: list[Path], cache_dir: Path = CACHE_DIR) -> Detector:
    """Fit once per set of good images and cache the result under results/cache/."""
    path = cache_dir / f"detector_{_cache_key(good_paths)}.npz"
    if path.exists():
        return Detector.load(path)
    det = Detector.fit(good_paths)
    det.save(path)
    return det
