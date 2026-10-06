"""Offline tests for the detector on synthetic textures (plumbing and geometry, not performance)."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from inspector.detector import Detector, _box_mean, get_detector, load_rgb, patch_features, top_regions


def _texture(seed: int, size: int = 256) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.array([0.45, 0.32, 0.22], np.float32)
    noise = rng.normal(0, 0.03, (size, size, 1)).astype(np.float32)
    return np.clip(base + noise, 0, 1)


def _save(rgb: np.ndarray, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(path)
    return path


@pytest.fixture(scope="module")
def good_dir(tmp_path_factory) -> list[Path]:
    root = tmp_path_factory.mktemp("good")
    return [_save(_texture(i), root / f"{i:03d}.png") for i in range(12)]


def test_box_mean_matches_naive():
    x = np.arange(100, dtype=float).reshape(10, 10)
    got = _box_mean(x, 4, 3)
    assert got.shape == (3, 3)
    assert got[1, 2] == pytest.approx(x[3:7, 6:10].mean())


def test_patch_features_shape():
    f = patch_features(_texture(0), 16, 8)
    assert f.shape == (31, 31, 7) and np.isfinite(f).all()


def test_dark_spot_scores_high_and_is_localised(good_dir, tmp_path):
    det = Detector.fit(good_dir)
    clean = det.detect(_texture(99))
    bad = _texture(100)
    bad[150:180, 40:70] *= 0.3  # dark spot, centre ~ (x=0.21, y=0.64)
    hit = det.detect(bad)
    assert hit.score > det.threshold > clean.score * 0.9
    x0, y0, x1, y1 = hit.regions[0].box
    assert x0 <= 0.21 <= x1 and y0 <= 0.64 <= y1
    assert all(0.0 <= v <= 1.0 for r in hit.regions for v in r.box)
    assert all(isinstance(v, float) for v in hit.regions[0].box)


def test_top_regions_are_distinct_and_sorted():
    amap = np.zeros((64, 64))
    amap[5:9, 5:9] = 10
    amap[40:44, 50:54] = 6
    regions = top_regions(amap, k=3)
    assert len(regions) == 3
    assert regions[0].score >= regions[1].score >= regions[2].score
    assert regions[0].box[0] < 0.2 and regions[1].box[0] > 0.7


def test_save_load_roundtrip_and_cache(good_dir, tmp_path):
    det = get_detector(good_dir, cache_dir=tmp_path)
    files = list(tmp_path.glob("detector_*.npz"))
    assert len(files) == 1
    again = get_detector(good_dir, cache_dir=tmp_path)
    img = _texture(5)
    assert again.detect(img).score == pytest.approx(det.detect(img).score)
    assert again.threshold == pytest.approx(det.threshold)


def test_fit_needs_some_images(tmp_path):
    with pytest.raises(ValueError):
        Detector.fit([])


def test_load_rgb_resizes(good_dir):
    assert load_rgb(good_dir[0], 64).shape == (64, 64, 3)
