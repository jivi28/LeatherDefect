"""Offline tests for inspector.data: taxonomy from folders, masks, stratified deterministic split, test lock."""

from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from inspector import data
from inspector.data import Sample, SplitLocked, get_split, load_samples, split, taxonomy

REAL = data.DEFAULT_ROOT


def _png(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), (100, 80, 60)).save(path)


@pytest.fixture
def tiny_root(tmp_path: Path) -> Path:
    counts = {"good": 7, "scratch": 5, "stain": 4}
    for label, n in counts.items():
        for i in range(n):
            _png(tmp_path / "test" / label / f"{i:03d}.png")
            if label != "good":
                _png(tmp_path / "ground_truth" / label / f"{i:03d}_mask.png")
    for i in range(3):
        _png(tmp_path / "train" / "good" / f"{i:03d}.png")
    (tmp_path / "test" / "empty_folder").mkdir()
    return tmp_path


def test_taxonomy_comes_from_folders(tiny_root):
    assert taxonomy(tiny_root) == ["scratch", "stain"]  # good and empty folders excluded


def test_load_samples_attaches_masks(tiny_root):
    samples = load_samples(tiny_root)
    assert len(samples) == 16
    assert all(s.mask is None for s in samples if s.label == "good")
    assert all(s.mask is not None and s.mask.exists() for s in samples if s.is_defect)


def test_split_is_stratified_complete_and_disjoint(tiny_root):
    samples = load_samples(tiny_root)
    parts = split(samples, seed=1)
    dev, test = parts["dev"], parts["test"]
    assert {s.path for s in dev}.isdisjoint({s.path for s in test})
    assert len(dev) + len(test) == len(samples)
    dev_c, test_c = Counter(s.label for s in dev), Counter(s.label for s in test)
    for label, n in Counter(s.label for s in samples).items():
        assert abs(dev_c[label] - test_c[label]) <= 1, label
        assert dev_c[label] >= 1 and test_c[label] >= 1


def test_odd_labels_alternate_the_extra_image(tiny_root):
    parts = split(load_samples(tiny_root), seed=1)
    assert abs(len(parts["dev"]) - len(parts["test"])) <= 1  # good=7 and scratch=5 are both odd


def test_split_is_deterministic_and_order_independent(tiny_root):
    samples = load_samples(tiny_root)
    a = split(samples, seed=3)
    b = split(list(reversed(samples)), seed=3)
    assert [s.path for s in a["dev"]] == [s.path for s in b["dev"]]
    c = split(samples, seed=4)
    assert [s.path for s in a["dev"]] != [s.path for s in c["dev"]]


def test_test_split_is_locked_by_default(tiny_root, monkeypatch):
    monkeypatch.delenv(data.ALLOW_TEST_ENV, raising=False)
    assert get_split("dev", tiny_root)
    with pytest.raises(SplitLocked):
        get_split("test", tiny_root)
    assert get_split("test", tiny_root, allow_test=True)
    monkeypatch.setenv(data.ALLOW_TEST_ENV, "1")
    assert get_split("test", tiny_root)


def test_unknown_split_name(tiny_root):
    with pytest.raises(ValueError):
        get_split("train", tiny_root)


def test_sample_id():
    assert Sample(Path("/x/test/cut/004.png"), "cut").id == "cut/004"


@pytest.mark.skipif(not (REAL / "test").is_dir() or data.is_fake(REAL), reason="real MVTec leather not imported")
def test_real_leather_split_shape():
    samples = load_samples(REAL)
    assert len(samples) == 124
    assert taxonomy(REAL) == ["color", "cut", "fold", "glue", "poke"]
    parts = split(samples)
    assert len(parts["dev"]) in (61, 62, 63) and len(parts["dev"]) + len(parts["test"]) == 124
    assert Counter(s.label for s in parts["dev"])["good"] == 16
