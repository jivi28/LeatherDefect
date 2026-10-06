import csv
import hashlib

import numpy as np
import pytest
from PIL import Image

from scripts.make_fake_leather import DEFECTS, MANIFEST, MARKER, make_dataset, render


def digest(arr):
    return hashlib.sha256(arr.tobytes()).hexdigest()


def test_render_is_deterministic():
    a, ma = render(11, 128, "cut", 0.7)
    b, mb = render(11, 128, "cut", 0.7)
    assert digest(a) == digest(b) and (ma == mb).all()
    c, _ = render(12, 128, "cut", 0.7)
    assert digest(a) != digest(c)


def test_good_image_has_empty_mask_and_right_shape():
    img, mask = render(3, 128)
    assert img.shape == (128, 128, 3) and img.dtype == np.uint8
    assert mask.shape == (128, 128) and not mask.any()


@pytest.mark.parametrize("defect", DEFECTS)
def test_defect_changes_pixels_mostly_inside_mask(defect):
    size = 256
    base, _ = render(21, size)
    img, mask = render(21, size, defect, 1.0)
    assert mask.sum() >= 20, "mask too small to be a real defect"
    assert mask.mean() < 0.35, "defect covers too much of the image"
    diff = np.abs(img.astype(int) - base.astype(int)).sum(axis=2)
    assert diff[mask].mean() > 3 * diff[~mask].mean() + 1


@pytest.mark.parametrize("defect", DEFECTS)
def test_lower_severity_is_less_visible(defect):
    size = 256
    base, _ = render(31, size)
    strong, m1 = render(31, size, defect, 1.0)
    weak, m2 = render(31, size, defect, 0.35)
    d_strong = np.abs(strong.astype(int) - base.astype(int)).sum()
    d_weak = np.abs(weak.astype(int) - base.astype(int)).sum()
    assert d_weak < d_strong


def test_unknown_defect_raises():
    with pytest.raises(ValueError):
        render(1, 64, "scratch")


@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    out = tmp_path_factory.mktemp("leather") / "leather"
    rows = make_dataset(out, seed=7, size=96, n_train=4, n_test_good=3, per_defect=2)
    return out, rows


def test_layout_matches_mvtec(dataset):
    out, rows = dataset
    assert (out / MARKER).exists() and (out / MANIFEST).exists()
    assert len(list((out / "train" / "good").glob("*.png"))) == 4
    assert len(list((out / "test" / "good").glob("*.png"))) == 3
    for defect in DEFECTS:
        images = sorted((out / "test" / defect).glob("*.png"))
        masks = sorted((out / "ground_truth" / defect).glob("*_mask.png"))
        assert [p.stem for p in images] == ["000", "001"]
        assert [p.stem for p in masks] == ["000_mask", "001_mask"]
        mask = np.array(Image.open(masks[0]))
        assert set(np.unique(mask)) <= {0, 255} and mask.any()
    assert not (out / "ground_truth" / "good").exists()  # like MVTec: good images have no mask
    assert len(rows) == 3 + 5 * 2


def test_manifest_rows_and_unique_images(dataset):
    out, rows = dataset
    with open(out / MANIFEST) as f:
        read = list(csv.DictReader(f))
    assert len(read) == len(rows) == 13
    assert {r["defect"] for r in read} == {"good", *DEFECTS}
    hashes = {hashlib.sha256((out / r["file"]).read_bytes()).hexdigest() for r in read}
    assert len(hashes) == len(read)


def test_refuses_to_overwrite_without_force_and_protects_real_data(tmp_path):
    out = tmp_path / "leather"
    make_dataset(out, size=64, n_train=1, n_test_good=1, per_defect=1)
    with pytest.raises(FileExistsError):
        make_dataset(out, size=64, n_train=1, n_test_good=1, per_defect=1)
    make_dataset(out, size=64, n_train=1, n_test_good=1, per_defect=1, force=True)  # fake data: replaceable

    real = tmp_path / "real"
    (real / "train" / "good").mkdir(parents=True)
    (real / "train" / "good" / "000.png").write_bytes(b"precious")
    with pytest.raises(FileExistsError, match="REAL"):
        make_dataset(real, size=64, n_train=1, n_test_good=1, per_defect=1, force=True)
    assert (real / "train" / "good" / "000.png").read_bytes() == b"precious"


def test_same_seed_same_dataset(tmp_path):
    a, b = tmp_path / "a", tmp_path / "b"
    make_dataset(a, seed=5, size=64, n_train=2, n_test_good=1, per_defect=1)
    make_dataset(b, seed=5, size=64, n_train=2, n_test_good=1, per_defect=1)
    for p in sorted(a.rglob("*.png")):
        assert p.read_bytes() == (b / p.relative_to(a)).read_bytes()
