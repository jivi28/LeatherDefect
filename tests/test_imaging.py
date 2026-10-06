"""Coordinate maths and image helpers."""

import numpy as np
import pytest
from PIL import Image

from inspector import imaging as im


def test_cell_to_box_and_back():
    assert im.cell_to_box("A1") == (0.0, 0.0, 0.125, 0.125)
    assert im.cell_to_box("c4") == (0.25, 0.375, 0.375, 0.5)
    assert im.cell_to_box(" H8 ") == (0.875, 0.875, 1.0, 1.0)
    assert im.box_to_cells(im.cell_to_box("C4")) == "C4"
    assert im.box_to_cells((0.26, 0.38, 0.49, 0.62)) == "C4-D5"


@pytest.mark.parametrize("bad", ["", "Z9", "A0", "A9", "44", "AA1", "C"])
def test_bad_cells_raise_with_help(bad):
    with pytest.raises(ValueError) as exc:
        im.cell_to_box(bad)
    assert "A1" in str(exc.value) or "column" in str(exc.value)


def test_clean_box_repairs_model_mistakes():
    assert im.clean_box([0.5, 0.6, 0.2, 0.1]) == (0.2, 0.1, 0.5, 0.6)  # swapped corners
    assert im.clean_box([-0.2, 0.0, 1.3, 1.0]) == (0.0, 0.0, 1.0, 1.0)  # clamped
    x0, y0, x1, y1 = im.clean_box([0.5, 0.5, 0.5, 0.5])  # zero size -> minimum size around the point
    assert x1 - x0 == pytest.approx(im.MIN_SIDE) and (x0 + x1) / 2 == pytest.approx(0.5)
    x0, _, x1, _ = im.clean_box([1.0, 0.2, 1.0, 0.3])  # zero size at the edge stays inside
    assert x1 == pytest.approx(1.0) and x0 == pytest.approx(1 - im.MIN_SIDE)


@pytest.mark.parametrize("bad", [None, [1, 2, 3], [0, 0, 512, 512], ["a", 0, 1, 1], [float("nan"), 0, 1, 1]])
def test_clean_box_rejects_uninterpretable(bad):
    with pytest.raises(ValueError):
        im.clean_box(bad)


def test_pixel_like_box_gets_a_hint():
    with pytest.raises(ValueError, match="normalised"):
        im.clean_box([100, 100, 300, 300])


def test_crop_box_maps_to_pixels():
    img = Image.new("RGB", (1000, 500))
    crop, used = im.crop_box(img, (0.1, 0.2, 0.3, 0.6))
    assert crop.size == (200, 200)
    assert used == (0.1, 0.2, 0.3, 0.6)
    crop, used = im.crop_box(img, (0.0, 0.0, 0.1, 0.1), pad=0.05)
    assert used == (0.0, 0.0, 0.15, 0.15) and crop.size == (150, 75)


def test_downscale_keeps_aspect_and_never_upscales():
    assert im.downscale(Image.new("RGB", (1024, 512)), 768).size == (768, 384)
    assert im.downscale(Image.new("RGB", (100, 50)), 768).size == (100, 50)


def test_overlays_keep_size_and_encode():
    img = Image.new("RGB", (256, 256), (120, 90, 60))
    assert im.draw_grid(img).size == img.size
    assert im.draw_ticks(img, (0.1, 0.1, 0.4, 0.4)).size == img.size
    heat = im.heatmap_overlay(img, np.random.rand(64, 64), [(0.1, 0.1, 0.3, 0.3)])
    assert heat.size == img.size
    pair = im.side_by_side(img, Image.new("RGB", (128, 128)))
    assert pair.height == 256 + 22
    data = im.to_image_data(img, "x")
    assert data.mime == "image/png" and data.label == "x" and len(data.b64) > 100
