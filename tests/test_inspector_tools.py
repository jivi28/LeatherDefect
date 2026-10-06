"""Inspector tools through the real ToolRegistry: they return images, and bad input never raises."""

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from inspector.detector import Detector
from inspector.tools import TOOL_NAMES, Inspection, ReferenceBank, build_tools, resolve_region
from agentkit import ToolError


def _texture(seed: int, size: int = 512) -> np.ndarray:
    rng = np.random.default_rng(seed)
    base = np.array([0.45, 0.32, 0.22], np.float32)
    return np.clip(base + rng.normal(0, 0.03, (size, size, 1)).astype(np.float32), 0, 1)


def _save(rgb, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((rgb * 255).astype(np.uint8)).save(path)
    return path


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    root = tmp_path_factory.mktemp("tools")
    goods = [_save(_texture(i), root / "good" / f"{i:03d}.png") for i in range(10)]
    det = Detector.fit(goods)
    bad = _texture(50)
    bad[300:340, 100:140] *= 0.3  # dark spot around x=0.23, y=0.62
    photo = _save(bad, root / "photo.png")
    return det, ReferenceBank(goods, n=4), photo


@pytest.fixture
def registry(setup):
    det, refs, photo = setup
    return build_tools(Inspection(photo, det, refs))


def test_registry_has_the_spec_tools(registry):
    assert set(registry.names()) == set(TOOL_NAMES)


def test_view_overview_returns_one_image(registry):
    out = registry.execute("view_overview", "{}")
    assert out.ok and len(out.images) == 1 and out.images[0].mime == "image/png"


def test_scan_anomalies_finds_the_spot(registry):
    out = registry.execute("scan_anomalies", json.dumps({"top_k": 2}))
    assert out.ok and len(out.images) == 1
    data = json.loads(out.content)
    assert data["image_level"] in ("above normal", "far above normal")
    x0, y0, x1, y1 = data["regions"][0]["box"]
    assert x0 <= 0.23 <= x1 and y0 <= 0.62 <= y1
    assert len(data["regions"]) == 2


def test_zoom_compare_measure_by_cell_and_box(registry):
    for name in ("zoom", "compare_reference"):
        assert registry.execute(name, json.dumps({"cell": "B5"})).images
        assert registry.execute(name, json.dumps({"box": [0.15, 0.55, 0.3, 0.7]})).images
    m = json.loads(registry.execute("measure", json.dumps({"box": [0.15, 0.55, 0.3, 0.7]})).content)
    assert m["peak_anomaly"] > m["normal_threshold"]
    assert m["brightness_z"] < -3  # the spot is much darker than normal leather
    calm = json.loads(registry.execute("measure", json.dumps({"cell": "H1"})).content)
    assert calm["peak_anomaly"] < m["peak_anomaly"]


@pytest.mark.parametrize(
    "name,args,hint",
    [
        ("zoom", {}, "grid cell"),
        ("zoom", {"cell": "Q9"}, "A1"),
        ("zoom", {"box": [0, 0, 600, 600]}, "normalised"),
        ("measure", {"box": [0.1, 0.2]}, "4 numbers"),
        ("compare_reference", {"cell": "nonsense"}, "column"),
        ("zoom", "not json", "valid JSON"),
        ("scan_anomalies", {"top_k": "many"}, "top_k"),
    ],
)
def test_bad_arguments_return_helpful_text(registry, name, args, hint):
    out = registry.execute(name, args if isinstance(args, str) else json.dumps(args))
    assert not out.ok and hint in out.content and not out.images


def test_unreadable_photo_does_not_raise(setup, tmp_path):
    det, refs, _ = setup
    broken = tmp_path / "broken.png"
    broken.write_bytes(b"not a png")
    out = build_tools(Inspection(broken, det, refs)).execute("view_overview", "{}")
    assert not out.ok and "Could not open" in out.content


def test_resolve_region_ranges():
    assert resolve_region(cell="A1-B2") == (0.0, 0.0, 0.25, 0.25)
    with pytest.raises(ToolError):
        resolve_region()
