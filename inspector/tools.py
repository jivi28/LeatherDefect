"""The agent's tools, bound to one photo. Built on imaging.py and detector.py.

Agent-facing: never reads labels, masks or the test split. References come from train/good only.
Every tool returns a short string (plus images); bad arguments become helpful ToolError messages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from agentkit import ToolError, ToolRegistry, ToolResult

from . import imaging as im
from .detector import MAP_SIZE, WORK_SIZE, Detection, Detector, load_rgb, patch_features

OVERVIEW_SIDE = 768
CROP_SIDE = 512
TOOL_NAMES = ("view_overview", "scan_anomalies", "zoom", "compare_reference", "measure")


class ReferenceBank:
    """A handful of known-good images; picks the one whose overall colour is closest to the photo."""

    def __init__(self, good_paths: list[Path], n: int = 16) -> None:
        paths = sorted(good_paths)
        step = max(1, len(paths) // n)
        self.paths = paths[::step][:n]
        self._means: np.ndarray | None = None

    def closest(self, rgb_small: np.ndarray) -> Path:
        if not self.paths:
            raise ToolError("No known-good reference images are available.")
        if self._means is None:
            self._means = np.array([load_rgb(p, 32).reshape(-1, 3).mean(0) for p in self.paths])
        target = rgb_small.reshape(-1, 3).mean(0)
        return self.paths[int(np.argmin(((self._means - target) ** 2).sum(1)))]


@dataclass
class Inspection:
    """Everything the tools need for one photo, computed lazily and cached for the run."""

    image_path: Path
    detector: Detector
    references: ReferenceBank
    _image: Image.Image | None = None
    _rgb: np.ndarray | None = None
    _detection: Detection | None = None
    _reference: Image.Image | None = None
    seen: list[str] = field(default_factory=list)  # tool names used, for traces

    @property
    def image(self) -> Image.Image:
        if self._image is None:
            try:
                with Image.open(self.image_path) as raw:
                    self._image = raw.convert("RGB")
            except (OSError, ValueError) as exc:
                raise ToolError(f"Could not open the photo: {exc}") from exc
        return self._image

    @property
    def rgb(self) -> np.ndarray:
        if self._rgb is None:
            self._rgb = np.asarray(self.image.resize((WORK_SIZE, WORK_SIZE), Image.BILINEAR), np.float32) / 255.0
        return self._rgb

    @property
    def detection(self) -> Detection:
        if self._detection is None:
            self._detection = self.detector.detect(self.rgb, top_k=5)
        return self._detection

    @property
    def reference(self) -> Image.Image:
        if self._reference is None:
            path = self.references.closest(self.rgb[::8, ::8])
            with Image.open(path) as raw:
                self._reference = raw.convert("RGB").resize(self.image.size)
        return self._reference


def resolve_region(cell: str = "", box: list[float] | None = None) -> im.Box:
    """Turn the model's cell or box into a clean normalised box, or raise ToolError with how to fix it."""
    if box:
        try:
            return im.clean_box(box)
        except ValueError as exc:
            raise ToolError(str(exc)) from None
    if cell:
        cells = [c.strip() for c in cell.replace(":", "-").split("-") if c.strip()]
        try:
            boxes = [im.cell_to_box(c) for c in cells[:2]]
        except ValueError as exc:
            raise ToolError(str(exc)) from None
        return im.clean_box((min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)))
    raise ToolError("Give either a grid cell like 'C4' (or a range 'C4-D5') or a box [x0, y0, x1, y1] in 0..1.")


def _level(score: float, threshold: float) -> str:
    if score >= 2 * threshold:
        return "far above normal"
    if score >= threshold:
        return "above normal"
    if score >= 0.7 * threshold:
        return "borderline"
    return "within normal range"


def measure_region(ins: Inspection, box: im.Box) -> dict:
    """Deterministic numbers for a region, in z-units relative to known-good leather patches."""
    det = ins.detector
    s0 = det.scales[0]
    feats = patch_features(ins.rgb, s0.patch, s0.stride)
    n = feats.shape[0]
    centres = (np.arange(n) * s0.stride + s0.patch / 2) / WORK_SIZE
    x0, y0, x1, y1 = box
    sel_x = (centres >= x0) & (centres <= x1)
    sel_y = (centres >= y0) & (centres <= y1)
    if not sel_x.any():
        sel_x[np.argmin(np.abs(centres - (x0 + x1) / 2))] = True
    if not sel_y.any():
        sel_y[np.argmin(np.abs(centres - (y0 + y1) / 2))] = True
    region = feats[np.ix_(sel_y, sel_x)].reshape(-1, feats.shape[-1])
    dist = s0.distances(region)
    top = region[dist >= np.quantile(dist, 0.75)] if len(region) > 3 else region
    std = np.sqrt(np.diag(np.linalg.inv(s0.inv_cov)))
    z = (top.mean(0) - s0.mean) / np.maximum(std, 1e-6)

    amap = ins.detection.anomaly_map
    ax0, ay0 = int(x0 * MAP_SIZE), int(y0 * MAP_SIZE)
    ax1, ay1 = max(ax0 + 1, int(np.ceil(x1 * MAP_SIZE))), max(ay0 + 1, int(np.ceil(y1 * MAP_SIZE)))
    sub = amap[ay0:ay1, ax0:ax1]
    peak = float(sub.max())
    hot = sub >= max(1.0, 0.5 * peak)
    elongation = None
    if hot.sum() >= 3:
        ys, xs = np.nonzero(hot)
        cov = np.cov(np.stack([xs, ys]).astype(float))
        ev = np.sort(np.linalg.eigvalsh(cov + 1e-6 * np.eye(2)))
        elongation = round(float(np.sqrt(ev[1] / max(ev[0], 1e-6))), 2)
    r = lambda v: round(float(v), 2)  # noqa: E731
    return {
        "box": list(box),
        "peak_anomaly": r(peak),
        "normal_threshold": r(det.threshold),
        "unusual_area_share": r((sub >= 1.0).mean()),
        "strong_area_share": r((sub >= det.threshold).mean()),
        "elongation": elongation,
        "brightness_z": r(z[0]),
        "red_green_shift_z": r(z[1]),
        "yellow_blue_shift_z": r(z[2]),
        "local_contrast_z": r(z[3]),
        "edge_energy_z": r(z[4]),
        "line_orientation_z": r(z[5]),
        "fine_detail_z": r(z[6]),
        "units": "z = standard deviations from normal leather patches; |z| > 3 is unusual. elongation 1 = round blob, > 3 = line-like.",
    }


def build_tools(ins: Inspection) -> ToolRegistry:
    registry = ToolRegistry()

    @registry.register
    def view_overview() -> ToolResult:
        """See the whole photo, downscaled, with a labelled 8x8 grid (A1 = top-left, H8 = bottom-right).
        Use the cell names to point at areas in other tools."""
        ins.seen.append("view_overview")
        img = im.draw_grid(im.downscale(ins.image, OVERVIEW_SIDE))
        return ToolResult(f"Overview of the photo with an 8x8 grid ({ins.image.width}x{ins.image.height} px original).",
                          [im.to_image_data(img, "overview with grid")])

    @registry.register
    def scan_anomalies(top_k: int = 3) -> ToolResult:
        """Run the deterministic texture detector (fitted on known-good leather only). Lists the most unusual
        regions with a normalised box, grid cells and score, plus a heatmap image. Scores near or below the
        normal threshold mean the area looks like ordinary leather. The detector cannot name defect types."""
        ins.seen.append("scan_anomalies")
        k = max(1, min(int(top_k), 5))
        d = ins.detection
        thr = ins.detector.threshold
        regions = [
            {"rank": i, "box": list(r.box), "cells": im.box_to_cells(r.box), "score": round(r.score, 2), "level": _level(r.score, thr)}
            for i, r in enumerate(d.regions[:k], 1)
        ]
        summary = {
            "image_score": round(d.score, 2),
            "normal_threshold": round(thr, 2),
            "image_level": _level(d.score, thr),
            "regions": regions,
        }
        heat = im.heatmap_overlay(im.downscale(ins.image, CROP_SIDE), d.anomaly_map, [r.box for r in d.regions[:k]], vmax=max(2 * thr, d.score))
        return ToolResult(json.dumps(summary), [im.to_image_data(heat, "anomaly heatmap, numbered boxes = ranked regions")])

    @registry.register
    def zoom(cell: str = "", box: list[float] | None = None) -> ToolResult:
        """Full-resolution close-up of one region. Give a grid cell like 'C4', a range like 'C4-D5', or a
        normalised box [x0, y0, x1, y1] (0..1, e.g. a box from scan_anomalies). The close-up has grid lines
        labelled with the global x/y coordinates so you can give precise boxes afterwards."""
        region = resolve_region(cell, box)
        ins.seen.append("zoom")
        crop, used = im.crop_box(ins.image, region, pad=0.03)
        crop = im.draw_ticks(im.downscale(crop, CROP_SIDE), used)
        return ToolResult(f"Close-up of box {list(used)} (cells {im.box_to_cells(used)}).",
                          [im.to_image_data(crop, f"zoom {im.box_to_cells(used)}")])

    @registry.register
    def compare_reference(cell: str = "", box: list[float] | None = None) -> ToolResult:
        """Show the same region from a known-good leather photo next to this photo (left = this photo,
        right = good reference). Use it to judge whether something is a defect or normal grain."""
        region = resolve_region(cell, box)
        ins.seen.append("compare_reference")
        mine, used = im.crop_box(ins.image, region, pad=0.03)
        ref, _ = im.crop_box(ins.reference, used)
        pair = im.side_by_side(im.downscale(mine, CROP_SIDE // 2 + 64), im.downscale(ref, CROP_SIDE // 2 + 64))
        return ToolResult(f"Left: this photo, right: known-good reference, region {list(used)}.",
                          [im.to_image_data(pair, f"compare {im.box_to_cells(used)}")])

    @registry.register
    def measure(cell: str = "", box: list[float] | None = None) -> str:
        """Deterministic measurements of a region: anomaly strength and area, elongation (line-like vs blob),
        brightness, colour shift, contrast, edge energy and line orientation, each as a z-score against normal
        leather. Use it to back up what you see with numbers."""
        region = resolve_region(cell, box)
        ins.seen.append("measure")
        return json.dumps(measure_region(ins, region))

    return registry
