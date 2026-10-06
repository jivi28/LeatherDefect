"""Pure image helpers: grids, normalised boxes, crops, side-by-side, heatmaps, encoding.

Coordinates are always normalised 0..1 (x0, y0, x1, y1) relative to the full original image.
Grid cells are named column letter + row number, e.g. "C4" = 3rd column, 4th row.
"""

from __future__ import annotations

import io
import re
from typing import Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from agentkit import ImageData, image_from_bytes

Box = tuple[float, float, float, float]
GRID = 8  # overview grid is GRID x GRID cells
MIN_SIDE = 0.02  # smallest crop side, normalised
_CELL_RE = re.compile(r"^\s*([A-Za-z])\s*(\d{1,2})\s*$")


def downscale(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    scale = min(1.0, max_side / max(w, h))
    if scale >= 1.0:
        return img.copy()
    return img.resize((max(1, round(w * scale)), max(1, round(h * scale))), Image.LANCZOS)


def cell_name(col: int, row: int) -> str:
    return f"{chr(ord('A') + col)}{row + 1}"


def cell_to_box(cell: str, grid: int = GRID) -> Box:
    """'C4' -> normalised box of that cell. Raises ValueError with a helpful message."""
    m = _CELL_RE.match(cell or "")
    last = cell_name(grid - 1, grid - 1)
    if not m:
        raise ValueError(f"'{cell}' is not a grid cell. Use a column letter A-{last[0]} and a row 1-{grid}, like 'C4'.")
    col, row = ord(m.group(1).upper()) - ord("A"), int(m.group(2)) - 1
    if not (0 <= col < grid and 0 <= row < grid):
        raise ValueError(f"Cell '{cell}' is outside the grid; valid cells are A1 to {last}.")
    return (col / grid, row / grid, (col + 1) / grid, (row + 1) / grid)


def box_to_cells(box: Sequence[float], grid: int = GRID) -> str:
    """Cells covered by a box, e.g. 'C4' or 'C4-D5'."""
    x0, y0, x1, y1 = box
    c0, r0 = min(grid - 1, int(x0 * grid)), min(grid - 1, int(y0 * grid))
    c1, r1 = min(grid - 1, int(max(x1 - 1e-9, 0) * grid)), min(grid - 1, int(max(y1 - 1e-9, 0) * grid))
    a, b = cell_name(c0, r0), cell_name(c1, r1)
    return a if a == b else f"{a}-{b}"


def clean_box(box: Sequence[float], min_side: float = MIN_SIDE) -> Box:
    """Repair a model-supplied box: accept swapped corners, clamp to 0..1, enforce a minimum size.

    Raises ValueError only when the box cannot be interpreted at all."""
    if box is None or len(box) != 4:
        raise ValueError("A box needs exactly 4 numbers [x0, y0, x1, y1], each between 0 and 1.")
    try:
        x0, y0, x1, y1 = (float(v) for v in box)
    except (TypeError, ValueError):
        raise ValueError("Box values must be numbers between 0 and 1.") from None
    if any(v != v for v in (x0, y0, x1, y1)):
        raise ValueError("Box values must be numbers between 0 and 1.")
    if max(abs(x0), abs(y0), abs(x1), abs(y1)) > 1.5:
        raise ValueError(
            "Box values look like pixels. Use normalised coordinates between 0 and 1 (0,0 = top-left, 1,1 = bottom-right)."
        )
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    x0, y0, x1, y1 = (min(1.0, max(0.0, v)) for v in (x0, y0, x1, y1))

    def widen(a: float, b: float) -> tuple[float, float]:
        if b - a >= min_side:
            return a, b
        c = min(1 - min_side / 2, max(min_side / 2, (a + b) / 2))
        return c - min_side / 2, c + min_side / 2

    x0, x1 = widen(x0, x1)
    y0, y1 = widen(y0, y1)
    return (round(x0, 4), round(y0, 4), round(x1, 4), round(y1, 4))


def pad_box(box: Box, pad: float) -> Box:
    x0, y0, x1, y1 = box
    return clean_box((x0 - pad, y0 - pad, x1 + pad, y1 + pad))


def crop_box(img: Image.Image, box: Sequence[float], pad: float = 0.0) -> tuple[Image.Image, Box]:
    """Crop a normalised box (optionally padded) out of img. Returns the crop and the box actually used."""
    used = pad_box(clean_box(box), pad) if pad else clean_box(box)
    w, h = img.size
    x0, y0, x1, y1 = used
    px = (int(x0 * w), int(y0 * h), max(int(x0 * w) + 1, round(x1 * w)), max(int(y0 * h) + 1, round(y1 * h)))
    return img.crop(px), used


def _font(size: int) -> ImageFont.ImageFont:
    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # very old Pillow
        return ImageFont.load_default()


def draw_grid(img: Image.Image, grid: int = GRID) -> Image.Image:
    """Overlay a labelled grid (A1 top-left). Labels sit in each cell's top-left corner."""
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out, "RGBA")
    w, h = out.size
    font = _font(max(10, w // 48))
    for i in range(1, grid):
        d.line([(i * w / grid, 0), (i * w / grid, h)], fill=(255, 255, 255, 110), width=1)
        d.line([(0, i * h / grid), (w, i * h / grid)], fill=(255, 255, 255, 110), width=1)
    for c in range(grid):
        for r in range(grid):
            x, y = c * w / grid + 3, r * h / grid + 2
            label = cell_name(c, r)
            d.rectangle(d.textbbox((x, y), label, font=font), fill=(0, 0, 0, 120))
            d.text((x, y), label, fill=(255, 255, 255, 230), font=font)
    return out


def draw_ticks(img: Image.Image, box: Box, n: int = 4) -> Image.Image:
    """Grid lines on a crop, labelled with the GLOBAL normalised coordinates they correspond to."""
    out = img.convert("RGB").copy()
    d = ImageDraw.Draw(out, "RGBA")
    w, h = out.size
    font = _font(max(10, w // 40))
    x0, y0, x1, y1 = box
    for i in range(1, n):
        fx, fy = i / n, i / n
        gx, gy = x0 + fx * (x1 - x0), y0 + fy * (y1 - y0)
        d.line([(fx * w, 0), (fx * w, h)], fill=(255, 255, 255, 100), width=1)
        d.line([(0, fy * h), (w, fy * h)], fill=(255, 255, 255, 100), width=1)
        for (tx, ty), text in (((fx * w + 2, 2), f"x={gx:.2f}"), ((2, fy * h + 2), f"y={gy:.2f}")):
            d.rectangle(d.textbbox((tx, ty), text, font=font), fill=(0, 0, 0, 120))
            d.text((tx, ty), text, fill=(255, 255, 255, 230), font=font)
    return out


def side_by_side(left: Image.Image, right: Image.Image, labels: tuple[str, str] = ("this photo", "known-good reference")) -> Image.Image:
    h = max(left.height, right.height)
    left = left.resize((round(left.width * h / left.height), h))
    right = right.resize((round(right.width * h / right.height), h))
    gap = 8
    out = Image.new("RGB", (left.width + right.width + gap, h + 22), (30, 30, 30))
    out.paste(left, (0, 22))
    out.paste(right, (left.width + gap, 22))
    d = ImageDraw.Draw(out)
    font = _font(14)
    d.text((4, 3), labels[0], fill=(255, 255, 255), font=font)
    d.text((left.width + gap + 4, 3), labels[1], fill=(255, 255, 255), font=font)
    return out


def _colormap(x: np.ndarray) -> np.ndarray:
    """0..1 -> RGB uint8, dark blue -> yellow -> red (no matplotlib needed)."""
    x = np.clip(x, 0, 1)[..., None]
    stops = np.array([[20, 20, 90], [40, 140, 200], [250, 220, 40], [220, 30, 30]], float)
    pos = np.array([0.0, 0.35, 0.7, 1.0])
    out = np.zeros(x.shape[:-1] + (3,))
    for i in range(3):
        a, b = pos[i], pos[i + 1]
        t = np.clip((x[..., 0] - a) / (b - a), 0, 1)
        sel = (x[..., 0] >= a) & (x[..., 0] <= b)
        out[sel] = (stops[i] + (stops[i + 1] - stops[i]) * t[..., None])[sel]
    return out.astype(np.uint8)


def heatmap_overlay(
    img: Image.Image, anomaly_map: np.ndarray, boxes: Sequence[Box] = (), vmax: float | None = None, alpha: float = 0.45
) -> Image.Image:
    """Blend an anomaly map over the image and draw numbered boxes (1 = most anomalous)."""
    base = img.convert("RGB")
    vmax = vmax or max(float(anomaly_map.max()), 1e-6)
    heat = Image.fromarray(_colormap(anomaly_map / vmax)).resize(base.size, Image.BILINEAR)
    out = Image.blend(base, heat, alpha)
    d = ImageDraw.Draw(out)
    w, h = out.size
    font = _font(max(12, w // 32))
    for i, (x0, y0, x1, y1) in enumerate(boxes, 1):
        d.rectangle([x0 * w, y0 * h, x1 * w, y1 * h], outline=(255, 255, 255), width=3)
        d.text((x0 * w + 4, y0 * h + 2), str(i), fill=(255, 255, 255), font=font)
    return out


def to_image_data(img: Image.Image, label: str) -> ImageData:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="PNG", optimize=True)
    return image_from_bytes(buf.getvalue(), "image/png", label)
