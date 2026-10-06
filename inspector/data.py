"""Samples, taxonomy and the deterministic dev/test split.

This is the only module (with metrics.py and evaluate.py) that knows about labels and `ground_truth/`.
Agent-facing code may call `taxonomy()` and `train_good()` (folder names and known-good images), nothing else.

Layout (MVTec AD style, one category):
    <root>/train/good/*.png
    <root>/test/<label>/*.png           label is "good" or a defect name
    <root>/ground_truth/<defect>/<stem>_mask.png
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_ROOT = ROOT / "data" / "leather"
GOOD = "good"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
SPLITS = ("dev", "test")
DEFAULT_SEED = 7
ALLOW_TEST_ENV = "INSPECTOR_ALLOW_TEST"


class SplitLocked(RuntimeError):
    """Raised when code asks for the test split without explicitly unlocking it."""


@dataclass(frozen=True)
class Sample:
    path: Path
    label: str  # "good" or a defect name
    mask: Path | None = None

    @property
    def is_defect(self) -> bool:
        return self.label != GOOD

    @property
    def id(self) -> str:
        return f"{self.label}/{self.path.stem}"


def _images(folder: Path) -> list[Path]:
    if not folder.is_dir():
        return []
    return sorted(p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)


def taxonomy(root: Path = DEFAULT_ROOT) -> list[str]:
    """Defect names, read from `test/<name>/` folders. Safe for agent code: names only, no labels per image."""
    test = Path(root) / "test"
    if not test.is_dir():
        return []
    return sorted(p.name for p in test.iterdir() if p.is_dir() and p.name != GOOD and _images(p))


def train_good(root: Path = DEFAULT_ROOT) -> list[Path]:
    """Known-good images used to fit the detector and as references. Safe for agent code."""
    return _images(Path(root) / "train" / GOOD)


def _mask_for(root: Path, label: str, image: Path) -> Path | None:
    if label == GOOD:
        return None
    folder = root / "ground_truth" / label
    for candidate in (folder / f"{image.stem}_mask.png", folder / f"{image.stem}.png"):
        if candidate.is_file():
            return candidate
    return None


def load_samples(root: Path = DEFAULT_ROOT) -> list[Sample]:
    """Every labelled image under test/, sorted by (label, filename)."""
    root = Path(root)
    samples: list[Sample] = []
    for label in [GOOD, *taxonomy(root)]:
        for image in _images(root / "test" / label):
            samples.append(Sample(image, label, _mask_for(root, label, image)))
    return samples


def split(samples: list[Sample], seed: int = DEFAULT_SEED) -> dict[str, list[Sample]]:
    """Stratified ~50/50 dev/test split, deterministic for a seed and independent of input order.

    Each label is shuffled on its own; odd-sized labels give the extra image to dev and test in turn,
    so neither side collects all the leftovers.
    """
    by_label: dict[str, list[Sample]] = {}
    for s in sorted(samples, key=lambda s: (s.label, s.path.name)):
        by_label.setdefault(s.label, []).append(s)
    rng = random.Random(seed)
    out: dict[str, list[Sample]] = {"dev": [], "test": []}
    odd_turn = 0
    for label in sorted(by_label):
        group = by_label[label][:]
        rng.shuffle(group)
        n_dev = len(group) // 2
        if len(group) % 2:
            n_dev += 1 - odd_turn
            odd_turn ^= 1
        out["dev"].extend(group[:n_dev])
        out["test"].extend(group[n_dev:])
    for name in out:
        out[name].sort(key=lambda s: (s.label, s.path.name))
    return out


def get_split(
    name: str, root: Path = DEFAULT_ROOT, seed: int = DEFAULT_SEED, *, allow_test: bool = False
) -> list[Sample]:
    """The dev or test split. The test split needs allow_test=True or INSPECTOR_ALLOW_TEST=1."""
    if name not in SPLITS:
        raise ValueError(f"unknown split '{name}', use one of {SPLITS}")
    if name == "test" and not (allow_test or os.environ.get(ALLOW_TEST_ENV) == "1"):
        raise SplitLocked(
            "The test split is locked. Tune on dev; unlock test only for the final, reported run "
            f"(allow_test=True or {ALLOW_TEST_ENV}=1)."
        )
    return split(load_samples(root), seed)[name]


def is_fake(root: Path = DEFAULT_ROOT) -> bool:
    return (Path(root) / "FAKE_DATA.txt").exists()
