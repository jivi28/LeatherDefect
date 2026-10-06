#!/usr/bin/env python
"""Check that your machine and this repo are ready. Safe to run any time; changes nothing.

    python scripts/check_setup.py
    python scripts/check_setup.py --data data/leather

Each line is [ok], [warn] or [FAIL] with the fix. Exit code 1 only if something blocking failed.
"""

from __future__ import annotations

import argparse
import importlib
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EXPECTED_DEFECTS = ("color", "cut", "fold", "glue", "poke")  # leather names as remembered; verify on real data
REQUIRED_MODULES = [("openai", "openai"), ("pydantic", "pydantic"), ("numpy", "numpy"), ("PIL", "pillow"), ("pytest", "pytest")]
OPTIONAL_MODULES = [("dotenv", "python-dotenv")]


@dataclass
class Check:
    name: str
    status: str  # ok | warn | fail
    detail: str
    fix: str = ""


def check_python(version: tuple[int, int] | None = None) -> Check:
    v = version or sys.version_info[:2]
    text = f"{v[0]}.{v[1]}"
    if v >= (3, 11):
        return Check("python", "ok", text)
    return Check("python", "fail", f"{text} is too old", "Install Python 3.11 or newer.")


def check_modules() -> list[Check]:
    checks = []
    for module, package in REQUIRED_MODULES + OPTIONAL_MODULES:
        required = (module, package) in REQUIRED_MODULES
        try:
            importlib.import_module(module)
            checks.append(Check(f"package {package}", "ok", "installed"))
        except ImportError:
            checks.append(
                Check(f"package {package}", "fail" if required else "warn", "missing", "pip install -r requirements-dev.txt")
            )
    return checks


def check_llm(env: Mapping[str, str]) -> Check:
    try:
        from agentkit.llm import ConfigError, from_env
    except ImportError as exc:
        return Check("llm config", "fail", f"cannot import agentkit: {exc}", "pip install -r requirements-dev.txt")
    try:
        llm = from_env(env)
    except ConfigError as exc:
        return Check(
            "llm config",
            "warn",
            str(exc),
            "Copy .env.example to .env and add a key. Tests and the fake dataset work without it; real agent runs do not.",
        )
    return Check("llm config", "ok", f"provider chain ready: {getattr(llm, 'label', None) or type(llm).__name__}", "Run: python scripts/smoke_llm.py")


def check_data(root: Path) -> list[Check]:
    from PIL import Image

    checks: list[Check] = []
    hint = "Generate practice data: python scripts/make_fake_leather.py   (real data: see START_HERE.md)"
    if not root.is_dir():
        return [Check("data folder", "fail", f"{root} not found", hint)]
    fake = (root / "FAKE_DATA.txt").exists()
    checks.append(Check("data kind", "warn" if fake else "ok", "FAKE generated data" if fake else "looks like real data"))

    train = sorted((root / "train" / "good").glob("*.png")) if (root / "train" / "good").is_dir() else []
    test_good = sorted((root / "test" / "good").glob("*.png")) if (root / "test" / "good").is_dir() else []
    if not train:
        checks.append(Check("train/good", "fail", "no png files", hint))
    else:
        checks.append(Check("train/good", "ok", f"{len(train)} images"))
    if not test_good:
        checks.append(Check("test/good", "fail", "no png files", hint))
    else:
        checks.append(Check("test/good", "ok", f"{len(test_good)} images"))

    test_dir = root / "test"
    defects = sorted(p.name for p in test_dir.iterdir() if p.is_dir() and p.name != "good") if test_dir.is_dir() else []
    if not defects:
        checks.append(Check("defect folders", "fail", "none under test/", hint))
        return checks
    counts = {d: len(list((test_dir / d).glob("*.png"))) for d in defects}
    checks.append(Check("defect folders", "ok", ", ".join(f"{d}={n}" for d, n in counts.items())))
    unexpected = [d for d in defects if d not in EXPECTED_DEFECTS]
    missing = [d for d in EXPECTED_DEFECTS if d not in defects]
    if unexpected or missing:
        checks.append(
            Check(
                "defect names",
                "warn",
                f"differ from the course (unexpected: {unexpected or 'none'}; missing: {missing or 'none'})",
                "Tell Claude Code the real folder names; the taxonomy in inspector/ must follow them (my names were from memory).",
            )
        )

    unmasked = []
    for d in defects:
        for img in (test_dir / d).glob("*.png"):
            if not (root / "ground_truth" / d / f"{img.stem}_mask.png").exists():
                unmasked.append(f"{d}/{img.name}")
    if unmasked:
        checks.append(Check("masks", "warn", f"{len(unmasked)} defect images without a mask, e.g. {unmasked[0]}", "Check the ground_truth folder."))
    else:
        checks.append(Check("masks", "ok", "every defect image has a mask"))

    sample = (train[:5] + test_good[:5]) if train and test_good else []
    sizes = set()
    modes = set()
    for p in sample:
        try:
            with Image.open(p) as im:
                sizes.add(im.size)
                modes.add(im.mode)
        except OSError:
            checks.append(Check("image files", "fail", f"cannot open {p.name}", "Re-download or regenerate the data."))
            return checks
    if sizes:
        checks.append(Check("image size", "ok" if len(sizes) == 1 else "warn", ", ".join(f"{w}x{h}" for w, h in sorted(sizes)) + f" ({'/'.join(sorted(modes))})"))
    return checks


def check_tools() -> list[Check]:
    checks = []
    for tool, why in [("git", "version control (Entire needs a git repo)"), ("claude", "Claude Code CLI")]:
        found = shutil.which(tool)
        checks.append(Check(f"tool {tool}", "ok" if found else "warn", found or f"not on PATH ({why})", "" if found else f"Install {tool}."))
    return checks


def run_checks(data_root: Path, env: Mapping[str, str] | None = None) -> list[Check]:
    env = os.environ if env is None else env
    checks = [check_python(), *check_modules(), check_llm(env), *check_data(data_root), *check_tools()]
    return checks


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv(ROOT / ".env")
    except ImportError:
        pass


def render(checks: list[Check]) -> str:
    marks = {"ok": "[ok]  ", "warn": "[warn]", "fail": "[FAIL]"}
    lines = []
    for c in checks:
        lines.append(f"{marks[c.status]} {c.name}: {c.detail}")
        if c.fix and c.status != "ok":
            lines.append(f"         fix: {c.fix}")
        elif c.fix and c.status == "ok":
            lines.append(f"         next: {c.fix}")
    fails = sum(c.status == "fail" for c in checks)
    warns = sum(c.status == "warn" for c in checks)
    lines.append("")
    lines.append(f"{fails} blocking problem(s), {warns} warning(s).")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default=str(ROOT / "data" / "leather"))
    a = p.parse_args(argv)
    _load_dotenv()
    checks = run_checks(Path(a.data))
    print(render(checks))
    return 1 if any(c.status == "fail" for c in checks) else 0


if __name__ == "__main__":
    raise SystemExit(main())
