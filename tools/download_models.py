#!/usr/bin/env python3
"""Download the original dlib model files into ``models/`` and verify them.

The models are fetched from https://dlib.net/files/ as ``.bz2`` archives, decompressed, and
checked against the SHA-256 digests pinned in ``python/manyfaced/_model_files.py`` (the single
source of truth, loaded here without importing the manyfaced package, so this works before manyfaced
is built).

Usage:
    python tools/download_models.py            # download missing models into models/
    python tools/download_models.py --verify   # only verify what is present
    python tools/download_models.py --bundle   # also copy the bundled models (5-point
                                               # predictor, ResNet) into python/manyfaced/_models/
                                               # so that wheels built afterwards include them

Only the standard library is used.
"""

from __future__ import annotations

import argparse
import importlib.util
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DIR = ROOT / "models"


def _load_models_module():
    spec = importlib.util.spec_from_file_location(
        "_manyfaced_models", ROOT / "python" / "manyfaced" / "_model_files.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses need the module to be registered
    spec.loader.exec_module(module)
    return module


models = _load_models_module()


def check(path: Path, expected: str) -> bool:
    actual = models.sha256_of(path)
    if actual != expected:
        print(
            f"{path.name}: sha256 mismatch\n  expected {expected}\n  actual   {actual}",
            file=sys.stderr,
        )
        return False
    print(f"{path.name}: ok", file=sys.stderr)
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dir", type=Path, default=DEFAULT_DIR, help="target directory")
    parser.add_argument("--verify", action="store_true", help="do not download, only verify")
    parser.add_argument(
        "--bundle",
        action="store_true",
        help="copy the models bundled in the wheels into python/manyfaced/_models/",
    )
    args = parser.parse_args()
    args.dir.mkdir(parents=True, exist_ok=True)

    ok = True
    for name, model in models.MODELS.items():
        path = args.dir / f"{name}.dat"
        if not path.exists():
            if args.verify:
                print(f"{path.name}: missing", file=sys.stderr)
                ok = False
                continue
            print(f"downloading {model.url(name)}", file=sys.stderr)
            # warnings about the 68-point license are for library users, not for this tool.
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                path = models.download_model(name, args.dir)
        ok &= check(path, model.sha256)
        if ok and args.bundle and model.bundled:
            models.BUNDLED_DIR.mkdir(parents=True, exist_ok=True)
            target = models.BUNDLED_DIR / path.name
            if not target.exists() or models.sha256_of(target) != model.sha256:
                shutil.copyfile(path, target)
            print(f"{path.name}: bundled in {target.relative_to(ROOT)}", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
