"""Helpers shared by the golden generators (tools/gen_golden.py and tools/golden_*.py)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tests"))

import testimages  # noqa: E402
from testimages import GOLDEN_DIR, MODELS_DIR, TestImage  # noqa: E402,F401

try:
    import dlib
except ImportError:
    sys.exit("gen_golden.py needs the original dlib: uv sync --group golden")


def meta(images: list[TestImage]) -> dict[str, np.ndarray]:
    return {
        "dlib_version": np.array(dlib.__version__),
        "image_ids": np.array([img.id for img in images]),
        "image_digests": np.array([testimages.digest(img.array) for img in images]),
    }


def rect(r: dlib.rectangle) -> list[int]:
    return [r.left(), r.top(), r.right(), r.bottom()]


def synthetic_boxes(img: np.ndarray) -> list[tuple[str, list[int]]]:
    """Boxes that exercise edge cases independently of any detector."""
    h, w = img.shape[:2]
    cx, cy = w // 2, h // 2
    return [
        ("whole_image", [0, 0, w - 1, h - 1]),
        ("partly_outside", [-w // 4, -h // 4, w // 2, h // 2]),
        ("tiny_center", [cx - 5, cy - 5, cx + 4, cy + 4]),
        ("beyond_image", [-w // 2, -h // 2, w + w // 2, h + h // 2]),
        ("single_pixel", [cx, cy, cx, cy]),
    ]


def landmark_cases(images: list[TestImage]) -> list[tuple[str, str, list[int]]]:
    """(image id, box source, [l, t, r, b]) for every landmark test case."""
    detector = dlib.get_frontal_face_detector()
    labeled = testimages.landmark_boxes()
    cases = []
    for img in images:
        arr = img.contiguous()
        for b in labeled:
            if b.image == img.id:
                cases.append((img.id, "xml", [b.left, b.top, b.right, b.bottom]))
        if min(arr.shape[:2]) >= 20:
            for det in detector(arr, 1):
                cases.append((img.id, "hog_up1", rect(det)))
        for name, box in synthetic_boxes(arr):
            cases.append((img.id, name, box))
    return cases
