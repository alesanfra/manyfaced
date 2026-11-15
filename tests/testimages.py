"""The shared set of test images, used both by tools/gen_golden.py and by the tests.

Natural images come from ``third_party/dlib/examples/faces`` (not committed:
their licenses are those of PASCAL VOC and of the original photographers).
JPEGs are decoded with Pillow, and the *decoded arrays* are what we feed to
both dlib and manyfaced, so JPEG decoder differences between dlib and Pillow do
not matter. Each golden file records a digest of every decoded image so a
Pillow/libjpeg change that alters pixels is detected instead of showing up as
a bogus parity failure.

Synthetic edge cases are generated from closed-form formulas (no RNG), so
they are bit-identical everywhere.
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
DLIB_DIR = ROOT / "third_party" / "dlib"
FACES_DIR = DLIB_DIR / "examples" / "faces"
MODELS_DIR = ROOT / "models"
GOLDEN_DIR = ROOT / "tests" / "golden"

FACE_IMAGES = [
    "2007_007763.jpg",
    "2008_001009.jpg",
    "2008_001322.jpg",
    "2008_002079.jpg",
    "2008_002470.jpg",
    "2008_002506.jpg",
    "2008_004176.jpg",
    "2008_007676.jpg",
    "2009_004587.jpg",
    "Tom_Cruise_avp_2014_4.jpg",
    "bald_guys.jpg",
    "dogs.jpg",
]

LANDMARK_XMLS = ["training_with_face_landmarks.xml", "testing_with_face_landmarks.xml"]


@dataclass(frozen=True)
class TestImage:
    """A test image. ``array`` may be a non-contiguous view on purpose."""

    id: str
    array: np.ndarray

    def contiguous(self) -> np.ndarray:
        return np.ascontiguousarray(self.array)


def digest(arr: np.ndarray) -> str:
    """SHA-256 over dtype, shape and C-order pixel bytes."""
    h = hashlib.sha256()
    h.update(f"{arr.dtype.str}{arr.shape}".encode())
    h.update(np.ascontiguousarray(arr).tobytes())
    return h.hexdigest()


def _load_jpeg(path: Path, mode: str) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as im:
        return np.asarray(im.convert(mode), dtype=np.uint8)


def _tiny_texture(h: int, w: int) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    return ((x * 37 + y * 91 + x * y * 7) % 256).astype(np.uint8)


def _gradient(h: int, w: int) -> np.ndarray:
    y, x = np.mgrid[0:h, 0:w]
    r = (x * 255) // (w - 1)
    g = (y * 255) // (h - 1)
    b = ((x + y) * 255) // (w + h - 2)
    return np.stack([r, g, b], axis=-1).astype(np.uint8)


def load_images() -> list[TestImage]:
    """Return every test image. Raises if third_party/dlib is missing."""
    if not FACES_DIR.is_dir():
        raise FileNotFoundError(
            f"{FACES_DIR} not found: clone dlib into third_party/dlib (see docs/development.md)"
        )
    images = [
        TestImage(f"faces/{name}", _load_jpeg(FACES_DIR / name, "RGB")) for name in FACE_IMAGES
    ]
    by_id = {img.id: img.array for img in images}

    tom = by_id["faces/Tom_Cruise_avp_2014_4.jpg"]
    pascal = by_id["faces/2008_001322.jpg"]
    images += [
        # Pillow's "L" conversion is used only to produce *some* grayscale input;
        # both libraries receive the same uint8 HxW array.
        TestImage("gray/2008_001322", _load_jpeg(FACES_DIR / "2008_001322.jpg", "L")),
        TestImage("tiny/texture_13x17", _tiny_texture(13, 17)),
        TestImage("tiny/texture_1x1", _tiny_texture(1, 1)),
        TestImage("noface/gradient_240x320", _gradient(240, 320)),
        # Nearest-neighbour 3x upscale: 2400x1695, deterministic.
        TestImage("large/tom_cruise_x3", np.repeat(np.repeat(tom, 3, axis=0), 3, axis=1)),
        # Non-contiguous views: negative strides and step slicing.
        TestImage("noncontig/2008_001322_flipped", pascal[:, ::-1]),
        TestImage("noncontig/tom_cruise_step2", tom[::2, ::2]),
    ]
    return images


@dataclass(frozen=True)
class LabeledBox:
    """A ground-truth face box from dlib's example XML files (dlib rectangle convention)."""

    image: str
    left: int
    top: int
    right: int
    bottom: int


def landmark_boxes() -> list[LabeledBox]:
    """Face boxes from the example datasets, keyed by ``faces/<file>`` image ids."""
    boxes = []
    for xml_name in LANDMARK_XMLS:
        root = ET.parse(FACES_DIR / xml_name).getroot()
        for image in root.iter("image"):
            for box in image.iter("box"):
                left, top = int(box.get("left")), int(box.get("top"))
                width, height = int(box.get("width")), int(box.get("height"))
                boxes.append(
                    LabeledBox(
                        f"faces/{image.get('file')}", left, top, left + width - 1, top + height - 1
                    )
                )
    return boxes
