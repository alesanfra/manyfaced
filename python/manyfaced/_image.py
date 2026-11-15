"""Image decoding with Pillow.

Pillow is imported only when it is necessary, so that ``import manyfaced`` stays fast.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import numpy as np

__all__ = ["is_pil_image", "load_rgb", "pil_to_rgb"]


def is_pil_image(obj: object) -> bool:
    """Give True if ``obj`` is a ``PIL.Image.Image``.

    This function does not import Pillow. If Pillow is not imported, ``obj`` cannot be a Pillow
    image.
    """
    module = sys.modules.get("PIL.Image")
    cls = getattr(module, "Image", None)
    return isinstance(cls, type) and isinstance(obj, cls)


def pil_to_rgb(image: Any, *, exif_transpose: bool) -> np.ndarray:
    """Change a Pillow image into an ``HxWx3`` ``uint8`` RGB array.

    If ``exif_transpose`` is True, the function first applies the EXIF Orientation tag. As a
    result, a photo from a turned camera comes out upright. ``ImageOps.exif_transpose`` removes
    the tag, so that an image that is already upright does not turn again.
    """
    from PIL import ImageOps

    oriented = ImageOps.exif_transpose(image) if exif_transpose else image
    return np.asarray(oriented.convert("RGB"), dtype=np.uint8)


def load_rgb(path: str | os.PathLike[str], *, exif_transpose: bool) -> np.ndarray:
    """Decode an image file into an ``HxWx3`` ``uint8`` RGB array with Pillow.

    ``exif_transpose`` is as in :func:`pil_to_rgb`.
    """
    from PIL import Image

    with Image.open(path) as im:
        return pil_to_rgb(im, exif_transpose=exif_transpose)
