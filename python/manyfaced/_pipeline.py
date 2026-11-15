"""The full pipeline in one call: detection, 5-point landmarks and face descriptors."""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, TypeVar, cast

import numpy as np

from ._image import is_pil_image, load_rgb, pil_to_rgb
from ._manyfaced import FhogDetector, Rect, ShapePredictor
from ._recognition import FaceRecognizer

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

__all__ = ["Face", "find_faces"]

_M = TypeVar("_M", FhogDetector, ShapePredictor, FaceRecognizer)

_lock = threading.Lock()
_defaults: dict[type, Any] = {}


def _default(cls: type[_M]) -> _M:
    """Give the instance of ``cls()`` for the process. The first call makes it.

    The models do not change after they load, so all threads can use the same instance.
    """
    with _lock:
        model = _defaults.get(cls)
        if model is None:
            model = _defaults[cls] = cls()
    return cast(_M, model)


@dataclass(frozen=True, eq=False)
class Face:
    """A face that :func:`find_faces` found.

    - ``rect``: the box of the detector.
    - ``landmarks``: the (5, 2) int64 ``(x, y)`` points of the bundled 5-point
      ``ShapePredictor``.
    - ``descriptor``: the (128,) float64 output of the bundled ``FaceRecognizer``.

    ``find_faces`` gives read-only arrays. Faces compare by identity, because the fields are
    numpy arrays. To compare two faces, use
    ``np.linalg.norm(a.descriptor - b.descriptor) <= 0.6``.
    """

    rect: Rect
    landmarks: np.ndarray
    descriptor: np.ndarray


def find_faces(
    image: np.ndarray | str | os.PathLike[str] | PILImage,
    *,
    upsample: int = 1,
    num_jitters: int = 0,
) -> list[Face]:
    """Find the faces in an image, with their landmarks and descriptors, in detector order.

    ``image`` can be one of these:

    - A ``uint8`` ``HxWx3`` (RGB) or ``HxW`` (grayscale) numpy array.
    - The path of an image file, as ``str`` or ``os.PathLike``. Pillow decodes the file.
    - A Pillow image (``PIL.Image.Image``), in any mode. It is changed into RGB.

    For a path or a Pillow image, the EXIF orientation is applied, so that a photo from a turned
    phone comes out upright. ``manyfaced.compat.dlib.load_rgb_image`` ignores the orientation,
    as dlib does.

    The function does ``FhogDetector()(image, upsample)``, then ``ShapePredictor()(image, rect)``
    for each face, then one ``FaceRecognizer().embed_many(image, landmarks, num_jitters)`` call.
    As a result, the chips are extracted together, as in dlib and face_recognition. The first
    call makes the three default models, and all threads use them again. For other models or
    options, use the models directly. If there are no faces, the result is an empty list.
    """
    array: np.ndarray
    if isinstance(image, str | os.PathLike):
        array = load_rgb(image, exif_transpose=True)
    elif is_pil_image(image):
        array = pil_to_rgb(image, exif_transpose=True)
    else:
        array = cast(np.ndarray, image)  # the native code checks the type and the shape
    rects = _default(FhogDetector)(array, upsample=upsample)
    if not rects:
        return []
    predictor = _default(ShapePredictor)
    landmarks = [predictor(array, rect) for rect in rects]
    descriptors = _default(FaceRecognizer).embed_many(array, landmarks, num_jitters=num_jitters)
    descriptors.flags.writeable = False
    for lm in landmarks:
        lm.flags.writeable = False
    return [Face(r, lm, d) for r, lm, d in zip(rects, landmarks, descriptors, strict=True)]
