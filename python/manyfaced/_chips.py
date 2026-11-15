"""Face chip extraction: dlib's ``get_face_chip_details`` and ``extract_image_chip(s)``."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from . import _manyfaced

ChipDetails = _manyfaced.ChipDetails

__all__ = [
    "ChipDetails",
    "extract_chip",
    "extract_chips",
    "face_chip",
    "face_chip_details",
    "face_chips",
]


def _landmarks(landmarks: npt.ArrayLike) -> np.ndarray:
    """Make sure that ``landmarks`` is a (5, 2) or (68, 2) array. Give a contiguous float64 copy."""
    pts = np.asarray(landmarks, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"landmarks must have shape (N, 2), got {pts.shape}")
    if pts.shape[0] not in (5, 68):
        raise ValueError(f"expected 5 or 68 landmarks, got {pts.shape[0]}")
    return np.ascontiguousarray(pts)


def face_chip_details(
    landmarks: npt.ArrayLike, size: int = 150, padding: float = 0.25
) -> ChipDetails:
    """Give the position of a ``size`` x ``size`` aligned face chip, as dlib's
    ``get_face_chip_details``.

    ``landmarks`` is an (N, 2) array of (x, y) from the 5-point or 68-point shape predictor.
    ``padding`` is the margin around the face, as a fraction of the face size.
    """
    return _manyfaced._face_chip_details(_landmarks(landmarks), int(size), float(padding))


def extract_chips(image: np.ndarray, details: Sequence[ChipDetails]) -> list[np.ndarray]:
    """Extract several chips, as dlib's ``extract_image_chips``.

    The chips of one call use the same image pyramid, as in dlib. As a result, the pixels can be
    slightly different from the pixels of one call for each chip.
    """
    return _manyfaced._extract_chips(image, list(details))


def extract_chip(image: np.ndarray, details: ChipDetails) -> np.ndarray:
    """Extract one chip, as dlib's ``extract_image_chip``.

    Gives a uint8 array with the same number of channels as the input (``rows x cols`` or
    ``rows x cols x 3``). Pixels outside the image are black.
    """
    return _manyfaced._extract_chips(image, [details])[0]


def face_chip(
    image: np.ndarray, landmarks: npt.ArrayLike, size: int = 150, padding: float = 0.25
) -> np.ndarray:
    """Give the aligned face chip for ``landmarks``, as ``dlib.get_face_chip``."""
    return extract_chip(image, face_chip_details(landmarks, size, padding))


def face_chips(
    image: np.ndarray,
    landmarks: Sequence[npt.ArrayLike],
    size: int = 150,
    padding: float = 0.25,
) -> list[np.ndarray]:
    """Give the aligned face chips for the faces of one image, as ``dlib.get_face_chips``."""
    return extract_chips(image, [face_chip_details(lm, size, padding) for lm in landmarks])
