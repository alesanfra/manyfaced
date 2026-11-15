"""Face descriptors: dlib's ``face_recognition_model_v1``."""

from __future__ import annotations

import os
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from . import _manyfaced
from ._chips import _landmarks

__all__ = ["FaceRecognizer"]


class FaceRecognizer:
    """dlib's face recognition ResNet. It changes a face into a 128-D descriptor.

    The descriptors of the same person are near each other. dlib and ``face_recognition`` find a
    match when the Euclidean distance between two descriptors is 0.6 or less.

    ``FaceRecognizer()`` uses the ResNet in the manyfaced wheel. ``FaceRecognizer(path)`` loads
    ``dlib_face_recognition_resnet_model_v1.dat`` from ``path``. Many threads can use the same
    instance. The GIL is released during the computation.

    When ``num_jitters > 1``, the descriptor is the mean of the descriptors of ``num_jitters``
    copies of the chip with random jitter, as in dlib. This is slower and slightly more accurate.
    For a given ``seed``, the jitter is deterministic. manyfaced uses its own random generator,
    so the jitter descriptors are not identical to the dlib descriptors.

    dlib accepts only RGB images. This class also accepts grayscale images and changes them into
    RGB.
    """

    def __init__(self, path: str | os.PathLike[str] | None = None) -> None:
        if path is None:
            from ._model_files import model_path

            path = model_path("dlib_face_recognition_resnet_model_v1")
        self._native = _manyfaced._FaceRecognizer.load(path)

    @property
    def chip_size(self) -> int:
        """The side of the aligned face chips that the network uses (150)."""
        return self._native.chip_size

    def embed(
        self,
        image: np.ndarray,
        landmarks: npt.ArrayLike,
        num_jitters: int = 0,
        padding: float = 0.25,
        seed: int = 0,
    ) -> np.ndarray:
        """Give the descriptor of one face, as dlib's ``compute_face_descriptor(img, face)``.

        ``landmarks`` is the (5, 2) or (68, 2) output of a ``ShapePredictor``. The result is a
        float64 array with shape (128,).
        """
        return self.embed_many(image, [landmarks], num_jitters, padding, seed)[0]

    def embed_many(
        self,
        image: np.ndarray,
        landmarks: Sequence[npt.ArrayLike],
        num_jitters: int = 0,
        padding: float = 0.25,
        seed: int = 0,
    ) -> np.ndarray:
        """Give the descriptors of the faces of one image, as dlib's
        ``compute_face_descriptor(img, faces)``.

        The chips are extracted together. As a result, the descriptors can be slightly different
        from the descriptors of one call for each face. The result is a float64 (N, 128) array.
        """
        lms = [_landmarks(lm) for lm in landmarks]
        return self._native._compute(image, lms, int(num_jitters), float(padding), int(seed))

    def embed_chip(self, chip: np.ndarray, num_jitters: int = 0, seed: int = 0) -> np.ndarray:
        """Give the descriptor of an aligned 150x150 face chip (``manyfaced.face_chip``), as
        dlib's ``compute_face_descriptor(chip)``. The result is a float64 array with shape (128,).
        """
        return self.embed_chips([chip], num_jitters, seed)[0]

    def embed_chips(
        self, chips: Sequence[np.ndarray], num_jitters: int = 0, seed: int = 0
    ) -> np.ndarray:
        """Give the descriptors of aligned face chips. The result is a float64 (N, 128) array."""
        return self._native._compute_chips(list(chips), int(num_jitters), int(seed))

    def __repr__(self) -> str:
        return repr(self._native)
