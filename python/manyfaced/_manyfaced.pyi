import os
from typing import final

import numpy as np
import numpy.typing as npt

__all__ = [
    "ChipDetails",
    "DLIB_REFERENCE_VERSION",
    "FhogDetector",
    "ModelFormatError",
    "Rect",
    "ShapePredictor",
    "_FaceRecognizer",
    "__version__",
    "_chinese_whispers",
    "_chinese_whispers_clustering",
    "_extract_chips",
    "_face_chip_details",
    "_image_info",
]

__version__: str
DLIB_REFERENCE_VERSION: str

class ModelFormatError(ValueError): ...

@final
class Rect:
    """An integer rectangle with inclusive bounds, as dlib's ``rectangle``."""

    def __new__(cls, left: int, top: int, right: int, bottom: int) -> Rect: ...
    @property
    def left(self) -> int: ...
    @property
    def top(self) -> int: ...
    @property
    def right(self) -> int: ...
    @property
    def bottom(self) -> int: ...
    @property
    def width(self) -> int: ...
    @property
    def height(self) -> int: ...
    @property
    def area(self) -> int: ...
    def is_empty(self) -> bool: ...
    def center(self) -> tuple[int, int]: ...
    def to_tuple(self) -> tuple[int, int, int, int]: ...

@final
class ShapePredictor:
    """dlib's shape_predictor (Ensemble of Regression Trees landmark predictor)."""

    def __new__(cls, path: str | os.PathLike[str] | None = None) -> ShapePredictor: ...
    @property
    def num_parts(self) -> int: ...
    def __call__(
        self,
        image: npt.NDArray[np.uint8],
        rect: Rect | tuple[int, int, int, int],
    ) -> npt.NDArray[np.int64]: ...

@final
class ChipDetails:
    """A rectangle, turned by ``angle`` radians and resampled to ``rows`` x ``cols`` pixels."""

    def __new__(
        cls, rect: tuple[float, float, float, float], rows: int, cols: int, angle: float = 0.0
    ) -> ChipDetails: ...
    @property
    def rect(self) -> tuple[float, float, float, float]: ...
    @property
    def angle(self) -> float: ...
    @property
    def rows(self) -> int: ...
    @property
    def cols(self) -> int: ...

def _face_chip_details(
    landmarks: npt.NDArray[np.float64], size: int, padding: float
) -> ChipDetails: ...
def _extract_chips(
    image: npt.NDArray[np.uint8], details: list[ChipDetails]
) -> list[npt.NDArray[np.uint8]]: ...

@final
class _FaceRecognizer:
    @staticmethod
    def load(path: str | os.PathLike[str]) -> _FaceRecognizer: ...
    @property
    def chip_size(self) -> int: ...
    def _compute(
        self,
        image: npt.NDArray[np.uint8],
        landmarks: list[npt.NDArray[np.float64]],
        num_jitters: int,
        padding: float,
        seed: int,
    ) -> npt.NDArray[np.float64]: ...
    def _compute_chips(
        self, chips: list[npt.NDArray[np.uint8]], num_jitters: int, seed: int
    ) -> npt.NDArray[np.float64]: ...

@final
class FhogDetector:
    def __new__(cls, path: str | os.PathLike[str] | None = None) -> FhogDetector: ...
    @property
    def num_detectors(self) -> int: ...
    @property
    def detection_window_width(self) -> int: ...
    @property
    def detection_window_height(self) -> int: ...
    def __call__(self, image: npt.NDArray[np.uint8], upsample: int = 0) -> list[Rect]: ...
    def run(
        self, image: npt.NDArray[np.uint8], upsample: int = 0, adjust_threshold: float = 0.0
    ) -> tuple[list[Rect], list[float], list[int]]: ...

def _chinese_whispers(
    edges: list[tuple[int, int] | tuple[int, int, float]], num_iterations: int = 100
) -> list[int]: ...
def _chinese_whispers_clustering(
    descriptors: npt.NDArray[np.float64], threshold: float
) -> list[int]: ...
def _image_info(image: npt.NDArray[np.uint8]) -> tuple[int, int, int, int]: ...
