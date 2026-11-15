"""manyfaced: a Rust port of the dlib face pipeline.

It contains face detection, landmarks, face chips, face descriptors and clustering. It reads the
original dlib ``.dat`` models and gives the same results as dlib.
"""

from . import _cpu

_cpu.check()

from ._chips import (  # noqa: E402
    ChipDetails,
    extract_chip,
    extract_chips,
    face_chip,
    face_chip_details,
    face_chips,
)
from ._cluster import chinese_whispers, chinese_whispers_clustering  # noqa: E402
from ._manyfaced import (  # noqa: E402
    DLIB_REFERENCE_VERSION,
    FhogDetector,
    ModelFormatError,
    Rect,
    ShapePredictor,
    __version__,
)
from ._model_files import download_model, model_cache_dir, model_path  # noqa: E402
from ._pipeline import Face, find_faces  # noqa: E402
from ._recognition import FaceRecognizer  # noqa: E402

__all__ = [
    "DLIB_REFERENCE_VERSION",
    "ChipDetails",
    "Face",
    "FaceRecognizer",
    "FhogDetector",
    "ModelFormatError",
    "Rect",
    "ShapePredictor",
    "__version__",
    "chinese_whispers",
    "chinese_whispers_clustering",
    "download_model",
    "extract_chip",
    "extract_chips",
    "face_chip",
    "face_chip_details",
    "face_chips",
    "find_faces",
    "model_cache_dir",
    "model_path",
]
