"""A replacement for the ``face_recognition_models`` package that uses the manyfaced models.

``face_recognition`` gets the model paths from this module when you import it:

- The 5-point predictor and the ResNet come from the manyfaced wheel.
- The 68-point predictor is not for commercial use. face_recognition loads it at import time.
  This module downloads it one time into ``manyfaced.model_cache_dir()``.
- manyfaced does not support the CNN detector. Its path is only a placeholder.

Do not import this module directly. Use :func:`manyfaced.compat.install`.
"""

from __future__ import annotations

from .._model_files import download_model, model_path


def pose_predictor_model_location() -> str:
    return str(download_model("shape_predictor_68_face_landmarks"))


def pose_predictor_five_point_model_location() -> str:
    return str(model_path("shape_predictor_5_face_landmarks"))


def face_recognition_model_location() -> str:
    return str(model_path("dlib_face_recognition_resnet_model_v1"))


def cnn_face_detector_model_location() -> str:
    # manyfaced.compat.dlib.cnn_face_detection_model_v1 does not open this file.
    return "mmod_human_face_detector.dat (not supported by manyfaced)"
