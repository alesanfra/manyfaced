"""Compatibility layers that let existing code use manyfaced without changes."""

from __future__ import annotations

import sys


def install() -> None:
    """Make ``import dlib`` and ``import face_recognition_models`` give the manyfaced modules.

    Call this function before you import code that uses dlib, for example ``face_recognition``::

        import manyfaced.compat
        manyfaced.compat.install()
        import face_recognition
    """
    from . import dlib, face_recognition_models

    sys.modules["dlib"] = dlib
    sys.modules["face_recognition_models"] = face_recognition_models


__all__ = ["install"]
