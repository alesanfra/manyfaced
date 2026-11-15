"""The dlib model files: the files in the manyfaced wheels, and downloads for the other files.

Public API (``manyfaced`` exports it): ``model_path``, ``download_model``, ``model_cache_dir``.

The wheels contain these models (public domain, refer to docs/models.md):

- ``shape_predictor_5_face_landmarks`` (for ``manyfaced.ShapePredictor()``)
- ``dlib_face_recognition_resnet_model_v1`` (for ``manyfaced.FaceRecognizer()``)

The wheels do not contain ``shape_predictor_68_face_landmarks``. It is approximately 100 MB, and
its training data (iBUG 300-W) has a license that does not permit commercial use. To get it, use
:func:`download_model`.

This module uses only the standard library, because ``tools/download_models.py`` loads it
directly.
"""

from __future__ import annotations

import bz2
import hashlib
import os
import sys
import tempfile
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path

__all__ = ["MODELS", "BUNDLED_DIR", "model_path", "download_model", "model_cache_dir", "sha256_of"]

BASE_URL = "https://dlib.net/files/"
BUNDLED_DIR = Path(__file__).resolve().parent / "_models"


@dataclass(frozen=True)
class Model:
    sha256: str  # SHA-256 of the decompressed .dat file
    size: int  # size of the decompressed .dat file, in bytes
    bundled: bool
    note: str = ""

    def url(self, name: str) -> str:
        return f"{BASE_URL}{name}.dat.bz2"


_NON_COMMERCIAL = (
    "The training data of shape_predictor_68_face_landmarks is the iBUG 300-W dataset. Its "
    'annotations are "for research purposes ONLY (NO commercial products)". For this reason, '
    "the dlib model page says that you cannot use this model in a commercial product. If "
    "possible, use shape_predictor_5_face_landmarks (in the manyfaced wheel)."
)

# Name (without .dat) -> pinned SHA-256 of the decompressed file (first download, 2026-10-20).
MODELS: dict[str, Model] = {
    "shape_predictor_5_face_landmarks": Model(
        "c4b1e9804792707d3a405c2c16a80a20269e6675021f64a41d30fffafbc41888",
        size=9_150_489,
        bundled=True,
    ),
    "dlib_face_recognition_resnet_model_v1": Model(
        "55533b28a95800a551ba546ba62fe69625c7e95a7061c338adffead08719da30",
        size=22_466_066,
        bundled=True,
    ),
    "shape_predictor_68_face_landmarks": Model(
        "fbdc2cb80eb9aa7a758672cbfdda32ba6300efe9b6e6c7a299ff7e736b11b92f",
        size=99_693_937,
        bundled=False,
        note=_NON_COMMERCIAL,
    ),
}

_CHUNK = 1 << 20


def _name(name: str) -> str:
    stem = name[:-4] if name.endswith(".dat") else name
    if stem not in MODELS:
        raise ValueError(f"unknown model {name!r}; known models: {', '.join(MODELS)}")
    return stem


def sha256_of(file: Path) -> str:
    h = hashlib.sha256()
    with Path(file).open("rb") as f:
        while chunk := f.read(_CHUNK):
            h.update(chunk)
    return h.hexdigest()


def model_path(name: str) -> Path:
    """Give the path of a model in the manyfaced wheel.

    Raises ``FileNotFoundError`` in these conditions:

    - The wheel does not contain the model. Use :func:`download_model`.
    - manyfaced was built from source without the models. Run
      ``python tools/download_models.py --bundle`` before the build.
    """
    stem = _name(name)
    if not MODELS[stem].bundled:
        raise FileNotFoundError(
            f"{stem} is not in the manyfaced wheel. Use manyfaced.download_model({stem!r})."
        )
    file = BUNDLED_DIR / f"{stem}.dat"
    if not file.exists():
        raise FileNotFoundError(
            f"{file} does not exist: this manyfaced build does not contain its models. Run "
            f"`python tools/download_models.py --bundle` before the build, or use "
            f"manyfaced.download_model({stem!r})."
        )
    return file


def model_cache_dir() -> Path:
    """Give the directory where :func:`download_model` keeps the models.

    If ``$MANYFACED_MODEL_DIR`` is set, it is that directory. If not, it is the cache directory of
    the user: ``~/Library/Caches/manyfaced`` (macOS), ``%LOCALAPPDATA%\\manyfaced\\Cache``
    (Windows), or ``$XDG_CACHE_HOME/manyfaced`` or ``~/.cache/manyfaced`` (other platforms).
    """
    if env := os.environ.get("MANYFACED_MODEL_DIR"):
        return Path(env)
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Caches" / "manyfaced"
    if sys.platform == "win32":
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
        return Path(base) / "manyfaced" / "Cache"
    return Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache") / "manyfaced"


def download_model(name: str, directory: str | os.PathLike[str] | None = None) -> Path:
    """Download a dlib model from dlib.net (``.bz2``), decompress it and examine its SHA-256.

    The result is the path of the ``.dat`` file. If a valid file is already in the directory,
    the function uses it and does not download again. The default ``directory`` is
    :func:`model_cache_dir`.

    For ``shape_predictor_68_face_landmarks``, the function gives a ``UserWarning`` about the
    non-commercial license.
    """
    stem = _name(name)
    model = MODELS[stem]
    dest_dir = Path(directory) if directory is not None else model_cache_dir()
    dest = dest_dir / f"{stem}.dat"
    if dest.exists() and sha256_of(dest) == model.sha256:
        return dest
    if model.note:
        warnings.warn(model.note, UserWarning, stacklevel=2)
    dest_dir.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=dest_dir, prefix=dest.name + ".", suffix=".part")
    tmp = Path(tmp_name)
    try:
        decomp = bz2.BZ2Decompressor()
        h = hashlib.sha256()
        written = 0
        with os.fdopen(fd, "wb") as out, urllib.request.urlopen(model.url(stem), timeout=60) as r:
            while chunk := r.read(_CHUNK):
                # Limit the output of each step, so that a bad server cannot fill the memory or
                # the disk before the SHA-256 check.
                data = decomp.decompress(chunk, max_length=model.size + 1 - written)
                written += len(data)
                if written > model.size:
                    raise OSError(f"{model.url(stem)}: the file is larger than expected")
                h.update(data)
                out.write(data)
        if not decomp.eof:
            raise OSError(f"{model.url(stem)}: the download is not complete")
        if h.hexdigest() != model.sha256:
            raise OSError(
                f"{model.url(stem)}: SHA-256 mismatch (expected {model.sha256}, got "
                f"{h.hexdigest()})"
            )
        tmp.chmod(0o644)
        try:
            tmp.replace(dest)
        except PermissionError:
            # Windows: another process has dest open, for example a parallel download that
            # examines its SHA-256. Use that file if it is valid.
            if not (dest.exists() and sha256_of(dest) == model.sha256):
                raise
    finally:
        tmp.unlink(missing_ok=True)
    return dest
