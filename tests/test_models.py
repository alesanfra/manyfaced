"""Bundled models and the model download helper (no network: downloads reuse valid files)."""

from __future__ import annotations

import bz2
import hashlib
import io
import shutil
import warnings

import pytest

import manyfaced
from manyfaced import _model_files as models


@pytest.mark.parametrize("name", [n for n, m in models.MODELS.items() if m.bundled])
def test_bundled_models_are_in_the_package(name):
    path = models.model_path(name)
    assert path.parent == models.BUNDLED_DIR
    assert models.sha256_of(path) == models.MODELS[name].sha256
    assert models.model_path(name + ".dat") == path


def test_default_loads_use_the_bundled_models():
    sp = manyfaced.ShapePredictor()
    assert sp.num_parts == 5
    rec = manyfaced.FaceRecognizer()
    assert rec.chip_size == 150


def test_unbundled_and_unknown_models():
    with pytest.raises(FileNotFoundError, match="download"):
        models.model_path("shape_predictor_68_face_landmarks")
    with pytest.raises(ValueError, match="unknown model"):
        models.model_path("mmod_human_face_detector")


def test_download_reuses_a_valid_file(tmp_path, monkeypatch):
    name = "shape_predictor_5_face_landmarks"
    shutil.copyfile(models.model_path(name), tmp_path / f"{name}.dat")
    monkeypatch.setenv("MANYFACED_MODEL_DIR", str(tmp_path))
    assert models.model_cache_dir() == tmp_path

    def no_network(*args, **kwargs):
        raise AssertionError("must not download a file that is already valid")

    monkeypatch.setattr(models.urllib.request, "urlopen", no_network)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the 5-point model has no license warning
        assert models.download_model(name) == tmp_path / f"{name}.dat"


def test_68_point_download_warns_about_the_license(tmp_path, monkeypatch):
    monkeypatch.setattr(
        models.urllib.request, "urlopen", lambda *a, **k: (_ for _ in ()).throw(OSError("offline"))
    )
    with pytest.warns(UserWarning, match="commercial"), pytest.raises(OSError, match="offline"):
        models.download_model("shape_predictor_68_face_landmarks", tmp_path)
    assert not list(tmp_path.iterdir()), "partial downloads are removed"


_PAYLOAD = b"not a real model, but enough bytes to need more than one read" * 1000


def _serve(monkeypatch, tmp_path, body, size=None, sha256=None):
    """Replace the model list with one fake model and the network with ``body``."""
    sha256 = sha256 or hashlib.sha256(_PAYLOAD).hexdigest()
    fake = models.Model(sha256, size=len(_PAYLOAD) if size is None else size, bundled=False)
    monkeypatch.setattr(models, "MODELS", {"fake": fake})
    monkeypatch.setattr(models, "_CHUNK", 1024)
    monkeypatch.setattr(models.urllib.request, "urlopen", lambda *a, **k: io.BytesIO(body))
    return tmp_path / "fake.dat"


def test_download_decompresses_and_verifies(tmp_path, monkeypatch):
    dest = _serve(monkeypatch, tmp_path, bz2.compress(_PAYLOAD))
    assert models.download_model("fake", tmp_path) == dest
    assert dest.read_bytes() == _PAYLOAD
    assert [p.name for p in tmp_path.iterdir()] == ["fake.dat"]


@pytest.mark.parametrize(
    ("body", "kwargs", "message"),
    [
        (bz2.compress(_PAYLOAD), {"sha256": "0" * 64}, "SHA-256 mismatch"),
        (bz2.compress(_PAYLOAD)[:-20], {}, "not complete"),
        (bz2.compress(_PAYLOAD), {"size": len(_PAYLOAD) - 1}, "larger than expected"),
    ],
    ids=["sha256", "truncated", "too-large"],
)
def test_bad_downloads_leave_no_file(tmp_path, monkeypatch, body, kwargs, message):
    _serve(monkeypatch, tmp_path, body, **kwargs)
    with pytest.raises(OSError, match=message):
        models.download_model("fake", tmp_path)
    assert not list(tmp_path.iterdir())


def test_license_warning_only_when_downloading(tmp_path, monkeypatch):
    name = "shape_predictor_68_face_landmarks"
    (tmp_path / f"{name}.dat").write_bytes(b"")
    monkeypatch.setattr(models, "sha256_of", lambda _: models.MODELS[name].sha256)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        assert models.download_model(name, tmp_path) == tmp_path / f"{name}.dat"
