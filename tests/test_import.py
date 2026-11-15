import importlib.metadata

import manyfaced
import manyfaced.compat.dlib


def test_version():
    # The Rust crate version (Cargo.toml) must be the version of the installed package.
    assert manyfaced.__version__ == importlib.metadata.version("manyfaced")
    assert manyfaced.DLIB_REFERENCE_VERSION == "20.0.1"


def test_native_module_location():
    from manyfaced import _manyfaced

    assert _manyfaced.__name__ == "manyfaced._manyfaced"


def test_flat_public_namespace():
    import importlib

    for name in manyfaced.__all__:
        assert hasattr(manyfaced, name), name
    for old in ("face", "detect", "cluster", "models"):
        assert not hasattr(manyfaced, old), old
        try:
            importlib.import_module(f"manyfaced.{old}")
        except ModuleNotFoundError:
            pass
        else:
            raise AssertionError(f"manyfaced.{old} should not exist")
    assert manyfaced.ShapePredictor.__module__ == "manyfaced"
    assert manyfaced.FhogDetector.__module__ == "manyfaced"
    assert manyfaced.ChipDetails.__module__ == "manyfaced"
