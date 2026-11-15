"""Integration: face_recognition running on manyfaced.compat.dlib gives the same results as on dlib.

face_recognition is imported twice in this process, once with ``sys.modules["dlib"]`` set to the
original dlib (dlib-bin) and once set to ``manyfaced.compat.dlib``. Its model package
(face_recognition_models, 100 MB of the same dlib files) is replaced by a stub that points to
models/. The real dlib cannot load the CNN detector without its model file, so on the reference
side ``cnn_face_detection_model_v1`` is replaced by a placeholder; only model="hog" is compared.
"""

from __future__ import annotations

import importlib
import importlib.util
import itertools
import sys
import types
import warnings

import numpy as np
import pytest

import manyfaced.compat.dlib as manyfaced_dlib
import testimages

pytestmark = pytest.mark.parity

M = testimages.MODELS_DIR


def _models_stub() -> types.ModuleType:
    stub = types.ModuleType("face_recognition_models")
    stub.pose_predictor_model_location = lambda: str(M / "shape_predictor_68_face_landmarks.dat")
    stub.pose_predictor_five_point_model_location = lambda: str(
        M / "shape_predictor_5_face_landmarks.dat"
    )
    stub.face_recognition_model_location = lambda: str(
        M / "dlib_face_recognition_resnet_model_v1.dat"
    )
    stub.cnn_face_detector_model_location = lambda: str(M / "mmod_human_face_detector.dat")
    return stub


def _reference_dlib() -> types.ModuleType:
    if importlib.util.find_spec("dlib") is None:
        pytest.fail("needs the original dlib as the reference: uv sync --group golden")
    import dlib as real

    proxy = types.ModuleType("dlib")
    proxy.__dict__.update({k: getattr(real, k) for k in dir(real) if not k.startswith("__")})
    proxy.cnn_face_detection_model_v1 = manyfaced_dlib.cnn_face_detection_model_v1
    return proxy


def _import_face_recognition(
    dlib_module: types.ModuleType, models_module: types.ModuleType | None = None
) -> types.ModuleType:
    # find_spec, not import: importing face_recognition without its models calls quit().
    if importlib.util.find_spec("face_recognition") is None:
        pytest.fail("needs face_recognition: uv sync --group integration")
    saved = {k: sys.modules.get(k) for k in ("dlib", "face_recognition_models")}
    for k in [
        k for k in sys.modules if k == "face_recognition" or k.startswith("face_recognition.")
    ]:
        del sys.modules[k]
    sys.modules["dlib"] = dlib_module
    sys.modules["face_recognition_models"] = models_module or _models_stub()
    try:
        return importlib.import_module("face_recognition")
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
        for k in [
            k for k in sys.modules if k == "face_recognition" or k.startswith("face_recognition.")
        ]:
            del sys.modules[k]


@pytest.fixture(scope="module")
def apis():
    # manyfaced's side uses manyfaced.compat.face_recognition_models (what
    # manyfaced.compat.install() registers): bundled 5-point and ResNet models, and the 68-point
    # model from the download cache, pointed at models/ so the already downloaded file is reused
    # (no network).
    import manyfaced.compat.face_recognition_models as fr_models

    mp = pytest.MonkeyPatch()
    mp.setenv("MANYFACED_MODEL_DIR", str(M))
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # the 68-point license warning
            ours = _import_face_recognition(manyfaced_dlib, fr_models)
    finally:
        mp.undo()
    return {"dlib": _import_face_recognition(_reference_dlib()), "manyfaced": ours}


@pytest.fixture(scope="module")
def photos(images):
    # face_recognition works on contiguous RGB arrays (load_image_file returns them); dlib-bin
    # misbehaves on strided views (see docs/internals.md), so both sides get contiguous copies here.
    return {
        k: v.contiguous()
        for k, v in images.items()
        if k.startswith(("faces/", "large/")) and v.array.ndim == 3
    }


def test_face_recognition_uses_manyfaced(apis):
    import manyfaced

    assert apis["manyfaced"].api.dlib is manyfaced_dlib
    assert apis["manyfaced"].api.predictor_5_point_model == str(
        manyfaced.model_path("shape_predictor_5_face_landmarks")
    )
    assert apis["dlib"].api.dlib is not manyfaced_dlib


def test_face_locations_landmarks_encodings_match(apis, photos):
    ref, ours = apis["dlib"], apis["manyfaced"]
    encodings = {"dlib": [], "manyfaced": []}
    worst = 0.0
    for image_id, img in photos.items():
        locs = ref.face_locations(img, number_of_times_to_upsample=1, model="hog")
        assert ours.face_locations(img, number_of_times_to_upsample=1, model="hog") == locs, (
            image_id
        )
        assert ours.face_locations(img) == ref.face_locations(img), image_id
        for model in ("large", "small"):
            assert ours.face_landmarks(img, locs, model=model) == ref.face_landmarks(
                img, locs, model=model
            ), (image_id, model)
        assert ours.face_landmarks(img) == ref.face_landmarks(img), image_id
        for model in ("small", "large"):
            a = ref.face_encodings(img, locs, num_jitters=1, model=model)
            b = ours.face_encodings(img, locs, num_jitters=1, model=model)
            assert len(a) == len(b)
            for x, y in zip(a, b, strict=True):
                assert isinstance(y, np.ndarray) and y.shape == (128,)
                worst = max(worst, float(np.linalg.norm(x - y)))
            if model == "small":
                encodings["dlib"] += a
                encodings["manyfaced"] += b
    assert worst < 1e-4, worst

    # compare_faces / face_distance on every pair of faces.
    ref_enc, our_enc = encodings["dlib"], encodings["manyfaced"]
    assert len(ref_enc) > 10
    pairs = 0
    for i, j in itertools.combinations(range(len(ref_enc)), 2):
        assert ours.compare_faces([our_enc[i]], our_enc[j]) == ref.compare_faces(
            [ref_enc[i]], ref_enc[j]
        ), (i, j)
        pairs += 1
    d_ref = ref.face_distance(np.array(ref_enc), ref_enc[0])
    d_our = ours.face_distance(np.array(our_enc), our_enc[0])
    np.testing.assert_allclose(d_our, d_ref, atol=1e-5)
    assert pairs == len(ref_enc) * (len(ref_enc) - 1) // 2


def test_cnn_model_is_reported_as_unsupported(apis, photos):
    img = next(iter(photos.values()))
    with pytest.raises(NotImplementedError, match="CNN"):
        apis["manyfaced"].face_locations(img, model="cnn")
