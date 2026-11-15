"""shape_predictor parity with dlib (golden files from tools/gen_golden.py) and API behaviour."""

import threading

import numpy as np
import pytest

import manyfaced
import testimages
from manyfaced import ShapePredictor

MODELS = {5: "shape_predictor_5_face_landmarks.dat", 68: "shape_predictor_68_face_landmarks.dat"}

pytestmark = pytest.mark.parity


@pytest.fixture(scope="module", params=sorted(MODELS))
def model(request):
    n = request.param
    return n, ShapePredictor(testimages.MODELS_DIR / MODELS[n])


def test_landmarks_match_dlib(model, images, golden):
    n, sp = model
    g = golden(f"landmarks_{n}")
    assert str(g["model"]) == MODELS[n]
    mismatches = []
    worst = 0.0
    for k, (image_id, source, rect, expected) in enumerate(
        zip(g["case_image"], g["case_source"], g["rects"], g["parts"], strict=True)
    ):
        # The original arrays, including the non-contiguous views.
        got = sp(images[str(image_id)].array, tuple(int(v) for v in rect))
        assert got.shape == (n, 2) and got.dtype == np.int64
        err = float(np.linalg.norm((got - expected).astype(np.float64), axis=1).max())
        worst = max(worst, err)
        if err > 0:
            mismatches.append((k, str(image_id), str(source), rect.tolist(), err))
    # Brief: max error < 0.5 px per point. Landmarks are integers, so this means exact.
    assert worst < 0.5, f"{len(mismatches)} of {len(g['rects'])} cases differ: {mismatches[:5]}"
    assert not mismatches


def test_rect_argument_forms(model):
    _, sp = model
    img = np.full((60, 80, 3), 128, dtype=np.uint8)
    a = sp(img, manyfaced.Rect(10, 10, 50, 50))
    b = sp(img, (10, 10, 50, 50))
    np.testing.assert_array_equal(a, b)
    with pytest.raises(TypeError, match="Rect"):
        sp(img, (10, 10, 50))
    with pytest.raises(TypeError, match="Rect"):
        sp(img, "box")


def test_odd_images_and_rects(model):
    n, sp = model
    for img in [
        np.zeros((0, 0), dtype=np.uint8),
        np.zeros((1, 1, 3), dtype=np.uint8),
        np.full((7, 3), 200, dtype=np.uint8),
    ]:
        for rect in [(0, 0, 10, 10), (5, 5, 4, 4), (-100, -100, -50, -50), (10**6,) * 4]:
            assert sp(img, rect).shape == (n, 2)


def test_bad_images(model):
    _, sp = model
    with pytest.raises(TypeError, match="uint8"):
        sp(np.zeros((10, 10), dtype=np.float64), (0, 0, 5, 5))
    with pytest.raises(ValueError, match="HxWx3"):
        sp(np.zeros((10, 10, 4), dtype=np.uint8), (0, 0, 5, 5))
    with pytest.raises(TypeError, match="numpy"):
        sp([[0, 0], [0, 0]], (0, 0, 1, 1))


def test_grayscale_equals_rgb_with_the_same_intensity(model):
    _, sp = model
    gray = np.add.outer(np.arange(90), np.arange(120)).astype(np.uint8)
    rgb = np.repeat(gray[..., None], 3, axis=2)
    np.testing.assert_array_equal(sp(gray, (20, 20, 80, 80)), sp(rgb, (20, 20, 80, 80)))


def test_concurrent_calls_share_one_predictor(model, images):
    _, sp = model
    img = images["faces/2008_001322.jpg"].array
    rect = (100, 50, 250, 200)
    expected = sp(img, rect)
    results, errors = [], []

    def work():
        try:
            for _ in range(20):
                results.append(sp(img, rect))
        except Exception as e:  # pragma: no cover - reported below
            errors.append(e)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(results) == 160
    assert all(np.array_equal(r, expected) for r in results)


def test_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        ShapePredictor(tmp_path / "missing.dat")
    bad = tmp_path / "bad.dat"
    bad.write_bytes(b"\x01\x01\x01")
    with pytest.raises(manyfaced.ModelFormatError, match="truncated"):
        ShapePredictor(bad)
    with pytest.raises(manyfaced.ModelFormatError, match="shape_predictor"):
        ShapePredictor(testimages.MODELS_DIR / "dlib_face_recognition_resnet_model_v1.dat")


def test_repr_and_parts(model):
    n, sp = model
    assert sp.num_parts == n
    assert repr(sp).startswith(f"ShapePredictor(num_parts={n}, path=")
