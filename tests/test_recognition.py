"""Face recognition parity with dlib (golden file from tools/golden_descriptors.py) and API."""

import itertools
import threading

import numpy as np
import pytest

import manyfaced
import testimages
from manyfaced import FaceRecognizer

pytestmark = pytest.mark.parity

MODEL = testimages.MODELS_DIR / "dlib_face_recognition_resnet_model_v1.dat"
# Brief: L2 distance to dlib's descriptor < 1e-4.
MAX_L2 = 1e-4
# face_recognition.compare_faces default tolerance.
TOLERANCE = 0.6


@pytest.fixture(scope="module")
def fr():
    return FaceRecognizer(MODEL)


@pytest.fixture(scope="module")
def g(golden):
    return golden("descriptors")


def _lm(g, i):
    return g["lm_points"][i, : g["lm_num_parts"][i]]


def _image(images, g, i):
    # The original arrays, including non-contiguous views.
    return images[str(g["lm_image"][i])].array


def _check(ours, expected, what):
    assert ours.shape == expected.shape and ours.dtype == np.float64
    dist = np.linalg.norm(ours - expected.astype(np.float64), axis=1)
    worst = int(dist.argmax())
    assert dist.max() < MAX_L2, f"{what}: max L2 {dist.max():.3g} at case {worst}"
    return dist


def test_single_descriptors_match_dlib(fr, g, images):
    ours = np.array([fr.embed(_image(images, g, i), _lm(g, i)) for i in range(len(g["single"]))])
    _check(ours, g["single"], "single")


def test_batch_descriptors_match_dlib(fr, g, images):
    ids = g["batch_ids"]
    for b in np.unique(ids):
        rows = np.flatnonzero(ids == b)
        faces = g["batch_landmarks"][rows]
        ours = fr.embed_many(_image(images, g, faces[0]), [_lm(g, i) for i in faces])
        _check(ours, g["batch"][rows], f"batch {b}")


def test_chip_descriptors_match_dlib(fr, g, images):
    chips = []
    for i in range(len(g["chip"])):
        chip = manyfaced.face_chip(_image(images, g, i), _lm(g, i))
        assert testimages.digest(chip) == g["chip_digests"][i], f"chip {i} differs from dlib's"
        chips.append(chip)
    _check(fr.embed_chips(chips), g["chip"], "chip")


def test_compare_faces_decisions_are_identical(fr, g, images):
    ours = np.array([fr.embed(_image(images, g, i), _lm(g, i)) for i in range(len(g["single"]))])
    theirs = g["single"].astype(np.float64)
    pairs = list(itertools.combinations(range(len(ours)), 2))
    a, b = np.array(pairs).T
    d_ours = np.linalg.norm(ours[a] - ours[b], axis=1)
    d_theirs = np.linalg.norm(theirs[a] - theirs[b], axis=1)
    differ = np.flatnonzero((d_ours <= TOLERANCE) != (d_theirs <= TOLERANCE))
    assert differ.size == 0, f"{differ.size} of {len(pairs)} compare_faces decisions differ"


def test_grayscale_equals_replicated_rgb(fr, g, images):
    rgb = images["faces/2008_001322.jpg"].array
    gray = rgb[..., 1].copy()
    replicated = np.repeat(gray[..., None], 3, axis=2)
    i = int(np.flatnonzero(g["lm_image"] == "faces/2008_001322.jpg")[0])
    np.testing.assert_array_equal(fr.embed(gray, _lm(g, i)), fr.embed(replicated, _lm(g, i)))
    chip = manyfaced.face_chip(gray, _lm(g, i))
    np.testing.assert_array_equal(
        fr.embed_chip(chip), fr.embed_chip(np.repeat(chip[..., None], 3, axis=2))
    )


def test_jitter(fr, g, images):
    img, lm = _image(images, g, 0), _lm(g, 0)
    plain = fr.embed(img, lm)
    # num_jitters <= 1 does not jitter, like dlib.
    np.testing.assert_array_equal(fr.embed(img, lm, num_jitters=1), plain)
    a = fr.embed(img, lm, num_jitters=3, seed=1)
    b = fr.embed(img, lm, num_jitters=3, seed=1)
    c = fr.embed(img, lm, num_jitters=3, seed=2)
    assert a.shape == (128,)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
    # Jittered copies are still the same person.
    assert np.linalg.norm(a - plain) < 0.2


def test_padding_is_a_float32_like_dlib(fr, g, images):
    img, lm = _image(images, g, 0), _lm(g, 0)
    np.testing.assert_array_equal(
        fr.embed(img, lm, padding=0.1), fr.embed(img, lm, padding=float(np.float32(0.1)))
    )


def test_errors(fr):
    img = np.zeros((100, 100, 3), np.uint8)
    with pytest.raises(ValueError, match="5 or 68"):
        fr.embed(img, np.zeros((4, 2)))
    with pytest.raises(ValueError, match=r"\(N, 2\)"):
        fr.embed(img, np.zeros(10))
    with pytest.raises(TypeError, match="uint8"):
        fr.embed(img.astype(np.float32), np.zeros((5, 2)))
    with pytest.raises(ValueError, match="150x150"):
        fr.embed_chip(np.zeros((100, 100, 3), np.uint8))
    with pytest.raises(ValueError, match="square"):
        fr.embed_chip(np.zeros((150, 151, 3), np.uint8), num_jitters=2)
    with pytest.raises(FileNotFoundError):
        FaceRecognizer("/nonexistent/model.dat")
    with pytest.raises(manyfaced.ModelFormatError):
        FaceRecognizer(testimages.MODELS_DIR / "shape_predictor_5_face_landmarks.dat")
    assert fr.embed_many(img, []).shape == (0, 128)
    assert fr.chip_size == 150
    assert "FaceRecognizer" in repr(fr)


def test_threads_share_one_model(fr, g, images):
    cases = list(range(8))
    expected = [fr.embed(_image(images, g, i), _lm(g, i)) for i in cases]
    results = {}

    def work(t):
        results[t] = [fr.embed(_image(images, g, i), _lm(g, i)) for i in cases]

    threads = [threading.Thread(target=work, args=(t,)) for t in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert sorted(results) == [0, 1, 2, 3], "a worker thread failed"
    for got in results.values():
        for a, b in zip(got, expected, strict=True):
            np.testing.assert_array_equal(a, b)
