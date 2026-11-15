"""manyfaced.compat.dlib against the original dlib (dlib-bin), side by side in one process."""

from __future__ import annotations

import itertools
import pickle

import numpy as np
import pytest

import manyfaced.compat.dlib as fd
import testimages

pytestmark = pytest.mark.parity

try:
    import dlib
except ImportError:  # pragma: no cover - the parity environment always has it
    dlib = None


@pytest.fixture(autouse=True)
def _needs_dlib():
    if dlib is None:
        pytest.fail("the compat tests compare against the original dlib: uv sync --group golden")


M = testimages.MODELS_DIR
SP5 = str(M / "shape_predictor_5_face_landmarks.dat")
SP68 = str(M / "shape_predictor_68_face_landmarks.dat")
RESNET = str(M / "dlib_face_recognition_resnet_model_v1.dat")
LIBS = {"dlib": dlib, "manyfaced": fd}


@pytest.fixture(scope="module")
def models():
    return {
        name: {
            "det": lib.get_frontal_face_detector(),
            "sp5": lib.shape_predictor(SP5),
            "sp68": lib.shape_predictor(SP68),
            "rec": lib.face_recognition_model_v1(RESNET),
        }
        for name, lib in LIBS.items()
    }


def _rects(rs):
    return [(r.left(), r.top(), r.right(), r.bottom()) for r in rs]


def _parts(det):
    return [(p.x, p.y) for p in det.parts()]


def _same_exception(f):
    """Both libraries raise the same exception type (or neither raises)."""
    outcomes = {}
    for name, lib in LIBS.items():
        try:
            f(lib)
            outcomes[name] = None
        except Exception as e:  # noqa: BLE001
            outcomes[name] = type(e)
    assert outcomes["dlib"] == outcomes["manyfaced"], outcomes


# --- geometry -------------------------------------------------------------------------------

COORDS = [-3, 0, 1, 4, 7]
RECTS = [(a, b, c, d) for a, b, c, d in itertools.product(COORDS, repeat=4) if (a + b + c + d) % 3]


def test_rectangle_methods_match_dlib():
    for args in RECTS[:200]:
        d, f = dlib.rectangle(*args), fd.rectangle(*args)
        for m in ["left", "top", "right", "bottom", "width", "height", "area", "is_empty"]:
            assert getattr(f, m)() == getattr(d, m)(), (args, m)
        for m in ["center", "dcenter", "tl_corner", "tr_corner", "bl_corner", "br_corner"]:
            pd, pf = getattr(d, m)(), getattr(f, m)()
            assert (pf.x, pf.y) == (pd.x, pd.y), (args, m)
        assert repr(f) == repr(d) and str(f) == str(d)
        for x, y in [(0, 0), (4, 1), (-3, 7), (8, 8)]:
            assert f.contains(x, y) == d.contains(x, y)
            assert f.contains(fd.point(x, y)) == d.contains(dlib.point(x, y))
    for a, b in itertools.islice(itertools.product(RECTS[:40], repeat=2), 400):
        da, db, fa, fb = dlib.rectangle(*a), dlib.rectangle(*b), fd.rectangle(*a), fd.rectangle(*b)
        assert _rects([fa.intersect(fb)]) == _rects([da.intersect(db)])
        assert _rects([fa + fb]) == _rects([da + db])
        assert fa.contains(fb) == da.contains(db)
        assert (fa == fb) == (da == db) and (fa != fb) == (da != db)


def test_drectangle_and_points_match_dlib():
    for args in [(0.4, 0.5, 2.5, 3.49), (-1.5, -2.5, 1.5, 2.5), (5.0, 5.0, 1.0, 1.0)]:
        d, f = dlib.drectangle(*args), fd.drectangle(*args)
        for m in ["left", "top", "right", "bottom", "width", "height", "area", "is_empty"]:
            assert getattr(f, m)() == getattr(d, m)(), (args, m)
        assert repr(f) == repr(d) and str(f) == str(d)
        assert _rects([fd.rectangle(f)]) == _rects([dlib.rectangle(d)])
        assert (f.center().x, f.center().y) == (d.center().x, d.center().y)
    assert repr(fd.point(3, -4)) == repr(dlib.point(3, -4))
    assert str(fd.dpoint(1.25, -2)) == str(dlib.dpoint(1.25, -2))
    assert repr(fd.dpoint(1.0 / 3, 2e-7)) == repr(dlib.dpoint(1.0 / 3, 2e-7))
    p = fd.point(fd.dpoint(2.5, -2.5))
    q = dlib.point(dlib.dpoint(2.5, -2.5))
    assert (p.x, p.y) == (q.x, q.y)
    assert repr(fd.rectangles([fd.rectangle(1, 2, 3, 4)])) == repr(
        dlib.rectangles([dlib.rectangle(1, 2, 3, 4)])
    )
    r = fd.rectangle(1, 2, 3, 4)
    r += fd.point(10, -5)
    s = dlib.rectangle(1, 2, 3, 4)
    s += dlib.point(10, -5)
    assert _rects([r]) == _rects([s])
    for obj in [fd.rectangle(1, 2, 3, 4), fd.drectangle(1, 2, 3, 4.5), fd.point(1, 2)]:
        assert repr(pickle.loads(pickle.dumps(obj))) == repr(obj)


# --- pipeline -------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def rgb_images(images):
    return {
        k: v.array for k, v in images.items() if v.array.ndim == 3 and min(v.array.shape[:2]) >= 20
    }


def test_detector_landmarks_chips_descriptors_match_dlib(models, images, rgb_images):
    worst = 0.0
    for image_id, view in rgb_images.items():
        out = {}
        for name, lib in LIBS.items():
            # dlib-bin gets contiguous copies: on strided views its results depend on earlier
            # calls in the process (seen on noncontig/tom_cruise_step2, see docs/internals.md).
            # manyfaced gets the original, possibly non-contiguous, views.
            img = np.ascontiguousarray(view) if lib is dlib else view
            m = models[name]
            rects = m["det"](img, 1)
            run = m["det"].run(img, 0, -0.5)
            shapes5, shapes68 = lib.full_object_detections(), lib.full_object_detections()
            for r in rects:
                shapes5.append(m["sp5"](img, r))
                shapes68.append(m["sp68"](img, r))
            chips = lib.get_face_chips(img, shapes5) if shapes5 else []
            details = lib.get_face_chip_details(shapes68) if shapes68 else []
            descs = (
                [np.array(v) for v in m["rec"].compute_face_descriptor(img, shapes5)]
                if shapes5
                else []
            )
            out[name] = {
                "rects": _rects(rects),
                "run": (_rects(run[0]), run[2]),
                "scores": np.array(run[1]),
                "p5": [_parts(s) for s in shapes5],
                "p68": [_parts(s) for s in shapes68],
                "chips": [np.asarray(c) for c in chips],
                "details": [
                    (_rects([dlib.rectangle(0, 0, 0, 0)]), d.rows, d.cols) for d in details
                ],
                "detail_rects": [(d.rect.left(), d.rect.top(), d.angle) for d in details],
                "descs": descs,
            }
        a, b = out["dlib"], out["manyfaced"]
        for k in ["rects", "run", "p5", "p68", "details"]:
            assert a[k] == b[k], (image_id, k)
        np.testing.assert_allclose(a["scores"], b["scores"], atol=1e-4, rtol=0)
        np.testing.assert_allclose(a["detail_rects"], b["detail_rects"], rtol=1e-9, atol=1e-9)
        for ca, cb in zip(a["chips"], b["chips"], strict=True):
            assert ca.shape == cb.shape
            assert np.abs(ca.astype(int) - cb.astype(int)).max(initial=0) <= 1
        for da, db in zip(a["descs"], b["descs"], strict=True):
            worst = max(worst, float(np.linalg.norm(da - db)))
    assert worst < 1e-4, worst


def test_compute_face_descriptor_overloads(models, images):
    img = images["faces/2008_001322.jpg"].contiguous()
    res = {}
    for name, lib in LIBS.items():
        m = models[name]
        shapes = lib.full_object_detections()
        for r in m["det"](img, 1):
            shapes.append(m["sp5"](img, r))
        chip = lib.get_face_chip(img, shapes[0])
        rec = m["rec"]
        res[name] = [
            np.array(rec.compute_face_descriptor(img, shapes[0])),
            np.array(rec.compute_face_descriptor(img, shapes[0], 0, 0.25)),
            np.array(rec.compute_face_descriptor(img=img, face=shapes[0], padding=0.4)),
            np.array(rec.compute_face_descriptor(chip)),
            np.array(rec.compute_face_descriptor(chip, 0)),
            np.array([np.array(v) for v in rec.compute_face_descriptor(img, shapes)]),
            np.array([np.array(v) for v in rec.compute_face_descriptor(img, faces=shapes)]),
            np.array([np.array(v) for v in rec.compute_face_descriptor([chip, chip])]),
            np.array(
                [np.array(v) for vs in rec.compute_face_descriptor([img], [shapes]) for v in vs]
            ),
        ]
        assert len(rec.compute_face_descriptor(img, shapes[0])) == 128
    for a, b in zip(res["dlib"], res["manyfaced"], strict=True):
        assert a.shape == b.shape
        assert np.abs(a - b).max() < 1e-5


def test_errors_match_dlib(models, images):
    img = images["faces/2008_001322.jpg"].contiguous()
    gray = images["gray/2008_001322"].contiguous()
    box = lambda lib: lib.rectangle(232, 218, 294, 281)  # noqa: E731

    def shape(lib):
        return lib.shape_predictor(SP5)(img, box(lib))

    _same_exception(lambda lib: lib.get_frontal_face_detector()(img.astype(np.float32)))
    _same_exception(lambda lib: lib.shape_predictor(SP5)(np.zeros((5, 5, 4), np.uint8), box(lib)))
    _same_exception(
        lambda lib: lib.face_recognition_model_v1(RESNET).compute_face_descriptor(gray, shape(lib))
    )
    _same_exception(
        lambda lib: lib.face_recognition_model_v1(RESNET).compute_face_descriptor(
            np.zeros((100, 100, 3), np.uint8)
        )
    )
    _same_exception(
        lambda lib: lib.get_face_chip(
            img, lib.full_object_detection(box(lib), [lib.point(1, 1)] * 4)
        )
    )
    _same_exception(lambda lib: lib.full_object_detection(box(lib), [lib.point(1, 1)]).part(3))
    _same_exception(lambda lib: lib.chinese_whispers([(1, 2, 3, 4)]))
    _same_exception(lambda lib: lib.shape_predictor("/nonexistent.dat"))
    _same_exception(lambda lib: lib.face_recognition_model_v1("/nonexistent.dat"))
    _same_exception(lambda lib: lib.fhog_object_detector("README.md"))
    _same_exception(lambda lib: lib.shape_predictor(RESNET))


def test_clustering_matches_dlib(golden):
    descs = golden("descriptors")["single"].astype(np.float64)
    for t in [0.4, 0.5, 0.6]:
        a = dlib.chinese_whispers_clustering([dlib.vector(d.tolist()) for d in descs], t)
        b = fd.chinese_whispers_clustering([fd.vector(d) for d in descs], t)
        assert list(a) == b
    edges = [(0, 1), (1, 2, 0.5), (3, 4), (5, 5)]
    assert list(dlib.chinese_whispers(edges)) == fd.chinese_whispers(edges)


def test_unsupported_names_are_explained():
    with pytest.raises(
        AttributeError, match="does not implement dlib.train_simple_object_detector"
    ):
        fd.train_simple_object_detector  # noqa: B018
    cnn = fd.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
    with pytest.raises(NotImplementedError, match="CNN"):
        cnn(np.zeros((10, 10, 3), np.uint8))
