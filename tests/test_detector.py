"""HOG frontal face detector: parity with dlib (tests/golden/detections.npz) and API behaviour."""

from __future__ import annotations

import threading

import numpy as np
import pytest

import manyfaced
import testimages
from manyfaced import FhogDetector

EMBEDDED_DAT = testimages.ROOT / "crates" / "manyfaced-core" / "data" / "frontal_face_detector.dat"


def _cases(g):
    for k in range(len(g["case_image"])):
        s, n = int(g["case_start"][k]), int(g["case_count"][k])
        yield (
            k,
            str(g["case_image"][k]),
            int(g["case_upsample"][k]),
            float(g["case_adjust"][k]),
            g["rects"][s : s + n],
            g["scores"][s : s + n],
            g["weight_indices"][s : s + n],
        )


@pytest.mark.parity
def test_detections_match_dlib(images, golden):
    g = golden("detections")
    det = FhogDetector()
    worst = 0.0
    failures = []
    for k, image_id, up, adj, rects, scores, idx in _cases(g):
        # The original arrays, including the non-contiguous views.
        got_rects, got_scores, got_idx = det.run(images[image_id].array, up, adj)
        got = np.array([r.to_tuple() for r in got_rects], dtype=np.int64).reshape(-1, 4)
        if got.shape != rects.shape or not np.array_equal(got, rects):
            failures.append((k, image_id, up, adj, "rects", got.tolist(), rects.tolist()))
            continue
        if list(got_idx) != idx.tolist():
            failures.append((k, image_id, up, adj, "filter indices", got_idx, idx.tolist()))
            continue
        if len(scores):
            worst = max(worst, float(np.abs(np.array(got_scores) - scores).max()))
        if adj == 0.0:
            plain = det(images[image_id].array, up)
            assert [r.to_tuple() for r in plain] == [tuple(r) for r in rects.tolist()]
    assert not failures, failures[:3]
    # Brief: identical rectangles and filter indices, scores within 1e-4.
    assert worst < 1e-4, worst


@pytest.mark.parity
def test_detector_is_thread_safe(images):
    det = FhogDetector()
    img = images["faces/bald_guys.jpg"].array
    expected = det.run(img, 0, -0.5)
    results = [None] * 8

    def work(i):
        results[i] = det.run(img, 0, -0.5)

    threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert all(r == expected for r in results)
    assert len(expected[0]) >= 20


def test_frontal_face_detector_properties():
    det = FhogDetector()
    assert det.num_detectors == 5
    assert (det.detection_window_width, det.detection_window_height) == (80, 80)
    assert "frontal_face" in repr(det)


def test_load_the_embedded_detector_file():
    det = FhogDetector(EMBEDDED_DAT)
    assert det.num_detectors == 5
    img = (np.arange(200 * 180, dtype=np.int64) * 7919 % 251).astype(np.uint8).reshape(200, 180)
    assert det.run(img, 1, -1.0) == FhogDetector().run(img, 1, -1.0)


def test_load_simple_object_detector_svm(tmp_path):
    # dlib.train_simple_object_detector appends `int version, unsigned upsampling_amount`
    # after the object_detector; the extra bytes are ignored like in dlib's loader.
    svm = tmp_path / "detector.svm"
    svm.write_bytes(EMBEDDED_DAT.read_bytes() + bytes([0x01, 0x01, 0x01, 0x00]))
    assert FhogDetector(svm).num_detectors == 5


def test_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        FhogDetector(tmp_path / "missing.svm")
    bad = tmp_path / "bad.svm"
    bad.write_bytes(EMBEDDED_DAT.read_bytes()[:1000])
    with pytest.raises(manyfaced.ModelFormatError, match="invalid dlib object_detector"):
        FhogDetector(bad)


@pytest.mark.parametrize(
    "shape",
    [(0, 0), (1, 1), (5, 7), (40, 30), (0, 0, 3), (1, 1, 3), (9, 9, 3), (63, 70, 3)],
)
def test_small_and_empty_images(shape):
    det = FhogDetector()
    img = np.full(shape, 128, dtype=np.uint8)
    for up in (0, 1, 2):
        rects, scores, idx = det.run(img, up, -10.0)
        assert len(rects) == len(scores) == len(idx)
        assert det(img, up) == [r for r, s in zip(rects, scores, strict=True) if s >= 0]


def test_bad_arguments():
    det = FhogDetector()
    with pytest.raises(TypeError, match="uint8"):
        det(np.zeros((10, 10), dtype=np.float32))
    with pytest.raises(ValueError, match="HxWx3"):
        det(np.zeros((10, 10, 4), dtype=np.uint8))
    with pytest.raises(ValueError, match="upsample"):
        det(np.zeros((10, 10), dtype=np.uint8), -1)
    with pytest.raises(ValueError, match="too much memory"):
        det.run(np.zeros((100, 100), dtype=np.uint8), 30)


def _dlib_int(value: int) -> bytes:
    """One integer in dlib's serialization format."""
    if value == 0:
        return bytes([1, 0])
    raw = abs(value).to_bytes((abs(value).bit_length() + 7) // 8, "little")
    return bytes([len(raw) | (0x80 if value < 0 else 0)]) + raw


def _detector_with(**fields: int) -> bytes:
    """The embedded detector with some of its first scanner fields replaced."""
    names = ["version", "scanner_version", "max_size", "size", "cell_size", "padding"]
    names += ["window_width", "window_height", "max_levels", "min_width", "min_height"]
    data, pos, out = EMBEDDED_DAT.read_bytes(), 0, b""
    for name in names:
        n = data[pos] & 0x0F
        value = int.from_bytes(data[pos + 1 : pos + 1 + n], "little")
        value = -value if data[pos] & 0x80 else value
        out += _dlib_int(fields.get(name, value))
        pos += 1 + n
    return out + data[pos:]


def test_hostile_detector_files(tmp_path):
    path = tmp_path / "hostile.svm"
    # dlib does not accept a minimum pyramid layer size of 0.
    path.write_bytes(_detector_with(min_width=0, min_height=0))
    with pytest.raises(manyfaced.ModelFormatError, match="minimum pyramid layer size"):
        FhogDetector(path)
    # 1x1 levels map to themselves: the pyramid must stop, not run 2^64 - 1 levels.
    path.write_bytes(_detector_with(min_width=1, min_height=1, max_levels=2**64 - 1))
    assert FhogDetector(path)(np.zeros((100, 100), np.uint8)) == []
    # A window so large that its FHOG size would overflow.
    path.write_bytes(_detector_with(padding=2**29, window_height=2**30))
    with pytest.raises(manyfaced.ModelFormatError, match="detection window"):
        FhogDetector(path)


def test_huge_broadcast_view_is_a_memory_error():
    view = np.broadcast_to(np.zeros(3, np.uint8), (10**7, 10**7, 3))
    with pytest.raises(MemoryError):
        FhogDetector()(view)
