"""Face chips: parity of chip details and pixels with dlib, plus API and edge cases."""

from __future__ import annotations

import numpy as np
import pytest

import manyfaced
import testimages

LOCAL_CHIPS = testimages.GOLDEN_DIR / "local" / "face_chips.npz"


def _landmarks(g, i):
    return g["lm_points"][i, : g["lm_num_parts"][i]]


@pytest.fixture(scope="module")
def chip_golden(golden):
    return golden("face_chip_details")


@pytest.fixture(scope="module")
def local_chips(chip_golden):
    if not LOCAL_CHIPS.exists():
        pytest.fail(
            f"{LOCAL_CHIPS} not found: chip pixels are not committed (image licenses); "
            "generate them with `python tools/gen_golden.py chips` (needs dlib-bin)"
        )
    with np.load(LOCAL_CHIPS) as data:
        chips = {k: data[k] for k in data.files}
    if not np.array_equal(chips["rec_chip_digest"], chip_golden["rec_chip_digest"]):
        pytest.fail(
            f"{LOCAL_CHIPS} is stale: regenerate it with `python tools/gen_golden.py chips`"
        )
    return chips


@pytest.mark.parity
def test_chip_details_match_dlib(chip_golden):
    g = chip_golden
    worst_rect = worst_angle = 0.0
    for k in range(len(g["rec_size"])):
        lm = _landmarks(g, g["rec_landmarks"][k])
        d = manyfaced.face_chip_details(lm, int(g["rec_size"][k]), float(g["rec_padding"][k]))
        expected = g["rec_rect"][k]
        rel = np.abs(np.array(d.rect) - expected) / np.maximum(1.0, np.abs(expected))
        worst_rect = max(worst_rect, float(rel.max()))
        worst_angle = max(worst_angle, abs(d.angle - g["rec_angle"][k]))
        assert (d.rows, d.cols) == (g["rec_rows"][k], g["rec_cols"][k]), k
    assert worst_rect < 1e-9, f"max relative rect difference {worst_rect:.3g}"
    assert worst_angle < 1e-9, f"max angle difference {worst_angle:.3g}"


def _manyfaced_chips(g, images):
    """Chip k for every golden record, extracting batches together like dlib.get_face_chips."""
    out = {}
    n = len(g["rec_size"])
    batch = g["rec_batch"]
    for k in range(n):
        if batch[k] >= 0:
            continue
        i = g["rec_landmarks"][k]
        img = images[str(g["lm_image"][i])].array
        out[k] = manyfaced.face_chip(
            img, _landmarks(g, i), int(g["rec_size"][k]), float(g["rec_padding"][k])
        )
    for b in np.unique(batch[batch >= 0]):
        ks = np.flatnonzero(batch == b)
        i0 = g["rec_landmarks"][ks[0]]
        img = images[str(g["lm_image"][i0])].array
        lms = [_landmarks(g, g["rec_landmarks"][k]) for k in ks]
        for k, chip in zip(ks, manyfaced.face_chips(img, lms, 150, 0.25), strict=True):
            out[int(k)] = chip
    return out


@pytest.mark.parity
def test_chips_match_dlib(chip_golden, local_chips, images):
    g = chip_golden
    ours = _manyfaced_chips(g, images)
    worst, differing, total = 0, 0, 0
    for k, chip in ours.items():
        expected = local_chips[f"chip_{k:05d}"]
        assert chip.shape == expected.shape and chip.dtype == np.uint8, k
        diff = np.abs(chip.astype(np.int16) - expected.astype(np.int16))
        worst = max(worst, int(diff.max(initial=0)))
        differing += int(np.count_nonzero(diff))
        total += diff.size
    assert len(ours) == len(g["rec_size"])
    assert worst <= 1, f"max pixel difference {worst} ({differing}/{total} values differ)"


def test_wrong_landmark_count():
    img = np.zeros((50, 50, 3), np.uint8)
    with pytest.raises(ValueError, match="5 or 68 landmarks"):
        manyfaced.face_chip(img, np.zeros((6, 2)))
    with pytest.raises(ValueError, match=r"shape \(N, 2\)"):
        manyfaced.face_chip_details(np.zeros((5, 3)))


def test_huge_chips_are_rejected():
    img = np.zeros((20, 20, 3), np.uint8)
    with pytest.raises(ValueError, match="too large"):
        manyfaced.face_chip(img, [[12, 8], [10, 8], [4, 8], [6, 8], [8, 12]], size=2**30)
    with pytest.raises(ValueError, match="too large"):
        manyfaced.extract_chip(img, manyfaced.ChipDetails((0, 0, 9, 9), 2**30, 2**30, 0.1))


def test_invalid_size_and_padding():
    lm = [[60, 40], [50, 41], [20, 40], [30, 41], [40, 60]]
    with pytest.raises(ValueError, match="size"):
        manyfaced.face_chip_details(lm, size=0)
    with pytest.raises(ValueError, match="padding"):
        manyfaced.face_chip_details(lm, padding=-0.5)


def test_bad_image():
    lm = [[60, 40], [50, 41], [20, 40], [30, 41], [40, 60]]
    with pytest.raises(TypeError, match="uint8"):
        manyfaced.face_chip(np.zeros((50, 50), np.float32), lm)
    with pytest.raises(ValueError, match="HxW"):
        manyfaced.face_chip(np.zeros((50, 50, 4), np.uint8), lm)


def test_shapes_and_edge_cases():
    lm = [[60, 40], [50, 41], [20, 40], [30, 41], [40, 60]]
    rgb = np.full((80, 80, 3), 200, np.uint8)
    chip = manyfaced.face_chip(rgb, lm, size=64)
    assert chip.shape == (64, 64, 3) and chip.dtype == np.uint8
    gray = manyfaced.face_chip(rgb[..., 0], lm, size=64)
    assert gray.shape == (64, 64)
    # An empty image or a face far outside the image gives a black chip.
    empty = manyfaced.face_chip(np.zeros((0, 0, 3), np.uint8), lm, size=20)
    assert empty.shape == (20, 20, 3) and not empty.any()
    far = np.array(lm) + 10_000
    assert not manyfaced.face_chip(rgb, far, size=20).any()
    chips = manyfaced.face_chips(rgb, [lm, lm], size=32)
    assert len(chips) == 2 and chips[0].shape == (32, 32, 3)
    assert manyfaced.face_chips(rgb, []) == []


def test_chip_details_object():
    d = manyfaced.ChipDetails((0.0, 0.0, 9.0, 9.0), rows=10, cols=10)
    assert (d.rows, d.cols, d.angle, d.rect) == (10, 10, 0.0, (0.0, 0.0, 9.0, 9.0))
    assert "ChipDetails(rect=" in repr(d)
    img = np.arange(100, dtype=np.uint8).reshape(10, 10)
    # No rotation and no scaling: a plain copy.
    assert np.array_equal(manyfaced.extract_chip(img, d), img)
    assert len(manyfaced.extract_chips(img, [d, d])) == 2


@pytest.mark.parametrize("rect", [(0.0, 0.0, float("inf"), 10.0), (float("nan"), 0.0, 9.0, 9.0)])
def test_non_finite_chip_details_are_rejected(rect):
    with pytest.raises(ValueError, match="finite"):
        manyfaced.ChipDetails(rect, 10, 10)
    with pytest.raises(ValueError, match="finite"):
        manyfaced.ChipDetails((0.0, 0.0, 9.0, 9.0), 10, 10, angle=float("nan"))
