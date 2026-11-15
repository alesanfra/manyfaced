import pickle

import numpy as np
import pytest

import manyfaced
from manyfaced import _manyfaced


def test_rect():
    r = manyfaced.Rect(10, 20, 29, 59)
    assert (r.left, r.top, r.right, r.bottom) == (10, 20, 29, 59)
    assert (r.width, r.height, r.area) == (20, 40, 800)
    assert r.center() == (20, 40)
    assert r == manyfaced.Rect(10, 20, 29, 59)
    assert r != manyfaced.Rect(10, 20, 29, 60)
    assert len({r, manyfaced.Rect(10, 20, 29, 59)}) == 1
    assert repr(r) == "Rect(left=10, top=20, right=29, bottom=59)"
    assert pickle.loads(pickle.dumps(r)) == r
    assert manyfaced.Rect(5, 5, 4, 4).is_empty()
    assert manyfaced.Rect(5, 5, 4, 4).area == 0


def test_image_conversion_contiguous_and_views():
    rgb = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
    assert _manyfaced._image_info(rgb) == (4, 5, 3, int(rgb.sum()))
    gray = rgb[..., 0]  # non-contiguous view
    assert _manyfaced._image_info(gray) == (4, 5, 1, int(gray.sum()))
    flipped = rgb[::-1, ::2]
    assert _manyfaced._image_info(flipped) == (4, 3, 3, int(flipped.sum()))
    assert _manyfaced._image_info(np.asfortranarray(rgb[..., 1])) == (
        4,
        5,
        1,
        int(rgb[..., 1].sum()),
    )
    empty = np.zeros((0, 0), dtype=np.uint8)
    assert _manyfaced._image_info(empty) == (0, 0, 1, 0)


@pytest.mark.parametrize(
    ("value", "exc", "message"),
    [
        (np.zeros((4, 4), dtype=np.float32), TypeError, "uint8"),
        (np.zeros((4, 4, 4), dtype=np.uint8), ValueError, "HxWx3"),
        (np.zeros((4,), dtype=np.uint8), ValueError, "HxW"),
        ([[1, 2], [3, 4]], TypeError, "numpy uint8 array"),
    ],
)
def test_image_conversion_errors(value, exc, message):
    with pytest.raises(exc, match=message):
        _manyfaced._image_info(value)


def test_rect_arithmetic_does_not_wrap():
    full = manyfaced.Rect(-(2**63), 0, 2**63 - 1, 0)
    assert full.width == 2**64 - 1  # saturates: the true width 2**64 does not fit in u64
    assert manyfaced.Rect(0, 0, 2**33, 2**33).area == 2**64 - 1
    assert manyfaced.Rect(2**62, 0, 2**62, 0).center() == (2**62, 0)
    assert manyfaced.Rect(2**63 - 1, 0, 2**63 - 1, 0).center() == (2**63 - 1, 0)
