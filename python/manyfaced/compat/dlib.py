"""dlib-compatible API: same names, signatures and defaults as dlib's Python bindings, for the
functionality manyfaced implements.

Usage::

    import manyfaced.compat.dlib as dlib

or, to run existing code (e.g. ``face_recognition``) unchanged::

    import sys, manyfaced.compat.dlib
    sys.modules["dlib"] = manyfaced.compat.dlib

Behaviour follows dlib 20.0.1's ``tools/python/src`` bindings. Differences:

- Only 8-bit images (``uint8`` ``HxW`` or ``HxWx3``) are accepted where dlib also accepts other
  pixel types (``get_face_chip``, ``extract_image_chip``).
- ``point`` and ``dpoint`` compare by value (dlib's compare by identity).
- With ``num_jitters > 1`` descriptors use manyfaced's own random generator (deterministic per
  model instance), so they are not bit-identical to dlib's jittered descriptors.
- ``cnn_face_detection_model_v1`` exists only so that code which creates one at import time
  (``face_recognition``) keeps working; calling it raises ``NotImplementedError``.
- ``load_rgb_image`` decodes with Pillow instead of dlib's bundled decoders.
- The models cannot be pickled. Create them again in each process.
- Names that manyfaced does not implement raise ``AttributeError`` with an explanation.
"""

from __future__ import annotations

import itertools
import math
import os
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from .. import _chips, _cluster, _image, _manyfaced, _recognition

__version__ = _manyfaced.DLIB_REFERENCE_VERSION
DLIB_USE_CUDA = False

_UNSUPPORTED_IMAGE = "Unsupported image type, must be 8bit gray or RGB image."
_RGB_REQUIRED = "compute_face_descriptor(): incompatible function arguments (RGB image required)"


def _lround(v: float) -> int:
    """C ``std::lround``: round half away from zero."""
    return int(math.copysign(math.floor(abs(v) + 0.5), v))


def _fmt(v: float) -> str:
    """C++ ``ostream << double`` with the default precision (6 significant digits)."""
    return format(v, "g")


def _load(loader: Any, filename: Any) -> Any:
    """dlib reports unreadable and invalid model files as RuntimeError."""
    try:
        return loader(filename)
    except (OSError, ValueError) as e:
        raise RuntimeError(f"Unable to open {os.fspath(filename)}: {e}") from None


def _check_image(img: Any, rgb_only: bool = False) -> np.ndarray:
    """dlib's ``is_image<unsigned char>`` / ``is_image<rgb_pixel>`` checks."""
    ok = (
        isinstance(img, np.ndarray)
        and img.dtype == np.uint8
        and (img.ndim == 2 or (img.ndim == 3 and img.shape[2] == 3))
    )
    if not ok or (rgb_only and img.ndim != 3):
        raise RuntimeError(_UNSUPPORTED_IMAGE)
    return img


# --------------------------------------------------------------------------------------------
# Geometry


class point:
    """A point with integer coordinates (dlib's ``point``)."""

    __slots__ = ("x", "y")

    x: int
    y: int

    def __init__(self, x: Any, y: int | None = None) -> None:
        if y is not None:
            self.x, self.y = int(x), int(y)
        elif isinstance(x, dpoint):
            self.x, self.y = _lround_floor(x.x), _lround_floor(x.y)
        else:
            v = np.asarray(x).ravel()
            self.x, self.y = int(v[0]), int(v[1])

    def __repr__(self) -> str:
        return f"point({self.x}, {self.y})"

    def __str__(self) -> str:
        return f"({self.x}, {self.y})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, point) and (self.x, self.y) == (other.x, other.y)

    __hash__ = None

    def __add__(self, other: point) -> point:
        return point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: point) -> point:
        return point(self.x - other.x, self.y - other.y)

    def __mul__(self, s: float) -> dpoint:
        return dpoint(self.x * s, self.y * s)

    __rmul__ = __mul__

    def __truediv__(self, s: float) -> dpoint:
        return dpoint(self.x / s, self.y / s)

    def normalize(self) -> dpoint:
        n = math.hypot(self.x, self.y)
        return dpoint(self.x / n, self.y / n) if n else dpoint(0.0, 0.0)

    def __reduce__(self):
        return (point, (self.x, self.y))


def _lround_floor(v: float) -> int:
    """dlib's float -> integer vector conversion: ``floor(v + 0.5)``."""
    return int(math.floor(v + 0.5))


class dpoint:
    """A point with floating point coordinates (dlib's ``dpoint``)."""

    __slots__ = ("x", "y")

    x: float
    y: float

    def __init__(self, x: Any, y: float | None = None) -> None:
        if y is not None:
            self.x, self.y = float(x), float(y)
        elif isinstance(x, point):
            self.x, self.y = float(x.x), float(x.y)
        else:
            v = np.asarray(x, dtype=np.float64).ravel()
            self.x, self.y = float(v[0]), float(v[1])

    def __repr__(self) -> str:
        return f"dpoint({_fmt(self.x)}, {_fmt(self.y)})"

    def __str__(self) -> str:
        return f"({_fmt(self.x)}, {_fmt(self.y)})"

    def __eq__(self, other: object) -> bool:
        return isinstance(other, dpoint) and (self.x, self.y) == (other.x, other.y)

    __hash__ = None

    def __add__(self, other: dpoint) -> dpoint:
        return dpoint(self.x + other.x, self.y + other.y)

    def __sub__(self, other: dpoint) -> dpoint:
        return dpoint(self.x - other.x, self.y - other.y)

    def __mul__(self, s: float) -> dpoint:
        return dpoint(self.x * s, self.y * s)

    __rmul__ = __mul__

    def __truediv__(self, s: float) -> dpoint:
        return dpoint(self.x / s, self.y / s)

    def normalize(self) -> dpoint:
        n = math.hypot(self.x, self.y)
        return dpoint(self.x / n, self.y / n) if n else dpoint(0.0, 0.0)

    def __reduce__(self):
        return (dpoint, (self.x, self.y))


class _vector_list(list):
    """pybind11 ``bind_vector`` containers: a list with ``clear``, ``resize`` and ``extend``."""

    _item: type = object

    def __init__(self, arg: int | Iterable[Any] = ()) -> None:
        if isinstance(arg, int):
            super().__init__(self._item() for _ in range(arg))
        else:
            super().__init__(arg)

    def resize(self, n: int) -> None:
        if n < len(self):
            del self[n:]
        else:
            self.extend(self._item() for _ in range(n - len(self)))

    def __repr__(self) -> str:
        # pybind11's bind_vector prints the elements with their C++ operator<< (their str).
        return f"{type(self).__name__}[{', '.join(str(x) for x in self)}]"


def length(p: point | dpoint) -> float:
    return math.hypot(p.x, p.y)


def dot(a: Any, b: Any) -> float:
    if isinstance(a, (point, dpoint)):
        return a.x * b.x + a.y * b.y
    return float(np.dot(np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)))


class rectangle:
    """An integer rectangle with inclusive bounds (dlib's ``rectangle``)."""

    __slots__ = ("_l", "_t", "_r", "_b")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if kwargs:
            if set(kwargs) == {"rect"}:
                args = (kwargs["rect"],)
            else:
                args = (kwargs["left"], kwargs["top"], kwargs["right"], kwargs["bottom"])
        if not args:
            self._l, self._t, self._r, self._b = 0, 0, -1, -1
        elif len(args) == 1 and isinstance(args[0], rectangle):
            r = args[0]
            self._l, self._t, self._r, self._b = r._l, r._t, r._r, r._b
        elif len(args) == 1 and isinstance(args[0], drectangle):
            d = args[0]
            self._l, self._t = _lround(d.left()), _lround(d.top())
            self._r, self._b = _lround(d.right()), _lround(d.bottom())
        elif len(args) == 4:
            self._l, self._t, self._r, self._b = (int(v) for v in args)
        else:
            raise TypeError(
                "rectangle(): expected (left, top, right, bottom), (rect) or no arguments"
            )

    @classmethod
    def _from_native(cls, r: _manyfaced.Rect) -> rectangle:
        return cls(r.left, r.top, r.right, r.bottom)

    def _tuple(self) -> tuple[int, int, int, int]:
        return (self._l, self._t, self._r, self._b)

    def left(self) -> int:
        return self._l

    def top(self) -> int:
        return self._t

    def right(self) -> int:
        return self._r

    def bottom(self) -> int:
        return self._b

    def is_empty(self) -> bool:
        return self._t > self._b or self._l > self._r

    def width(self) -> int:
        return 0 if self.is_empty() else self._r - self._l + 1

    def height(self) -> int:
        return 0 if self.is_empty() else self._b - self._t + 1

    def area(self) -> int:
        return self.width() * self.height()

    def tl_corner(self) -> point:
        return point(self._l, self._t)

    def tr_corner(self) -> point:
        return point(self._r, self._t)

    def bl_corner(self) -> point:
        return point(self._l, self._b)

    def br_corner(self) -> point:
        return point(self._r, self._b)

    def center(self) -> point:
        x, y = self._l + self._r + 1, self._t + self._b + 1
        if x < 0:
            x -= 1
        if y < 0:
            y -= 1
        # C++ integer division truncates towards zero.
        return point(int(x / 2), int(y / 2))

    def dcenter(self) -> point:
        # The binding converts dlib's dcenter (a dpoint) to a point.
        return point(dpoint((self._l + self._r) / 2.0, (self._t + self._b) / 2.0))

    def contains(self, *args: Any, **kwargs: Any) -> bool:
        if "x" in kwargs or len(args) == 2:
            x, y = (kwargs["x"], kwargs["y"]) if kwargs else args
            p = point(x, y)
        else:
            p = args[0] if args else next(iter(kwargs.values()))
        if isinstance(p, rectangle):
            return self + p == self
        if isinstance(p, dpoint):
            p = point(p)
        return not (p.x < self._l or p.x > self._r or p.y < self._t or p.y > self._b)

    def intersect(self, rect: rectangle) -> rectangle:
        return rectangle(
            max(self._l, rect._l),
            max(self._t, rect._t),
            min(self._r, rect._r),
            min(self._b, rect._b),
        )

    def __add__(self, other: rectangle | point) -> rectangle:
        if isinstance(other, point):
            other = rectangle(other.x, other.y, other.x, other.y)
        if other.is_empty():
            return rectangle(self)
        if self.is_empty():
            return rectangle(other)
        return rectangle(
            min(self._l, other._l),
            min(self._t, other._t),
            max(self._r, other._r),
            max(self._b, other._b),
        )

    def __iadd__(self, other: rectangle | point) -> rectangle:
        r = self + other
        self._l, self._t, self._r, self._b = r._l, r._t, r._r, r._b
        return self

    def __eq__(self, other: object) -> bool:
        return isinstance(other, rectangle) and self._tuple() == other._tuple()

    def __ne__(self, other: object) -> bool:
        return not self == other

    __hash__ = None

    def __repr__(self) -> str:
        return f"rectangle({self._l},{self._t},{self._r},{self._b})"

    def __str__(self) -> str:
        return f"[({self._l}, {self._t}) ({self._r}, {self._b})]"

    def __reduce__(self):
        return (rectangle, self._tuple())


class drectangle:
    """A rectangle with floating point coordinates (dlib's ``drectangle``)."""

    __slots__ = ("_l", "_t", "_r", "_b")

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        if kwargs:
            if set(kwargs) == {"rect"}:
                args = (kwargs["rect"],)
            else:
                args = (kwargs["left"], kwargs["top"], kwargs["right"], kwargs["bottom"])
        if not args:
            self._l, self._t, self._r, self._b = 0.0, 0.0, -1.0, -1.0
        elif len(args) == 1 and isinstance(args[0], (rectangle, drectangle)):
            r = args[0]
            self._l, self._t = float(r.left()), float(r.top())
            self._r, self._b = float(r.right()), float(r.bottom())
        elif len(args) == 4:
            self._l, self._t, self._r, self._b = (float(v) for v in args)
        else:
            raise TypeError(
                "drectangle(): expected (left, top, right, bottom), (rect) or no arguments"
            )

    def _tuple(self) -> tuple[float, float, float, float]:
        return (self._l, self._t, self._r, self._b)

    def left(self) -> float:
        return self._l

    def top(self) -> float:
        return self._t

    def right(self) -> float:
        return self._r

    def bottom(self) -> float:
        return self._b

    def is_empty(self) -> bool:
        return self._t > self._b or self._l > self._r

    def width(self) -> float:
        return 0.0 if self.is_empty() else self._r - self._l + 1

    def height(self) -> float:
        return 0.0 if self.is_empty() else self._b - self._t + 1

    def area(self) -> float:
        return self.width() * self.height()

    def tl_corner(self) -> dpoint:
        return dpoint(self._l, self._t)

    def tr_corner(self) -> dpoint:
        return dpoint(self._r, self._t)

    def bl_corner(self) -> dpoint:
        return dpoint(self._l, self._b)

    def br_corner(self) -> dpoint:
        return dpoint(self._r, self._b)

    def center(self) -> point:
        return point(dpoint((self._l + self._r) / 2.0, (self._t + self._b) / 2.0))

    def dcenter(self) -> point:
        return self.center()

    def contains(self, *args: Any, **kwargs: Any) -> bool:
        if "x" in kwargs or len(args) == 2:
            x, y = (kwargs["x"], kwargs["y"]) if kwargs else args
            p: Any = dpoint(x, y)
        else:
            p = args[0] if args else next(iter(kwargs.values()))
        if isinstance(p, drectangle):
            return p.is_empty() or (
                p._l >= self._l and p._t >= self._t and p._r <= self._r and p._b <= self._b
            )
        return not (p.x < self._l or p.x > self._r or p.y < self._t or p.y > self._b)

    def intersect(self, rect: drectangle) -> drectangle:
        return drectangle(
            max(self._l, rect._l),
            max(self._t, rect._t),
            min(self._r, rect._r),
            min(self._b, rect._b),
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, drectangle) and self._tuple() == other._tuple()

    def __ne__(self, other: object) -> bool:
        return not self == other

    __hash__ = None

    def __repr__(self) -> str:
        return "drectangle(" + ",".join(_fmt(v) for v in self._tuple()) + ")"

    def __str__(self) -> str:
        left, top, right, bottom = (_fmt(v) for v in self._tuple())
        return f"[({left}, {top}) ({right}, {bottom})]"

    def __reduce__(self):
        return (drectangle, self._tuple())


class points(_vector_list):
    """An array of point objects."""

    _item = point


class dpoints(_vector_list):
    """An array of dpoint objects."""

    _item = dpoint


class rectangles(_vector_list):
    """An array of rectangle objects."""

    _item = rectangle


class rectangless(_vector_list):
    """An array of arrays of rectangle objects."""

    _item = rectangles


def _as_rect(r: Any) -> rectangle:
    if isinstance(r, rectangle):
        return r
    raise TypeError(f"expected a dlib.rectangle, got {type(r).__name__}")


# --------------------------------------------------------------------------------------------
# Vectors (face descriptors)


class vector:
    """A column vector of doubles (dlib's ``vector``); converts to numpy with ``np.array(v)``."""

    __slots__ = ("_v",)

    def __init__(self, arg: int | Iterable[float] | None = None) -> None:
        if arg is None:
            self._v = np.zeros(0)
        elif isinstance(arg, int):
            self._v = np.zeros(arg)
        else:
            self._v = np.array(
                list(arg) if not isinstance(arg, np.ndarray) else arg, dtype=np.float64
            )
            self._v = self._v.ravel().copy()

    @classmethod
    def _wrap(cls, arr: np.ndarray) -> vector:
        v = cls.__new__(cls)
        v._v = np.ascontiguousarray(arr, dtype=np.float64)
        return v

    def __array__(self, dtype: Any = None, copy: Any = None) -> np.ndarray:
        arr = self._v if dtype is None else self._v.astype(dtype)
        return arr.copy() if copy and arr is self._v else arr

    def __len__(self) -> int:
        return len(self._v)

    def __getitem__(self, i: Any) -> Any:
        if isinstance(i, slice):
            return vector._wrap(self._v[i])
        return float(self._v[i])

    def __setitem__(self, i: int, value: float) -> None:
        self._v[i] = value

    def __iter__(self):
        return (float(x) for x in self._v)

    def set_size(self, n: int) -> None:
        self._v = np.zeros(n)

    resize = set_size

    @property
    def shape(self) -> tuple[int, int]:
        return (len(self._v), 1)

    def __repr__(self) -> str:
        return "dlib.vector([" + ", ".join(_fmt(x) for x in self._v) + "])"

    def __str__(self) -> str:
        return "\n".join(_fmt(x) for x in self._v)

    def __reduce__(self):
        return (vector, (self._v.tolist(),))


class vectors(_vector_list):
    """An array of vector objects."""

    _item = vector


class vectorss(_vector_list):
    """An array of arrays of vector objects."""

    _item = vectors


# --------------------------------------------------------------------------------------------
# Landmarks


class full_object_detection:
    """An object's bounding box and the positions of its parts (dlib's
    ``full_object_detection``)."""

    __slots__ = ("_rect", "_parts")

    def __init__(self, rect: rectangle, parts: Iterable[point]) -> None:
        self._rect = rectangle(_as_rect(rect))
        self._parts = [p if isinstance(p, point) else point(p) for p in parts]

    @classmethod
    def _from_native(cls, rect: rectangle, xy: np.ndarray) -> full_object_detection:
        d = cls.__new__(cls)
        d._rect = rect
        d._parts = [point(int(x), int(y)) for x, y in xy]
        return d

    @property
    def rect(self) -> rectangle:
        return self._rect

    @property
    def num_parts(self) -> int:
        return len(self._parts)

    def part(self, idx: int) -> point:
        if not 0 <= idx < len(self._parts):
            raise IndexError("Index out of range")
        return self._parts[idx]

    def parts(self) -> points:
        return points(self._parts)

    def _landmarks(self) -> np.ndarray:
        return np.array([(p.x, p.y) for p in self._parts], dtype=np.float64).reshape(-1, 2)

    def __reduce__(self):
        return (full_object_detection, (self._rect, list(self._parts)))


class full_object_detections(_vector_list):
    """An array of full_object_detection objects."""


class shape_predictor:
    """dlib's ``shape_predictor``: ``shape_predictor(path)``, then ``sp(image, box)``."""

    def __init__(self, filename: str | os.PathLike[str] | None = None) -> None:
        self._native = None if filename is None else _load(_manyfaced.ShapePredictor, filename)

    def __call__(self, image: np.ndarray, box: rectangle) -> full_object_detection:
        box = _as_rect(box)
        _check_image(image)
        if self._native is None:
            return full_object_detection(box, [])
        xy = self._native(image, box._tuple())
        return full_object_detection._from_native(rectangle(box), xy)


# --------------------------------------------------------------------------------------------
# Detection


class fhog_object_detector:
    """A HOG sliding-window detector (dlib's ``fhog_object_detector``)."""

    def __init__(self, filename: str | os.PathLike[str]) -> None:
        self._native = _load(_manyfaced.FhogDetector, filename)

    @classmethod
    def _wrap(cls, native: Any) -> fhog_object_detector:
        d = cls.__new__(cls)
        d._native = native
        return d

    def __call__(self, image: np.ndarray, upsample_num_times: int = 0) -> rectangles:
        _check_image(image)
        return rectangles(
            rectangle._from_native(r) for r in self._native(image, upsample_num_times)
        )

    def run(
        self, image: np.ndarray, upsample_num_times: int = 0, adjust_threshold: float = 0.0
    ) -> tuple[rectangles, list[float], list[int]]:
        _check_image(image)
        rects, scores, idx = self._native.run(image, upsample_num_times, adjust_threshold)
        return (rectangles(rectangle._from_native(r) for r in rects), list(scores), list(idx))

    @property
    def detection_window_width(self) -> int:
        return self._native.detection_window_width

    @property
    def detection_window_height(self) -> int:
        return self._native.detection_window_height

    @property
    def num_detectors(self) -> int:
        return self._native.num_detectors


def get_frontal_face_detector() -> fhog_object_detector:
    """The HOG frontal face detector embedded in dlib."""
    return fhog_object_detector._wrap(_manyfaced.FhogDetector())


class cnn_face_detection_model_v1:
    """Placeholder for dlib's CNN (MMOD) face detector, which manyfaced does not implement.

    It can be created (``face_recognition`` creates one at import time) but not used.
    """

    def __init__(self, filename: str | os.PathLike[str] | None = None) -> None:
        self._filename = filename

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError(
            "manyfaced does not implement dlib's CNN face detector (cnn_face_detection_model_v1); "
            "use get_frontal_face_detector() (HOG), e.g. face_recognition's model='hog'"
        )


# --------------------------------------------------------------------------------------------
# Face chips


class chip_dims:
    """``chip_dims(rows, cols)``."""

    def __init__(self, rows: int, cols: int) -> None:
        self.rows, self.cols = int(rows), int(cols)

    def __repr__(self) -> str:
        return f"chip_dims(rows={self.rows}, cols={self.cols})"

    def __str__(self) -> str:
        return f"rows={self.rows}, cols={self.cols}"


class chip_details:
    """Where to extract an image chip: a rectangle, a rotation angle and an output size."""

    def __init__(
        self, rect: Any = None, size_or_dims: Any = None, angle: float | None = None, **kwargs: Any
    ) -> None:
        rect = kwargs.pop("rect", rect)
        size_or_dims = kwargs.pop("size", kwargs.pop("dims", size_or_dims))
        if angle is None:
            angle = kwargs.pop("angle", 0.0)
        if kwargs:
            raise TypeError(f"chip_details(): unexpected arguments {sorted(kwargs)}")
        self.angle = float(angle)
        if rect is None:
            self.rect, self.rows, self.cols = drectangle(), 0, 0
            return
        is_int_rect = isinstance(rect, rectangle)
        self.rect = drectangle(rect)
        if size_or_dims is None:
            if is_int_rect:
                self.rows, self.cols = rect.height(), rect.width()
            else:
                self.rows, self.cols = _lround(self.rect.height()), _lround(self.rect.width())
        elif isinstance(size_or_dims, chip_dims):
            self.rows, self.cols = size_or_dims.rows, size_or_dims.cols
        else:
            self._dims_from_size(int(size_or_dims))

    def _dims_from_size(self, size: int) -> None:
        # chip_details::compute_dims_from_size (interpolation.h).
        if self.rect.is_empty():
            self.rows = self.cols = round(math.sqrt(size))
        else:
            relative = math.sqrt(size / self.rect.area())
            rows = int(self.rect.height() * relative + 0.5)
            cols = int(size / rows + 0.5) if rows else 0
            self.rows, self.cols = max(1, rows), max(1, cols)

    @classmethod
    def _from_native(cls, d: _manyfaced.ChipDetails) -> chip_details:
        c = cls.__new__(cls)
        c.rect = drectangle(*d.rect)
        c.angle, c.rows, c.cols = d.angle, d.rows, d.cols
        return c

    def _native(self) -> _manyfaced.ChipDetails:
        return _manyfaced.ChipDetails(self.rect._tuple(), self.rows, self.cols, self.angle)

    def __repr__(self) -> str:
        r = ",".join(_fmt(v) for v in self.rect._tuple())
        return (
            f"chip_details(rect=drectangle({r}), angle={_fmt(self.angle)}, "
            f"dims=chip_dims(rows={self.rows}, cols={self.cols}))"
        )

    def __str__(self) -> str:
        return f"rect={self.rect}, angle={_fmt(self.angle)}, rows={self.rows}, cols={self.cols}"


def _face_landmarks(det: Any) -> np.ndarray:
    if not isinstance(det, full_object_detection):
        raise TypeError(f"expected a dlib.full_object_detection, got {type(det).__name__}")
    return det._landmarks()


def _chip_errors(fn: Any) -> Any:
    """dlib asserts (RuntimeError) for wrong part counts and invalid sizes."""
    try:
        return fn()
    except ValueError as e:
        raise RuntimeError(str(e)) from None


def get_face_chip_details(
    det: full_object_detection | Sequence[full_object_detection],
    size: int = 200,
    padding: float = 0.2,
) -> chip_details | list[chip_details]:
    """``get_face_chip_details(det, size=200, padding=0.2)``, also for a list of detections."""

    def one(d: full_object_detection) -> chip_details:
        native = _chip_errors(lambda: _chips.face_chip_details(_face_landmarks(d), size, padding))
        return chip_details._from_native(native)

    if isinstance(det, full_object_detection):
        return one(det)
    return [one(d) for d in det]


def _float32(v: float) -> float:
    # get_face_chip / get_face_chips take padding as a C++ float.
    return float(np.float32(v))


def get_face_chip(
    img: np.ndarray, face: full_object_detection, size: int = 150, padding: float = 0.25
) -> np.ndarray:
    """The face as an upright ``size`` x ``size`` chip (``dlib.get_face_chip``)."""
    _check_image(img)
    return _chip_errors(
        lambda: _chips.face_chip(img, _face_landmarks(face), size, _float32(padding))
    )


def get_face_chips(
    img: np.ndarray, faces: Sequence[full_object_detection], size: int = 150, padding: float = 0.25
) -> list[np.ndarray]:
    """The faces as upright chips, extracted together (``dlib.get_face_chips``)."""
    _check_image(img)
    lms = [_face_landmarks(f) for f in faces]
    return _chip_errors(lambda: _chips.face_chips(img, lms, size, _float32(padding)))


def extract_image_chip(img: np.ndarray, chip_location: chip_details) -> np.ndarray:
    _check_image(img)
    return _chip_errors(lambda: _chips.extract_chip(img, chip_location._native()))


def extract_image_chips(
    img: np.ndarray, chip_locations: Sequence[chip_details]
) -> list[np.ndarray]:
    _check_image(img)
    return _chip_errors(lambda: _chips.extract_chips(img, [c._native() for c in chip_locations]))


# --------------------------------------------------------------------------------------------
# Face recognition


def _is_rgb(img: Any) -> bool:
    return (
        isinstance(img, np.ndarray)
        and img.dtype == np.uint8
        and img.ndim == 3
        and img.shape[2] == 3
    )


class face_recognition_model_v1:
    """dlib's face recognition ResNet (``dlib_face_recognition_resnet_model_v1.dat``)."""

    def __init__(self, filename: str | os.PathLike[str]) -> None:
        self._native = _load(_recognition.FaceRecognizer, filename)
        # dlib's model holds a stateful random generator for jittering. To copy that behaviour
        # (deterministically), each jittered call uses the next seed. next() on a count is
        # atomic, so two threads never get the same seed.
        self._jitter_seeds = itertools.count(1)

    def _jitters(self, num_jitters: int) -> tuple[int, int]:
        if num_jitters <= 1:
            return 0, 0
        return int(num_jitters), next(self._jitter_seeds)

    def compute_face_descriptor(self, *args: Any, **kwargs: Any) -> Any:
        """The overloads of dlib's ``compute_face_descriptor``:

        - ``(img, face, num_jitters=0, padding=0.25)`` -> ``vector``
        - ``(img, num_jitters=0)`` with ``img`` an aligned 150x150 chip -> ``vector``
        - ``(img, faces, num_jitters=0, padding=0.25)`` -> ``vectors``
        - ``(batch_img, batch_faces, num_jitters=0, padding=0.25)`` -> ``vectorss``
        - ``(batch_img, num_jitters=0)`` with aligned chips -> ``vectors``

        Images must be RGB (``HxWx3`` ``uint8``), like dlib.
        """
        names = ["img", "face", "num_jitters", "padding"]
        if "batch_img" in kwargs:
            kwargs["img"] = kwargs.pop("batch_img")
        for k in ("faces", "batch_faces"):
            if k in kwargs:
                kwargs["face"] = kwargs.pop(k)
        # Positional (img, num_jitters) for the aligned-chip overloads.
        if (
            len(args) >= 2
            and isinstance(args[1], (int, np.integer))
            and not isinstance(args[1], bool)
        ):
            args = (args[0], None, *args[1:])
        bound: dict[str, Any] = dict(zip(names, args, strict=False))
        overlap = set(bound) & set(kwargs)
        if overlap or len(args) > len(names) or set(kwargs) - set(names):
            raise TypeError("compute_face_descriptor(): incompatible function arguments")
        bound.update(kwargs)
        img = bound.get("img")
        face = bound.get("face")
        if img is None:
            raise TypeError("compute_face_descriptor(): incompatible function arguments")
        num_jitters = int(bound.get("num_jitters", 0))
        padding = float(bound.get("padding", 0.25))

        if face is None:
            if isinstance(img, np.ndarray):
                return self._chip_descriptors([img], num_jitters)[0]
            return vectors(self._chip_descriptors(list(img), num_jitters))
        if isinstance(img, np.ndarray):
            if isinstance(face, full_object_detection):
                return self._descriptors(img, [face], num_jitters, padding)[0]
            return vectors(self._descriptors(img, list(face), num_jitters, padding))
        imgs, faces = list(img), list(face)
        if len(imgs) != len(faces):
            raise RuntimeError(
                "The length of the input image list must match the length of the faces list."
            )
        return vectorss(
            vectors(self._descriptors(i, list(f), num_jitters, padding))
            for i, f in zip(imgs, faces, strict=True)
        )

    def _descriptors(
        self, img: Any, faces: list[full_object_detection], num_jitters: int, padding: float
    ) -> list[vector]:
        if not _is_rgb(img):
            raise TypeError(_RGB_REQUIRED)
        if not faces:
            return []
        lms = [_face_landmarks(f) for f in faces]
        for lm in lms:
            if len(lm) not in (5, 68):
                raise RuntimeError(
                    "The full_object_detection must use the iBUG 300W 68 point face landmark "
                    "style or dlib's 5 point style."
                )
        n, seed = self._jitters(num_jitters)
        out = self._native.embed_many(img, lms, num_jitters=n, padding=padding, seed=seed)
        return [vector._wrap(row) for row in out]

    def _chip_descriptors(self, chips: list[Any], num_jitters: int) -> list[vector]:
        for c in chips:
            if not _is_rgb(c):
                raise TypeError(_RGB_REQUIRED)
            if c.shape[:2] != (150, 150):
                raise RuntimeError(
                    "Unsupported image size, it should be of size 150x150. Also cropping must be "
                    "done as `dlib.get_face_chip` would do it. That is, centered and scaled "
                    "essentially the same way."
                )
        if not chips:
            return []
        n, seed = self._jitters(num_jitters)
        out = self._native.embed_chips(chips, num_jitters=n, seed=seed)
        return [vector._wrap(row) for row in out]


# --------------------------------------------------------------------------------------------
# Clustering


def chinese_whispers_clustering(descriptors: Sequence[Any], threshold: float) -> list[int]:
    """Labels for ``descriptors`` (dlib's ``chinese_whispers_clustering``)."""
    if not threshold > 0:
        raise RuntimeError("threshold must be > 0")
    if len(descriptors) == 0:
        return []
    arr = np.array([np.asarray(d, dtype=np.float64).ravel() for d in descriptors])
    # The native function, not _cluster's: dlib accepts descriptors that are not finite.
    return _manyfaced._chinese_whispers_clustering(np.ascontiguousarray(arr), threshold)


def chinese_whispers(edges: Sequence[Any]) -> list[int]:
    """dlib's raw ``chinese_whispers(edges)``: ``(i, j)`` or ``(i, j, distance)`` tuples."""
    for e in edges:
        if len(e) not in (2, 3):
            raise IndexError("Input must be a list of tuples with 2 or 3 elements.")
    return _cluster.chinese_whispers([tuple(e) for e in edges]).tolist()


# --------------------------------------------------------------------------------------------
# Images


def load_rgb_image(filename: str | os.PathLike[str]) -> np.ndarray:
    """Load an image as an ``HxWx3`` ``uint8`` RGB array.

    Pillow decodes the image. dlib uses its own bundled decoders, so the JPEG pixels can be
    slightly different. As in dlib, the EXIF orientation is ignored.
    """
    return _image.load_rgb(filename, exif_transpose=False).copy()


def __getattr__(name: str) -> Any:
    raise AttributeError(
        f"manyfaced.compat.dlib does not implement dlib.{name} (manyfaced covers detection, "
        "landmarks, face chips, face descriptors and chinese whispers; see the README)"
    )
