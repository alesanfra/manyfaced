//! `manyfaced.Rect` and conversions from Python rectangle-like objects.

use manyfaced_core::geometry::Rect;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;

/// An integer rectangle with inclusive bounds, as dlib's `rectangle`.
#[pyclass(module = "manyfaced", name = "Rect", frozen, eq, hash)]
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
pub struct PyRect(pub Rect);

#[pymethods]
impl PyRect {
    #[new]
    fn new(left: i64, top: i64, right: i64, bottom: i64) -> Self {
        PyRect(Rect::new(left, top, right, bottom))
    }

    #[getter]
    fn left(&self) -> i64 {
        self.0.left
    }

    #[getter]
    fn top(&self) -> i64 {
        self.0.top
    }

    #[getter]
    fn right(&self) -> i64 {
        self.0.right
    }

    #[getter]
    fn bottom(&self) -> i64 {
        self.0.bottom
    }

    /// `right - left + 1`, or 0 if the rectangle is empty.
    #[getter]
    fn width(&self) -> u64 {
        self.0.width()
    }

    /// `bottom - top + 1`, or 0 if the rectangle is empty.
    #[getter]
    fn height(&self) -> u64 {
        self.0.height()
    }

    #[getter]
    fn area(&self) -> u64 {
        self.0.area()
    }

    fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// The integer centre `(x, y)`, as dlib's `center(rect)`.
    fn center(&self) -> (i64, i64) {
        let c = self.0.center();
        (c.x, c.y)
    }

    /// `(left, top, right, bottom)`.
    #[pyo3(name = "to_tuple")]
    fn as_tuple(&self) -> (i64, i64, i64, i64) {
        (self.0.left, self.0.top, self.0.right, self.0.bottom)
    }

    fn __repr__(&self) -> String {
        let r = self.0;
        format!(
            "Rect(left={}, top={}, right={}, bottom={})",
            r.left, r.top, r.right, r.bottom
        )
    }

    fn __getnewargs__(&self) -> (i64, i64, i64, i64) {
        self.as_tuple()
    }
}

/// Accepts a `manyfaced.Rect` or a `(left, top, right, bottom)` tuple of integers.
pub fn extract_rect(obj: &Bound<'_, PyAny>) -> PyResult<Rect> {
    if let Ok(r) = obj.cast::<PyRect>() {
        return Ok(r.get().0);
    }
    obj.extract::<(i64, i64, i64, i64)>()
        .map(|(l, t, r, b)| Rect::new(l, t, r, b))
        .map_err(|_| {
            PyTypeError::new_err(
                "expected a manyfaced.Rect or a (left, top, right, bottom) tuple of integers",
            )
        })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyRect>()
}
