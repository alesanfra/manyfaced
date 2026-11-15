//! Bindings for `manyfaced_core::face::ShapePredictor`.

use std::path::PathBuf;

use manyfaced_core::face::ShapePredictor;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use pyo3::prelude::*;

use crate::errors::to_py_err;
use crate::geometry::extract_rect;
use crate::image::{AnyImage, with_image};

/// dlib's `shape_predictor` (Ensemble of Regression Trees landmark predictor).
///
/// ``ShapePredictor()`` uses the 5-point model in the manyfaced wheel. ``ShapePredictor(path)``
/// loads an original dlib ``.dat`` file, for example ``shape_predictor_68_face_landmarks.dat``.
/// Many threads can use the same instance.
#[pyclass(module = "manyfaced", name = "ShapePredictor", frozen)]
pub struct PyShapePredictor {
    inner: ShapePredictor,
    path: PathBuf,
}

#[pymethods]
impl PyShapePredictor {
    /// Load a dlib ``shape_predictor`` ``.dat`` file. Without ``path``, load the 5-point model
    /// in the manyfaced wheel (``manyfaced.model_path("shape_predictor_5_face_landmarks")``).
    #[new]
    #[pyo3(signature = (path = None))]
    fn new(py: Python<'_>, path: Option<PathBuf>) -> PyResult<Self> {
        let path = match path {
            Some(p) => p,
            None => py
                .import("manyfaced._model_files")?
                .call_method1("model_path", ("shape_predictor_5_face_landmarks",))?
                .extract::<PathBuf>()?,
        };
        let inner = py
            .detach(|| ShapePredictor::load(&path))
            .map_err(to_py_err)?;
        Ok(PyShapePredictor { inner, path })
    }

    /// The number of landmarks for each object.
    #[getter]
    fn num_parts(&self) -> usize {
        self.inner.num_parts()
    }

    /// Find the landmarks of the object in ``rect``, in a ``uint8`` ``HxW`` or ``HxWx3`` (RGB)
    /// image. ``rect`` is a ``manyfaced.Rect`` or a ``(left, top, right, bottom)`` tuple.
    ///
    /// The result is an ``int64`` array with shape ``(num_parts, 2)``. Each row is an ``(x, y)``
    /// point in image coordinates, as in dlib's ``full_object_detection.parts()``.
    fn __call__<'py>(
        &self,
        py: Python<'py>,
        image: &Bound<'py, PyAny>,
        rect: &Bound<'py, PyAny>,
    ) -> PyResult<Bound<'py, PyArray2<i64>>> {
        let rect = extract_rect(rect)?;
        let sp = &self.inner;
        let points = with_image(image, |img| match img {
            AnyImage::Gray(v) => sp.predict(v, rect),
            AnyImage::Rgb(v) => sp.predict(v, rect),
        })?;
        let flat: Vec<i64> = points.iter().flat_map(|p| [p.x, p.y]).collect();
        PyArray1::from_vec(py, flat).reshape([points.len(), 2])
    }

    fn __repr__(&self) -> String {
        format!(
            "ShapePredictor(num_parts={}, path={:?})",
            self.inner.num_parts(),
            self.path.display().to_string()
        )
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyShapePredictor>()
}
