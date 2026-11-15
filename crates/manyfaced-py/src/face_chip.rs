//! Bindings for face chip extraction.

use manyfaced_core::face::chip::{self, ChipError};
use manyfaced_core::geometry::{DPoint, DRect};
use manyfaced_core::image::Image;
use numpy::{PyArray, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::image::{AnyImage, with_image};

/// The position of a chip in an image, as dlib's ``chip_details``.
///
/// The chip is `rect`, turned by `angle` radians around its centre and resampled to
/// `rows × cols` pixels.
#[pyclass(module = "manyfaced", name = "ChipDetails", frozen)]
#[derive(Clone, Copy)]
pub struct PyChipDetails(pub chip::ChipDetails);

#[pymethods]
impl PyChipDetails {
    #[new]
    #[pyo3(signature = (rect, rows, cols, angle = 0.0))]
    fn new(rect: (f64, f64, f64, f64), rows: u64, cols: u64, angle: f64) -> PyResult<Self> {
        let (l, t, r, b) = rect;
        let details = chip::ChipDetails {
            rect: DRect::new(l, t, r, b),
            angle,
            rows,
            cols,
        };
        details.validate().map_err(chip_err)?;
        Ok(PyChipDetails(details))
    }

    /// The `(left, top, right, bottom)` of the source rectangle, before the rotation.
    #[getter]
    fn rect(&self) -> (f64, f64, f64, f64) {
        let r = self.0.rect;
        (r.left, r.top, r.right, r.bottom)
    }

    /// The rotation, in radians.
    #[getter]
    fn angle(&self) -> f64 {
        self.0.angle
    }

    #[getter]
    fn rows(&self) -> u64 {
        self.0.rows
    }

    #[getter]
    fn cols(&self) -> u64 {
        self.0.cols
    }

    fn __repr__(&self) -> String {
        let d = self.0;
        format!(
            "ChipDetails(rect=({}, {}, {}, {}), rows={}, cols={}, angle={})",
            d.rect.left, d.rect.top, d.rect.right, d.rect.bottom, d.rows, d.cols, d.angle
        )
    }

    fn __getnewargs__(&self) -> ((f64, f64, f64, f64), u64, u64, f64) {
        (self.rect(), self.0.rows, self.0.cols, self.0.angle)
    }
}

fn chip_err(e: ChipError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// `_face_chip_details(landmarks, size, padding)`: `landmarks` is a float64 (N, 2) array.
#[pyfunction]
fn _face_chip_details(
    landmarks: PyReadonlyArray2<'_, f64>,
    size: u64,
    padding: f64,
) -> PyResult<PyChipDetails> {
    if landmarks.shape()[1] != 2 {
        return Err(PyValueError::new_err(format!(
            "landmarks must have shape (N, 2), got {:?}",
            landmarks.shape()
        )));
    }
    let pts: Vec<DPoint> = landmarks
        .as_array()
        .rows()
        .into_iter()
        .map(|r| DPoint::new(r[0], r[1]))
        .collect();
    chip::face_chip_details(&pts, size, padding)
        .map(PyChipDetails)
        .map_err(chip_err)
}

fn to_numpy<const C: usize>(py: Python<'_>, img: Image<C>) -> PyResult<Py<PyAny>> {
    let (h, w) = (img.height(), img.width());
    let flat = PyArray::from_vec(py, img.into_vec());
    Ok(if C == 1 {
        flat.reshape([h, w])?.into_any().unbind()
    } else {
        flat.reshape([h, w, C])?.into_any().unbind()
    })
}

enum Chips {
    Gray(Vec<Image<1>>),
    Rgb(Vec<Image<3>>),
}

/// `_extract_chips(image, details) -> list[np.ndarray]`: all chips extracted together, like
/// dlib's `extract_image_chips` (they share one image pyramid).
#[pyfunction]
fn _extract_chips<'py>(
    image: &Bound<'py, PyAny>,
    details: Vec<PyRef<'py, PyChipDetails>>,
) -> PyResult<Bound<'py, PyList>> {
    let py = image.py();
    let details: Vec<chip::ChipDetails> = details.iter().map(|d| d.0).collect();
    let chips = with_image(image, move |img| match img {
        AnyImage::Gray(v) => chip::extract_image_chips(v, &details).map(Chips::Gray),
        AnyImage::Rgb(v) => chip::extract_image_chips(v, &details).map(Chips::Rgb),
    })?
    .map_err(chip_err)?;
    let out = match chips {
        Chips::Gray(c) => c
            .into_iter()
            .map(|i| to_numpy(py, i))
            .collect::<PyResult<Vec<_>>>()?,
        Chips::Rgb(c) => c
            .into_iter()
            .map(|i| to_numpy(py, i))
            .collect::<PyResult<Vec<_>>>()?,
    };
    PyList::new(py, out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyChipDetails>()?;
    m.add_function(wrap_pyfunction!(_face_chip_details, m)?)?;
    m.add_function(wrap_pyfunction!(_extract_chips, m)?)?;
    Ok(())
}
