//! Bindings for the face recognition ResNet.

use std::path::PathBuf;

use manyfaced_core::face::FaceRecognitionModel;
use manyfaced_core::face::recognition::{JitterRng, RecognitionError, gray_to_rgb};
use manyfaced_core::geometry::DPoint;
use numpy::{PyArray1, PyArray2, PyArrayMethods, PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::errors::to_py_err;
use crate::image::{AnyImage, with_image};

fn recognition_err(e: RecognitionError) -> PyErr {
    PyValueError::new_err(e.to_string())
}

/// dlib's `num_jitters`: values <= 1 mean no jittering.
fn jitters(n: i64) -> u32 {
    u32::try_from(n.max(0)).unwrap_or(u32::MAX)
}

/// Descriptors as a float64 `(N, dim)` array. The values are dlib's float32 results, widened
/// exactly like dlib's `matrix_cast<double>`.
fn to_numpy<'py>(
    py: Python<'py>,
    descriptors: Vec<Vec<f32>>,
    dim: usize,
) -> PyResult<Bound<'py, PyArray2<f64>>> {
    let n = descriptors.len();
    let flat: Vec<f64> = descriptors.into_iter().flatten().map(f64::from).collect();
    PyArray1::from_vec(py, flat).reshape([n, dim])
}

/// The native face recognizer; use `manyfaced.FaceRecognizer`, which validates arguments.
#[pyclass(module = "manyfaced._manyfaced", name = "_FaceRecognizer", frozen)]
pub struct PyFaceRecognizer {
    inner: FaceRecognitionModel,
    path: PathBuf,
}

impl PyFaceRecognizer {
    fn dim(&self) -> usize {
        self.inner
            .network()
            .layers
            .last()
            .map(|l| match l {
                manyfaced_core::dnn::Layer::Fc(f) => f.num_outputs,
                _ => 0,
            })
            .unwrap_or(0)
    }
}

#[pymethods]
impl PyFaceRecognizer {
    #[staticmethod]
    fn load(py: Python<'_>, path: PathBuf) -> PyResult<Self> {
        let inner = py
            .detach(|| FaceRecognitionModel::load(&path))
            .map_err(to_py_err)?;
        Ok(PyFaceRecognizer { inner, path })
    }

    #[getter]
    fn chip_size(&self) -> usize {
        self.inner.chip_size()
    }

    /// `_compute(image, landmarks, num_jitters, padding, seed)`: one descriptor per float64
    /// `(N, 2)` landmark array, chips extracted together.
    fn _compute<'py>(
        &self,
        py: Python<'py>,
        image: &Bound<'py, PyAny>,
        landmarks: Vec<PyReadonlyArray2<'py, f64>>,
        num_jitters: i64,
        padding: f32,
        seed: u64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let faces = landmarks
            .iter()
            .map(|a| {
                if a.shape()[1] != 2 {
                    return Err(PyValueError::new_err(format!(
                        "landmarks must have shape (N, 2), got {:?}",
                        a.shape()
                    )));
                }
                Ok(a.as_array()
                    .rows()
                    .into_iter()
                    .map(|r| DPoint::new(r[0], r[1]))
                    .collect::<Vec<_>>())
            })
            .collect::<PyResult<Vec<_>>>()?;
        let model = &self.inner;
        let n = jitters(num_jitters);
        let out = with_image(image, |img| match img {
            AnyImage::Rgb(v) => model.compute_face_descriptors(v, &faces, n, padding, seed),
            AnyImage::Gray(v) => {
                let rgb = gray_to_rgb(v);
                model.compute_face_descriptors(rgb.view(), &faces, n, padding, seed)
            }
        })?
        .map_err(recognition_err)?;
        to_numpy(py, out, self.dim())
    }

    /// `_compute_chips(chips, num_jitters, seed)`: descriptors of aligned chips.
    fn _compute_chips<'py>(
        &self,
        py: Python<'py>,
        chips: Vec<Bound<'py, PyAny>>,
        num_jitters: i64,
        seed: u64,
    ) -> PyResult<Bound<'py, PyArray2<f64>>> {
        let model = &self.inner;
        let n = jitters(num_jitters);
        let mut rng = JitterRng::new(seed);
        let mut out = Vec::with_capacity(chips.len());
        for chip in &chips {
            let rng_ref = &mut rng;
            let d = with_image(chip, |img| match img {
                AnyImage::Rgb(v) => model.jittered_descriptor(v, n, rng_ref),
                AnyImage::Gray(v) => {
                    let rgb = gray_to_rgb(v);
                    model.jittered_descriptor(rgb.view(), n, rng_ref)
                }
            })?
            .map_err(recognition_err)?;
            out.push(d);
        }
        to_numpy(py, out, self.dim())
    }

    fn __repr__(&self) -> String {
        format!("FaceRecognizer(path={:?})", self.path.display().to_string())
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFaceRecognizer>()
}
