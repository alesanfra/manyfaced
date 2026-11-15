//! Bindings for the HOG object detector.

use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use manyfaced_core::detect::{Detection, FhogObjectDetector};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::errors::to_py_err;
use crate::geometry::PyRect;
use crate::image::{AnyImage, with_image};

/// A FHOG sliding-window object detector, as dlib's ``fhog_object_detector``.
///
/// ``FhogDetector()`` is the HOG face detector that dlib contains
/// (``get_frontal_face_detector()``). ``FhogDetector(path)`` reads a detector that dlib saved
/// (``fhog_object_detector.save`` or ``train_simple_object_detector``). The detector does not
/// change after it loads, and many threads can use it. This is not the case in dlib. The GIL is
/// released while the detector runs.
#[pyclass(module = "manyfaced", name = "FhogDetector", frozen)]
pub struct PyFhogDetector {
    inner: Arc<FhogObjectDetector>,
    source: Option<PathBuf>,
}

fn upsample_arg(upsample: i64) -> PyResult<u32> {
    u32::try_from(upsample)
        .map_err(|_| PyValueError::new_err(format!("upsample must be >= 0, got {upsample}")))
}

impl PyFhogDetector {
    fn detect(
        &self,
        image: &Bound<'_, PyAny>,
        upsample: i64,
        adjust_threshold: f64,
    ) -> PyResult<Vec<Detection>> {
        let upsample = upsample_arg(upsample)?;
        let det = &self.inner;
        with_image(image, |img| match img {
            AnyImage::Gray(v) => det.detect_upsampled(v, upsample, adjust_threshold),
            AnyImage::Rgb(v) => det.detect_upsampled(v, upsample, adjust_threshold),
        })?
        .map_err(|e| PyValueError::new_err(e.to_string()))
    }
}

#[pymethods]
impl PyFhogDetector {
    /// Without ``path``, give the HOG frontal face detector of dlib, as one shared instance. It
    /// has 5 filters: front, left, right, front rotated left and front rotated right. With
    /// ``path``, load a serialized dlib ``object_detector<scan_fhog_pyramid<pyramid_down<6>>>``.
    #[new]
    #[pyo3(signature = (path = None))]
    fn new(py: Python<'_>, path: Option<PathBuf>) -> PyResult<Self> {
        let Some(path) = path else {
            static FRONTAL: OnceLock<Arc<FhogObjectDetector>> = OnceLock::new();
            return Ok(PyFhogDetector {
                inner: FRONTAL
                    .get_or_init(|| Arc::new(FhogObjectDetector::frontal_face()))
                    .clone(),
                source: None,
            });
        };
        let inner = py
            .detach(|| FhogObjectDetector::load(&path))
            .map_err(to_py_err)?;
        Ok(PyFhogDetector {
            inner: Arc::new(inner),
            source: Some(path),
        })
    }

    /// The number of filters (dlib's ``num_detectors``).
    #[getter]
    fn num_detectors(&self) -> usize {
        self.inner.num_detectors()
    }

    /// The width of the sliding window, in pixels.
    #[getter]
    fn detection_window_width(&self) -> u64 {
        self.inner.scanner().window_width()
    }

    /// The height of the sliding window, in pixels.
    #[getter]
    fn detection_window_height(&self) -> u64 {
        self.inner.scanner().window_height()
    }

    /// Find objects in a ``uint8`` ``HxW`` or ``HxWx3`` (RGB) image.
    ///
    /// The detector first doubles the image size ``upsample`` times. The result is the list of
    /// rectangles, with the best rectangle first.
    #[pyo3(signature = (image, upsample = 0))]
    fn __call__(&self, image: &Bound<'_, PyAny>, upsample: i64) -> PyResult<Vec<PyRect>> {
        Ok(self
            .detect(image, upsample, 0.0)?
            .into_iter()
            .map(|d| PyRect(d.rect))
            .collect())
    }

    /// Find objects, as a call to the detector does, with ``adjust_threshold`` added to the
    /// detection threshold.
    ///
    /// The result is ``(rects, scores, filter_indices)``, with the best detection first. A score
    /// is the distance of the window above the threshold of the filter that found it.
    #[pyo3(signature = (image, upsample = 0, adjust_threshold = 0.0))]
    fn run(
        &self,
        image: &Bound<'_, PyAny>,
        upsample: i64,
        adjust_threshold: f64,
    ) -> PyResult<(Vec<PyRect>, Vec<f64>, Vec<usize>)> {
        let dets = self.detect(image, upsample, adjust_threshold)?;
        Ok((
            dets.iter().map(|d| PyRect(d.rect)).collect(),
            dets.iter().map(|d| d.confidence).collect(),
            dets.iter().map(|d| d.weight_index).collect(),
        ))
    }

    fn __repr__(&self) -> String {
        match &self.source {
            None => format!(
                "FhogDetector(frontal_face, num_detectors={})",
                self.inner.num_detectors()
            ),
            Some(p) => format!(
                "FhogDetector(num_detectors={}, path={:?})",
                self.inner.num_detectors(),
                p.display().to_string()
            ),
        }
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyFhogDetector>()
}
