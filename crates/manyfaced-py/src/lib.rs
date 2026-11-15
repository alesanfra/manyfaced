//! PyO3 bindings for `manyfaced-core`.
//!
//! This crate contains no algorithms. It converts numpy arrays and Python objects to core
//! types, releases the GIL during the computations and changes core errors into Python
//! exceptions.

mod cluster;
mod detector;
mod errors;
mod face_chip;
mod geometry;
mod image;
mod recognition;
mod shape_predictor;

use pyo3::prelude::*;

#[pymodule]
fn _manyfaced(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    m.add(
        "DLIB_REFERENCE_VERSION",
        manyfaced_core::DLIB_REFERENCE_VERSION,
    )?;
    errors::register(m)?;
    geometry::register(m)?;
    image::register(m)?;
    shape_predictor::register(m)?;
    face_chip::register(m)?;
    recognition::register(m)?;
    detector::register(m)?;
    cluster::register(m)?;
    Ok(())
}
