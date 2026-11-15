//! manyfaced-core errors → Python exceptions.

use pyo3::exceptions::{PyFileNotFoundError, PyOSError, PyValueError};
use pyo3::prelude::*;

pyo3::create_exception!(
    manyfaced,
    ModelFormatError,
    PyValueError,
    "A model file is not a valid dlib file of the expected type (corrupt, truncated or a different model)."
);

pub fn to_py_err(e: manyfaced_core::Error) -> PyErr {
    match &e {
        manyfaced_core::Error::Io { source, .. }
            if source.kind() == std::io::ErrorKind::NotFound =>
        {
            PyFileNotFoundError::new_err(e.to_string())
        }
        manyfaced_core::Error::Io { .. } => PyOSError::new_err(e.to_string()),
        manyfaced_core::Error::Format { .. } => ModelFormatError::new_err(e.to_string()),
    }
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add("ModelFormatError", m.py().get_type::<ModelFormatError>())
}
