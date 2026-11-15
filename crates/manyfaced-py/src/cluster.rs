//! Bindings for `manyfaced_core::cluster` (chinese whispers).

use manyfaced_core::cluster::{self, Edge};
use numpy::PyReadonlyArray2;
use pyo3::exceptions::{PyMemoryError, PyTypeError, PyValueError};
use pyo3::prelude::*;

/// `chinese_whispers(edges, num_iterations)`: `edges` are `(i, j)` or `(i, j, weight)` tuples.
/// The memory is proportional to the largest node index: a very large index gives `MemoryError`.
#[pyfunction]
#[pyo3(signature = (edges, num_iterations = 100))]
fn _chinese_whispers(
    py: Python<'_>,
    edges: Vec<Bound<'_, PyAny>>,
    num_iterations: u64,
) -> PyResult<Vec<usize>> {
    let edges = edges
        .iter()
        .map(|e| {
            if let Ok((a, b)) = e.extract::<(usize, usize)>() {
                Ok(Edge::new(a, b))
            } else if let Ok((a, b, w)) = e.extract::<(usize, usize, f64)>() {
                Ok(Edge::weighted(a, b, w))
            } else {
                Err(PyTypeError::new_err(
                    "edges must be (i, j) or (i, j, weight) tuples with non-negative integer indices",
                ))
            }
        })
        .collect::<PyResult<Vec<_>>>()?;
    py.detach(|| cluster::chinese_whispers(&edges, num_iterations))
        .map_err(|_| {
            PyMemoryError::new_err("not enough memory for a graph with this largest node index")
        })
}

/// `chinese_whispers_clustering(descriptors, threshold)` on an `(N, D)` float64 array.
#[pyfunction]
fn _chinese_whispers_clustering(
    py: Python<'_>,
    descriptors: PyReadonlyArray2<'_, f64>,
    threshold: f32,
) -> PyResult<Vec<usize>> {
    // dlib asserts `threshold > 0`, which NaN also fails.
    if threshold.is_nan() || threshold <= 0.0 {
        return Err(PyValueError::new_err(format!(
            "threshold must be > 0, got {threshold}"
        )));
    }
    let array = descriptors.as_array();
    let rows: Vec<Vec<f64>> = array.rows().into_iter().map(|r| r.to_vec()).collect();
    Ok(py.detach(|| cluster::chinese_whispers_clustering(&rows, threshold)))
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_chinese_whispers, m)?)?;
    m.add_function(wrap_pyfunction!(_chinese_whispers_clustering, m)?)
}
