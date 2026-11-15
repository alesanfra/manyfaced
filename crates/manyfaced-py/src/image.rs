//! numpy `uint8` arrays → core image views, with the GIL released while the view is used.

use manyfaced_core::image::{GrayView, RgbView};
use numpy::{
    PyArrayDescrMethods, PyArrayDyn, PyArrayMethods, PyUntypedArray, PyUntypedArrayMethods,
};
use pyo3::exceptions::{PyMemoryError, PyTypeError, PyValueError};
use pyo3::prelude::*;

/// A grayscale or RGB view of a numpy array.
#[derive(Clone, Copy)]
pub enum AnyImage<'a> {
    Gray(GrayView<'a>),
    Rgb(RgbView<'a>),
}

/// Validates `obj` as an `H×W` or `H×W×3` `uint8` numpy array and runs `f` on a view of it with
/// the GIL released. C-contiguous arrays are used in place; other layouts (slices, flips,
/// transposes) are copied once into a contiguous buffer.
///
/// The array must not be modified by another thread while `f` runs (the same contract as any
/// native extension that releases the GIL).
pub fn with_image<T, F>(obj: &Bound<'_, PyAny>, f: F) -> PyResult<T>
where
    T: Send,
    F: for<'a> FnOnce(AnyImage<'a>) -> T + Send,
{
    let py = obj.py();
    let untyped = obj.cast::<PyUntypedArray>().map_err(|_| {
        PyTypeError::new_err(format!(
            "expected a numpy uint8 array, got {}",
            obj.get_type()
                .name()
                .map(|n| n.to_string())
                .unwrap_or_else(|_| "an unknown type".into())
        ))
    })?;
    let dtype = untyped.dtype();
    if !dtype.is_equiv_to(&numpy::dtype::<u8>(py)) {
        return Err(PyTypeError::new_err(format!(
            "expected a uint8 array, got dtype {dtype}"
        )));
    }
    let shape = untyped.shape().to_vec();
    let channels = match shape.as_slice() {
        [_, _] => 1,
        [_, _, 3] => 3,
        _ => {
            return Err(PyValueError::new_err(format!(
                "expected an HxW (grayscale) or HxWx3 (RGB) uint8 array, got shape {shape:?}"
            )));
        }
    };
    let (height, width) = (shape[0], shape[1]);
    let array = obj.cast::<PyArrayDyn<u8>>()?;
    let readonly = array
        .try_readonly()
        .map_err(|e| PyValueError::new_err(format!("cannot read the image array: {e}")))?;
    let mut copied = Vec::new();
    let data: &[u8] = match readonly.as_slice() {
        Ok(slice) => slice,
        Err(_) => {
            // A view (for example from np.broadcast_to) can be much larger than its memory: make
            // sure that the copy fits before the copy starts, so that the process does not abort.
            let view = readonly.as_array();
            copied.try_reserve_exact(view.len()).map_err(|_| {
                PyMemoryError::new_err(format!(
                    "not enough memory to copy a non-contiguous {shape:?} image"
                ))
            })?;
            // Logical (C) order iteration copies any stride pattern correctly.
            copied.extend(view.iter().copied());
            &copied
        }
    };
    let layout_err =
        |e: manyfaced_core::image::ImageLayoutError| PyValueError::new_err(e.to_string());
    let image = if channels == 1 {
        AnyImage::Gray(GrayView::packed(data, width, height).map_err(layout_err)?)
    } else {
        AnyImage::Rgb(RgbView::packed(data, width, height).map_err(layout_err)?)
    };
    Ok(py.detach(move || f(image)))
}

/// `_image_info(image) -> (height, width, channels, sum_of_bytes)`: exercises the numpy
/// conversion path from Python tests (dtype/shape errors, non-contiguous copies).
#[pyfunction]
pub fn _image_info(image: &Bound<'_, PyAny>) -> PyResult<(usize, usize, usize, u64)> {
    with_image(image, |img| {
        fn sum<const C: usize>(v: &manyfaced_core::image::ImageView<'_, C>) -> u64 {
            (0..v.height())
                .map(|y| v.row(y).iter().map(|&b| u64::from(b)).sum::<u64>())
                .sum()
        }
        match img {
            AnyImage::Gray(v) => (v.height(), v.width(), 1, sum(&v)),
            AnyImage::Rgb(v) => (v.height(), v.width(), 3, sum(&v)),
        }
    })
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(_image_info, m)?)
}
