//! Pure Rust port of a subset of [dlib](https://github.com/davisking/dlib).
//!
//! This crate has no Python dependency. The Python bindings are in `manyfaced-py`.
//! The dlib source code is the specification for each algorithm and file format in this crate.
//! `docs/internals.md` gives the dlib tag.

pub mod cluster;
pub mod detect;
pub mod dnn;
pub mod face;
pub mod geometry;
pub mod image;
pub mod linalg;
pub mod serialize;

use std::path::{Path, PathBuf};

/// The dlib release whose behaviour this crate reproduces.
pub const DLIB_REFERENCE_VERSION: &str = "20.0.1";

/// Errors returned by manyfaced-core.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    #[error("cannot read {}: {source}", path.display())]
    Io {
        path: PathBuf,
        #[source]
        source: std::io::Error,
    },
    #[error("{}invalid {what}: {source}", path.as_ref().map(|p| format!("{}: ", p.display())).unwrap_or_default())]
    Format {
        /// What was being loaded, e.g. "dlib shape_predictor".
        what: &'static str,
        path: Option<PathBuf>,
        #[source]
        source: serialize::Error,
    },
}

pub type Result<T> = std::result::Result<T, Error>;

/// Reads a whole model file and parses it with `parse`, attaching the path to errors.
pub(crate) fn load_file<T>(
    path: &Path,
    what: &'static str,
    parse: impl FnOnce(&[u8]) -> serialize::Result<T>,
) -> Result<T> {
    let data = std::fs::read(path).map_err(|source| Error::Io {
        path: path.to_owned(),
        source,
    })?;
    parse(&data).map_err(|source| Error::Format {
        what,
        path: Some(path.to_owned()),
        source,
    })
}

/// Parses an in-memory model with `parse`.
pub(crate) fn load_bytes<T>(
    data: &[u8],
    what: &'static str,
    parse: impl FnOnce(&[u8]) -> serialize::Result<T>,
) -> Result<T> {
    parse(data).map_err(|source| Error::Format {
        what,
        path: None,
        source,
    })
}
