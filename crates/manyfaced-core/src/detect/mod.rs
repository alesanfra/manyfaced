//! Object detection: FHOG features, `scan_fhog_pyramid` and non-maximum suppression.

pub mod fhog;
pub mod neon;
pub mod object_detector;
pub mod scan;

pub use object_detector::{
    BoxOverlap, DetectError, Detection, FhogObjectDetector, ScanFhogPyramidConfig,
};
