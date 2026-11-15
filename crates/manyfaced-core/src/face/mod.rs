//! Face pipeline: `shape_predictor`, face chips and face recognition.

pub mod chip;
pub mod recognition;
pub mod shape_predictor;

pub use chip::{
    ChipDetails, ChipError, extract_image_chip, extract_image_chips, face_chip_details,
};
pub use recognition::FaceRecognitionModel;
pub use shape_predictor::ShapePredictor;
