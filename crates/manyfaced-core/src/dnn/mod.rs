//! The minimal set of dlib DNN layers needed to run inference on the face recognition ResNet.

pub mod forward;
pub mod network;
pub mod spec;

pub use forward::{Backend, ForwardError, Tensor};
pub use network::{Layer, Network};
pub use spec::{LayerSpec, NetSpec};
