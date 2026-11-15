//! Graph clustering (`chinese_whispers`).

pub mod chinese_whispers;
pub mod rand;

pub use chinese_whispers::{Edge, chinese_whispers, chinese_whispers_clustering};
