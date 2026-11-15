//! Face recognition ResNet inference on the real model (`python3 tools/download_models.py`).
//! Parity with dlib is tested from Python against golden descriptors (tests/test_recognition.py).

use std::path::PathBuf;
use std::sync::Arc;

use manyfaced_core::dnn::Backend;
use manyfaced_core::face::recognition::{JitterRng, RecognitionError};
use manyfaced_core::face::{ChipError, FaceRecognitionModel};
use manyfaced_core::geometry::DPoint;
use manyfaced_core::image::RgbView;

fn model() -> FaceRecognitionModel {
    let dir = std::env::var_os("MANYFACED_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models"));
    let path = dir.join("dlib_face_recognition_resnet_model_v1.dat");
    assert!(
        path.exists(),
        "{} not found: run `python3 tools/download_models.py`",
        path.display()
    );
    FaceRecognitionModel::load(path).unwrap()
}

/// A deterministic, face-sized RGB pattern.
fn chip_pixels(seed: usize) -> Vec<u8> {
    (0..150 * 150 * 3)
        .map(|i| {
            let (p, ch) = (i / 3, i % 3);
            let (y, x) = (p / 150, p % 150);
            ((x * (3 + ch) + y * (5 + seed) + (x * y) / 37) % 256) as u8
        })
        .collect()
}

fn norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[test]
fn fast_backends_match_the_naive_reference() {
    let m = model();
    for seed in 0..2 {
        let data = chip_pixels(seed);
        let chip = RgbView::packed(&data, 150, 150).unwrap();
        let slow = m.descriptor_with(chip, Backend::Naive).unwrap();
        // Backend::default() is Accelerate on macOS (feature `accelerate`), Gemm elsewhere.
        for backend in [Backend::Gemm, Backend::default()] {
            let fast = m.descriptor_with(chip, backend).unwrap();
            assert_eq!(fast.len(), 128);
            let diff: f32 = fast
                .iter()
                .zip(&slow)
                .map(|(a, b)| (a - b) * (a - b))
                .sum::<f32>()
                .sqrt();
            assert!(diff < 1e-5, "L2({backend:?} - naive) = {diff}");
            // dlib descriptors are unnormalized but have norms around 0.4 - 0.6.
            assert!((0.1..2.0).contains(&norm(&fast)), "{}", norm(&fast));
        }
        let a = m.descriptor(chip).unwrap();
        assert_eq!(a, m.descriptor(chip).unwrap(), "deterministic");
        assert_eq!(a, m.descriptor_with(chip, Backend::default()).unwrap());
    }
}

#[test]
fn wrong_inputs_are_errors() {
    let m = model();
    let data = vec![0u8; 100 * 100 * 3];
    let small = RgbView::packed(&data, 100, 100).unwrap();
    assert!(matches!(
        m.descriptor(small),
        Err(RecognitionError::Forward(_))
    ));
    let image = RgbView::packed(&data, 100, 100).unwrap();
    let four = vec![DPoint::new(1.0, 1.0); 4];
    assert!(matches!(
        m.compute_face_descriptors(image, &[four], 0, 0.25, 0),
        Err(RecognitionError::Chip(ChipError::WrongLandmarkCount(4)))
    ));
    let none: [Vec<DPoint>; 0] = [];
    assert!(
        m.compute_face_descriptors(image, &none, 0, 0.25, 0)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn jitter_averages_and_is_seeded() {
    let m = model();
    let data = chip_pixels(3);
    let chip = RgbView::packed(&data, 150, 150).unwrap();
    let plain = m.descriptor(chip).unwrap();
    // num_jitters <= 1 means no jitter, like dlib.
    assert_eq!(
        m.jittered_descriptor(chip, 1, &mut JitterRng::new(5))
            .unwrap(),
        plain
    );
    let a = m
        .jittered_descriptor(chip, 3, &mut JitterRng::new(5))
        .unwrap();
    let b = m
        .jittered_descriptor(chip, 3, &mut JitterRng::new(5))
        .unwrap();
    assert_eq!(a, b);
    assert_ne!(a, plain);
}

#[test]
fn shared_between_threads() {
    let m = Arc::new(model());
    let data = Arc::new(chip_pixels(1));
    let expected = m
        .descriptor(RgbView::packed(&data, 150, 150).unwrap())
        .unwrap();
    let handles: Vec<_> = (0..4)
        .map(|_| {
            let (m, data) = (Arc::clone(&m), Arc::clone(&data));
            std::thread::spawn(move || {
                m.descriptor(RgbView::packed(&data, 150, 150).unwrap())
                    .unwrap()
            })
        })
        .collect();
    for h in handles {
        assert_eq!(h.join().unwrap(), expected);
    }
}
