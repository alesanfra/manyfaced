//! Behaviour tests for the FHOG detector that need no external data (parity with dlib is
//! tested from Python against tests/golden/detections.npz).

use std::sync::Arc;

use manyfaced_core::detect::{DetectError, FhogObjectDetector};
use manyfaced_core::image::ImageView;

const _: () = {
    const fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<FhogObjectDetector>();
};

/// A deterministic textured image.
fn texture(w: usize, h: usize, c: usize) -> Vec<u8> {
    (0..w * h * c)
        .map(|i| ((i * 7919 + (i / (w * c)) * 104_729) % 251) as u8)
        .collect()
}

#[test]
fn frontal_face_detector_geometry() {
    let d = FhogObjectDetector::frontal_face();
    assert_eq!(d.num_detectors(), 5);
    assert_eq!(d.fhog_window(), (10, 10));
    assert!(d.filter_banks().iter().all(|b| b.use_separable()));
    let ranks: Vec<usize> = d
        .filter_banks()
        .iter()
        .map(|b| b.num_separable_filters())
        .collect();
    // Same ranks as dlib's build_fhog_filterbank (tools/cpp_probe/hog_probe.cpp).
    assert_eq!(ranks, [61, 58, 54, 66, 66]);
}

#[test]
fn any_image_size_is_accepted() {
    let d = FhogObjectDetector::frontal_face();
    for (w, h) in [
        (0, 0),
        (1, 1),
        (0, 5),
        (7, 0),
        (3, 2),
        (17, 13),
        (64, 64),
        (81, 79),
        (200, 31),
    ] {
        let gray = texture(w, h, 1);
        let rgb = texture(w, h, 3);
        let g = ImageView::<1>::packed(&gray, w, h).unwrap();
        let c = ImageView::<3>::packed(&rgb, w, h).unwrap();
        for up in 0..3 {
            for adjust in [0.0, -5.0, f64::NEG_INFINITY, f64::NAN] {
                let a = d.detect_upsampled(g, up, adjust).unwrap();
                let b = d.detect_upsampled(c, up, adjust).unwrap();
                for det in a.iter().chain(&b) {
                    assert!(det.weight_index < 5);
                    assert!(!det.rect.is_empty());
                }
            }
        }
    }
}

#[test]
fn lower_threshold_finds_more_candidates() {
    let d = FhogObjectDetector::frontal_face();
    let px = texture(160, 120, 1);
    let img = ImageView::<1>::packed(&px, 160, 120).unwrap();
    let strict = d.detect(&img, 0.0);
    let loose = d.detect(&img, -3.0);
    assert!(loose.len() >= strict.len());
    assert!(!loose.is_empty());
    // Best first.
    assert!(loose.windows(2).all(|w| w[0].confidence >= w[1].confidence));
    // Non-maximum suppression: no two kept boxes overlap by dlib's test.
    for (i, a) in loose.iter().enumerate() {
        for b in &loose[i + 1..] {
            assert!(!d.overlap().overlaps(a.rect, b.rect));
        }
    }
}

#[test]
fn shared_between_threads() {
    let d = Arc::new(FhogObjectDetector::frontal_face());
    let px = texture(240, 180, 3);
    let img = ImageView::<3>::packed(&px, 240, 180).unwrap();
    let expected = d.detect_upsampled(img, 1, -2.0).unwrap();
    std::thread::scope(|s| {
        let handles: Vec<_> = (0..8)
            .map(|_| {
                let d = Arc::clone(&d);
                s.spawn(move || d.detect_upsampled(img, 1, -2.0).unwrap())
            })
            .collect();
        for h in handles {
            assert_eq!(h.join().unwrap(), expected);
        }
    });
}

#[test]
fn huge_upsampling_is_refused() {
    let d = FhogObjectDetector::frontal_face();
    let px = texture(100, 100, 1);
    let img = ImageView::<1>::packed(&px, 100, 100).unwrap();
    assert!(matches!(
        d.detect_upsampled(img, 20, 0.0),
        Err(DetectError::UpsampleTooLarge { .. })
    ));
    // An empty image never grows.
    let empty = ImageView::<1>::packed(&[], 0, 0).unwrap();
    assert_eq!(d.detect_upsampled(empty, 40, 0.0).unwrap(), []);
}

#[test]
fn wrong_dimensions_are_rejected() {
    // The frontal detector's file with the stored dimension count changed from 3100
    // (02 1c 0c) to 3101: dlib's deserialize rejects it, so do we.
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("data/frontal_face_detector.dat");
    let mut bytes = std::fs::read(path).unwrap();
    let pos = bytes
        .windows(3)
        .position(|w| w == [0x02, 0x1c, 0x0c])
        .unwrap();
    bytes[pos + 1] = 0x1d;
    let e = FhogObjectDetector::from_bytes(&bytes)
        .unwrap_err()
        .to_string();
    assert!(e.contains("3101 dimensions"), "{e}");
}
