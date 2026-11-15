//! Inference robustness on the real 5-point model (parity with dlib is checked by
//! tests/test_landmarks.py against the golden files).

use std::path::PathBuf;
use std::sync::Arc;

use manyfaced_core::face::ShapePredictor;
use manyfaced_core::geometry::Rect;
use manyfaced_core::image::{GrayView, RgbView};

fn predictor() -> ShapePredictor {
    let dir = std::env::var_os("MANYFACED_MODELS_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../models"));
    let path = dir.join("shape_predictor_5_face_landmarks.dat");
    assert!(
        path.exists(),
        "{} not found: run `python3 tools/download_models.py`",
        path.display()
    );
    ShapePredictor::load(path).unwrap()
}

fn texture(w: usize, h: usize) -> Vec<u8> {
    (0..w * h * 3)
        .map(|i| ((i * 37 + i / 7 * 11) % 256) as u8)
        .collect()
}

#[test]
fn any_rect_and_image_size_is_handled() {
    let sp = predictor();
    let data = texture(64, 48);
    let rgb = RgbView::packed(&data, 64, 48).unwrap();
    let one = GrayView::packed(&data[..1], 1, 1).unwrap();
    let empty = GrayView::packed(&[], 0, 0).unwrap();
    let rects = [
        Rect::new(10, 10, 40, 40),
        Rect::new(5, 5, 4, 4), // empty
        Rect::new(7, 7, 7, 7), // single pixel
        Rect::new(-500, -500, -400, -400),
        Rect::new(1000, 1000, 2000, 2000),
        Rect::new(-1 << 40, -1 << 40, 1 << 40, 1 << 40),
        Rect::new(i64::MIN, i64::MIN, i64::MAX, i64::MAX),
    ];
    for rect in rects {
        assert_eq!(sp.predict(rgb, rect).len(), 5);
        assert_eq!(sp.predict(one, rect).len(), 5);
        assert_eq!(sp.predict(empty, rect).len(), 5);
    }
}

#[test]
fn predictions_are_deterministic_across_threads() {
    let sp = Arc::new(predictor());
    let data = Arc::new(texture(200, 150));
    let rect = Rect::new(40, 30, 140, 130);
    let reference = sp.predict(RgbView::packed(&data, 200, 150).unwrap(), rect);
    let handles: Vec<_> = (0..8)
        .map(|_| {
            let (sp, data) = (Arc::clone(&sp), Arc::clone(&data));
            std::thread::spawn(move || {
                (0..20)
                    .map(|_| sp.predict(RgbView::packed(&data, 200, 150).unwrap(), rect))
                    .collect::<Vec<_>>()
            })
        })
        .collect();
    for h in handles {
        for r in h.join().unwrap() {
            assert_eq!(r, reference);
        }
    }
}
