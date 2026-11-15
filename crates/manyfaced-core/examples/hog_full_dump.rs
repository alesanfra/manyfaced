//! Rust counterpart of `tools/cpp_probe/hog_full_probe.cpp` (non-separable filter path):
//!
//!     cargo run --release --example hog_full_dump -- IMAGE.raw WEIGHTS.f64 OUT.f32
use manyfaced_core::detect::fhog::extract_fhog;
use manyfaced_core::detect::scan::{FilterBank, FilterScratch, FloatImage, apply_filters};
use manyfaced_core::image::ImageView;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let raw = std::fs::read(&args[1]).unwrap();
    let hdr = |i: usize| i32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap()) as usize;
    let (w, h) = (hdr(0), hdr(1));
    let img = ImageView::<1>::packed(&raw[12..], w, h).unwrap();
    let weights: Vec<f64> = std::fs::read(&args[2])
        .unwrap()
        .chunks_exact(8)
        .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
        .collect();
    let fb = FilterBank::build(&weights, 10, 10);
    eprintln!(
        "separable: {}, {} components",
        fb.use_separable(),
        fb.num_separable_filters()
    );
    let feats = extract_fhog(&img, 8, 10, 10);
    let (mut sal, mut scratch) = (FloatImage::default(), FilterScratch::default());
    let area = apply_filters(&fb, &feats, &mut sal, &mut scratch);
    eprintln!("area {area:?} {}x{}", sal.rows, sal.cols);
    let bytes: Vec<u8> = sal.data.iter().flat_map(|v| v.to_le_bytes()).collect();
    std::fs::write(&args[3], bytes).unwrap();
}
