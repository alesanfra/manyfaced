//! Dumps the frontal face detector's intermediates in the same `.npy` layout as
//! `tools/cpp_probe/hog_probe.cpp`, to compare manyfaced with dlib stage by stage:
//!
//!     cargo run --release --example hog_dump -- IMAGE.raw OUT_DIR [upsample] [adjust_threshold]
use std::io::Write;

use manyfaced_core::detect::FhogObjectDetector;
use manyfaced_core::detect::fhog::{NUM_PLANES, extract_fhog};
use manyfaced_core::detect::scan::{FilterScratch, FloatImage, apply_filters};
use manyfaced_core::image::interpolation::Pixels;
use manyfaced_core::image::pyramid::{pyramid_down6, pyramid_up};
use manyfaced_core::image::{Image, ImageView};

fn write_npy(path: &str, descr: &str, shape: &[usize], bytes: &[u8]) {
    let shape_s = if shape.len() == 1 {
        format!("({},)", shape[0])
    } else {
        format!(
            "({})",
            shape
                .iter()
                .map(|s| s.to_string())
                .collect::<Vec<_>>()
                .join(", ")
        )
    };
    let mut header =
        format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': {shape_s}, }}");
    let total = 10 + header.len() + 1;
    header.push_str(&" ".repeat((64 - total % 64) % 64));
    header.push('\n');
    let mut f = std::fs::File::create(path).unwrap();
    f.write_all(b"\x93NUMPY\x01\x00").unwrap();
    f.write_all(&(header.len() as u16).to_le_bytes()).unwrap();
    f.write_all(header.as_bytes()).unwrap();
    f.write_all(bytes).unwrap();
}

fn f32_bytes(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|x| x.to_le_bytes()).collect()
}

fn dump_image<const C: usize>(path: &str, img: &impl Pixels<C>) {
    let mut bytes = Vec::new();
    for y in 0..img.height() {
        bytes.extend_from_slice(img.row(y));
    }
    let mut shape = vec![img.height(), img.width()];
    if C == 3 {
        shape.push(3);
    }
    write_npy(path, "|u1", &shape, &bytes);
}

fn run<const C: usize>(view: ImageView<'_, C>, out: &str, upsample: u32, adjust: f64) {
    let det = FhogObjectDetector::frontal_face();
    let mut img: Image<C> = Image::from_vec(
        (0..view.height())
            .flat_map(|y| view.row(y).to_vec())
            .collect(),
        view.width(),
        view.height(),
    )
    .unwrap();
    if upsample > 0 {
        for _ in 0..upsample {
            img = pyramid_up(&img);
        }
        dump_image(&format!("{out}/up_image.npy"), &img);
    }
    let (w, h) = det.fhog_window();
    let mut levels = vec![img.clone()];
    loop {
        // Same level count as the detector (min layer 64x64).
        let next = pyramid_down6(levels.last().unwrap());
        let r = manyfaced_core::image::pyramid::pyramid_down6_rect_down(
            manyfaced_core::geometry::Rect::new(
                0,
                0,
                levels.last().unwrap().width() as i64 - 1,
                levels.last().unwrap().height() as i64 - 1,
            ),
        );
        if !(r.width() >= 64 && r.height() >= 64) {
            break;
        }
        levels.push(next);
    }
    for (l, li) in levels.iter().enumerate() {
        dump_image(&format!("{out}/level{l}_image.npy"), li);
        let f = extract_fhog(li, 8, h, w);
        write_npy(
            &format!("{out}/level{l}_fhog.npy"),
            "<f4",
            &[NUM_PLANES, f.rows, f.cols],
            &f32_bytes(&f.data),
        );
        for (i, bank) in det.filter_banks().iter().enumerate() {
            let (mut sal, mut scratch) = (FloatImage::default(), FilterScratch::default());
            let area = apply_filters(bank, &f, &mut sal, &mut scratch);
            write_npy(
                &format!("{out}/level{l}_sal{i}.npy"),
                "<f4",
                &[sal.rows, sal.cols],
                &f32_bytes(&sal.data),
            );
            let a: Vec<u8> = [area.left, area.top, area.right, area.bottom]
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect();
            write_npy(&format!("{out}/level{l}_area{i}.npy"), "<i8", &[4], &a);
        }
    }
    for (i, bank) in det.filter_banks().iter().enumerate() {
        let rank: Vec<u8> = bank
            .row_filters
            .iter()
            .flat_map(|r| (r.len() as i32).to_le_bytes())
            .collect();
        write_npy(
            &format!("{out}/filter{i}_rank.npy"),
            "<i4",
            &[NUM_PLANES],
            &rank,
        );
        let rows: Vec<f32> = bank
            .row_filters
            .iter()
            .flatten()
            .flatten()
            .copied()
            .collect();
        let cols: Vec<f32> = bank
            .col_filters
            .iter()
            .flatten()
            .flatten()
            .copied()
            .collect();
        write_npy(
            &format!("{out}/filter{i}_row.npy"),
            "<f4",
            &[rows.len()],
            &f32_bytes(&rows),
        );
        write_npy(
            &format!("{out}/filter{i}_col.npy"),
            "<f4",
            &[cols.len()],
            &f32_bytes(&cols),
        );
    }
    let dets = det.detect_upsampled(view, upsample, adjust).unwrap();
    let rows: Vec<u8> = dets
        .iter()
        .flat_map(|d| {
            [
                d.rect.left as f64,
                d.rect.top as f64,
                d.rect.right as f64,
                d.rect.bottom as f64,
                d.confidence,
                d.weight_index as f64,
            ]
        })
        .flat_map(|v| v.to_le_bytes())
        .collect();
    write_npy(&format!("{out}/dets.npy"), "<f8", &[dets.len(), 6], &rows);
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let raw = std::fs::read(&args[1]).unwrap();
    let out = &args[2];
    let upsample = args.get(3).map_or(0, |s| s.parse().unwrap());
    let adjust = args.get(4).map_or(0.0, |s| s.parse().unwrap());
    let hdr = |i: usize| i32::from_le_bytes(raw[i * 4..i * 4 + 4].try_into().unwrap()) as usize;
    let (w, h, c) = (hdr(0), hdr(1), hdr(2));
    std::fs::create_dir_all(out).unwrap();
    if c == 1 {
        run(
            ImageView::<1>::packed(&raw[12..], w, h).unwrap(),
            out,
            upsample,
            adjust,
        );
    } else {
        run(
            ImageView::<3>::packed(&raw[12..], w, h).unwrap(),
            out,
            upsample,
            adjust,
        );
    }
}
