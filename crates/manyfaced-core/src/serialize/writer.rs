//! Test-only encoder that mirrors dlib's serializers (`ser_helper::pack_int`,
//! `float_details::convert_from_T`), used to build synthetic streams for the reader tests.

use super::TensorShape;

#[derive(Default)]
pub(crate) struct Writer {
    pub buf: Vec<u8>,
}

/// `std::frexp` for finite nonzero `v`: v = f * 2^e with 0.5 <= |f| < 1.
fn frexp(v: f64) -> (f64, i32) {
    let bits = v.to_bits();
    let biased = ((bits >> 52) & 0x7ff) as i32;
    if biased == 0 {
        // Subnormal: normalize first.
        let (f, e) = frexp(v * 2f64.powi(64));
        return (f, e - 64);
    }
    let f = f64::from_bits((bits & !(0x7ff << 52)) | (1022 << 52));
    (f, biased - 1022)
}

impl Writer {
    fn int(&mut self, negative: bool, mut magnitude: u64) {
        let mut bytes = Vec::new();
        loop {
            bytes.push((magnitude & 0xff) as u8);
            magnitude >>= 8;
            if magnitude == 0 {
                break;
            }
        }
        self.buf
            .push(bytes.len() as u8 | if negative { 0x80 } else { 0 });
        self.buf.extend(bytes);
    }

    pub fn i64(&mut self, v: i64) {
        self.int(v < 0, v.unsigned_abs());
    }

    pub fn u64(&mut self, v: u64) {
        self.int(false, v);
    }

    pub fn bool(&mut self, v: bool) {
        self.buf.push(if v { b'1' } else { b'0' });
    }

    fn float_details(&mut self, v: f64, digits: i32) {
        let (mantissa, exponent) = if v == f64::INFINITY {
            (0, 32000)
        } else if v == f64::NEG_INFINITY {
            (0, 32001)
        } else if v.is_nan() {
            (0, 32002)
        } else if v == 0.0 {
            (0, 0)
        } else {
            let (f, e) = frexp(v);
            let mut m = (f * 2f64.powi(digits)) as i64;
            let mut exp = e - digits;
            for _ in 0..8 {
                if m & 0xff != 0 {
                    break;
                }
                m >>= 8;
                exp += 8;
            }
            (m, exp as i64)
        };
        self.i64(mantissa);
        self.i64(exponent);
    }

    pub fn f64(&mut self, v: f64) {
        self.float_details(v, 53);
    }

    pub fn f32(&mut self, v: f32) {
        self.float_details(f64::from(v), 24);
    }

    pub fn string(&mut self, s: &str) {
        self.u64(s.len() as u64);
        self.buf.extend_from_slice(s.as_bytes());
    }

    pub fn matrix_f32(&mut self, rows: usize, cols: usize, data: &[f32]) {
        self.i64(-(rows as i64));
        self.i64(-(cols as i64));
        for &v in data {
            self.f32(v);
        }
    }

    pub fn resizable_tensor(&mut self, shape: TensorShape, data: &[f32]) {
        self.i64(2);
        for d in shape {
            self.i64(d as i64);
        }
        for v in data {
            self.buf.extend_from_slice(&v.to_le_bytes());
        }
    }

    pub fn alias_tensor(&mut self, shape: TensorShape) {
        self.i64(1);
        for d in shape {
            self.i64(d as i64);
        }
    }
}
