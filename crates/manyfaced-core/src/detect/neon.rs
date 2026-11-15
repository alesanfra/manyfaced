//! Software emulation of the arithmetic dlib's `simd4f` uses on ARM NEON.
//!
//! On NEON, dlib's `simd4f` division and square root (`dlib/simd/simd4f.h`) are not IEEE
//! operations: they start from the `FRECPE` / `FRSQRTE` estimate instructions and refine them
//! with two `FRECPS` / `FRSQRTS` Newton–Raphson steps. The reference outputs come from such a
//! build (`dlib-bin` on arm64), so the FHOG code reproduces these steps. The estimates follow the
//! `FPRecipEstimate` / `FPRSqrtEstimate` pseudocode of the Arm Architecture Reference Manual
//! (default FPCR: no flush-to-zero, round to nearest) and are checked against the hardware
//! instructions in the tests when running on aarch64.

/// `RecipEstimate(a)` for `a` in 256..512: `round(2^19 / (a + 0.5))` in 256..512.
const fn recip_estimate(a: u32) -> u32 {
    let a = a * 2 + 1;
    let b = (1 << 19) / a;
    // (b + 1) DIV 2 in the pseudocode.
    b.div_ceil(2)
}

/// `RecipSqrtEstimate(a)` for `a` in 128..512.
const fn recip_sqrt_estimate(a: u32) -> u32 {
    let a = if a < 256 {
        a * 2 + 1
    } else {
        ((a >> 1) << 1) * 2 + 2
    };
    let mut b: u64 = 512;
    while (a as u64) * (b + 1) * (b + 1) < (1 << 28) {
        b += 1;
    }
    // (b + 1) DIV 2 in the pseudocode.
    b.div_ceil(2) as u32
}

const fn build_recip_table() -> [u16; 256] {
    let mut t = [0u16; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = recip_estimate(256 + i as u32) as u16;
        i += 1;
    }
    t
}

const fn build_rsqrt_table() -> [u16; 384] {
    let mut t = [0u16; 384];
    let mut i = 0;
    while i < 384 {
        t[i] = recip_sqrt_estimate(128 + i as u32) as u16;
        i += 1;
    }
    t
}

static RECIP_TABLE: [u16; 256] = build_recip_table();
static RSQRT_TABLE: [u16; 384] = build_rsqrt_table();

/// `FRECPE` (single precision).
pub fn frecpe(x: f32) -> f32 {
    let bits = x.to_bits();
    let sign = bits & 0x8000_0000;
    if x.is_nan() {
        return f32::NAN;
    }
    if x.is_infinite() {
        return f32::from_bits(sign);
    }
    if x == 0.0 {
        return f32::from_bits(sign | 0x7f80_0000);
    }
    if x.abs() < f32::from_bits(0x0020_0000) {
        // |x| < 2^-128: the reciprocal overflows (round to nearest gives infinity).
        return f32::from_bits(sign | 0x7f80_0000);
    }
    let mut exp = ((bits >> 23) & 0xff) as i32;
    // 52-bit fraction as in the pseudocode (23 bits followed by 29 zeros).
    let mut fraction = u64::from(bits & 0x7f_ffff) << 29;
    if exp == 0 {
        if fraction & (1 << 51) == 0 {
            exp = -1;
            fraction = (fraction & ((1 << 50) - 1)) << 2;
        } else {
            fraction = (fraction & ((1 << 51) - 1)) << 1;
        }
    }
    let scaled = 256 + ((fraction >> 44) & 0xff) as usize;
    let mut result_exp = 253 - exp;
    let estimate = u64::from(RECIP_TABLE[scaled - 256]);
    let mut fraction = (estimate & 0xff) << 44;
    if result_exp == 0 {
        fraction = (1 << 51) | (fraction >> 1);
    } else if result_exp == -1 {
        fraction = (1 << 50) | (fraction >> 2);
        result_exp = 0;
    }
    f32::from_bits(
        sign | ((result_exp as u32 & 0xff) << 23) | ((fraction >> 29) as u32 & 0x7f_ffff),
    )
}

/// `FRSQRTE` (single precision).
pub fn frsqrte(x: f32) -> f32 {
    let bits = x.to_bits();
    if x.is_nan() {
        return f32::NAN;
    }
    if x == 0.0 {
        return f32::from_bits((bits & 0x8000_0000) | 0x7f80_0000);
    }
    if x < 0.0 {
        return f32::NAN;
    }
    if x.is_infinite() {
        return 0.0;
    }
    let mut exp = ((bits >> 23) & 0xff) as i32;
    let mut fraction = u64::from(bits & 0x7f_ffff) << 29;
    if exp == 0 {
        while fraction & (1 << 51) == 0 {
            fraction = (fraction & ((1 << 51) - 1)) << 1;
            exp -= 1;
        }
        fraction = (fraction & ((1 << 51) - 1)) << 1;
    }
    let scaled = if exp & 1 == 0 {
        256 + ((fraction >> 44) & 0xff) as usize
    } else {
        128 + ((fraction >> 45) & 0x7f) as usize
    };
    let result_exp = (380 - exp) / 2;
    let estimate = u32::from(RSQRT_TABLE[scaled - 128]);
    f32::from_bits(((result_exp as u32 & 0xff) << 23) | ((estimate & 0xff) << 15))
}

/// `FRECPS`: `2 - a * b` with a single rounding (`inf * 0` gives 2).
#[inline]
fn frecps(a: f32, b: f32) -> f32 {
    if (a.is_infinite() && b == 0.0) || (a == 0.0 && b.is_infinite()) {
        return 2.0;
    }
    (-a).mul_add(b, 2.0)
}

/// `FRSQRTS`: `(3 - a * b) / 2` with a single rounding (`inf * 0` gives 1.5).
#[inline]
fn frsqrts(a: f32, b: f32) -> f32 {
    if (a.is_infinite() && b == 0.0) || (a == 0.0 && b.is_infinite()) {
        return 1.5;
    }
    let r = (-a).mul_add(b, 3.0);
    // Halving is exact unless the result is subnormal, which cannot happen for the finite,
    // positive operands dlib produces here.
    r * 0.5
}

/// dlib's NEON `simd4f` division `lhs / rhs`: two Newton–Raphson refinements of `FRECPE`.
#[inline]
pub fn div(lhs: f32, rhs: f32) -> f32 {
    let mut r = frecpe(rhs);
    r *= frecps(rhs, r);
    r *= frecps(rhs, r);
    lhs * r
}

/// The reciprocal dlib's NEON division multiplies by.
#[inline]
pub fn reciprocal(rhs: f32) -> f32 {
    let mut r = frecpe(rhs);
    r *= frecps(rhs, r);
    r *= frecps(rhs, r);
    r
}

/// dlib's NEON `simd4f` `sqrt`: two Newton–Raphson refinements of `FRSQRTE`, multiplied back by
/// the input; `sqrt(0) = 0`.
#[inline]
pub fn sqrt(x: f32) -> f32 {
    let q0 = frsqrte(x);
    let p0 = x * q0;
    let r0 = frsqrts(p0, q0);
    let q1 = q0 * r0;
    let p1 = x * q1;
    let r1 = frsqrts(p1, q1);
    let q2 = q1 * r1;
    let res = x * q2;
    if x == 0.0 { x } else { res }
}

#[cfg(test)]
mod tests {
    use super::*;

    // Only the hardware comparison (aarch64) needs the samples.
    #[cfg(target_arch = "aarch64")]
    fn sample_bits() -> impl Iterator<Item = u32> {
        // Every exponent with a spread of mantissas, plus every value near 1 and the special
        // values.
        (0u32..256)
            .flat_map(|e| (0u32..(1 << 23)).step_by(4099).map(move |m| (e << 23) | m))
            .chain(0x3f00_0000..0x3f00_4000)
            .chain([
                0,
                1,
                0x7f80_0000,
                0x7f7f_ffff,
                0x0080_0000,
                0x0020_0000,
                0x001f_ffff,
            ])
    }

    #[test]
    fn known_values() {
        // 1/8 and 1/sqrt(1): estimates are within 2^-8 of the true values.
        assert!((frecpe(8.0) - 0.125).abs() < 0.125 / 256.0);
        assert!((frsqrte(1.0) - 1.0).abs() < 1.0 / 256.0);
        assert_eq!(sqrt(0.0), 0.0);
        assert!((sqrt(2.0) - std::f32::consts::SQRT_2).abs() <= 2.0 * f32::EPSILON);
        assert!((div(1.0, 3.0) - 1.0 / 3.0).abs() <= f32::EPSILON);
    }

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn estimates_match_the_hardware() {
        use std::arch::aarch64::*;
        for bits in sample_bits() {
            for x in [f32::from_bits(bits), -f32::from_bits(bits)] {
                // SAFETY: NEON is part of the aarch64 baseline.
                let (hw_re, hw_rs, hw_sqrt, hw_div) = unsafe {
                    let v = vdupq_n_f32(x);
                    let re = vgetq_lane_f32::<0>(vrecpeq_f32(v));
                    let rs = vgetq_lane_f32::<0>(vrsqrteq_f32(v));
                    // dlib's sqrt and division sequences.
                    let q0 = vrsqrteq_f32(v);
                    let q1 = vmulq_f32(q0, vrsqrtsq_f32(vmulq_f32(v, q0), q0));
                    let q2 = vmulq_f32(q1, vrsqrtsq_f32(vmulq_f32(v, q1), q1));
                    let s = vmulq_f32(v, q2);
                    let s = vbslq_f32(vceqq_f32(vdupq_n_f32(0.0), v), v, s);
                    let mut r = vrecpeq_f32(v);
                    r = vmulq_f32(vrecpsq_f32(v, r), r);
                    r = vmulq_f32(vrecpsq_f32(v, r), r);
                    let d = vmulq_f32(vdupq_n_f32(0.1), r);
                    (re, rs, vgetq_lane_f32::<0>(s), vgetq_lane_f32::<0>(d))
                };
                let same =
                    |a: f32, b: f32| a.to_bits() == b.to_bits() || (a.is_nan() && b.is_nan());
                assert!(
                    same(frecpe(x), hw_re),
                    "frecpe({x:e}) = {:e}, hw {hw_re:e}",
                    frecpe(x)
                );
                assert!(
                    same(frsqrte(x), hw_rs),
                    "frsqrte({x:e}) = {:e}, hw {hw_rs:e}",
                    frsqrte(x)
                );
                if x >= 0.0 {
                    assert!(
                        same(sqrt(x), hw_sqrt),
                        "sqrt({x:e}) = {:e}, hw {hw_sqrt:e}",
                        sqrt(x)
                    );
                }
                if x.is_finite() && x != 0.0 {
                    assert!(
                        same(div(0.1, x), hw_div),
                        "0.1/{x:e} = {:e}, hw {hw_div:e}",
                        div(0.1, x)
                    );
                }
            }
        }
    }
}
