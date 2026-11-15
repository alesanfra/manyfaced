//! `float_details` → `float`/`double`, ported from `float_details::convert_to_T`
//! (`dlib/float_details.h`).

const IS_INF: i16 = 32000;
const IS_NINF: i16 = 32001;

/// `ldexp((float)mantissa, exponent)`, with dlib's special exponents for inf and NaN.
pub(super) fn to_f32(mantissa: i64, exponent: i16) -> f32 {
    match exponent {
        e if e < IS_INF => ldexp_f32(mantissa as f32, i32::from(e)),
        IS_INF => f32::INFINITY,
        IS_NINF => f32::NEG_INFINITY,
        _ => f32::NAN,
    }
}

/// `ldexp((double)mantissa, exponent)`, with dlib's special exponents for inf and NaN.
pub(super) fn to_f64(mantissa: i64, exponent: i16) -> f64 {
    match exponent {
        e if e < IS_INF => ldexp_f64(mantissa as f64, i32::from(e)),
        IS_INF => f64::INFINITY,
        IS_NINF => f64::NEG_INFINITY,
        _ => f64::NAN,
    }
}

/// Exact power of two for `e` in the normal `f64` exponent range.
fn pow2(e: i32) -> f64 {
    debug_assert!((-1022..=1023).contains(&e));
    f64::from_bits(((e + 1023) as u64) << 52)
}

/// Correctly rounded `x * 2^e` for `x = (float)i64`, like C `ldexpf`.
fn ldexp_f32(x: f32, e: i32) -> f32 {
    if x == 0.0 {
        return x;
    }
    // |x| is in [1, 2^63] with a 24-bit significand, so x * 2^e is exact in f64 for
    // e in [-1000, 1000] and the final cast is the only rounding, as in ldexpf. Outside that
    // range the result is 0 or inf in f32 either way.
    let e = e.clamp(-1000, 1000);
    (f64::from(x) * pow2(e)) as f32
}

/// Correctly rounded `x * 2^e` for a normal `x`, like C `ldexp`, including the single
/// round-half-to-even step when the result is subnormal.
fn ldexp_f64(x: f64, e: i32) -> f64 {
    if x == 0.0 || !x.is_finite() {
        return x;
    }
    let bits = x.to_bits();
    let sign = bits & (1 << 63);
    let biased = ((bits >> 52) & 0x7ff) as i32;
    // x comes from a nonzero i64, so it is normal.
    debug_assert!(biased != 0);
    let new_biased = biased + e;
    if new_biased >= 0x7ff {
        return f64::from_bits(sign | f64::INFINITY.to_bits());
    }
    let frac = bits & ((1 << 52) - 1);
    if new_biased >= 1 {
        return f64::from_bits(sign | ((new_biased as u64) << 52) | frac);
    }
    // Subnormal result: shift the full 53-bit significand right and round half to even.
    let sig = frac | (1 << 52);
    let shift = (1 - new_biased) as u32;
    if shift > 53 {
        return f64::from_bits(sign);
    }
    let mut q = sig >> shift;
    let rem = sig & ((1 << shift) - 1);
    let half = 1 << (shift - 1);
    if rem > half || (rem == half && q & 1 == 1) {
        q += 1;
    }
    // A carry into bit 52 correctly produces the smallest normal number.
    f64::from_bits(sign | q)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ldexp_f64_normal_and_overflow() {
        assert_eq!(ldexp_f64(3.0, 4), 48.0);
        assert_eq!(ldexp_f64(-3.0, -2), -0.75);
        assert_eq!(ldexp_f64(1.0, 1023), f64::MAX / (2.0 - f64::EPSILON));
        assert_eq!(ldexp_f64(1.0, 1024), f64::INFINITY);
        assert_eq!(ldexp_f64(-1.0, 30000), f64::NEG_INFINITY);
    }

    #[test]
    fn ldexp_f64_subnormal_rounding() {
        let min = f64::from_bits(1);
        assert_eq!(ldexp_f64(1.0, -1074), min);
        // 1.5 * 2^-1074 is a tie: rounds to even (2 * min).
        assert_eq!(ldexp_f64(3.0, -1075), 2.0 * min);
        // 0.5 * min is a tie that rounds to even (0).
        assert_eq!(ldexp_f64(1.0, -1075), 0.0);
        // Slightly above the tie rounds up.
        assert_eq!(ldexp_f64(1.0 + f64::EPSILON, -1075), min);
        assert_eq!(ldexp_f64(-1.0, -1200), -0.0);
        assert!(ldexp_f64(-1.0, -1200).is_sign_negative());
    }

    #[test]
    fn ldexp_f32_matches_reference_cases() {
        assert_eq!(ldexp_f32(1.0, -149), f32::from_bits(1));
        assert_eq!(ldexp_f32(1.0, -150), 0.0);
        assert_eq!(ldexp_f32(3.0, -150), f32::from_bits(2));
        assert_eq!(ldexp_f32(1.0, 128), f32::INFINITY);
        assert_eq!(ldexp_f32(5.0, 3), 40.0);
        assert_eq!(ldexp_f32(1.0, -30000), 0.0);
    }
}
