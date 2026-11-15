//! `point_transform_affine`, `find_affine_transform` and `find_similarity_transform`, ported from
//! `dlib/geometry/point_transforms.h` with dlib's order of floating point operations.

use super::DPoint;
use crate::linalg::{Mat, pinv, svd};

/// `p -> m * p + b`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PointTransformAffine {
    /// Row-major 2×2 matrix.
    pub m: [[f64; 2]; 2],
    pub b: [f64; 2],
}

impl Default for PointTransformAffine {
    /// The identity.
    fn default() -> Self {
        PointTransformAffine {
            m: [[1.0, 0.0], [0.0, 1.0]],
            b: [0.0, 0.0],
        }
    }
}

impl PointTransformAffine {
    pub fn new(m: [[f64; 2]; 2], b: [f64; 2]) -> Self {
        PointTransformAffine { m, b }
    }

    /// `m * p + b`, evaluated as dlib does: `(m(r,0) * x + m(r,1) * y) + b(r)`.
    #[inline]
    pub fn apply(&self, p: DPoint) -> DPoint {
        DPoint::new(
            (self.m[0][0] * p.x + self.m[0][1] * p.y) + self.b[0],
            (self.m[1][0] * p.x + self.m[1][1] * p.y) + self.b[1],
        )
    }

    /// dlib's `inv(point_transform_affine)`: `(inv(m), -inv(m) * b)`, with dlib's closed-form
    /// 2×2 inverse (`inv_helper<EXP,2>`), which returns the identity for a singular matrix.
    pub fn inverse(&self) -> PointTransformAffine {
        let [[a, b], [c, d]] = self.m;
        let det = a * d - b * c;
        let im = if det != 0.0 {
            let d_inv = 1.0 / det;
            [[d * d_inv, b * -d_inv], [c * -d_inv, a * d_inv]]
        } else {
            [[1.0, 0.0], [0.0, 1.0]]
        };
        // (-im) * b; negation is exact, so this equals -(im * b).
        let nb = [
            -(im[0][0] * self.b[0] + im[0][1] * self.b[1]),
            -(im[1][0] * self.b[0] + im[1][1] * self.b[1]),
        ];
        PointTransformAffine { m: im, b: nb }
    }
}

/// dlib's `find_affine_transform`: the least squares affine map `Q * pinv(P)` with
/// `P = [from; 1]` (3×N) and `Q = to` (2×N). Requires at least 3 point pairs.
pub fn find_affine_transform(from: &[DPoint], to: &[DPoint]) -> PointTransformAffine {
    assert!(
        from.len() == to.len() && from.len() >= 3,
        "find_affine_transform needs >= 3 point pairs"
    );
    let n = from.len();
    let mut p = Mat::zeros(3, n);
    let mut q = Mat::zeros(2, n);
    for (i, (f, t)) in from.iter().zip(to).enumerate() {
        p.set(0, i, f.x);
        p.set(1, i, f.y);
        p.set(2, i, 1.0);
        q.set(0, i, t.x);
        q.set(1, i, t.y);
    }
    let m = q.mul(&pinv(&p));
    PointTransformAffine::new(
        [[m.get(0, 0), m.get(0, 1)], [m.get(1, 0), m.get(1, 1)]],
        [m.get(0, 2), m.get(1, 2)],
    )
}

fn det2(m: &Mat) -> f64 {
    // matrix_la.h det() for 2x2: m(0,0)*m(1,1) - m(0,1)*m(1,0).
    m.get(0, 0) * m.get(1, 1) - m.get(0, 1) * m.get(1, 0)
}

/// dlib's `find_similarity_transform` (Umeyama 1991, equations 34–43): the least squares
/// rotation + uniform scale + translation mapping `from` onto `to`. Requires at least 2 pairs.
pub fn find_similarity_transform(from: &[DPoint], to: &[DPoint]) -> PointTransformAffine {
    assert!(
        from.len() == to.len() && from.len() >= 2,
        "find_similarity_transform needs >= 2 point pairs"
    );
    let n = from.len() as f64;
    let (mut mean_from, mut mean_to) = (DPoint::default(), DPoint::default());
    for (f, t) in from.iter().zip(to) {
        mean_from.x += f.x;
        mean_from.y += f.y;
        mean_to.x += t.x;
        mean_to.y += t.y;
    }
    mean_from = DPoint::new(mean_from.x / n, mean_from.y / n);
    mean_to = DPoint::new(mean_to.x / n, mean_to.y / n);

    let (mut sigma_from, mut sigma_to) = (0.0f64, 0.0f64);
    let mut cov = [[0.0f64; 2]; 2];
    for (f, t) in from.iter().zip(to) {
        let df = *f - mean_from;
        let dt = *t - mean_to;
        sigma_from += df.length_squared();
        sigma_to += dt.length_squared();
        // cov += (to - mean_to) * trans(from - mean_from)
        cov[0][0] += dt.x * df.x;
        cov[0][1] += dt.x * df.y;
        cov[1][0] += dt.y * df.x;
        cov[1][1] += dt.y * df.y;
    }
    sigma_from /= n;
    // sigma_to is computed by dlib but not used in the result.
    let _ = sigma_to / n;
    // `cov /= n` on a dlib matrix is `cov * (1.0 / n)` (matrix.h, operator/ on matrix_exp).
    let inv_n = 1.0 / n;
    let cov = Mat::from_rows(
        2,
        2,
        vec![
            cov[0][0] * inv_n,
            cov[0][1] * inv_n,
            cov[1][0] * inv_n,
            cov[1][1] * inv_n,
        ],
    );

    let s_ = svd(&cov);
    let (u, v) = (s_.u, s_.v);
    let d = [s_.w[0], s_.w[1]];
    let mut s = [1.0f64, 1.0f64]; // diagonal of s = identity_matrix(cov)
    let det_cov = det2(&cov);
    if det_cov < 0.0 || (det_cov == 0.0 && det2(&u) * det2(&v) < 0.0) {
        if d[1] < d[0] {
            s[1] = -1.0;
        } else {
            s[0] = -1.0;
        }
    }

    // r = u * s * trans(v), evaluated as (u * s) * trans(v) with dlib's full products.
    let s_mat = Mat::from_rows(2, 2, vec![s[0], 0.0, 0.0, s[1]]);
    let r = u.mul(&s_mat).mul(&v.transpose());
    let mut c = 1.0f64;
    if sigma_from != 0.0 {
        // trace(d * s) with d = diagm(w).
        let d_mat = Mat::from_rows(2, 2, vec![d[0], 0.0, 0.0, d[1]]);
        let ds = d_mat.mul(&s_mat);
        c = 1.0 / sigma_from * (ds.get(0, 0) + ds.get(1, 1));
    }
    // dlib moves scalar factors outside matrix products (matrix.h, "percolate up"), so
    // `c * r * mean_from` is `(r * mean_from) * c`, and `c * r` is `r(i,j) * c`.
    let cr = [
        [r.get(0, 0) * c, r.get(0, 1) * c],
        [r.get(1, 0) * c, r.get(1, 1) * c],
    ];
    let r_mean = [
        r.get(0, 0) * mean_from.x + r.get(0, 1) * mean_from.y,
        r.get(1, 0) * mean_from.x + r.get(1, 1) * mean_from.y,
    ];
    let t = [mean_to.x - r_mean[0] * c, mean_to.y - r_mean[1] * c];
    PointTransformAffine::new(cr, t)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close(a: DPoint, b: DPoint) -> bool {
        (a.x - b.x).abs() < 1e-9 && (a.y - b.y).abs() < 1e-9
    }

    #[test]
    fn affine_recovers_an_exact_map() {
        let from = [
            DPoint::new(0.0, 0.0),
            DPoint::new(1.0, 0.0),
            DPoint::new(1.0, 1.0),
        ];
        let to = [
            DPoint::new(10.0, 20.0),
            DPoint::new(29.0, 20.0),
            DPoint::new(29.0, 59.0),
        ];
        let t = find_affine_transform(&from, &to);
        for (f, g) in from.iter().zip(&to) {
            assert!(close(t.apply(*f), *g), "{:?}", t.apply(*f));
        }
        let inv = t.inverse();
        assert!(close(
            inv.apply(DPoint::new(29.0, 59.0)),
            DPoint::new(1.0, 1.0)
        ));
    }

    #[test]
    fn similarity_recovers_rotation_scale_translation() {
        let (angle, scale) = (0.3f64, 2.5);
        let (s, c) = angle.sin_cos();
        let map = |p: DPoint| {
            DPoint::new(
                scale * (c * p.x - s * p.y) + 7.0,
                scale * (s * p.x + c * p.y) - 3.0,
            )
        };
        let from = [
            DPoint::new(0.0, 0.0),
            DPoint::new(1.0, 0.2),
            DPoint::new(0.3, 0.9),
            DPoint::new(-0.5, 0.4),
        ];
        let to: Vec<_> = from.iter().map(|&p| map(p)).collect();
        let t = find_similarity_transform(&from, &to);
        for (f, g) in from.iter().zip(&to) {
            assert!(close(t.apply(*f), *g));
        }
    }

    #[test]
    fn similarity_handles_reflection_and_degenerate_input() {
        // A mirrored point set cannot be matched by a rotation: dlib flips the smaller axis.
        let from = [
            DPoint::new(0.0, 0.0),
            DPoint::new(1.0, 0.0),
            DPoint::new(0.0, 1.0),
        ];
        let to = [
            DPoint::new(0.0, 0.0),
            DPoint::new(-1.0, 0.0),
            DPoint::new(0.0, 1.0),
        ];
        let t = find_similarity_transform(&from, &to);
        let det = t.m[0][0] * t.m[1][1] - t.m[0][1] * t.m[1][0];
        assert!(det >= 0.0);
        // All points identical: sigma_from = 0, the scale stays 1.
        let same = [DPoint::new(2.0, 3.0); 3];
        let t = find_similarity_transform(&same, &same);
        assert!(t.m.iter().flatten().all(|v| v.is_finite()));
    }
}
