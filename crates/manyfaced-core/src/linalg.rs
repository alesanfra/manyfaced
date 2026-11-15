//! The small dense linear algebra dlib's geometry code relies on: `svd4` (Golub–Reinsch, via
//! `svd3` with a skinny U) and `pinv`.
//!
//! Ported from `dlib/matrix/matrix_la.h` (dlib v20.0.1, Boost Software License 1.0, see
//! `THIRD_PARTY_NOTICES`), following the non-LAPACK code path operation by operation so the
//! rounding matches dlib built without LAPACK. dlib built with LAPACK uses `gesvd` for dynamic
//! matrices (e.g. inside `pinv`), which can differ in the last bits; 2×2 fixed-size SVDs (as in
//! `find_similarity_transform`) always take this path.

/// A dense row-major `f64` matrix.
#[derive(Debug, Clone, PartialEq)]
pub struct Mat {
    pub rows: usize,
    pub cols: usize,
    pub data: Vec<f64>,
}

impl Mat {
    pub fn zeros(rows: usize, cols: usize) -> Self {
        Mat {
            rows,
            cols,
            data: vec![0.0; rows * cols],
        }
    }

    pub fn from_rows(rows: usize, cols: usize, data: Vec<f64>) -> Self {
        assert_eq!(data.len(), rows * cols, "matrix data has the wrong length");
        Mat { rows, cols, data }
    }

    #[inline]
    pub fn get(&self, r: usize, c: usize) -> f64 {
        self.data[r * self.cols + c]
    }

    #[inline]
    pub fn set(&mut self, r: usize, c: usize, v: f64) {
        self.data[r * self.cols + c] = v;
    }

    #[inline]
    fn at(&mut self, r: usize, c: usize) -> &mut f64 {
        &mut self.data[r * self.cols + c]
    }

    pub fn transpose(&self) -> Mat {
        let mut t = Mat::zeros(self.cols, self.rows);
        for r in 0..self.rows {
            for c in 0..self.cols {
                t.set(c, r, self.get(r, c));
            }
        }
        t
    }

    /// dlib's matrix product: `out(r,c) = sum_k a(r,k) * b(k,c)`, accumulated in order of `k`
    /// starting from the first product.
    pub fn mul(&self, b: &Mat) -> Mat {
        assert_eq!(self.cols, b.rows, "matrix product dimension mismatch");
        let mut out = Mat::zeros(self.rows, b.cols);
        for r in 0..self.rows {
            for c in 0..b.cols {
                let mut acc = self.get(r, 0) * b.get(0, c);
                for k in 1..self.cols {
                    acc += self.get(r, k) * b.get(k, c);
                }
                out.set(r, c, acc);
            }
        }
        out
    }
}

/// Result of [`svd`]: `a = u * diag(w) * vᵀ`, with `u` of size m×n (skinny), `w` of length n and
/// `v` of size n×n.
#[derive(Debug, Clone)]
pub struct Svd {
    pub u: Mat,
    pub w: Vec<f64>,
    pub v: Mat,
    /// 0 on success, otherwise `k` if the k-th singular value did not converge in 300
    /// iterations (as dlib's `svd4` return value; dlib's callers ignore it).
    pub status: usize,
}

/// dlib's `svd3(m, u, w, v)` for any shape: `svd4(SVD_SKINNY_U, true, m)` when `rows >= cols`,
/// otherwise the SVD of the transpose with `u`/`v` swapped and zero padding, as dlib does.
pub fn svd(a: &Mat) -> Svd {
    if a.rows >= a.cols {
        svd4_skinny(a)
    } else {
        // svd4(SVD_FULL_U, true, trans(m), v, w, u); then pad u and w with zeros.
        let t = svd4_full(&a.transpose());
        let (m, n) = (a.rows, a.cols);
        // t.u is n×n (full U of the n×m transpose), t.v is m×m, t.w has m values.
        let mut u = Mat::zeros(m, n);
        for r in 0..m {
            for c in 0..m {
                u.set(r, c, t.v.get(r, c));
            }
        }
        let mut w = t.w;
        w.resize(n, 0.0);
        Svd {
            u,
            w,
            v: t.u,
            status: t.status,
        }
    }
}

fn svd4_skinny(a: &Mat) -> Svd {
    svd4(a, false)
}

fn svd4_full(a: &Mat) -> Svd {
    svd4(a, true)
}

/// `svd4(u_mode, withv = true, a, u, q, v)` for `m >= n`, with `u_mode` either `SVD_SKINNY_U`
/// or `SVD_FULL_U`. A literal translation of dlib's Algol-derived code, gotos included (as
/// labelled loops).
// Index loops are kept to mirror the C code line by line.
#[allow(clippy::many_single_char_names, clippy::needless_range_loop)]
fn svd4(a: &Mat, full_u: bool) -> Svd {
    let m = a.rows;
    let n = a.cols;
    assert!(m >= n, "svd4 requires rows >= cols");
    let mut eps = f64::EPSILON;
    let tol = f64::MIN_POSITIVE / eps;

    let mut e = vec![0.0f64; n];
    let mut q = vec![0.0f64; n];
    let ucols = if full_u { m } else { n };
    let mut u = Mat::zeros(m, ucols);
    let mut v = Mat::zeros(n, n);
    let mut retval = 0usize;

    // Copy 'a' to 'u'.
    for i in 0..m {
        for j in 0..n {
            u.set(i, j, a.get(i, j));
        }
    }

    // Householder's reduction to bidiagonal form.
    let (mut g, mut x) = (0.0f64, 0.0f64);
    let mut l = 0usize;
    let (mut f, mut h, mut s);
    for i in 0..n {
        e[i] = g;
        s = 0.0;
        l = i + 1;
        for j in i..m {
            s += u.get(j, i) * u.get(j, i);
        }
        if s < tol {
            g = 0.0;
        } else {
            f = u.get(i, i);
            g = if f < 0.0 { s.sqrt() } else { -s.sqrt() };
            h = f * g - s;
            u.set(i, i, f - g);
            for j in l..n {
                s = 0.0;
                for k in i..m {
                    s += u.get(k, i) * u.get(k, j);
                }
                f = s / h;
                for k in i..m {
                    *u.at(k, j) += f * u.get(k, i);
                }
            }
        }
        q[i] = g;
        s = 0.0;
        for j in l..n {
            s += u.get(i, j) * u.get(i, j);
        }
        if s < tol {
            g = 0.0;
        } else {
            f = u.get(i, i + 1);
            g = if f < 0.0 { s.sqrt() } else { -s.sqrt() };
            h = f * g - s;
            u.set(i, i + 1, f - g);
            for j in l..n {
                e[j] = u.get(i, j) / h;
            }
            for j in l..m {
                s = 0.0;
                for k in l..n {
                    s += u.get(j, k) * u.get(i, k);
                }
                for k in l..n {
                    *u.at(j, k) += s * e[k];
                }
            }
        }
        let y = q[i].abs() + e[i].abs();
        if y > x {
            x = y;
        }
    }

    // Accumulation of right-hand transformations. `g` and `l` carry over from above.
    for i in (0..n).rev() {
        if g != 0.0 {
            h = u.get(i, i + 1) * g;
            for j in l..n {
                v.set(j, i, u.get(i, j) / h);
            }
            for j in l..n {
                s = 0.0;
                for k in l..n {
                    s += u.get(i, k) * v.get(k, j);
                }
                for k in l..n {
                    let vki = v.get(k, i);
                    *v.at(k, j) += s * vki;
                }
            }
        }
        for j in l..n {
            v.set(i, j, 0.0);
            v.set(j, i, 0.0);
        }
        v.set(i, i, 1.0);
        g = e[i];
        l = i;
    }

    // Accumulation of left-hand transformations.
    for i in n..u.rows {
        for j in n..u.cols {
            u.set(i, j, 0.0);
        }
        if i < u.cols {
            u.set(i, i, 1.0);
        }
    }
    for i in (0..n).rev() {
        l = i + 1;
        g = q[i];
        for j in l..u.cols {
            u.set(i, j, 0.0);
        }
        if g != 0.0 {
            h = u.get(i, i) * g;
            for j in l..u.cols {
                s = 0.0;
                for k in l..m {
                    s += u.get(k, i) * u.get(k, j);
                }
                f = s / h;
                for k in i..m {
                    let uki = u.get(k, i);
                    *u.at(k, j) += f * uki;
                }
            }
            for j in i..m {
                *u.at(j, i) /= g;
            }
        } else {
            for j in i..m {
                u.set(j, i, 0.0);
            }
        }
        *u.at(i, i) += 1.0;
    }

    // Diagonalization of the bidiagonal form.
    eps *= x;
    let (mut c, mut y, mut z);
    'k_loop: for k in (0..n).rev() {
        let mut iter = 0;
        // test_f_splitting
        loop {
            // Find l such that e[l] is negligible (convergence test) or q[l-1] is negligible
            // (cancellation). e[0] is always 0, so the search stops at l = 0 at the latest.
            let mut cancel = false;
            let mut ll = k;
            loop {
                if e[ll].abs() <= eps {
                    break;
                }
                if ll == 0 {
                    // Unreachable for finite input (e[0] == 0); dlib would read q[-1].
                    break;
                }
                if q[ll - 1].abs() <= eps {
                    cancel = true;
                    break;
                }
                ll -= 1;
            }
            l = ll;

            if cancel {
                // Cancellation of e[l] if l > 0.
                c = 0.0;
                s = 1.0;
                let l1 = l - 1;
                for i in l..=k {
                    f = s * e[i];
                    e[i] *= c;
                    if f.abs() <= eps {
                        break;
                    }
                    g = q[i];
                    h = (f * f + g * g).sqrt();
                    q[i] = h;
                    c = g / h;
                    s = -f / h;
                    for j in 0..m {
                        y = u.get(j, l1);
                        z = u.get(j, i);
                        u.set(j, l1, y * c + z * s);
                        u.set(j, i, -y * s + z * c);
                    }
                }
            }

            // test_f_convergence
            z = q[k];
            if l == k {
                break;
            }

            // Shift from bottom 2x2 minor.
            iter += 1;
            if iter > 300 {
                retval = k;
                break 'k_loop;
            }
            x = q[l];
            y = q[k - 1];
            g = e[k - 1];
            h = e[k];
            f = ((y - z) * (y + z) + (g - h) * (g + h)) / (2.0 * h * y);
            g = (f * f + 1.0).sqrt();
            f = ((x - z) * (x + z) + h * (y / (if f < 0.0 { f - g } else { f + g }) - h)) / x;

            // Next QR transformation.
            c = 1.0;
            s = 1.0;
            for i in l + 1..=k {
                g = e[i];
                y = q[i];
                h = s * g;
                g *= c;
                z = (f * f + h * h).sqrt();
                e[i - 1] = z;
                c = f / z;
                s = h / z;
                f = x * c + g * s;
                g = -x * s + g * c;
                h = y * s;
                y *= c;
                for j in 0..n {
                    x = v.get(j, i - 1);
                    z = v.get(j, i);
                    v.set(j, i - 1, x * c + z * s);
                    v.set(j, i, -x * s + z * c);
                }
                z = (f * f + h * h).sqrt();
                q[i - 1] = z;
                if z != 0.0 {
                    c = f / z;
                    s = h / z;
                }
                f = c * g + s * y;
                x = -s * g + c * y;
                for j in 0..m {
                    y = u.get(j, i - 1);
                    z = u.get(j, i);
                    u.set(j, i - 1, y * c + z * s);
                    u.set(j, i, -y * s + z * c);
                }
            }
            e[l] = 0.0;
            e[k] = f;
            q[k] = x;
        }

        // convergence: make q[k] non-negative.
        if z < 0.0 {
            q[k] = -z;
            for j in 0..n {
                let vjk = v.get(j, k);
                v.set(j, k, -vjk);
            }
        }
    }

    Svd {
        u,
        w: q,
        v,
        status: retval,
    }
}

/// dlib's `pinv(m)` with `tol = 0`: `v * diag(1 / round_zeros(w, eps)) * uᵀ` from the SVD of `m`
/// (or of its transpose when it has more columns than rows), with
/// `eps = machine_eps * max(rows, cols) * max(w)`.
pub fn pinv(m: &Mat) -> Mat {
    if m.cols > m.rows {
        pinv_helper(&m.transpose()).transpose()
    } else {
        pinv_helper(m)
    }
}

#[allow(clippy::needless_range_loop)]
fn pinv_helper(m: &Mat) -> Mat {
    let Svd { u, w, v, .. } = svd(m);
    let max_w = w.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let eps = f64::EPSILON * (m.rows.max(m.cols) as f64) * max_w;
    // reciprocal(round_zeros(w, eps)): values with |w| < eps become 0, and 1/0 stays 0.
    let inv_w: Vec<f64> = w
        .iter()
        .map(|&x| {
            let x = if x.abs() >= eps { x } else { 0.0 };
            if x != 0.0 { 1.0 / x } else { 0.0 }
        })
        .collect();
    // tmp(scale_columns(v, inv_w)) * trans(u)
    let mut scaled = v.clone();
    for r in 0..scaled.rows {
        for c in 0..scaled.cols {
            let val = scaled.get(r, c) * inv_w[c];
            scaled.set(r, c, val);
        }
    }
    scaled.mul(&u.transpose())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn assert_close(a: &Mat, b: &Mat, tol: f64) {
        assert_eq!((a.rows, a.cols), (b.rows, b.cols));
        for (x, y) in a.data.iter().zip(&b.data) {
            assert!((x - y).abs() <= tol, "{a:?} vs {b:?}");
        }
    }

    fn reconstruct(s: &Svd) -> Mat {
        let mut uw = s.u.clone();
        for r in 0..uw.rows {
            for (c, &w) in s.w.iter().enumerate().take(uw.cols) {
                let val = uw.get(r, c) * w;
                uw.set(r, c, val);
            }
        }
        uw.mul(&s.v.transpose())
    }

    #[test]
    fn svd_reconstructs_various_shapes() {
        let cases = [
            Mat::from_rows(2, 2, vec![3.0, 1.0, -2.0, 4.0]),
            Mat::from_rows(2, 2, vec![0.0, 0.0, 0.0, 0.0]),
            Mat::from_rows(2, 2, vec![1.0, 2.0, 2.0, 4.0]),
            Mat::from_rows(3, 3, vec![10.0, 0.0, 1.0, 20.0, 0.0, 1.0, 20.0, 30.0, 1.0]),
            Mat::from_rows(
                4,
                3,
                vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 0.5, -1.0, 2.0],
            ),
            Mat::from_rows(2, 3, vec![1.0, 0.0, 2.0, -1.0, 3.0, 1.0]),
        ];
        for a in cases {
            let s = svd(&a);
            assert_eq!(s.status, 0);
            assert!(s.w.iter().all(|&w| w >= 0.0));
            assert_close(&reconstruct(&s), &a, 1e-12);
            // v is orthogonal.
            let vtv = s.v.transpose().mul(&s.v);
            let mut id = Mat::zeros(s.v.cols, s.v.cols);
            for i in 0..id.rows {
                id.set(i, i, 1.0);
            }
            assert_close(&vtv, &id, 1e-12);
        }
    }

    #[test]
    fn pinv_of_invertible_matrix_is_its_inverse() {
        let a = Mat::from_rows(3, 3, vec![2.0, 0.0, 1.0, 1.0, 3.0, 0.0, 0.0, 1.0, 4.0]);
        let p = pinv(&a);
        let mut id = Mat::zeros(3, 3);
        for i in 0..3 {
            id.set(i, i, 1.0);
        }
        assert_close(&a.mul(&p), &id, 1e-14);
    }

    #[test]
    fn pinv_of_wide_and_singular_matrices() {
        // Wide: a * pinv(a) = I (full row rank).
        let a = Mat::from_rows(2, 3, vec![1.0, 2.0, 3.0, 0.0, 1.0, 4.0]);
        let p = pinv(&a);
        assert_eq!((p.rows, p.cols), (3, 2));
        let mut id = Mat::zeros(2, 2);
        id.set(0, 0, 1.0);
        id.set(1, 1, 1.0);
        assert_close(&a.mul(&p), &id, 1e-13);
        // Rank 1: Moore-Penrose condition a * p * a = a.
        let a = Mat::from_rows(2, 2, vec![1.0, 2.0, 2.0, 4.0]);
        let p = pinv(&a);
        assert_close(&a.mul(&p).mul(&a), &a, 1e-13);
    }
}
