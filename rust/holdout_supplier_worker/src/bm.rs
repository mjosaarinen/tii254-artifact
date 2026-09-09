//! Block Berlekamp--Massey over F2, in the order-basis formulation.
//!
//! Given a sequence of `b x b` matrices `A_0 .. A_{L-1}` over F2, we want a
//! matrix generator `F(x) = sum_j F_j x^j` with
//!
//! Equivalently, a minimal left order basis of `H(x) = [A(x); -I]` to order `L`.
//! We maintain a `2b x 2b` basis `Pi` together with its residual `R = Pi * H`,
//! consume one order per step, and keep the nominal degrees `delta` that make
//! the basis minimal.  This is Coppersmith's iteration; it is quadratic in `L`
//! and is here to be *correct*, not fast -- the recursive PM-basis ("lingen")
//! variant is what a target-scale run needs.
//!
//! # Contract and limits
//!
//! What this returns is a **single left row-relation**, not a `b x b` matrix
//! generator: one row `P_j` per degree with
//!
//! ```text
//! sum_j P_j A_{i+j} = 0,   0 <= i < L - m.
//! ```
//!
//! Block Wiedemann ultimately needs the full right generator `sum_j A_{i+j}F_j`;
//! since `Y` and `Z` are independently random the projected terms are not
//! symmetric, so the two orientations are genuinely different and this is not a
//! drop-in substitute.  It is a validated reference for the scalar case only.
//!
//! A basis row carries `2b` bits in one `u64`, so `b <= 32` here.  Target width
//! `b = 512` needs 1024-bit basis rows.

/// Sequence of `b x b` matrices, row-major, one `u64` per row (`b <= 64`).
pub struct Seq {
    pub b: usize,
    pub terms: Vec<Vec<u64>>,
}

/// A left row-relation: `coeffs[j]` is the `b`-bit row vector `P_j`.
pub struct LeftRelation {
    pub b: usize,
    pub coeffs: Vec<u64>,
}

#[inline]
fn mat_mul(b: usize, x: &[u64], y: &[u64]) -> Vec<u64> {
    // (x * y)[i] = XOR over k with x[i]_k set of y[k]
    let mut out = vec![0u64; b];
    for i in 0..b {
        let mut r = 0u64;
        let mut m = x[i];
        while m != 0 {
            let k = m.trailing_zeros() as usize;
            r ^= y[k];
            m &= m - 1;
        }
        out[i] = r;
    }
    out
}

/// Coppersmith's block Berlekamp--Massey.
///
/// Returns a generator of degree at most `L/2 + b`, or `None` if the iteration
/// failed to produce one within the supplied sequence.
/// The finished order basis: `pi[deg][row]` carries `2b` bits, `delta` the
/// nominal degrees.
pub struct OrderBasis {
    pub b: usize,
    pub pi: Vec<Vec<u64>>,
    pub delta: Vec<usize>,
}

/// Per-row reversal length `m = max(deg P, deg Q + 1)` and the left block
/// coefficients `P_0..P_degP`.  `None` if the row's left block is zero.
pub fn row_relation(ob: &OrderBasis, row: usize) -> Option<(usize, Vec<u64>)> {
    let b = ob.b;
    let deg = ob.pi.len() - 1;
    let bmask = if b == 64 { !0u64 } else { (1u64 << b) - 1 };
    let deg_p = (0..=deg).rev().find(|&dg| ob.pi[dg][row] & bmask != 0)?;
    let deg_q = (0..=deg)
        .rev()
        .find(|&dg| (ob.pi[dg][row] >> b) & bmask != 0);
    let m = match deg_q {
        Some(q) => deg_p.max(q + 1),
        None => deg_p,
    };
    let p: Vec<u64> = (0..=deg).map(|dg| ob.pi[dg][row] & bmask).collect();
    Some((m, p))
}

pub fn order_basis(seq: &Seq) -> OrderBasis {
    let b = seq.b;
    let l = seq.terms.len();
    assert!(
        b <= 32,
        "a Pi row holds 2b bits in one u64, so this reference path needs b <= 32 \
         (b = 512 needs 1024-bit basis rows, not merely wider matrix rows)"
    );
    let n = 2 * b;

    // Pi is n x n over F2[x]; store coefficient-major: pi[deg][row] : u64 (n bits)
    let mut pi: Vec<Vec<u64>> = vec![vec![0u64; n]];
    for i in 0..n {
        pi[0][i] = 1u64 << i;
    }
    // Residual R = Pi * H truncated to order L, where H = [A ; -I] is n x b.
    // Row i of H at degree t is A_t row i for i < b, and (i == b + j) gives -I,
    // i.e. the identity at degree 0 only.
    let mut r: Vec<Vec<u64>> = Vec::with_capacity(l);
    for t in 0..l {
        let mut ct = vec![0u64; n];
        for i in 0..b {
            ct[i] = seq.terms[t][i];
        }
        if t == 0 {
            for j in 0..b {
                ct[b + j] = 1u64 << j;
            }
        }
        r.push(ct);
    }

    // Shift semantics, ported from rust/blrt_xl_worker/src/main.rs:2141:
    //     initial_shifts[right_width..].fill(1)
    // The top (relation) block starts at shift 0 and the numerator block at 1.
    // Minimal-degree pivoting then prefers top-block rows, which is what makes
    // the nominal degree a genuine shifted degree rather than a pivot counter.
    // With a flat zero shift the two blocks compete equally, delta degenerates,
    // and the boundary exponent e_r = delta_r - d_r is inert.
    let mut delta: Vec<usize> = vec![0; n];
    delta[b..].fill(1);

    for t in 0..l {
        // discrepancy is coefficient t of R
        let dt = r[t].clone();
        // order rows by increasing nominal degree so pivots are minimal-degree
        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&i| delta[i]);

        let mut pivot_of_col: Vec<Option<usize>> = vec![None; b];
        let mut cur = dt;
        let bmask = if b == 64 { !0u64 } else { (1u64 << b) - 1 };
        let mut rows_pivot: Vec<usize> = Vec::new();
        for &i in &order {
            // Keep eliminating until this row's discrepancy is zero or it
            // becomes the pivot for its own leading column.  Doing a single
            // pass leaves a nonzero discrepancy behind and breaks the order.
            loop {
                let masked = cur[i] & bmask;
                if masked == 0 {
                    break;
                }
                let col = masked.trailing_zeros() as usize;
                match pivot_of_col[col] {
                    None => {
                        pivot_of_col[col] = Some(i);
                        rows_pivot.push(i);
                        break;
                    }
                    Some(p) => {
                        if p == i {
                            break;
                        }
                        for dg in 0..pi.len() {
                            pi[dg][i] ^= pi[dg][p];
                        }
                        for tt in t..l {
                            r[tt][i] ^= r[tt][p];
                        }
                        cur[i] ^= cur[p];
                    }
                }
            }
        }
        // multiply every pivot row by x: shifts both Pi and R up one degree
        if !rows_pivot.is_empty() {
            pi.push(vec![0u64; n]);
            let top = pi.len() - 1;
            for &p in &rows_pivot {
                for dg in (1..=top).rev() {
                    pi[dg][p] = pi[dg - 1][p];
                }
                pi[0][p] = 0;
                for tt in (t + 1..l).rev() {
                    r[tt][p] = r[tt - 1][p];
                }
                r[t][p] = 0;
                delta[p] += 1;
            }
        }
    }

    OrderBasis { b, pi, delta }
}

/// Minimal-degree single left row-relation (the validated scalar contract).
pub fn block_bm(seq: &Seq) -> Option<LeftRelation> {
    let b = seq.b;
    let ob = order_basis(seq);
    let n = 2 * b;
    let mut best: Option<(usize, usize)> = None;
    for i in 0..n {
        if row_relation(&ob, i).is_none() {
            continue;
        }
        match best {
            None => best = Some((ob.delta[i], i)),
            Some((dd, _)) if ob.delta[i] < dd => best = Some((ob.delta[i], i)),
            _ => {}
        }
    }
    let (_, row) = best?;
    let (m, p) = row_relation(&ob, row)?;
    // The relation is certified only for i in [0, L - m).  If m >= L that window
    // is empty and the order basis guarantees nothing usable, so refuse rather
    // than hand back a vacuous generator.
    if m >= seq.terms.len() {
        return None;
    }
    let coeffs: Vec<u64> = (0..=m).map(|j| *p.get(m - j).unwrap_or(&0)).collect();
    Some(LeftRelation { b, coeffs })
}

/// Reference relation of **exact** degree `degree` by direct linear solve:
/// row vectors `P_0..P_D` with `sum_j P_j A_{i+j} = 0` for every admissible `i`
/// **and** `P_D != 0`.  Without the exact-degree requirement the solve happily
/// returns a padded lower-degree relation that is valid on the degree-`D`
/// window but not on the shorter window its true degree implies.
///
/// This is Gaussian elimination on `(D+1)*b` unknowns -- polynomial time, not
/// exponential.  It is the oracle only because it is transparently correct.
pub fn reference_relation(seq: &Seq, degree: usize) -> Option<LeftRelation> {
    let b = seq.b;
    let l = seq.terms.len();
    if l <= degree {
        return None;
    }
    let nunk = (degree + 1) * b;
    assert!(nunk <= 128, "reference solve is for small blocks only");
    let mut rows: Vec<u128> = Vec::new();
    for i in 0..(l - degree) {
        for c in 0..b {
            let mut eq: u128 = 0;
            for j in 0..=degree {
                let a = &seq.terms[i + j];
                for k in 0..b {
                    if a[k] >> c & 1 == 1 {
                        eq ^= 1u128 << (j * b + k);
                    }
                }
            }
            if eq != 0 {
                rows.push(eq);
            }
        }
    }
    let mut piv: Vec<(usize, u128)> = Vec::new();
    for mut e in rows {
        for (pp, pr) in &piv {
            if e >> *pp & 1 == 1 {
                e ^= *pr;
            }
        }
        if e != 0 {
            let lead = e.trailing_zeros() as usize;
            piv.push((lead, e));
        }
    }
    piv.sort_by_key(|x| x.0);
    let pivset: std::collections::HashSet<usize> = piv.iter().map(|x| x.0).collect();
    let top_lo = degree * b;

    // Walk every null-space basis vector and keep one with a nonzero top block.
    for free in (0..nunk).filter(|q| !pivset.contains(q)) {
        let mut sol: u128 = 1u128 << free;
        for (pp, pr) in piv.iter().rev() {
            let rest = *pr ^ (1u128 << *pp);
            if (rest & sol).count_ones() & 1 == 1 {
                sol |= 1u128 << *pp;
            } else {
                sol &= !(1u128 << *pp);
            }
        }
        let top = (sol >> top_lo) & ((1u128 << b) - 1);
        if top == 0 {
            continue;
        }
        let mut coeffs: Vec<u64> = Vec::with_capacity(degree + 1);
        for j in 0..=degree {
            let mut row = 0u64;
            for k in 0..b {
                if sol >> (j * b + k) & 1 == 1 {
                    row |= 1u64 << k;
                }
            }
            coeffs.push(row);
        }
        let cand = LeftRelation { b, coeffs };
        if verify_generator(seq, &cand) {
            return Some(cand);
        }
    }
    None
}

/// Smallest `D` for which an exact-degree-`D` verified relation exists.
pub fn minimal_left_relation(seq: &Seq, max_degree: usize) -> Option<(usize, LeftRelation)> {
    for d in 0..=max_degree.min(seq.terms.len().saturating_sub(1)) {
        if (d + 1) * seq.b > 128 {
            break;
        }
        if let Some(r) = reference_relation(seq, d) {
            return Some((d, r));
        }
    }
    None
}

/// Check the left generator relation `sum_j P_j A_{i+j} = 0` for every
/// admissible `i`, where `P_j = coeffs[j]` is a `b`-bit row vector.
pub fn verify_generator(seq: &Seq, g: &LeftRelation) -> bool {
    let l = seq.terms.len();
    let dg = g.coeffs.len() - 1;
    if l <= dg {
        return false;
    }
    if g.coeffs.iter().all(|&c| c == 0) {
        return false;
    }
    for i in 0..(l - dg) {
        let mut acc = 0u64;
        for j in 0..=dg {
            // row vector P_j times matrix A_{i+j}
            let mut mrow = g.coeffs[j];
            let a = &seq.terms[i + j];
            while mrow != 0 {
                let k = mrow.trailing_zeros() as usize;
                acc ^= a[k];
                mrow &= mrow - 1;
            }
        }
        if acc != 0 {
            return false;
        }
    }
    true
}

#[cfg(test)]
pub mod tests_support {
    use super::*;
    pub struct Rng(pub u64);
    impl Rng {
        pub fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }
    pub fn sequence(n: usize, b: usize, terms: usize, seed: u64) -> Seq {
        let mut rng = Rng(seed);
        let mask = if n == 64 { !0u64 } else { (1u64 << n) - 1 };
        let bm: Vec<u64> = (0..n).map(|_| rng.next() & mask).collect();
        let ymask = (1u64 << b) - 1;
        let y: Vec<u64> = (0..n).map(|_| rng.next() & ymask).collect();
        let z: Vec<u64> = (0..n).map(|_| rng.next() & ymask).collect();
        let mut v = y.clone();
        let mut out = Vec::with_capacity(terms);
        for _ in 0..terms {
            let mut a = vec![0u64; b];
            for k in 0..n {
                let (zk, vk) = (z[k], v[k]);
                let mut mm = zk;
                while mm != 0 {
                    let r = mm.trailing_zeros() as usize;
                    a[r] ^= vk;
                    mm &= mm - 1;
                }
            }
            out.push(a);
            let mut nv = vec![0u64; n];
            for i in 0..n {
                let mut acc = 0u64;
                let mut mm = bm[i];
                while mm != 0 {
                    let k = mm.trailing_zeros() as usize;
                    acc ^= v[k];
                    mm &= mm - 1;
                }
                nv[i] = acc;
            }
            v = nv;
        }
        Seq { b, terms: out }
    }
}

#[cfg(test)]
mod tests {
    use super::tests_support::sequence;
    use super::*;

    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    #[test]
    fn generator_annihilates_the_sequence() {
        let (mut found, mut total) = (0, 0);
        for seed in 1..=60u64 {
            for b in 2..=6usize {
                let n = 8 + (seed as usize * 3 + b) % 24;
                let l = 2 * n / b + 4 * b + 8;
                let s = sequence(n, b, l, seed.wrapping_mul(0x9e3779b97f4a7c15) ^ b as u64);
                total += 1;
                match block_bm(&s) {
                    Some(g) => {
                        assert!(
                            verify_generator(&s, &g),
                            "non-generator at seed {seed} b={b} n={n}: deg {}",
                            g.coeffs.len() - 1
                        );
                        found += 1;
                    }
                    None => {}
                }
            }
        }
        assert!(
            found * 100 >= total * 95,
            "block_bm produced a generator only {found}/{total} times"
        );
        eprintln!("block_bm: {found}/{total} sequences generated and verified");
    }

    #[test]
    fn degree_equals_the_minimal_oracle_degree() {
        let mut compared = 0;
        for seed in 1..=15u64 {
            let (n, b) = (10usize, 2usize);
            let l = 2 * n / b + 4 * b + 8;
            let s = sequence(n, b, l, seed.wrapping_mul(0x2545f4914f6cdd1d));
            let oracle = minimal_left_relation(&s, n / b + 2 * b);
            let got = block_bm(&s);
            let (od, _) = oracle.expect("oracle found no relation at all");
            let g = got.expect("block_bm returned nothing where a relation exists");
            assert!(
                verify_generator(&s, &g),
                "seed {seed}: block_bm relation invalid"
            );
            assert_eq!(
                g.coeffs.len() - 1,
                od,
                "seed {seed}: block_bm degree {} != minimal {od}",
                g.coeffs.len() - 1
            );
            compared += 1;
        }
        assert_eq!(compared, 15);
    }
}

#[cfg(test)]
mod exhaustive {
    use super::*;

    /// Every b=2 sequence of length 2..=4.  If block_bm returns a generator it
    /// must satisfy its own verifier; anything else is a self-inconsistency.
    #[test]
    fn small_sequences_are_self_consistent() {
        let b = 2usize;
        let mut returned = 0u64;
        let mut invalid = 0u64;
        for l in 2..=4usize {
            let total: u64 = 1 << (4 * l); // 4 bits per 2x2 term
            for code in 0..total {
                let mut terms = Vec::with_capacity(l);
                for t in 0..l {
                    let nib = (code >> (4 * t)) & 0xf;
                    terms.push(vec![(nib & 0b11) as u64, ((nib >> 2) & 0b11) as u64]);
                }
                let s = Seq { b, terms };
                if let Some(g) = block_bm(&s) {
                    returned += 1;
                    if !verify_generator(&s, &g) {
                        invalid += 1;
                    }
                }
            }
        }
        eprintln!("exhaustive b=2 L=2..4: {returned} generators returned, {invalid} invalid");
        assert_eq!(
            invalid, 0,
            "block_bm returned {invalid} generators its own verifier rejects"
        );
    }
}

#[cfg(test)]
mod debug {
    use super::tests_support::*;
    use super::*;

    #[test]
    #[ignore]
    fn diagnose() {
        let (n, b) = (12usize, 3usize);
        let l = 2 * n / b + 2 * b + 6;
        let s = sequence(n, b, l, 0x9e3779b9);
        eprintln!("n={n} b={b} l={l}");
        match block_bm(&s) {
            None => eprintln!("block_bm: None"),
            Some(g) => {
                eprintln!(
                    "block_bm deg={} coeffs={:?}",
                    g.coeffs.len() - 1,
                    g.coeffs.clone()
                );
                eprintln!("verify: {}", verify_generator(&s, &g));
                let dg = g.coeffs.len() - 1;
                for i in 0..(l - dg) {
                    let mut acc = 0u64;
                    for j in 0..=dg {
                        let mut m = g.coeffs[j];
                        while m != 0 {
                            let k = m.trailing_zeros() as usize;
                            acc ^= s.terms[i + j][k];
                            m &= m - 1;
                        }
                    }
                    if acc != 0 {
                        eprintln!("  first failure at i={i}: {acc:#x}");
                        break;
                    }
                }
            }
        }
        for d in 1..=(n / b + b).min(l - 1) {
            if let Some(r) = reference_relation(&s, d) {
                if verify_generator(&s, &r) {
                    eprintln!("oracle deg={d} coeffs={:?}", r.coeffs.clone());
                    break;
                }
            }
        }
    }
}

/// A right-oriented finite-sequence recurrence: `coeffs[j]` is the `b x b`
/// matrix `F_j`, one `u64` per row, satisfying
///
/// ```text
/// sum_j A_{i+j} F_j = 0,   0 <= i < L - m.
/// ```
///
/// # This is NOT yet a validated kernel generator
///
/// With `a_i = Z^T B^(i+1) Y` the recurrence gives only `Z^T B^i (B W) = 0` for
/// `W = sum_j B^j Y F_j`.  It does **not** establish `B W = 0`, and empirically
/// it usually fails to.  Survey over 400 symmetric singular `B = M^T M` cases
/// (`reconstruction::reconstruction_survey`, run with `--ignored`):
///
/// ```text
/// offset 0: 400 verified | row-reduced 299 | full-rank kernel   2 | BW != 0 54 | deficient 344
/// offset 1: 400 verified | row-reduced 350 | full-rank kernel  29 | BW != 0 66 | deficient 305
/// ```
///
/// Two necessary corrections are already applied: the observed sequence starts
/// at `B Y` (as `rust/blrt_xl_worker` does), and per-row padding is appended
/// past each `m_r` rather than prepended -- prepending replaces `W_r` by
/// `B^(M - m_r) W_r`, which annihilates exactly the vectors wanted.  Neither is
/// sufficient.  Row-reducedness is likewise necessary but not binding: of the
/// 350 row-reduced bases only 29 gave a full-rank kernel block.
///
/// What is still missing is the per-column boundary exponent of Coppersmith's
/// reconstruction -- candidates are not simply `sum_j B^j Y F_j[:,r]` for every
/// column -- together with a degree-bound argument that forces `B W = 0`.
pub struct MatrixGenerator {
    pub b: usize,
    pub coeffs: Vec<Vec<u64>>,
    /// per-selected-row provenance, in output-column order
    pub rows: Vec<RowInfo>,
}

/// What a selected order-basis row contributes.  `reversal_length` is its own
/// `m_r`; zero padding beyond it is appended, never prepended, so the
/// reconstruction `W_r = sum_j B^j Y F_j[:,r]` is unchanged by padding.
#[derive(Clone, Copy, Debug)]
pub struct RowInfo {
    pub basis_row: usize,
    /// deg P_r: highest nonzero left-block coefficient
    pub true_degree: usize,
    /// delta_r maintained by the iteration
    pub nominal_degree: usize,
    /// m_r = max(deg P_r, deg Q_r + 1)
    pub reversal_length: usize,
}

fn transpose_square(b: usize, m: &[u64]) -> Vec<u64> {
    let mut out = vec![0u64; b];
    for (r, &row) in m.iter().enumerate().take(b) {
        let mut x = row;
        while x != 0 {
            let c = x.trailing_zeros() as usize;
            out[c] |= 1u64 << r;
            x &= x - 1;
        }
    }
    out
}

/// Transposed sequence `A_i^T`.
pub fn transpose_seq(seq: &Seq) -> Seq {
    Seq {
        b: seq.b,
        terms: seq
            .terms
            .iter()
            .map(|t| transpose_square(seq.b, t))
            .collect(),
    }
}

/// Right matrix generator, via `sum_j P_j A^T_{i+j} = 0  <=>  sum_j A_{i+j} P_j^T = 0`.
///
/// The order basis is run on the transposed sequence; the `b` minimal-degree
/// rows are stacked into `P_j` and each coefficient transposed.  Every column of
/// `F` inherits its own row's relation, so the matrix identity holds column by
/// column.  All rows are padded to a common reversal length
/// `m = max_r max(deg P_r, deg Q_r + 1)`, which only prepends zero coefficients
/// and so preserves validity from `i = 0`.
pub fn right_generator(seq: &Seq) -> Option<MatrixGenerator> {
    let b = seq.b;
    let tseq = transpose_seq(seq);
    let ob = order_basis(&tseq);
    let n = 2 * b;

    let mut cand: Vec<(usize, usize, usize, Vec<u64>)> = Vec::new(); // (delta, row, m, P)
    for r in 0..n {
        if let Some((m, p)) = row_relation(&ob, r) {
            cand.push((ob.delta[r], r, m, p));
        }
    }
    if cand.len() < b {
        return None;
    }
    cand.sort_by_key(|x| (x.0, x.1));
    cand.truncate(b);
    let m = cand.iter().map(|x| x.2).max()?;
    // Padding every selected row to the common reversal length is only sound
    // while the window [0, L - m) is non-empty.
    if m >= seq.terms.len() {
        return None;
    }

    let mut coeffs = Vec::with_capacity(m + 1);
    for j in 0..=m {
        // Each row uses ITS OWN reversal length m_r; padding is appended past
        // m_r, never prepended.  Prepending would replace the candidate
        // W_r by B^(m - m_r) W_r, which annihilates it exactly when
        // W_r already lies in ker B -- i.e. precisely on the vectors wanted.
        let mut pj = vec![0u64; b];
        for (rr, c) in cand.iter().enumerate() {
            let m_r = c.2;
            if j <= m_r {
                pj[rr] = *c.3.get(m_r - j).unwrap_or(&0);
            }
        }
        coeffs.push(transpose_square(b, &pj));
    }
    let rows = cand
        .iter()
        .map(|c| RowInfo {
            basis_row: c.1,
            true_degree: c.3.iter().rposition(|&w| w != 0).unwrap_or(0),
            nominal_degree: c.0,
            reversal_length: c.2,
        })
        .collect();
    Some(MatrixGenerator { b, coeffs, rows })
}

/// Check `sum_j A_{i+j} F_j = 0` for every admissible `i`.
pub fn verify_right_generator(seq: &Seq, g: &MatrixGenerator) -> bool {
    let b = seq.b;
    let l = seq.terms.len();
    let dg = g.coeffs.len() - 1;
    if l <= dg {
        return false;
    }
    if g.coeffs.iter().all(|c| c.iter().all(|&w| w == 0)) {
        return false;
    }
    for i in 0..(l - dg) {
        let mut acc = vec![0u64; b];
        for j in 0..=dg {
            let p = mat_mul(b, &seq.terms[i + j], &g.coeffs[j]);
            for k in 0..b {
                acc[k] ^= p[k];
            }
        }
        if acc.iter().any(|&w| w != 0) {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod right_tests {
    use super::tests_support::sequence;
    use super::*;

    #[test]
    fn right_generator_annihilates_the_sequence() {
        let (mut found, mut total) = (0, 0);
        for seed in 1..=40u64 {
            for b in 2..=5usize {
                let n = 8 + (seed as usize * 3 + b) % 20;
                let l = 2 * n / b + 4 * b + 8;
                let s = sequence(n, b, l, seed.wrapping_mul(0x9e3779b97f4a7c15) ^ b as u64);
                total += 1;
                if let Some(g) = right_generator(&s) {
                    assert!(
                        verify_right_generator(&s, &g),
                        "seed {seed} b={b} n={n}: right generator invalid, deg {}",
                        g.coeffs.len() - 1
                    );
                    found += 1;
                }
            }
        }
        eprintln!("right_generator: {found}/{total} sequences generated and verified");
        assert!(found * 100 >= total * 90, "only {found}/{total}");
    }

    /// The gate that caught the trailing-zero bug, applied to the new path.
    #[test]
    fn exhaustive_small_sequences_are_self_consistent() {
        let b = 2usize;
        let (mut returned, mut invalid) = (0u64, 0u64);
        for l in 2..=4usize {
            for code in 0..(1u64 << (4 * l)) {
                let terms: Vec<Vec<u64>> = (0..l)
                    .map(|t| {
                        let nib = (code >> (4 * t)) & 0xf;
                        vec![(nib & 0b11) as u64, ((nib >> 2) & 0b11) as u64]
                    })
                    .collect();
                let s = Seq { b, terms };
                if let Some(g) = right_generator(&s) {
                    returned += 1;
                    if !verify_right_generator(&s, &g) {
                        invalid += 1;
                    }
                }
            }
        }
        eprintln!("right exhaustive b=2 L=2..4: {returned} returned, {invalid} invalid");
        if invalid > 0 {
            // re-run and dump the first offender
            'outer: for l in 2..=4usize {
                for code in 0..(1u64 << (4 * l)) {
                    let terms: Vec<Vec<u64>> = (0..l)
                        .map(|t| {
                            let nib = (code >> (4 * t)) & 0xf;
                            vec![(nib & 0b11) as u64, ((nib >> 2) & 0b11) as u64]
                        })
                        .collect();
                    let s = Seq {
                        b,
                        terms: terms.clone(),
                    };
                    if let Some(g) = right_generator(&s) {
                        if !verify_right_generator(&s, &g) {
                            eprintln!("  offender L={l} code={code:#x} terms={terms:?}");
                            eprintln!("  F deg={} coeffs={:?}", g.coeffs.len() - 1, g.coeffs);
                            let ts = transpose_seq(&s);
                            let ob = order_basis(&ts);
                            eprintln!("  basis deg={} delta={:?}", ob.pi.len() - 1, ob.delta);
                            for r in 0..2 * b {
                                eprintln!("    row {r}: {:?}", row_relation(&ob, r));
                            }
                            break 'outer;
                        }
                    }
                }
            }
        }
        assert_eq!(invalid, 0);
    }

    /// Left and right paths must agree on the transposed sequence.
    #[test]
    fn agrees_with_the_left_path_under_transposition() {
        for seed in 1..=20u64 {
            for b in 2..=4usize {
                let n = 10 + (seed as usize) % 8;
                let l = 2 * n / b + 4 * b + 8;
                let s = sequence(n, b, l, seed.wrapping_mul(0x2545f4914f6cdd1d) ^ b as u64);
                let ts = transpose_seq(&s);
                if let (Some(lr), Some(rg)) = (block_bm(&ts), right_generator(&s)) {
                    assert!(verify_generator(&ts, &lr));
                    assert!(verify_right_generator(&s, &rg));
                    assert!(
                        rg.coeffs.len() >= lr.coeffs.len(),
                        "seed {seed} b={b}: right deg {} below minimal left deg {}",
                        rg.coeffs.len() - 1,
                        lr.coeffs.len() - 1
                    );
                }
            }
        }
    }
}

#[cfg(test)]
mod reconstruction {
    use super::tests_support::Rng;
    use super::*;

    /// `x` is `rows` x `b`, `f` is `b` x `b`; result is `rows` x `b`.
    fn mul_nb(rows: usize, x: &[u64], f: &[u64]) -> Vec<u64> {
        let mut out = vec![0u64; rows];
        for r in 0..rows {
            let mut acc = 0u64;
            let mut m = x[r];
            while m != 0 {
                let k = m.trailing_zeros() as usize;
                acc ^= f[k];
                m &= m - 1;
            }
            out[r] = acc;
        }
        out
    }

    fn apply_sq(n: usize, bmat: &[u64], v: &[u64]) -> Vec<u64> {
        let mut out = vec![0u64; n];
        for i in 0..n {
            let mut acc = 0u64;
            let mut m = bmat[i];
            while m != 0 {
                let k = m.trailing_zeros() as usize;
                acc ^= v[k];
                m &= m - 1;
            }
            out[i] = acc;
        }
        out
    }

    fn f2_rank(rows: &[u64]) -> usize {
        let mut piv: Vec<u64> = Vec::new();
        for &x in rows {
            let mut cur = x;
            for p in &piv {
                let h = p.leading_zeros();
                let _ = h;
                if cur ^ p < cur {
                    cur ^= p;
                }
            }
            if cur != 0 {
                piv.push(cur);
                piv.sort_unstable_by(|a, b| b.cmp(a));
            }
        }
        // recompute properly with leading-bit pivots
        let mut lead: std::collections::HashMap<u32, u64> = std::collections::HashMap::new();
        let mut r = 0;
        for &x in rows {
            let mut cur = x;
            while cur != 0 {
                let h = 63 - cur.leading_zeros();
                match lead.get(&h) {
                    Some(&p) => cur ^= p,
                    None => {
                        lead.insert(h, cur);
                        r += 1;
                        break;
                    }
                }
            }
        }
        r
    }

    struct Case {
        n: usize,
        b: usize,
        mmat: Vec<u64>,
        bmat: Vec<u64>,
        y: Vec<u64>,
        z: Vec<u64>,
    }

    fn make_case(n: usize, b: usize, seed: u64) -> Case {
        let mut rng = Rng(seed | 1);
        let mask = if n == 64 { !0u64 } else { (1u64 << n) - 1 };
        let m: Vec<u64> = (0..n).map(|_| rng.next() & mask).collect();
        // B = M^T M : symmetric and generically singular over F2
        let mut mt = vec![0u64; n];
        for (r, &row) in m.iter().enumerate() {
            let mut x = row;
            while x != 0 {
                let c = x.trailing_zeros() as usize;
                mt[c] |= 1u64 << r;
                x &= x - 1;
            }
        }
        let bmat = mul_nb(n, &mt, &m);
        let mmat = m.clone();
        let ymask = (1u64 << b) - 1;
        let y: Vec<u64> = (0..n).map(|_| rng.next() & ymask).collect();
        let z: Vec<u64> = (0..n).map(|_| rng.next() & ymask).collect();
        Case {
            n,
            b,
            mmat,
            bmat,
            y,
            z,
        }
    }

    /// Sequence a_i = Z^T B^(i+offset) Y.
    fn seq_of(c: &Case, terms: usize, offset: usize) -> Seq {
        let mut v = c.y.clone();
        for _ in 0..offset {
            v = apply_sq(c.n, &c.bmat, &v);
        }
        let mut out = Vec::with_capacity(terms);
        for _ in 0..terms {
            let mut a = vec![0u64; c.b];
            for k in 0..c.n {
                let mut mm = c.z[k];
                while mm != 0 {
                    let r = mm.trailing_zeros() as usize;
                    a[r] ^= v[k];
                    mm &= mm - 1;
                }
            }
            out.push(a);
            v = apply_sq(c.n, &c.bmat, &v);
        }
        Seq { b: c.b, terms: out }
    }

    /// Co94 reconstruction for one approximant row:
    ///   raw = sum_j B^j (Y * relation_j^T),  then candidate = B^(e_r) * raw.
    /// `Y * c^T` is an n-bit column whose k-th bit is parity(Y[k] & c).
    fn reconstruct_row(c: &Case, r: &ApproximantRow, shift: usize) -> u64 {
        let mut acc = 0u64; // n-bit column
        let mut vj = c.y.clone(); // B^j Y as n x b
        for &cj in &r.relation {
            for k in 0..c.n {
                if (vj[k] & cj).count_ones() & 1 == 1 {
                    acc ^= 1u64 << k;
                }
            }
            vj = {
                // B * (B^j Y), column block
                let mut nv = vec![0u64; c.n];
                for i in 0..c.n {
                    let mut a = 0u64;
                    let mut m = c.bmat[i];
                    while m != 0 {
                        let k = m.trailing_zeros() as usize;
                        a ^= vj[k];
                        m &= m - 1;
                    }
                    nv[i] = a;
                }
                nv
            };
        }
        // apply B^shift
        for _ in 0..shift {
            let mut nv = 0u64;
            for i in 0..c.n {
                let mut par = 0u32;
                let mut m = c.bmat[i] & acc;
                while m != 0 {
                    par ^= 1;
                    m &= m - 1;
                }
                if par == 1 {
                    nv |= 1u64 << i;
                }
            }
            acc = nv;
        }
        acc
    }

    fn apply_col(c: &Case, v: u64) -> u64 {
        let mut out = 0u64;
        for i in 0..c.n {
            if (c.bmat[i] & v).count_ones() & 1 == 1 {
                out |= 1u64 << i;
            }
        }
        out
    }

    /// Which power of B to apply after the raw candidate.
    #[derive(Clone, Copy)]
    enum Shift {
        None,
        EMinus1,
        E,
        EPlus1,
    }
    fn shift_of(r: &ApproximantRow, s: Shift) -> usize {
        match s {
            Shift::None => 0,
            Shift::EMinus1 => r.boundary_exponent.saturating_sub(1),
            Shift::E => r.boundary_exponent,
            Shift::EPlus1 => r.boundary_exponent + 1,
        }
    }

    fn survey_grid(offset: usize, sh: Shift) -> (usize, usize, usize) {
        let (mut valid, mut kern, mut full) = (0, 0, 0);
        for seed in 1..=400u64 {
            let b = 2 + (seed as usize % 3);
            let n = 10 + (seed as usize * 7) % 14;
            let c = make_case(n, b, seed.wrapping_mul(0x9e3779b97f4a7c15));
            let l = 2 * n / b + 4 * b + 8;
            let s = seq_of(&c, l, offset);
            let ts = transpose_seq(&s);
            let mut verified: Vec<u64> = Vec::new();
            for r in &approximant_rows(&ts) {
                if !verify_homogeneous_from_zero(&ts, r) {
                    continue;
                }
                valid += 1;
                let w = reconstruct_row(&c, r, shift_of(r, sh));
                if w != 0 && apply_col(&c, w) == 0 {
                    kern += 1;
                    verified.push(w);
                }
            }
            if f2_rank(&verified) >= b {
                full += 1;
            }
        }
        (valid, kern, full)
    }

    /// Is `rank >= b` even attainable?  Compare what was recovered against the
    /// exact nullities of `B` and `M`, and separate genuine `M`-kernel vectors
    /// from `B`-kernel vectors that are false for the original problem.
    #[test]
    #[ignore]
    fn kernel_availability() {
        let (mut cases, mut sum_nb, mut sum_nm, mut sum_rec, mut sum_mok, mut sum_false) =
            (0usize, 0usize, 0usize, 0usize, 0usize, 0usize);
        let (mut hit_b, mut hit_m, mut b_ge) = (0usize, 0usize, 0usize);
        for seed in 1..=400u64 {
            let b = 2 + (seed as usize % 3);
            let n = 10 + (seed as usize * 7) % 14;
            let c = make_case(n, b, seed.wrapping_mul(0x9e3779b97f4a7c15));
            let l = 2 * n / b + 4 * b + 8;
            let ts = transpose_seq(&seq_of(&c, l, 1));
            // exact nullities
            let null_b = n - f2_rank(&c.bmat);
            let null_m = n - f2_rank(&c.mmat);
            let mut recovered: Vec<u64> = Vec::new();
            let mut mkernel: Vec<u64> = Vec::new();
            let mut falsepos = 0usize;
            for r in &approximant_rows(&ts) {
                let w = reconstruct_row(&c, r, r.boundary_exponent);
                if w == 0 {
                    continue;
                }
                if apply_col(&c, w) != 0 {
                    continue;
                } // B w != 0
                recovered.push(w);
                // decisive residual: M w
                let mut mw = 0u64;
                for i in 0..c.n {
                    if (c.mmat[i] & w).count_ones() & 1 == 1 {
                        mw |= 1u64 << i;
                    }
                }
                if mw == 0 {
                    mkernel.push(w);
                } else {
                    falsepos += 1;
                }
            }
            let rb = f2_rank(&recovered);
            let rm = f2_rank(&mkernel);
            cases += 1;
            sum_nb += null_b;
            sum_nm += null_m;
            sum_rec += rb;
            sum_mok += rm;
            sum_false += falsepos;
            if rb >= null_b && null_b > 0 {
                hit_b += 1;
            }
            if rm >= null_m && null_m > 0 {
                hit_m += 1;
            }
            if null_b >= b {
                b_ge += 1;
            }
        }
        eprintln!("cases {cases}");
        eprintln!(
            "  mean nullity   B {:.2}   M {:.2}",
            sum_nb as f64 / cases as f64,
            sum_nm as f64 / cases as f64
        );
        eprintln!(
            "  mean recovered rank: B-kernel {:.2}   M-kernel {:.2}",
            sum_rec as f64 / cases as f64,
            sum_mok as f64 / cases as f64
        );
        eprintln!("  cases attaining FULL ker B: {hit_b}   FULL ker M: {hit_m}");
        eprintln!(
            "  cases where nullity(B) >= b at all: {b_ge} / {cases}   <- ceiling on 'rank >= b'"
        );
        eprintln!("  false B-kernel vectors (Bw=0 but Mw!=0): {sum_false}");
    }

    #[test]
    #[ignore]
    fn boundary_exponent_distribution() {
        for off in [0usize, 1, 2] {
            let mut hist = std::collections::BTreeMap::new();
            let mut delta_eq_true = 0usize;
            let mut rows = 0usize;
            let mut shift_diverges = 0usize;
            let mut sd_gt = 0usize;
            for seed in 1..=400u64 {
                let b = 2 + (seed as usize % 3);
                let n = 10 + (seed as usize * 7) % 14;
                let c = make_case(n, b, seed.wrapping_mul(0x9e3779b97f4a7c15));
                let l = 2 * n / b + 4 * b + 8;
                let ts = transpose_seq(&seq_of(&c, l, off));
                for r in &approximant_rows(&ts) {
                    rows += 1;
                    *hist.entry(r.boundary_exponent).or_insert(0usize) += 1;
                    if r.nominal_degree == r.true_degree {
                        delta_eq_true += 1;
                    }
                    if r.nominal_degree != r.basis_shifted_degree {
                        shift_diverges += 1;
                    }
                    if r.basis_shifted_degree > r.nominal_degree {
                        sd_gt += 1;
                    }
                }
            }
            eprintln!("offset {off}: rows {rows}, e histogram {hist:?}");
            eprintln!("          nominal degree == true degree: {delta_eq_true}/{rows}");
            eprintln!(
                "          running shift != basis shifted degree: {shift_diverges}/{rows} ({:.1}%), of which sd>shift: {sd_gt}",
                100.0 * shift_diverges as f64 / rows as f64
            );
        }
    }

    #[test]
    #[ignore]
    fn shift_grid() {
        eprintln!(
            "{:>7} {:>9} {:>7} {:>8} {:>9}",
            "offset", "shift", "valid", "B*w==0", "rank>=b"
        );
        for off in [0usize, 1, 2] {
            for (name, sh) in [
                ("0", Shift::None),
                ("e-1", Shift::EMinus1),
                ("e", Shift::E),
                ("e+1", Shift::EPlus1),
            ] {
                let (v, k, f) = survey_grid(off, sh);
                eprintln!("{off:>7} {name:>9} {v:>7} {k:>8} {f:>9}");
            }
        }
    }

    fn survey(offset: usize) -> (usize, usize, usize, usize, usize) {
        let (mut cases, mut rows_total, mut rows_valid, mut kernel_rows, mut fullrank) =
            (0, 0, 0, 0, 0);
        for seed in 1..=400u64 {
            let b = 2 + (seed as usize % 3);
            let n = 10 + (seed as usize * 7) % 14;
            let c = make_case(n, b, seed.wrapping_mul(0x9e3779b97f4a7c15));
            let l = 2 * n / b + 4 * b + 8;
            let s = seq_of(&c, l, offset);
            let ts = transpose_seq(&s);
            let rows = approximant_rows(&ts);
            if rows.is_empty() {
                continue;
            }
            cases += 1;
            let mut verified: Vec<u64> = Vec::new();
            for r in &rows {
                rows_total += 1;
                if !verify_homogeneous_from_zero(&ts, r) {
                    continue;
                }
                rows_valid += 1;
                let w = reconstruct_row(&c, r, r.boundary_exponent);
                if w != 0 && apply_col(&c, w) == 0 {
                    kernel_rows += 1;
                    verified.push(w);
                }
            }
            if f2_rank(&verified) >= b {
                fullrank += 1;
            }
        }
        (cases, rows_total, rows_valid, kernel_rows, fullrank)
    }

    /// RETIRED: `rank >= b` is not a sound success statistic for these random
    /// square matrices.  Only 70 of 400 cases have nullity(B) >= b at all, so
    /// the metric had a 17.5% ceiling and said nothing about recovery quality.
    /// `kernel_availability` compares recovered rank against the exact nullities
    /// of `B` and `M` instead, and separates genuine `M`-kernel vectors from
    /// `B`-kernel vectors that are false for the original problem.
    #[test]
    #[ignore]
    fn reconstruction_survey_unsound_metric() {
        for off in [0usize, 1] {
            let (cases, rows, valid, kern, full) = survey(off);
            eprintln!(
                "offset {off}: {cases} cases | rows {rows} | approximant-valid {valid} | B*w == 0 {kern} | rank>=b {full}"
            );
        }
    }
}

// ---------------------------------------------------------------------------
// Co94 approximant-row contract, ported from `src/blrt_mceliece/wiedemann.py`
// (`_order_basis_relation_metadata`) and `rust/blrt_xl_worker`.
// ---------------------------------------------------------------------------

/// One complete approximant row with the metadata Coppersmith's reconstruction
/// needs.  The relation is the row's own top block reversed to forward
/// reconstruction order -- reversal length is `true_degree`, **not** a common
/// span -- and the shift is carried separately as `boundary_exponent`, applied
/// as `B^e_r` after the raw candidate is formed.
///
/// Folding `e_r` into the polynomial instead (a common reversal length, or
/// trimming/padding coefficients) changes which power of `B` multiplies each
/// coefficient and is exactly what made the earlier reconstruction fail.
#[derive(Clone, Debug)]
pub struct ApproximantRow {
    pub basis_row: usize,
    /// `state.shifts[row]`: the running shift the iteration maintains.  This is
    /// what `_order_basis_relation_metadata` uses for `boundary_exponent`.
    pub nominal_degree: usize,
    /// column-shifted degree read off the finished basis, which the Sage oracle
    /// uses for row ordering.  In a correctly maintained shifted order basis the
    /// two coincide; any divergence is a defect in the shift update.
    pub basis_shifted_degree: usize,
    /// `d_r`, the highest degree at which the top block is nonzero
    pub true_degree: usize,
    /// `e_r = delta_r - d_r`
    pub boundary_exponent: usize,
    /// top block in forward reconstruction order: `P_{d}, P_{d-1}, ..., P_0`
    pub relation: Vec<u64>,
    /// lower (numerator) block in ascending degree, trailing zeros trimmed
    pub numerator: Vec<u64>,
    /// deg Q_r, or `None` when the lower block vanishes
    pub numerator_degree: Option<usize>,
}

/// Shifted row degree as the established oracle computes it:
///
/// ```text
/// shifted_degree(row) = max over columns c with basis[row][c] != 0
///                       of  deg(basis[row][c]) + shift[c],
/// shift[c] = 0 for the top block, 1 for the numerator block.
/// ```
///
/// Read off the finished basis.  Two independent surveys (1,200 rows here and
/// 1,700 rows in review) found this equal to the running shift the iteration
/// maintains in **every** case, so the iteration does maintain the intended
/// shifted-degree invariant; an earlier comment here claimed otherwise and was
/// wrong.  The mostly-zero boundary exponent is genuine for these sequences.
pub fn shifted_degree(ob: &OrderBasis, row: usize) -> Option<usize> {
    let b = ob.b;
    let deg = ob.pi.len() - 1;
    let mut best: Option<usize> = None;
    for c in 0..2 * b {
        if let Some(dc) = (0..=deg).rev().find(|&dg| (ob.pi[dg][row] >> c) & 1 == 1) {
            let v = dc + if c < b { 0 } else { 1 };
            best = Some(best.map_or(v, |x: usize| x.max(v)));
        }
    }
    best
}

/// Production BLRT selection: the first `right_width` complete approximant rows
/// in `(shift, row)` order.  Production deliberately does **not** require a
/// homogeneous recurrence from index zero: rows with a boundary numerator are
/// corrected by `B^e` during reconstruction and then checked against the exact
/// original operator.  Filtering here would make every retained exponent zero.
pub fn approximant_rows(seq: &Seq) -> Vec<ApproximantRow> {
    let all = complete_rows(seq);
    select_production(seq, all, Some(seq.b))
}

/// Production extraction from rows already ordered by `(shift, row)`: skip a
/// top block whose degree reaches the requested order, then retain the prefix.
/// No homogeneous-recurrence predicate is applied.
pub fn select_production(
    seq: &Seq,
    rows: Vec<ApproximantRow>,
    cap: Option<usize>,
) -> Vec<ApproximantRow> {
    if cap == Some(0) {
        return Vec::new();
    }
    let mut output = Vec::new();
    for row in rows {
        if row.true_degree >= seq.terms.len() {
            continue;
        }
        assert!(
            row.boundary_exponent < seq.terms.len() - row.true_degree,
            "row {} has a vacuous post-boundary recurrence window: e={} d={} L={}",
            row.basis_row,
            row.boundary_exponent,
            row.true_degree,
            seq.terms.len(),
        );
        output.push(row);
        if cap.is_some_and(|cap| output.len() >= cap) {
            break;
        }
    }
    output
}

/// **Unconditional** audit: every basis row with a nonzero top block, ordered by
/// `(shift, row)`.  No relation-validity filter, degree cutoff, or row cap is
/// applied.  This is the complete-row view; the full `2b`-row basis state remains
/// available separately for rows whose top block is zero.
pub fn complete_rows(seq: &Seq) -> Vec<ApproximantRow> {
    let ob = order_basis_blrt(seq);
    complete_rows_from_basis(seq, &ob)
}

/// Extract the unconditional complete-row audit from an already computed basis.
/// Keeping this operation explicit prevents callers from computing one basis and
/// accidentally verifying row indices against another representative.
pub fn complete_rows_from_basis(seq: &Seq, ob: &OrderBasis) -> Vec<ApproximantRow> {
    let b = seq.b;
    assert_eq!(ob.b, b, "sequence and order basis widths differ");
    let deg = ob.pi.len() - 1;
    let bmask = if b == 64 { !0u64 } else { (1u64 << b) - 1 };
    let n = 2 * b;

    let mut order: Vec<(usize, usize)> = (0..n).map(|r| (ob.delta[r], r)).collect();
    order.sort_unstable();

    let mut out = Vec::new();
    for (nominal, row) in order {
        let Some(true_degree) = (0..=deg).rev().find(|&dg| ob.pi[dg][row] & bmask != 0) else {
            continue;
        };
        let sd = shifted_degree(ob, row)
            .expect("a row with a nonzero top block must have a shifted degree");
        assert_eq!(
            nominal, sd,
            "row {row}: maintained shift {nominal} differs from basis shifted degree {sd}"
        );
        // A shifted order basis must never leave a nominal degree below its top
        // degree.  Clamping here would mask an invalid ordering, so refuse.
        let boundary_exponent = nominal.checked_sub(true_degree).unwrap_or_else(|| {
            panic!("row {row}: nominal degree {nominal} below true degree {true_degree}")
        });
        let relation: Vec<u64> = (0..=true_degree)
            .map(|off| ob.pi[true_degree - off][row] & bmask)
            .collect();
        let mut numerator: Vec<u64> = (0..=deg).map(|dg| (ob.pi[dg][row] >> b) & bmask).collect();
        while numerator.len() > 1 && *numerator.last().unwrap() == 0 {
            numerator.pop();
        }
        let numerator_degree = numerator.iter().rposition(|&w| w != 0);
        out.push(ApproximantRow {
            basis_row: row,
            nominal_degree: nominal,
            basis_shifted_degree: sd,
            true_degree,
            boundary_exponent,
            relation,
            numerator,
            numerator_degree,
        });
    }
    out
}

/// Standalone BLRT diagnostic selection: keep only rows whose homogeneous
/// relation holds from index zero over its full window, at most `cap` of them,
/// in `(shift, row)` order.  This is deliberately not the production policy:
/// applying it there would discard boundary rows and force every retained
/// exponent to zero.
pub fn select_valid_from_zero(
    seq: &Seq,
    rows: Vec<ApproximantRow>,
    cap: Option<usize>,
) -> Vec<ApproximantRow> {
    let l = seq.terms.len();
    if cap == Some(0) {
        return Vec::new();
    }
    let mut out = Vec::new();
    for r in rows {
        let d = r.true_degree;
        if l <= d {
            continue;
        }
        let mut valid = true;
        'w: for i in 0..(l - d) {
            let mut acc = 0u64;
            for (j, &cj) in r.relation.iter().enumerate() {
                let a = &seq.terms[i + j];
                let mut m = cj;
                while m != 0 {
                    let k = m.trailing_zeros() as usize;
                    acc ^= a[k];
                    m &= m - 1;
                }
            }
            if acc != 0 {
                valid = false;
                break 'w;
            }
        }
        if valid {
            out.push(r);
            if let Some(c) = cap {
                if out.len() >= c {
                    break;
                }
            }
        }
    }
    out
}

/// **Self-contained** complete-approximant check: uses only the row's own
/// retained coefficients, so it cannot be pointed at a different basis.
///
/// `relation[off] = P_{d-off}`, `numerator[t] = Q_t`; verifies
/// `Q_t + sum_j P_j A_{t-j} = 0` for every `0 <= t < L`.
pub fn verify_row_complete_identity(seq: &Seq, r: &ApproximantRow) -> bool {
    let l = seq.terms.len();
    let d = r.true_degree;
    for t in 0..l {
        let mut acc = *r.numerator.get(t).unwrap_or(&0);
        for j in 0..=t.min(d) {
            let pj = r.relation[d - j];
            if pj == 0 {
                continue;
            }
            let a = &seq.terms[t - j];
            let mut m = pj;
            while m != 0 {
                let k = m.trailing_zeros() as usize;
                acc ^= a[k];
                m &= m - 1;
            }
        }
        if acc != 0 {
            return false;
        }
    }
    true
}

/// One row of the small dense Gate-A2 reconstruction audit.
#[derive(Clone, Debug)]
pub struct DenseReconstructionRow {
    pub raw_candidate: u64,
    pub shifted_candidate: u64,
    pub square_residual: u64,
    pub original_residual: u64,
    pub square_kernel: bool,
    pub original_kernel: bool,
}

#[derive(Clone, Debug)]
pub struct DenseReconstruction {
    pub rows: Vec<DenseReconstructionRow>,
    /// Nonzero, original-operator-verified candidates retained in input order
    /// when they increase the running GF(2) span.
    pub independent_verified: Vec<u64>,
}

fn apply_dense_vector(rows: &[u64], value: u64) -> u64 {
    let mut output = 0u64;
    for (row, &mask) in rows.iter().enumerate() {
        if (mask & value).count_ones() & 1 == 1 {
            output |= 1u64 << row;
        }
    }
    output
}

fn apply_dense_block(rows: &[u64], value: &[u64]) -> Vec<u64> {
    rows.iter()
        .map(|&mask| {
            let mut output = 0u64;
            let mut columns = mask;
            while columns != 0 {
                let column = columns.trailing_zeros() as usize;
                output ^= value[column];
                columns &= columns - 1;
            }
            output
        })
        .collect()
}

/// Projected sequence `U^T B^(i+1) Y` for the small dense Gate-A2 fixture.
/// Each returned term stores one `right_width`-bit row per left vector.
pub fn dense_krylov_sequence(
    square_rows: &[u64],
    left_vectors: &[u64],
    right_rows: &[u64],
    steps: usize,
) -> Vec<Vec<u64>> {
    let n = square_rows.len();
    assert!(n > 0 && n <= 64);
    assert_eq!(right_rows.len(), n);
    assert!(!left_vectors.is_empty() && left_vectors.len() <= 64);
    assert!(steps > 0);
    let nmask = if n == 64 { !0u64 } else { (1u64 << n) - 1 };
    assert!(
        square_rows
            .iter()
            .chain(left_vectors)
            .all(|&row| row & !nmask == 0)
    );
    let mut current = apply_dense_block(square_rows, right_rows);
    let mut sequence = Vec::with_capacity(steps);
    for _ in 0..steps {
        let term = left_vectors
            .iter()
            .map(|&left| {
                let mut row = 0u64;
                let mut coordinates = left;
                while coordinates != 0 {
                    let coordinate = coordinates.trailing_zeros() as usize;
                    row ^= current[coordinate];
                    coordinates &= coordinates - 1;
                }
                row
            })
            .collect();
        sequence.push(term);
        current = apply_dense_block(square_rows, &current);
    }
    sequence
}

fn insert_independent(basis: &mut [u64; 64], mut value: u64) -> bool {
    for pivot in (0..64).rev() {
        if value >> pivot & 1 == 0 {
            continue;
        }
        if basis[pivot] == 0 {
            basis[pivot] = value;
            return true;
        }
        value ^= basis[pivot];
    }
    false
}

/// Exact small dense reconstruction oracle for the Holdout side of Gate A2.
///
/// `square_rows` is the square proxy `B`, `original_rows` is the decisive
/// operator `M`, `right_rows[k]` packs row `k` of the initial block `Y`, and
/// each relation is in forward reconstruction order.  The raw candidate is
/// `sum_j B^j Y c_j`; the emitted candidate is `B^e` times that value.
pub fn reconstruct_dense(
    square_rows: &[u64],
    original_rows: &[u64],
    right_rows: &[u64],
    relations: &[Vec<u64>],
    boundary_exponents: &[usize],
) -> DenseReconstruction {
    let n = square_rows.len();
    assert!(
        n > 0 && n <= 64,
        "dense A2 square dimension must be in 1..=64"
    );
    assert!(
        original_rows.len() <= 64,
        "dense A2 original residual dimension must be at most 64"
    );
    assert_eq!(
        right_rows.len(),
        n,
        "right block length differs from square dimension"
    );
    assert_eq!(
        relations.len(),
        boundary_exponents.len(),
        "one exponent is required per relation"
    );
    let nmask = if n == 64 { !0u64 } else { (1u64 << n) - 1 };
    assert!(square_rows.iter().all(|&row| row & !nmask == 0));
    assert!(original_rows.iter().all(|&row| row & !nmask == 0));

    let mut rows = Vec::with_capacity(relations.len());
    let mut independent_verified = Vec::new();
    let mut verified_basis = [0u64; 64];
    for (relation, &boundary) in relations.iter().zip(boundary_exponents) {
        assert!(
            !relation.is_empty(),
            "a reconstruction relation cannot be empty"
        );
        let mut current = right_rows.to_vec();
        let mut raw_candidate = 0u64;
        for (degree, &coefficient) in relation.iter().enumerate() {
            for (coordinate, &block_row) in current.iter().enumerate() {
                if (block_row & coefficient).count_ones() & 1 == 1 {
                    raw_candidate ^= 1u64 << coordinate;
                }
            }
            if degree + 1 < relation.len() {
                current = apply_dense_block(square_rows, &current);
            }
        }
        let mut shifted_candidate = raw_candidate;
        for _ in 0..boundary {
            shifted_candidate = apply_dense_vector(square_rows, shifted_candidate);
        }
        let square_residual = apply_dense_vector(square_rows, shifted_candidate);
        let original_residual = apply_dense_vector(original_rows, shifted_candidate);
        let row = DenseReconstructionRow {
            raw_candidate,
            shifted_candidate,
            square_residual,
            original_residual,
            square_kernel: square_residual == 0,
            original_kernel: original_residual == 0,
        };
        if row.original_kernel
            && shifted_candidate != 0
            && insert_independent(&mut verified_basis, shifted_candidate)
        {
            independent_verified.push(shifted_candidate);
        }
        rows.push(row);
    }
    DenseReconstruction {
        rows,
        independent_verified,
    }
}

/// Standalone homogeneous recurrence predicate from index zero:
/// `sum_j relation_j A_{i+j} = 0` for `0 <= i < L - d_r`.
///
/// This is neither the complete `[P|Q]` identity nor the production selection
/// rule.  A valid boundary row can fail here before its `B^e` correction.
pub fn verify_homogeneous_from_zero(seq: &Seq, r: &ApproximantRow) -> bool {
    let l = seq.terms.len();
    let d = r.true_degree;
    if l <= d {
        return false;
    }
    if r.relation.iter().all(|&c| c == 0) {
        return false;
    }
    // The established predicate is validity from index 0 over the full window.
    // Widening the window to accommodate a failing row manufactures validity
    // that the oracle rejects; such rows are discarded at selection instead.
    for i in 0..(l - d) {
        let mut acc = 0u64;
        for (j, &cj) in r.relation.iter().enumerate() {
            let a = &seq.terms[i + j];
            let mut m = cj;
            while m != 0 {
                let k = m.trailing_zeros() as usize;
                acc ^= a[k];
                m &= m - 1;
            }
        }
        if acc != 0 {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod complete_identity {
    use super::tests_support::sequence;
    use super::*;

    #[test]
    #[ignore]
    fn survey_complete_vs_tail() {
        let (mut rows, mut tail_ok, mut full_ok, mut both) = (0, 0, 0, 0);
        for seed in 1..=400u64 {
            let b = 2 + (seed as usize % 3);
            let n = 10 + (seed as usize * 7) % 14;
            let l = 2 * n / b + 4 * b + 8;
            let s = sequence(n, b, l, seed.wrapping_mul(0x9e3779b97f4a7c15));
            let ars = complete_rows(&s);
            for r in &ars {
                rows += 1;
                let t = verify_homogeneous_from_zero(&s, r);
                let f = verify_row_complete_identity(&s, r);
                if t {
                    tail_ok += 1;
                }
                if f {
                    full_ok += 1;
                }
                if t && f {
                    both += 1;
                }
            }
        }
        eprintln!(
            "rows {rows} | homogeneous-tail ok {tail_ok} | COMPLETE identity ok {full_ok} | both {both}"
        );
    }
}

// ---------------------------------------------------------------------------
// Faithful port of the established M-basis from rust/blrt_xl_worker::order_basis.
//
// The established implementation builds a per-step transformation matrix -- a
// `constant` part for rows whose discrepancy is killed by a combination of
// pivots, and a `linear` part for pivot rows, which is additionally multiplied
// by x -- and applies it to the whole basis.  Reducing rows in place, as an
// earlier revision here did, keeps the basis valid but does not reproduce the
// same (minimal) basis: measured degree sequences differed by one, and a native
// differential mismatched on 4/100, 19/100, 33/100 sequences at b = 2, 3, 4.
//
// Over F2 the field arithmetic collapses: mul is AND, every pivot is 1, so the
// inverse is the identity and reduction is a plain XOR.
// ---------------------------------------------------------------------------

/// Row `r` of `t * u`, where both are `n x n` over F2 packed one row per `u64`.
#[inline]
fn f2_rowmul(t_row: u64, u: &[u64]) -> u64 {
    let mut acc = 0u64;
    let mut m = t_row;
    while m != 0 {
        let j = m.trailing_zeros() as usize;
        acc ^= u[j];
        m &= m - 1;
    }
    acc
}

/// M-basis of `[S^T ; I]` to order `L` with shift `[0..0, 1..1]`, following
/// `rust/blrt_xl_worker` step for step.
pub fn order_basis_blrt(seq: &Seq) -> OrderBasis {
    let b = seq.b;
    let l = seq.terms.len();
    assert!(b <= 32, "a basis row holds 2b bits in one u64");
    let n = 2 * b;
    let bmask = if b == 64 { !0u64 } else { (1u64 << b) - 1 };

    let mut shifts: Vec<usize> = vec![0; n];
    shifts[b..].fill(1);
    // basis[degree][row] : 2b bits
    let mut basis: Vec<Vec<u64>> = vec![(0..n).map(|i| 1u64 << i).collect()];

    for precision in 0..l {
        // coefficient x^precision of Pi * [S^T ; I]
        let mut discrepancy = vec![0u64; n];
        for (basis_degree, coeff) in basis.iter().enumerate() {
            if basis_degree > precision {
                break;
            }
            let seq_index = precision - basis_degree;
            let term = &seq.terms[seq_index];
            for row in 0..n {
                let mut m = coeff[row] & bmask;
                while m != 0 {
                    let inner = m.trailing_zeros() as usize;
                    discrepancy[row] ^= term[inner];
                    m &= m - 1;
                }
                if seq_index == 0 {
                    discrepancy[row] ^= (coeff[row] >> b) & bmask;
                }
            }
        }

        let mut order: Vec<usize> = (0..n).collect();
        order.sort_by_key(|&row| (shifts[row], row));

        // (pivot column, reduced residual, accumulated combination)
        let mut pivots: Vec<(usize, u64, u64)> = Vec::new();
        let mut constant = vec![0u64; n];
        let mut linear = vec![0u64; n];
        let mut pivot_rows = vec![false; n];

        for row in order {
            let mut residual = discrepancy[row] & bmask;
            let mut combination = 1u64 << row;
            for (pcol, presid, pcomb) in &pivots {
                if (residual >> *pcol) & 1 == 1 {
                    residual ^= *presid;
                    combination ^= *pcomb;
                }
            }
            if residual != 0 {
                let pcol = residual.trailing_zeros() as usize;
                linear[row] = combination;
                pivot_rows[row] = true;
                pivots.push((pcol, residual, combination));
            } else {
                constant[row] = combination;
            }
        }

        let mut updated: Vec<Vec<u64>> = vec![vec![0u64; n]; basis.len() + 1];
        for degree in 0..updated.len() {
            for row in 0..n {
                if degree < basis.len() && constant[row] != 0 {
                    updated[degree][row] ^= f2_rowmul(constant[row], &basis[degree]);
                }
                if degree > 0 && linear[row] != 0 {
                    updated[degree][row] ^= f2_rowmul(linear[row], &basis[degree - 1]);
                }
            }
        }
        while updated.len() > 1 && updated.last().unwrap().iter().all(|&v| v == 0) {
            updated.pop();
        }
        basis = updated;
        for (row, is_pivot) in pivot_rows.into_iter().enumerate() {
            if is_pivot {
                shifts[row] += 1;
            }
        }
    }

    OrderBasis {
        b,
        pi: basis,
        delta: shifts,
    }
}

#[cfg(test)]
mod basis_differential {
    use super::tests_support::sequence;
    use super::*;

    /// The in-place reduction against the ported established M-basis.
    #[test]
    #[ignore]
    fn inplace_vs_blrt() {
        let (mut cases, mut same_shift, mut same_basis, mut same_degs) = (0, 0, 0, 0);
        for seed in 1..=200u64 {
            for b in 2..=4usize {
                let n = 8 + (seed as usize) % 12;
                let l = 2 * n / b + 4 * b + 8;
                let s = sequence(n, b, l, seed.wrapping_mul(0x9e3779b97f4a7c15) ^ b as u64);
                let a = order_basis(&s);
                let c = order_basis_blrt(&s);
                cases += 1;
                if a.delta == c.delta {
                    same_shift += 1;
                }
                if a.pi == c.pi {
                    same_basis += 1;
                }
                let dg = |ob: &OrderBasis| {
                    let bm = (1u64 << b) - 1;
                    let d = ob.pi.len() - 1;
                    let mut v: Vec<usize> = (0..2 * b)
                        .filter_map(|r| (0..=d).rev().find(|&g| ob.pi[g][r] & bm != 0))
                        .collect();
                    v.sort_unstable();
                    v
                };
                if dg(&a) == dg(&c) {
                    same_degs += 1;
                }
            }
        }
        eprintln!(
            "cases {cases} | identical shifts {same_shift} | identical basis {same_basis} | identical top degrees {same_degs}"
        );
    }
}
