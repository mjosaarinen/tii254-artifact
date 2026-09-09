//! The binary hold-out operator, matrix-free and slice-blocked.
//!
//! For a constrained public column `p` with support `P` and a retained Hasse
//! level `j`, the operator row indexed by `T` with `|T| = j` acts as
//!
//! ```text
//! M[(p,T), S] = [ T subset S subset T union P ],      |S| = d.
//! ```
//!
//! Writing `Q = [0,k) \ P` and splitting `S = S_P | S_Q`, `T = T_P | T_Q`, the
//! condition `S \ T subset P` forces `S_Q = T_Q` exactly, while `T subset S`
//! forces `T_P subset S_P`.  Hence
//!
//! ```text
//! M_p = zeta_P  (x)  id_Q ,
//! ```
//!
//! block diagonal over *slices* `A = S_Q = T_Q`.  Inside one slice the block is
//! the inclusion matrix of `(j - |A|)`-subsets versus `(d - |A|)`-subsets of the
//! `w`-element set `P`, which depends on `|A|` only -- never on `A` itself.
//!
//! Two consequences drive the whole implementation:
//!
//! * the local incidence pattern is built once per `|A|` and reused across all
//!   `C(|Q|, |A|)` slices, so the index stream is cache-resident rather than
//!   streamed from DRAM;
//! * a slice touches only `C(w, d-|A|)` monomials, so the block-vector working
//!   set stays inside L2 and the inner loop runs at the cache-resident rate.

use crate::subsets::{Binom, colex_rank, subsets_of};
#[cfg(feature = "block64")]
use rayon::prelude::*;

/// Bits carried per monomial in one pass.  The production Krylov worker uses
/// 512, while the targeted GF(2^64) reference enables the additive `block64`
/// feature and carries one extension-field word per coordinate.
#[cfg(not(feature = "block64"))]
pub const BLOCK_BITS: usize = 512;
#[cfg(feature = "block64")]
pub const BLOCK_BITS: usize = 64;
/// `u64` words per monomial.
pub const BLOCK_WORDS: usize = BLOCK_BITS / 64;

/// One constrained public column.
#[derive(Clone, Debug)]
pub struct Point {
    /// `supp(p)` as a bitmask over `[0,k)`.
    pub support: u64,
    /// Retained Hasse levels `|T|`, Lucas-minimal and strictly increasing.
    pub levels: Vec<usize>,
}

/// The local incidence pattern for one value of `a = |A|`.
///
/// `row_cols[t]` lists the *local* column positions (indices into the slice's
/// `d-a`-subsets of `P`) incident to the `t`-th local row.  Rows are laid out
/// level by level in the order of `levels`.
struct LocalPattern {
    /// flattened incidence lists
    cols: Vec<u32>,
    /// `starts[t] .. starts[t+1]` delimits row `t`
    starts: Vec<u32>,
    /// number of local rows contributed by each retained level
    rows_per_level: Vec<usize>,
}

/// A dimension-admitted cell, ready to apply.
pub struct Operator {
    pub k: usize,
    pub d: usize,
    pub n_cols: u64,
    pub points: Vec<Point>,
    /// row offset of each point in the stacked operator
    pub point_row_offset: Vec<u64>,
    pub n_rows: u64,
    binom: Binom,
    /// `patterns[pi][a]` -- built lazily per point, since `w` may differ
    patterns: Vec<Vec<Option<LocalPattern>>>,
    /// per-point plans, built once by [`Operator::prepare`]
    plans: Vec<PointPlan>,
}

impl Operator {
    pub fn new(k: usize, d: usize, points: Vec<Point>) -> Self {
        assert!(k <= 63, "ground set must fit a u64 mask");
        let binom = Binom::new(k + 1);
        let n_cols = binom.c(k, d);
        let mut point_row_offset = Vec::with_capacity(points.len());
        let mut acc = 0u64;
        for p in &points {
            point_row_offset.push(acc);
            for &j in &p.levels {
                acc += binom.c(k, j);
            }
        }
        let n_rows = acc;
        let patterns = points
            .iter()
            .map(|_| (0..=d).map(|_| None).collect())
            .collect();
        Self {
            k,
            d,
            n_cols,
            points,
            point_row_offset,
            n_rows,
            binom,
            patterns,
            plans: Vec::new(),
        }
    }

    /// Exact nonzero count, for cross-checking against the reference model.
    pub fn nnz(&self) -> u64 {
        let mut total = 0u64;
        for p in &self.points {
            let w = p.support.count_ones() as usize;
            for &j in &p.levels {
                // rows with |T cap P| = i contribute C(w-i, d-j) columns each
                for i in 0..=j.min(w) {
                    let rows = self.binom.c(w, i) * self.binom.c(self.k - w, j - i);
                    if self.d >= j {
                        total += rows * self.binom.c(w - i, self.d - j);
                    }
                }
            }
        }
        total
    }

    fn build_pattern(&self, pi: usize, a: usize) -> LocalPattern {
        let p = &self.points[pi];
        let support = p.support;
        let cols_local = subsets_of(support, self.d - a);
        // local column position, keyed by mask
        let mut col_pos = std::collections::HashMap::with_capacity(cols_local.len());
        for (i, &m) in cols_local.iter().enumerate() {
            col_pos.insert(m, i as u32);
        }
        let mut cols = Vec::new();
        let mut starts = vec![0u32];
        let mut rows_per_level = Vec::new();
        for &j in &p.levels {
            if j < a || self.d < j {
                rows_per_level.push(0);
                continue;
            }
            let tp_size = j - a;
            let tps = subsets_of(support, tp_size);
            rows_per_level.push(tps.len());
            for &tp in &tps {
                for &b in &cols_local {
                    if tp & !b == 0 {
                        cols.push(col_pos[&b]);
                    }
                }
                starts.push(cols.len() as u32);
            }
        }
        LocalPattern {
            cols,
            starts,
            rows_per_level,
        }
    }

    fn pattern(&mut self, pi: usize, a: usize) -> &LocalPattern {
        if self.patterns[pi][a].is_none() {
            let pat = self.build_pattern(pi, a);
            self.patterns[pi][a] = Some(pat);
        }
        self.patterns[pi][a].as_ref().unwrap()
    }

    /// Per-point plan: everything that does not depend on the slice `A`.
    fn build_plan(&self, pi: usize) -> PointPlan {
        let support = self.points[pi].support;
        let levels = self.points[pi].levels.clone();
        let d = self.d;
        let complement = ((1u64 << self.k) - 1) & !support;
        let qsize = complement.count_ones() as usize;
        let mut per_a = Vec::new();
        for a in 0..=d.min(qsize) {
            let pat = self.build_pattern(pi, a);
            let slices = subsets_of(complement, a);
            let cols_local = subsets_of(support, d - a);
            let tps: Vec<Vec<u64>> = levels
                .iter()
                .map(|&j| {
                    if j >= a && d >= j {
                        subsets_of(support, j - a)
                    } else {
                        Vec::new()
                    }
                })
                .collect();
            let mut gcol = Vec::with_capacity(slices.len() * cols_local.len());
            for &aa in &slices {
                for &b in &cols_local {
                    gcol.push(colex_rank(b | aa, &self.binom) as u32);
                }
            }
            let grow: Vec<Vec<u32>> = tps
                .iter()
                .map(|tl| {
                    let mut v = Vec::with_capacity(slices.len() * tl.len());
                    for &aa in &slices {
                        for &tp in tl {
                            v.push(colex_rank(tp | aa, &self.binom) as u32);
                        }
                    }
                    v
                })
                .collect();
            per_a.push(SlicePlan {
                slices,
                cols_local,
                gcol,
                grow,
                tps,
                pat,
            });
        }
        PointPlan { levels, per_a }
    }

    /// Build every per-point plan.  Call once; the plans are reused by every
    /// [`Operator::apply`], which is what makes a long Krylov sequence viable.
    pub fn prepare(&mut self, threads: usize) {
        if !self.plans.is_empty() {
            return;
        }
        let npoints = self.points.len();
        let mut out: Vec<Option<PointPlan>> = (0..npoints).map(|_| None).collect();
        let next = std::sync::atomic::AtomicUsize::new(0);
        let cells: Vec<std::sync::Mutex<&mut Option<PointPlan>>> =
            out.iter_mut().map(std::sync::Mutex::new).collect();
        let this: &Operator = self;
        std::thread::scope(|scope| {
            for _ in 0..threads.max(1) {
                let next = &next;
                let cells = &cells;
                scope.spawn(move || {
                    loop {
                        let pi = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if pi >= npoints {
                            break;
                        }
                        let plan = this.build_plan(pi);
                        **cells[pi].lock().unwrap() = Some(plan);
                    }
                });
            }
        });
        drop(cells);
        self.plans = out.into_iter().map(|p| p.unwrap()).collect();
    }

    /// `y <- M x`, with `x` of `n_cols` blocks and `y` of `n_rows` blocks.
    ///
    /// Rows of distinct points are disjoint, so points are applied in parallel
    /// with no synchronisation.  Inside a point, each slice is gathered into a
    /// contiguous buffer of `C(w, d-|A|)` blocks -- at most a few hundred KiB,
    /// so the inner loop runs L2-resident.
    pub fn apply(&mut self, x: &[u64], y: &mut [u64], threads: usize) {
        assert_eq!(x.len() as u64, self.n_cols * BLOCK_WORDS as u64);
        assert_eq!(y.len() as u64, self.n_rows * BLOCK_WORDS as u64);
        self.prepare(threads);
        y.fill(0);
        let npoints = self.points.len();

        // Work items are (point, |A|-class, slice range).  A row `(p,T)` lies in
        // the class `a = |T cap Q_p|` and the slice `A = T cap Q_p`, so distinct
        // (point, a, A) triples write *disjoint* rows.  That makes the fan-out
        // safe without locking, and -- unlike a per-point split -- it produces
        // thousands of items instead of one per public column, which is what the
        // 29-point granularity ceiling of 51.8% on 28 threads demanded.
        let target = threads.max(1) * 16;
        let mut items: Vec<WorkItem> = Vec::new();
        for pi in 0..npoints {
            let base = self.point_row_offset[pi];
            for (ai, sp) in self.plans[pi].per_a.iter().enumerate() {
                if sp.slices.is_empty() || sp.cols_local.is_empty() || sp.pat.cols.is_empty() {
                    continue;
                }
                let per_slice = sp.pat.cols.len() + sp.cols_local.len();
                let total = per_slice * sp.slices.len();
                let chunk = ((sp.slices.len() * threads.max(1) * 4) / target.max(1)).max(1);
                let mut si = 0usize;
                while si < sp.slices.len() {
                    let end = (si + chunk).min(sp.slices.len());
                    items.push(WorkItem {
                        pi,
                        ai,
                        si_start: si,
                        si_end: end,
                        row_base: base,
                        weight: per_slice * (end - si),
                    });
                    si = end;
                }
                let _ = total;
            }
        }
        // longest-processing-time-first: schedule the heavy chunks early
        items.sort_unstable_by(|a, b| b.weight.cmp(&a.weight));

        let binom = &self.binom;
        let plans = &self.plans;
        let levels_of: Vec<Vec<usize>> = self.points.iter().map(|p| p.levels.clone()).collect();
        let k = self.k;
        #[cfg(not(feature = "block64"))]
        let next = std::sync::atomic::AtomicUsize::new(0);
        let yptr = YPtr(y.as_mut_ptr());

        #[cfg(feature = "block64")]
        items
            .par_iter()
            .for_each_init(Vec::<u64>::new, |gathered, it| {
                // SAFETY: items carry disjoint (point, a, slice) ranges and a
                // row's (point, a, A) is fixed by its own index, so no two items
                // address the same block of `y`.  Rayon retains one narrow
                // scratch buffer per worker and only the base pointer is shared.
                unsafe {
                    apply_slice_range(
                        &plans[it.pi],
                        &levels_of[it.pi],
                        k,
                        binom,
                        x,
                        yptr.get(),
                        it.row_base,
                        it.ai,
                        it.si_start,
                        it.si_end,
                        gathered,
                    );
                }
            });

        #[cfg(not(feature = "block64"))]
        std::thread::scope(|scope| {
            for _ in 0..threads.max(1) {
                let next = &next;
                let items = &items;
                let levels_of = &levels_of;
                let yptr = &yptr;
                scope.spawn(move || {
                    let mut gathered: Vec<u64> = Vec::new();
                    loop {
                        let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if i >= items.len() {
                            break;
                        }
                        let it = &items[i];
                        // SAFETY: items carry disjoint (point, a, slice) ranges and a
                        // row's (point, a, A) is fixed by its own index, so no two items
                        // address the same block of `y`.  Only the base pointer crosses
                        // the thread boundary; `&mut` is materialised per block inside,
                        // never over the whole array, so no two live `&mut` overlap.
                        unsafe {
                            apply_slice_range(
                                &plans[it.pi],
                                &levels_of[it.pi],
                                k,
                                binom,
                                x,
                                yptr.0,
                                it.row_base,
                                it.ai,
                                it.si_start,
                                it.si_end,
                                &mut gathered,
                            );
                        }
                    }
                });
            }
        });
    }

    /// `x <- M^T y`.  `x` is overwritten.
    pub fn apply_transpose(&mut self, y: &[u64], x: &mut [u64], threads: usize) {
        assert_eq!(y.len() as u64, self.n_rows * BLOCK_WORDS as u64);
        assert_eq!(x.len() as u64, self.n_cols * BLOCK_WORDS as u64);
        self.prepare(threads);
        x.fill(0);
        let binom = &self.binom;
        let plans = &self.plans;
        let levels_of: Vec<Vec<usize>> = self.points.iter().map(|p| p.levels.clone()).collect();
        let k = self.k;
        let xptr = YPtr(x.as_mut_ptr());
        for pi in 0..self.points.len() {
            let base = self.point_row_offset[pi];
            let mut items: Vec<(usize, usize, usize)> = Vec::new();
            for (ai, sp) in plans[pi].per_a.iter().enumerate() {
                if sp.slices.is_empty() || sp.cols_local.is_empty() || sp.pat.cols.is_empty() {
                    continue;
                }
                let chunk = (sp.slices.len() / (threads.max(1) * 8)).max(1);
                let mut si = 0usize;
                while si < sp.slices.len() {
                    let e = (si + chunk).min(sp.slices.len());
                    items.push((ai, si, e));
                    si = e;
                }
            }
            #[cfg(feature = "block64")]
            items
                .par_iter()
                .for_each_init(Vec::<u64>::new, |scratch, &(ai, s0, e0)| {
                    // SAFETY: within one point the slices partition the
                    // monomials, so these ranges touch disjoint columns;
                    // points are stepped sequentially.
                    unsafe {
                        transpose_slice_range(
                            &plans[pi],
                            &levels_of[pi],
                            k,
                            binom,
                            y,
                            xptr.get(),
                            base,
                            ai,
                            s0,
                            e0,
                            scratch,
                        );
                    }
                });

            #[cfg(not(feature = "block64"))]
            {
                let next = std::sync::atomic::AtomicUsize::new(0);
                std::thread::scope(|scope| {
                    for _ in 0..threads.max(1) {
                        let next = &next;
                        let items = &items;
                        let levels_of = &levels_of;
                        let xptr = &xptr;
                        scope.spawn(move || {
                            let mut scratch: Vec<u64> = Vec::new();
                            loop {
                                let i = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                if i >= items.len() {
                                    break;
                                }
                                let (ai, s0, e0) = items[i];
                                // SAFETY: within one point the slices partition the
                                // monomials, so these ranges touch disjoint columns; points
                                // are stepped sequentially.  As above, only the base pointer
                                // is shared and `&mut` is taken per block.
                                unsafe {
                                    transpose_slice_range(
                                        &plans[pi],
                                        &levels_of[pi],
                                        k,
                                        binom,
                                        y,
                                        xptr.0,
                                        base,
                                        ai,
                                        s0,
                                        e0,
                                        &mut scratch,
                                    );
                                }
                            }
                        });
                    }
                });
            }
        }
    }

    fn point_rows(&self, pi: usize) -> u64 {
        self.points[pi]
            .levels
            .iter()
            .map(|&j| self.binom.c(self.k, j))
            .sum()
    }

    /// Dense `F2` image of the operator, for differential testing only.
    /// Row `r` is returned as a bitmask over columns; requires `n_cols <= 64`.
    pub fn dense_rows(&mut self) -> Vec<u64> {
        assert!(self.n_cols <= 64, "dense form is for tiny cells only");
        let ncols = self.n_cols as usize;
        let mut rows = vec![0u64; self.n_rows as usize];
        let mut x = vec![0u64; ncols * BLOCK_WORDS];
        let mut y = vec![0u64; self.n_rows as usize * BLOCK_WORDS];
        // one indicator column at a time, using bit 0 of the block
        for c in 0..ncols {
            x.fill(0);
            x[c * BLOCK_WORDS] = 1;
            self.apply(&x, &mut y, 1);
            for r in 0..self.n_rows as usize {
                if y[r * BLOCK_WORDS] & 1 == 1 {
                    rows[r] |= 1u64 << c;
                }
            }
        }
        rows
    }
}

#[cfg(all(target_arch = "x86_64", not(feature = "block64")))]
#[inline]
fn has_avx512() -> bool {
    use std::sync::OnceLock;
    static F: OnceLock<bool> = OnceLock::new();
    *F.get_or_init(|| is_x86_feature_detected!("avx512f"))
}

#[cfg(all(target_arch = "x86_64", feature = "block64"))]
#[inline]
fn has_avx512() -> bool {
    // The AVX-512 helpers below deliberately load/store one complete
    // 512-bit monomial block.  A block64 consumer has only one u64 per
    // monomial, so using those helpers would cross coordinate boundaries.
    false
}

/// Portable fallback: XOR `cols` of `src` into `dst`, one 512-bit block.
fn accumulate_scalar(src: &[u64], cols: &[u32], dst: &mut [u64]) {
    let mut a = [0u64; BLOCK_WORDS];
    for &c in cols {
        let o = c as usize * BLOCK_WORDS;
        for wi in 0..BLOCK_WORDS {
            a[wi] ^= src[o + wi];
        }
    }
    dst[..BLOCK_WORDS].copy_from_slice(&a);
}

/// One 512-bit block per monomial is exactly one `zmm`, so the whole
/// accumulation is a dependency chain of `vpxorq`.  Four accumulators keep the
/// load ports busy; the generic loop above compiled to scalar code and lost 6x.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn accumulate_avx512(src: *const u64, cols: &[u32], dst: *mut u64) {
    use core::arch::x86_64::*;
    // SAFETY: the caller has checked AVX-512F. `src` covers every block named
    // by `cols`, `dst` covers one BLOCK_WORDS block, and both may be unaligned.
    unsafe {
        let mut a0 = _mm512_setzero_si512();
        let mut a1 = _mm512_setzero_si512();
        let mut a2 = _mm512_setzero_si512();
        let mut a3 = _mm512_setzero_si512();
        let n = cols.len();
        let mut i = 0usize;
        while i + 4 <= n {
            a0 = _mm512_xor_si512(
                a0,
                _mm512_loadu_si512(
                    src.add(*cols.get_unchecked(i) as usize * BLOCK_WORDS) as *const __m512i
                ),
            );
            a1 = _mm512_xor_si512(
                a1,
                _mm512_loadu_si512(
                    src.add(*cols.get_unchecked(i + 1) as usize * BLOCK_WORDS) as *const __m512i
                ),
            );
            a2 = _mm512_xor_si512(
                a2,
                _mm512_loadu_si512(
                    src.add(*cols.get_unchecked(i + 2) as usize * BLOCK_WORDS) as *const __m512i
                ),
            );
            a3 = _mm512_xor_si512(
                a3,
                _mm512_loadu_si512(
                    src.add(*cols.get_unchecked(i + 3) as usize * BLOCK_WORDS) as *const __m512i
                ),
            );
            i += 4;
        }
        while i < n {
            a0 = _mm512_xor_si512(
                a0,
                _mm512_loadu_si512(
                    src.add(*cols.get_unchecked(i) as usize * BLOCK_WORDS) as *const __m512i
                ),
            );
            i += 1;
        }
        let acc = _mm512_xor_si512(_mm512_xor_si512(a0, a1), _mm512_xor_si512(a2, a3));
        // Each row is written exactly once per apply: its slice class b = |T cap Q|,
        // its slice A = T cap Q and its level |T| are all determined by the row index,
        // so no other work item targets it.  A plain store therefore replaces a
        // read-modify-write and removes 45.3M random 64-byte reads per matvec.
        _mm512_storeu_si512(dst as *mut __m512i, acc);
    }
}

/// 0 = full, 1 = gather only, 2 = arithmetic only.  Instrumentation for
/// attributing wall time between the memory-bound and compute-bound stages.
fn phase_mode() -> u8 {
    use std::sync::OnceLock;
    static M: OnceLock<u8> = OnceLock::new();
    *M.get_or_init(|| match std::env::var("MCX_PHASE").as_deref() {
        Ok("gather") => 1,
        Ok("xor") => 2,
        _ => 0,
    })
}

struct WorkItem {
    pi: usize,
    ai: usize,
    si_start: usize,
    si_end: usize,
    row_base: u64,
    weight: usize,
}

/// Raw handle to the output vector.  Sound only because work items are
/// row-disjoint by construction; see the call site.
struct YPtr(*mut u64);
unsafe impl Send for YPtr {}
unsafe impl Sync for YPtr {}

impl YPtr {
    #[inline]
    fn get(&self) -> *mut u64 {
        self.0
    }
}

/// Slice-independent data for one value of `a = |A|`.
struct SlicePlan {
    slices: Vec<u64>,
    cols_local: Vec<u64>,
    /// global colex rank of `B | A`, laid out slice-major:
    /// `gcol[si * cols_local.len() + i]` for slice `si`, local column `i`.
    /// Precomputed once per point, reused by every matvec.
    gcol: Vec<u32>,
    /// global colex rank of `T_P | A`, per level, laid out slice-major.
    grow: Vec<Vec<u32>>,
    /// `tps[li]` = the `T_P` subsets for retained level `li`
    tps: Vec<Vec<u64>>,
    pat: LocalPattern,
}

struct PointPlan {
    levels: Vec<usize>,
    per_a: Vec<SlicePlan>,
}

/// Apply one `(point, |A|-class, slice range)` work item.
///
/// Writes only rows whose `T cap Q` equals one of the slices in
/// `si_start..si_end`, which is why concurrent items never collide.
/// # Safety
/// `y_base` must point at the output vector, and `(ai, si_start..si_end)` must
/// be a range no concurrently running call also covers for this point.
unsafe fn apply_slice_range(
    plan: &PointPlan,
    levels: &[usize],
    k: usize,
    binom: &Binom,
    x: &[u64],
    y_base: *mut u64,
    row_base: u64,
    ai: usize,
    si_start: usize,
    si_end: usize,
    gathered: &mut Vec<u64>,
) {
    let sp = &plan.per_a[ai];
    let phase = phase_mode();
    let ncl = sp.cols_local.len();
    gathered.resize(ncl * BLOCK_WORDS, 0);
    for si in si_start..si_end {
        let gbase = si * ncl;
        // The gather is the memory-bound stage: `ncl` random 64-byte reads out
        // of a multi-gigabyte block vector.  Issue the address stream ahead of
        // use so several DRAM transactions are in flight at once.
        const PF: usize = 24;
        #[cfg(target_arch = "x86_64")]
        for i in 0..PF.min(ncl) {
            let g = sp.gcol[gbase + i] as usize * BLOCK_WORDS;
            unsafe {
                core::arch::x86_64::_mm_prefetch(
                    x.as_ptr().add(g) as *const i8,
                    core::arch::x86_64::_MM_HINT_T0,
                );
            }
        }
        if phase != 2 {
            for i in 0..ncl {
                #[cfg(target_arch = "x86_64")]
                if i + PF < ncl {
                    let gn = sp.gcol[gbase + i + PF] as usize * BLOCK_WORDS;
                    unsafe {
                        core::arch::x86_64::_mm_prefetch(
                            x.as_ptr().add(gn) as *const i8,
                            core::arch::x86_64::_MM_HINT_T0,
                        );
                    }
                }
                let g = sp.gcol[gbase + i] as usize * BLOCK_WORDS;
                gathered[i * BLOCK_WORDS..(i + 1) * BLOCK_WORDS]
                    .copy_from_slice(&x[g..g + BLOCK_WORDS]);
            }
        }
        if phase == 1 {
            continue;
        }
        let mut t = 0usize;
        let mut level_row_base = row_base;
        for (li, &j) in levels.iter().enumerate() {
            let tps = &sp.tps[li];
            if tps.is_empty() {
                level_row_base += binom.c(k, j);
                continue;
            }
            let rowtab = &sp.grow[li];
            let rbase = si * tps.len();
            for ti in 0..tps.len() {
                let s0 = sp.pat.starts[t] as usize;
                let e0 = sp.pat.starts[t + 1] as usize;
                let grow = level_row_base + rowtab[rbase + ti] as u64;
                let yo = grow as usize * BLOCK_WORDS;
                let cols = &sp.pat.cols[s0..e0];
                // SAFETY: this block is written by no other work item, so the
                // narrow slice below is the only live reference to it.
                let dst = unsafe { y_base.add(yo) };
                #[cfg(target_arch = "x86_64")]
                {
                    if has_avx512() {
                        unsafe { accumulate_avx512(gathered.as_ptr(), cols, dst) }
                    } else {
                        let d = unsafe { std::slice::from_raw_parts_mut(dst, BLOCK_WORDS) };
                        accumulate_scalar(gathered, cols, d);
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    let d = unsafe { std::slice::from_raw_parts_mut(dst, BLOCK_WORDS) };
                    accumulate_scalar(gathered, cols, d);
                }
                t += 1;
            }
            level_row_base += binom.c(k, j);
        }
    }
}

/// Transpose action `x <- M^T y`, accumulating.
///
/// Within one point the slices partition the monomials, so slices of a single
/// point never collide.  Across points they do -- every point block spans every
/// column -- so points are stepped sequentially and the parallel fan-out is over
/// slices.  With ~90k slices per point that is ample width, and it avoids both
/// per-thread N-sized accumulators and atomics on 1.6e10 updates.
/// # Safety
/// `x_base` must point at the column vector, and the slice range must be one no
/// concurrently running call also covers for this point.
unsafe fn transpose_slice_range(
    plan: &PointPlan,
    levels: &[usize],
    k: usize,
    binom: &Binom,
    y: &[u64],
    x_base: *mut u64,
    row_base: u64,
    ai: usize,
    si_start: usize,
    si_end: usize,
    scratch: &mut Vec<u64>,
) {
    let sp = &plan.per_a[ai];
    let ncl = sp.cols_local.len();
    scratch.resize(ncl * BLOCK_WORDS, 0);
    for si in si_start..si_end {
        for v in scratch.iter_mut() {
            *v = 0;
        }
        let mut t = 0usize;
        let mut level_row_base = row_base;
        for (li, &j) in levels.iter().enumerate() {
            let tps = &sp.tps[li];
            if tps.is_empty() {
                level_row_base += binom.c(k, j);
                continue;
            }
            let rowtab = &sp.grow[li];
            let rbase = si * tps.len();
            for ti in 0..tps.len() {
                let s0 = sp.pat.starts[t] as usize;
                let e0 = sp.pat.starts[t + 1] as usize;
                let grow = level_row_base + rowtab[rbase + ti] as u64;
                let yo = grow as usize * BLOCK_WORDS;
                let cols = &sp.pat.cols[s0..e0];
                #[cfg(target_arch = "x86_64")]
                {
                    if has_avx512() {
                        // SAFETY: avx512f checked at run time; offsets in range
                        // by construction of the pattern.
                        unsafe { scatter_avx512(y.as_ptr().add(yo), cols, scratch.as_mut_ptr()) }
                    } else {
                        scatter_scalar(&y[yo..yo + BLOCK_WORDS], cols, scratch);
                    }
                }
                #[cfg(not(target_arch = "x86_64"))]
                scatter_scalar(&y[yo..yo + BLOCK_WORDS], cols, scratch);
                t += 1;
            }
            level_row_base += binom.c(k, j);
        }
        // fold the slice's local columns back into the global accumulator
        let gbase = si * ncl;
        for i in 0..ncl {
            let g = sp.gcol[gbase + i] as usize * BLOCK_WORDS;
            let o = i * BLOCK_WORDS;
            // SAFETY: slices of one point partition the monomials, so this block
            // is touched by no concurrent item.
            unsafe {
                let d = x_base.add(g);
                for wi in 0..BLOCK_WORDS {
                    *d.add(wi) ^= scratch[o + wi];
                }
            }
        }
    }
}

fn scatter_scalar(src: &[u64], cols: &[u32], dst: &mut [u64]) {
    for &c in cols {
        let o = c as usize * BLOCK_WORDS;
        for wi in 0..BLOCK_WORDS {
            dst[o + wi] ^= src[wi];
        }
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn scatter_avx512(src: *const u64, cols: &[u32], dst: *mut u64) {
    use core::arch::x86_64::*;
    // SAFETY: the caller has checked AVX-512F. `src` covers one BLOCK_WORDS
    // block and `dst` covers every block named by `cols`; accesses may be unaligned.
    unsafe {
        let v = _mm512_loadu_si512(src as *const __m512i);
        for &c in cols {
            let p = dst.add(c as usize * BLOCK_WORDS) as *mut __m512i;
            _mm512_storeu_si512(p, _mm512_xor_si512(_mm512_loadu_si512(p), v));
        }
    }
}

/// `a = Z^T V`, the full `BLOCK_BITS x BLOCK_BITS` projection.
///
/// `a[r]` is row `r`, `BLOCK_WORDS` words wide:  `a[r][c] = XOR_k Z[k]_r & V[k]_c`.
/// Realised as a scatter -- for every `k`, every set bit `r` of `Z[k]` takes a
/// copy of `V[k]` -- so the cost is `N * popcount(Z) * BLOCK_WORDS`, about five
/// matvecs at width 512.  That is the price of the block projection and it is
/// charged nowhere in the arithmetic model.
pub fn project(z: &[u64], v: &[u64], n_cols: u64, threads: usize) -> Vec<u64> {
    let ncols = n_cols as usize;
    let per = ncols.div_ceil(threads.max(1));
    let mut partials: Vec<Vec<u64>> = (0..threads.max(1))
        .map(|_| vec![0u64; BLOCK_BITS * BLOCK_WORDS])
        .collect();
    std::thread::scope(|scope| {
        for (t, part) in partials.iter_mut().enumerate() {
            let lo = t * per;
            let hi = ((t + 1) * per).min(ncols);
            scope.spawn(move || {
                if lo >= hi {
                    return;
                }
                for k in lo..hi {
                    let zk = &z[k * BLOCK_WORDS..(k + 1) * BLOCK_WORDS];
                    let vk = &v[k * BLOCK_WORDS..(k + 1) * BLOCK_WORDS];
                    for (wi, &zw) in zk.iter().enumerate() {
                        let mut m = zw;
                        while m != 0 {
                            let r = wi * 64 + m.trailing_zeros() as usize;
                            let dst = &mut part[r * BLOCK_WORDS..(r + 1) * BLOCK_WORDS];
                            for c in 0..BLOCK_WORDS {
                                dst[c] ^= vk[c];
                            }
                            m &= m - 1;
                        }
                    }
                }
            });
        }
    });
    let mut out = vec![0u64; BLOCK_BITS * BLOCK_WORDS];
    for p in &partials {
        for i in 0..out.len() {
            out[i] ^= p[i];
        }
    }
    out
}

#[cfg(test)]
mod project_tests {
    use super::*;

    #[test]
    fn matches_a_direct_bit_product() {
        let n = 37usize;
        let mut seed = 0x12345678u64;
        let mut rng = || {
            seed ^= seed << 13;
            seed ^= seed >> 7;
            seed ^= seed << 17;
            seed
        };
        let z: Vec<u64> = (0..n * BLOCK_WORDS).map(|_| rng()).collect();
        let v: Vec<u64> = (0..n * BLOCK_WORDS).map(|_| rng()).collect();
        for th in [1usize, 3, 8] {
            let got = project(&z, &v, n as u64, th);
            // direct: a[r][c] = parity over k of Z[k]_r & V[k]_c
            let probes = [0usize, 1, 5, 31, 63, 64, 300, 511];
            for r in probes.into_iter().filter(|&index| index < BLOCK_BITS) {
                for c in probes.into_iter().filter(|&index| index < BLOCK_BITS) {
                    let mut bit = 0u64;
                    for k in 0..n {
                        let zb = z[k * BLOCK_WORDS + r / 64] >> (r % 64) & 1;
                        let vb = v[k * BLOCK_WORDS + c / 64] >> (c % 64) & 1;
                        bit ^= zb & vb;
                    }
                    let got_bit = got[r * BLOCK_WORDS + c / 64] >> (c % 64) & 1;
                    assert_eq!(got_bit, bit, "threads={th} r={r} c={c}");
                }
            }
        }
    }
}
