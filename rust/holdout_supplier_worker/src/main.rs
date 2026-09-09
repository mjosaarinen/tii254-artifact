//! Matrix-free hold-out relation supplier: operator stage.
//!
//! Canonical JSON request on stdin, canonical JSON response on stdout, in the
//! style of the other native workers in this repository.  Sage and the Python
//! reference remain the exact oracle; this binary only changes representation.

mod bm;
mod operator;
mod subsets;

use bm::Seq;
use operator::{BLOCK_BITS, BLOCK_WORDS, Operator, Point};
use serde::{Deserialize, Serialize};
use std::io::Read;

#[derive(Deserialize)]
struct PointSpec {
    support: u64,
    levels: Vec<usize>,
}

#[derive(Deserialize)]
struct Request {
    schema: String,
    k: usize,
    d: usize,
    points: Vec<PointSpec>,
    /// Worker operation; the match below is the authoritative mode list.
    mode: String,
    /// for mode "apply": column indices set in bit 0 of the input block
    #[serde(default)]
    probe_columns: Vec<u64>,
    /// for mode "bench": worker threads
    #[serde(default)]
    threads: usize,
    /// for mode "bench": matvec repetitions
    #[serde(default)]
    reps: usize,
    /// for mode "krylov": number of sequence terms to generate
    #[serde(default)]
    steps: usize,
    /// for mode "approximant": block sequence in the BLRT orientation,
    /// `sequence_terms[i][left][right]`, entries in {0,1}
    #[serde(default)]
    sequence_terms: Vec<Vec<Vec<u16>>>,
    /// diagnostic: emit every complete row instead of the selected right_width
    #[serde(default)]
    select_all: bool,
    /// Standalone diagnostic policy matching `blrt_xl_worker`'s default.  The
    /// production reconstruction path leaves this false and relies on boundary
    /// shifts plus the exact original-operator residual.
    #[serde(default)]
    verify_order_relations: bool,
    /// Small exact Gate-A2 reconstruction inputs.  These fields are used only
    /// by mode `reconstruct` and are deliberately dense and bounded.
    #[serde(default)]
    square_operator: Vec<Vec<u16>>,
    #[serde(default)]
    original_operator: Vec<Vec<u16>>,
    #[serde(default)]
    right_vectors: Vec<Vec<u16>>,
    #[serde(default)]
    left_vectors: Vec<Vec<u16>>,
    #[serde(default)]
    relation_coefficients: Vec<Vec<Vec<u16>>>,
    #[serde(default)]
    relation_boundary_exponents: Vec<usize>,
}

#[derive(Serialize)]
struct Response {
    schema: String,
    k: usize,
    d: usize,
    n_cols: u64,
    n_rows: u64,
    nnz: u64,
    #[serde(skip_serializing_if = "Option::is_none")]
    dense_rows: Option<Vec<String>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    image_rows: Option<Vec<u64>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    bench: Option<Bench>,
    /// adjoint mode: <Mx,y> and <x,M^T y>, one bit per block plane
    #[serde(skip_serializing_if = "Option::is_none")]
    adjoint_lhs: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    adjoint_rhs: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    krylov: Option<Krylov>,
    #[serde(skip_serializing_if = "Option::is_none")]
    approximant: Option<Approximant>,
    #[serde(skip_serializing_if = "Option::is_none")]
    reconstruction: Option<Reconstruction>,
    /// Small dense `U^T B^(i+1)Y` sequence used by Gate A2.
    #[serde(skip_serializing_if = "Option::is_none")]
    dense_sequence: Option<Vec<Vec<Vec<u16>>>>,
}

/// Co94 metadata for every complete approximant row, for the Gate A2
/// differential against the Sage oracle and `rust/blrt_xl_worker`.
#[derive(Serialize)]
struct Approximant {
    /// `[S^T ; I]` with shift `[0..0,1..1]`, as GiLe14 sec. 6 / blrt_xl_worker
    convention: String,
    left_width: usize,
    right_width: usize,
    order: usize,
    /// Maintained shifted degrees for every row of the full `2b x 2b` basis.
    basis_shifts: Vec<usize>,
    /// Full basis in `[degree][row][column]` order for the exact native A2 audit.
    basis_coefficients: Vec<Vec<Vec<u16>>>,
    rows: Vec<ApproximantRowOut>,
}

#[derive(Serialize)]
struct ApproximantRowOut {
    basis_row: usize,
    nominal_degree: usize,
    basis_shifted_degree: usize,
    true_degree: usize,
    boundary_exponent: usize,
    numerator_degree: Option<usize>,
    /// forward reconstruction order: `P_d, P_{d-1}, ..., P_0`, one row per degree
    relation: Vec<Vec<u16>>,
    /// ascending degree, trailing zeros trimmed
    numerator: Vec<Vec<u16>>,
    complete_identity: bool,
}

#[derive(Serialize)]
struct Reconstruction {
    dimension: usize,
    right_width: usize,
    rows: Vec<ReconstructionRow>,
    independent_verified: Vec<Vec<u16>>,
}

#[derive(Serialize)]
struct ReconstructionRow {
    raw_candidate: Vec<u16>,
    shifted_candidate: Vec<u16>,
    square_residual: Vec<u16>,
    original_residual: Vec<u16>,
    square_kernel: bool,
    original_kernel: bool,
}

#[derive(Serialize)]
struct Krylov {
    steps: usize,
    block_bits: usize,
    seconds_per_step: f64,
    seconds_projection: f64,
    seconds_operator: f64,
    /// one digest per emitted term a_i = Z^T B^(i+1) Y (full BLOCK_BITS^2 matrix)
    term_digests: Vec<String>,
    projected_full_sequence_length: u64,
    projected_seconds: f64,
}

#[derive(Serialize)]
struct Bench {
    threads: usize,
    reps: usize,
    seconds_per_matvec: f64,
    nnz_per_second: f64,
    plan_seconds: f64,
    seconds_per_transpose: f64,
    /// FNV-1a over the whole output block vector: identical for any thread
    /// count if and only if the concurrent fan-out is race free.
    y_checksum: String,
    x_bytes: u64,
    y_bytes: u64,
}

fn build(req: &Request) -> Operator {
    let points = req
        .points
        .iter()
        .map(|p| Point {
            support: p.support,
            levels: p.levels.clone(),
        })
        .collect();
    Operator::new(req.k, req.d, points)
}

fn main() {
    let mut raw = String::new();
    std::io::stdin()
        .read_to_string(&mut raw)
        .expect("read stdin");
    let req: Request = serde_json::from_str(&raw).expect("parse request");
    assert_eq!(req.schema, "mcelix-holdout-supplier-operator-request-v1");
    let mut op = build(&req);

    let mut dense_rows = None;
    let mut image_rows = None;
    let mut bench = None;
    let mut adjoint_lhs = None;
    let mut adjoint_rhs = None;
    let mut krylov = None;
    let mut approximant = None;
    let mut reconstruction = None;
    let mut dense_sequence = None;
    match req.mode.as_str() {
        "nnz" => {}
        "dense" => {
            dense_rows = Some(op.dense_rows().iter().map(|r| format!("{r:#x}")).collect());
        }
        "apply" => {
            let mut x = vec![0u64; op.n_cols as usize * BLOCK_WORDS];
            for &c in &req.probe_columns {
                x[c as usize * BLOCK_WORDS] ^= 1;
            }
            let mut y = vec![0u64; op.n_rows as usize * BLOCK_WORDS];
            op.apply(&x, &mut y, 1);
            let hit: Vec<u64> = (0..op.n_rows)
                .filter(|&r| y[r as usize * BLOCK_WORDS] & 1 == 1)
                .collect();
            image_rows = Some(hit);
        }
        "adjoint" => {
            // <Mx, y> = <x, M^T y> on every one of the BLOCK_BITS planes at once:
            // XOR-accumulating (a & b) yields, in bit i, the parity of the plane-i
            // inner product.  A single run therefore checks 512 adjoint identities.
            let threads = if req.threads == 0 { 1 } else { req.threads };
            let nx = op.n_cols as usize * BLOCK_WORDS;
            let ny = op.n_rows as usize * BLOCK_WORDS;
            let mut x = vec![0u64; nx];
            let mut yv = vec![0u64; ny];
            let mut seed = 0x243f6a8885a308d3u64 ^ (req.k as u64) << 32 ^ req.d as u64;
            let mut rng = move || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            };
            for v in x.iter_mut() {
                *v = rng();
            }
            for v in yv.iter_mut() {
                *v = rng();
            }
            let mut u = vec![0u64; ny];
            let mut v = vec![0u64; nx];
            op.apply(&x, &mut u, threads);
            op.apply_transpose(&yv, &mut v, threads);
            let mut lhs = [0u64; BLOCK_WORDS];
            for r in 0..op.n_rows as usize {
                for w in 0..BLOCK_WORDS {
                    lhs[w] ^= u[r * BLOCK_WORDS + w] & yv[r * BLOCK_WORDS + w];
                }
            }
            let mut rhs = [0u64; BLOCK_WORDS];
            for c in 0..op.n_cols as usize {
                for w in 0..BLOCK_WORDS {
                    rhs[w] ^= x[c * BLOCK_WORDS + w] & v[c * BLOCK_WORDS + w];
                }
            }
            let hex = |a: &[u64; BLOCK_WORDS]| {
                a.iter()
                    .map(|w| format!("{w:016x}"))
                    .collect::<Vec<_>>()
                    .join("")
            };
            adjoint_lhs = Some(hex(&lhs));
            adjoint_rhs = Some(hex(&rhs));
        }
        "approximant" => {
            // BLRT and the Sage oracle both build the basis of [S^T ; I].  Our
            // order_basis consumes H = [A ; I] directly, so the incoming terms
            // are transposed here and the convention is echoed in the response.
            let terms = &req.sequence_terms;
            assert!(!terms.is_empty(), "approximant needs a nonempty sequence");
            let left = terms[0].len();
            assert!(left > 0, "approximant terms need at least one row");
            let right = terms[0][0].len();
            assert_eq!(left, right, "this reference path needs a square block");
            assert!(left <= 32, "a Pi row holds 2b bits in one u64");
            assert!(
                terms.iter().all(|term| {
                    term.len() == left
                        && term
                            .iter()
                            .all(|row| row.len() == right && row.iter().all(|&value| value <= 1))
                }),
                "approximant terms must all be square binary matrices of one width"
            );
            let packed: Vec<Vec<u64>> = terms
                .iter()
                .map(|t| {
                    // transpose: our A[r][c] = incoming term[c][r]
                    (0..right)
                        .map(|r| {
                            let mut w = 0u64;
                            for c in 0..left {
                                if t[c][r] != 0 {
                                    w |= 1u64 << c;
                                }
                            }
                            w
                        })
                        .collect()
                })
                .collect();
            let seq = Seq {
                b: left,
                terms: packed,
            };
            let bits =
                |w: u64, n: usize| (0..n).map(|i| ((w >> i) & 1) as u16).collect::<Vec<u16>>();
            let basis = bm::order_basis_blrt(&seq);
            let complete = bm::complete_rows_from_basis(&seq, &basis);
            let emitted = if req.select_all {
                complete
            } else if req.verify_order_relations {
                bm::select_valid_from_zero(&seq, complete, Some(left))
            } else {
                bm::select_production(&seq, complete, Some(left))
            };
            let rows = emitted
                .into_iter()
                .map(|r| ApproximantRowOut {
                    // verified from the row's OWN retained coefficients, so it
                    // cannot be checked against a different basis than emitted
                    complete_identity: bm::verify_row_complete_identity(&seq, &r),
                    relation: r.relation.iter().map(|&w| bits(w, left)).collect(),
                    numerator: r.numerator.iter().map(|&w| bits(w, left)).collect(),
                    basis_row: r.basis_row,
                    nominal_degree: r.nominal_degree,
                    basis_shifted_degree: r.basis_shifted_degree,
                    true_degree: r.true_degree,
                    boundary_exponent: r.boundary_exponent,
                    numerator_degree: r.numerator_degree,
                })
                .collect();
            approximant = Some(Approximant {
                convention: "[S^T; I], shift [0..0,1..1] (GiLe14 sec.6)".to_string(),
                left_width: left,
                right_width: right,
                order: seq.terms.len(),
                basis_shifts: basis.delta.clone(),
                basis_coefficients: basis
                    .pi
                    .iter()
                    .map(|coefficient| coefficient.iter().map(|&row| bits(row, 2 * left)).collect())
                    .collect(),
                rows,
            });
        }
        "reconstruct" => {
            let square = &req.square_operator;
            let n = square.len();
            assert!(
                n > 0 && n <= 64,
                "reconstruction square dimension must be in 1..=64"
            );
            assert!(
                square
                    .iter()
                    .all(|row| { row.len() == n && row.iter().all(|&value| value <= 1) }),
                "square_operator must be a square binary matrix"
            );
            assert!(!req.original_operator.is_empty() && req.original_operator.len() <= 64);
            assert!(
                req.original_operator
                    .iter()
                    .all(|row| { row.len() == n && row.iter().all(|&value| value <= 1) }),
                "original_operator must be a binary matrix with the same column count"
            );
            let width = req.right_vectors.len();
            assert!(
                width > 0 && width <= 64,
                "right-vector width must be in 1..=64"
            );
            assert!(
                req.right_vectors
                    .iter()
                    .all(|row| { row.len() == n && row.iter().all(|&value| value <= 1) }),
                "right_vectors must be a rectangular binary block"
            );
            assert!(
                !req.relation_coefficients.is_empty(),
                "at least one relation is required"
            );
            assert_eq!(
                req.relation_coefficients.len(),
                req.relation_boundary_exponents.len(),
                "one boundary exponent is required per relation"
            );
            assert!(
                req.relation_coefficients.iter().all(|relation| {
                    !relation.is_empty()
                        && relation.iter().all(|coefficient| {
                            coefficient.len() == width
                                && coefficient.iter().all(|&value| value <= 1)
                        })
                }),
                "relation coefficients must be nonempty binary right-width blocks"
            );

            let pack = |row: &[u16]| {
                row.iter()
                    .enumerate()
                    .fold(0u64, |word, (i, &bit)| word | ((bit as u64) << i))
            };
            let square_rows: Vec<u64> = square.iter().map(|row| pack(row)).collect();
            let original_rows: Vec<u64> =
                req.original_operator.iter().map(|row| pack(row)).collect();
            let right_rows: Vec<u64> = (0..n)
                .map(|coordinate| {
                    req.right_vectors
                        .iter()
                        .enumerate()
                        .fold(0u64, |word, (i, vector)| {
                            word | ((vector[coordinate] as u64) << i)
                        })
                })
                .collect();
            let relations: Vec<Vec<u64>> = req
                .relation_coefficients
                .iter()
                .map(|relation| {
                    relation
                        .iter()
                        .map(|coefficient| pack(coefficient))
                        .collect()
                })
                .collect();
            let result = bm::reconstruct_dense(
                &square_rows,
                &original_rows,
                &right_rows,
                &relations,
                &req.relation_boundary_exponents,
            );
            let unpack = |word: u64, length: usize| {
                (0..length)
                    .map(|i| ((word >> i) & 1) as u16)
                    .collect::<Vec<_>>()
            };
            reconstruction = Some(Reconstruction {
                dimension: n,
                right_width: width,
                rows: result
                    .rows
                    .into_iter()
                    .map(|row| ReconstructionRow {
                        raw_candidate: unpack(row.raw_candidate, n),
                        shifted_candidate: unpack(row.shifted_candidate, n),
                        square_residual: unpack(row.square_residual, n),
                        original_residual: unpack(
                            row.original_residual,
                            req.original_operator.len(),
                        ),
                        square_kernel: row.square_kernel,
                        original_kernel: row.original_kernel,
                    })
                    .collect(),
                independent_verified: result
                    .independent_verified
                    .into_iter()
                    .map(|row| unpack(row, n))
                    .collect(),
            });
        }
        "dense_sequence" => {
            let square = &req.square_operator;
            let n = square.len();
            assert!(
                n > 0 && n <= 64,
                "dense-sequence square dimension must be in 1..=64"
            );
            assert!(
                square
                    .iter()
                    .all(|row| { row.len() == n && row.iter().all(|&value| value <= 1) }),
                "square_operator must be a square binary matrix"
            );
            let left_width = req.left_vectors.len();
            let right_width = req.right_vectors.len();
            assert!(left_width > 0 && left_width <= 64);
            assert!(right_width > 0 && right_width <= 64);
            assert!(
                req.left_vectors
                    .iter()
                    .chain(&req.right_vectors)
                    .all(|row| { row.len() == n && row.iter().all(|&value| value <= 1) }),
                "dense-sequence vectors must be rectangular binary blocks"
            );
            assert!(req.steps > 0, "dense_sequence needs a positive step count");
            let pack = |row: &[u16]| {
                row.iter()
                    .enumerate()
                    .fold(0u64, |word, (i, &bit)| word | ((bit as u64) << i))
            };
            let square_rows: Vec<u64> = square.iter().map(|row| pack(row)).collect();
            let left_rows: Vec<u64> = req.left_vectors.iter().map(|row| pack(row)).collect();
            let right_rows: Vec<u64> = (0..n)
                .map(|coordinate| {
                    req.right_vectors
                        .iter()
                        .enumerate()
                        .fold(0u64, |word, (i, vector)| {
                            word | ((vector[coordinate] as u64) << i)
                        })
                })
                .collect();
            dense_sequence = Some(
                bm::dense_krylov_sequence(&square_rows, &left_rows, &right_rows, req.steps)
                    .into_iter()
                    .map(|term| {
                        term.into_iter()
                            .map(|row| (0..right_width).map(|i| ((row >> i) & 1) as u16).collect())
                            .collect()
                    })
                    .collect(),
            );
        }
        "krylov" => {
            // Block Krylov sequence for B = M^T M (symmetric, N x N):
            //   a_i = Z^T B^(i+1) Y,   B v = M^T (M v)
            // Each term is BLOCK_BITS x BLOCK_BITS and feeds block Berlekamp-Massey.
            // Terms are emitted incrementally so a generator that stabilises early
            // can stop the sequence rather than running the modelled bound.
            let threads = if req.threads == 0 { 1 } else { req.threads };
            let steps = if req.steps == 0 { 4 } else { req.steps };
            let nx = op.n_cols as usize * BLOCK_WORDS;
            let ny = op.n_rows as usize * BLOCK_WORDS;
            let mut seed = 0xb5026f5aa96619e9u64 ^ (req.k as u64) << 20;
            let mut rng = move || {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                seed
            };
            let mut v = vec![0u64; nx];
            let mut z = vec![0u64; nx];
            for t in v.iter_mut() {
                *t = rng();
            }
            for t in z.iter_mut() {
                *t = rng();
            }
            let mut tmp = vec![0u64; ny];
            // Both working vectors are allocated once.  Allocating (and zeroing)
            // the 2.47 GB column vector per step cost more than the transpose.
            let mut nv = vec![0u64; nx];
            let mut term_digests = Vec::with_capacity(steps);
            let mut t_proj = 0.0f64;
            let mut t_op = 0.0f64;
            // Coppersmith's reconstruction observes the sequence from y = B z, so
            // the recurrence constrains B W rather than W.  rust/blrt_xl_worker
            // does the same; projecting Y before the first application would give
            // a_0 = Z^T B^0 Y and leave B W unconstrained.
            {
                op.apply(&v, &mut tmp, threads);
                op.apply_transpose(&tmp, &mut nv, threads);
                std::mem::swap(&mut v, &mut nv);
            }
            let t0 = std::time::Instant::now();
            for _ in 0..steps {
                // a_i = Z^T B^(i+1) Y : the full BLOCK_BITS x BLOCK_BITS matrix.
                let tp = std::time::Instant::now();
                let a = operator::project(&z, &v, op.n_cols, threads);
                t_proj += tp.elapsed().as_secs_f64();
                let mut h: u64 = 0xcbf29ce484222325;
                for &w in a.iter() {
                    h ^= w;
                    h = h.wrapping_mul(0x100000001b3);
                }
                term_digests.push(format!("{h:016x}"));
                let to = std::time::Instant::now();
                op.apply(&v, &mut tmp, threads);
                op.apply_transpose(&tmp, &mut nv, threads);
                t_op += to.elapsed().as_secs_f64();
                std::mem::swap(&mut v, &mut nv);
            }
            let el = t0.elapsed().as_secs_f64() / steps as f64;
            let full = 2 * (op.n_cols / BLOCK_BITS as u64 + 1);
            krylov = Some(Krylov {
                steps,
                block_bits: BLOCK_BITS,
                seconds_per_step: el,
                seconds_projection: t_proj / steps as f64,
                seconds_operator: t_op / steps as f64,
                term_digests,
                projected_full_sequence_length: full,
                projected_seconds: el * full as f64,
            });
        }
        "bench" => {
            let threads = if req.threads == 0 { 1 } else { req.threads };
            let reps = if req.reps == 0 { 1 } else { req.reps };
            let nx = op.n_cols as usize * BLOCK_WORDS;
            let ny = op.n_rows as usize * BLOCK_WORDS;
            let mut x = vec![0u64; nx];
            let mut y = vec![0u64; ny];
            let mut seed = 0x9e3779b97f4a7c15u64;
            for v in x.iter_mut() {
                seed ^= seed << 13;
                seed ^= seed >> 7;
                seed ^= seed << 17;
                *v = seed;
            }
            let t0 = std::time::Instant::now();
            op.apply(&x, &mut y, threads);
            let plan_seconds = t0.elapsed().as_secs_f64();
            let t1 = std::time::Instant::now();
            for _ in 0..reps {
                op.apply(&x, &mut y, threads);
            }
            let el = t1.elapsed().as_secs_f64() / reps as f64;
            let mut xt = vec![0u64; nx];
            op.apply_transpose(&y, &mut xt, threads);
            let t2 = std::time::Instant::now();
            for _ in 0..reps {
                op.apply_transpose(&y, &mut xt, threads);
            }
            let elt = t2.elapsed().as_secs_f64() / reps as f64;
            std::hint::black_box(&xt);
            let mut h: u64 = 0xcbf29ce484222325;
            for &v in y.iter() {
                h ^= v;
                h = h.wrapping_mul(0x100000001b3);
            }
            let nnz = op.nnz() as f64;
            bench = Some(Bench {
                threads,
                reps,
                seconds_per_matvec: el,
                nnz_per_second: nnz / el,
                plan_seconds,
                seconds_per_transpose: elt,
                y_checksum: format!("{h:016x}"),
                x_bytes: (nx * 8) as u64,
                y_bytes: (ny * 8) as u64,
            });
            std::hint::black_box(&y);
        }
        other => panic!("unknown mode {other}"),
    }

    let resp = Response {
        schema: "mcelix-holdout-supplier-operator-response-v1".to_string(),
        k: op.k,
        d: op.d,
        n_cols: op.n_cols,
        n_rows: op.n_rows,
        nnz: op.nnz(),
        dense_rows,
        image_rows,
        bench,
        adjoint_lhs,
        adjoint_rhs,
        krylov,
        approximant,
        reconstruction,
        dense_sequence,
    };
    println!("{}", serde_json::to_string(&resp).expect("serialize"));
}
