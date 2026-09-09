//! Colexicographic ranking of fixed-size subsets of `[0,k)`.
//!
//! A `d`-subset is carried as a `u64` bitmask.  Colex order ranks
//! `S = {s_0 < s_1 < ... < s_{d-1}}` as `sum_i C(s_i, i+1)`, which is the
//! order the reference model and the Python driver both use.

/// Binomial coefficients up to `MAXK`, as exact `u64`.
pub struct Binom {
    table: Vec<u64>,
    n: usize,
}

impl Binom {
    pub fn new(n: usize) -> Self {
        let mut table = vec![0u64; (n + 1) * (n + 1)];
        for i in 0..=n {
            table[i * (n + 1)] = 1;
            for j in 1..=i {
                let a = table[(i - 1) * (n + 1) + j - 1];
                let b = table[(i - 1) * (n + 1) + j];
                table[i * (n + 1) + j] = a + b;
            }
        }
        Self { table, n }
    }

    #[inline]
    pub fn c(&self, a: usize, b: usize) -> u64 {
        if b > a || a > self.n {
            0
        } else {
            self.table[a * (self.n + 1) + b]
        }
    }
}

/// Colex rank of the `d`-subset given by `mask`.
#[inline]
pub fn colex_rank(mask: u64, binom: &Binom) -> u64 {
    let mut m = mask;
    let mut rank = 0u64;
    let mut i = 1usize;
    while m != 0 {
        let s = m.trailing_zeros() as usize;
        rank += binom.c(s, i);
        m &= m - 1;
        i += 1;
    }
    rank
}

/// Inverse of [`colex_rank`] for subsets of size `d`.
pub fn colex_unrank(mut rank: u64, d: usize, k: usize, binom: &Binom) -> u64 {
    let mut mask = 0u64;
    for i in (1..=d).rev() {
        // largest s with C(s, i) <= rank
        let mut s = i - 1;
        while s + 1 <= k && binom.c(s + 1, i) <= rank {
            s += 1;
        }
        mask |= 1u64 << s;
        rank -= binom.c(s, i);
    }
    mask
}

/// All `size`-subsets of the ground set `ground` (a bitmask), in colex order
/// of their *local* positions within `ground`.
pub fn subsets_of(ground: u64, size: usize) -> Vec<u64> {
    let bits: Vec<u32> = (0..64).filter(|b| ground >> b & 1 == 1).collect();
    let w = bits.len();
    let mut out = Vec::new();
    if size > w {
        return out;
    }
    let mut idx: Vec<usize> = (0..size).collect();
    loop {
        let mut m = 0u64;
        for &i in &idx {
            m |= 1u64 << bits[i];
        }
        out.push(m);
        // next combination in colex order
        let mut i = 0usize;
        while i < size {
            let limit = if i + 1 == size { w } else { idx[i + 1] };
            if idx[i] + 1 < limit {
                idx[i] += 1;
                for j in 0..i {
                    idx[j] = j;
                }
                break;
            }
            i += 1;
        }
        if i == size {
            break;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rank_unrank_roundtrip() {
        let binom = Binom::new(24);
        for d in 1..=5usize {
            let total = binom.c(20, d);
            for r in 0..total {
                let m = colex_unrank(r, d, 20, &binom);
                assert_eq!(m.count_ones() as usize, d);
                assert_eq!(colex_rank(m, &binom), r, "d={d} r={r}");
            }
        }
    }

    #[test]
    fn colex_is_monotone() {
        let binom = Binom::new(16);
        let all = subsets_of((1u64 << 12) - 1, 4);
        let mut ranks: Vec<u64> = all.iter().map(|&m| colex_rank(m, &binom)).collect();
        let sorted = {
            let mut s = ranks.clone();
            s.sort();
            s
        };
        assert_eq!(ranks, sorted, "subsets_of must emit colex order");
        ranks.dedup();
        assert_eq!(ranks.len(), binom.c(12, 4) as usize);
    }
}
