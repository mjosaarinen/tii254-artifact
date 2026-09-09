# Recovery method

## Relation supplier

The upstream calculation uses the degree-six Hermite/Holdout relation
operator.  After removing the explicit `L_star` nuisance family and
conditioning on one ordinary point, the implementation works with the square
binary operator

```text
A = R E J,       dim A = 15,527,170.
```

`J` is the coefficient-bearing section of the nuisance quotient, `E` is the
literal Tensor--Bier equation map, and `R` is a systematic Toeplitz range
compression.  Every reconstructed relation is subsequently replayed under
the uncompressed `E`, so compression cannot create a false accepted key.

A width-512 block Wiedemann computation supplies a projected sequence.  CADO
`lingen` computes a shifted order basis, and the extracted right generator
reconstructs

```text
Q = sum_j A^j Y F_j.
```

The finite-Hankel rank and reconstructed rank both give 121 for each of the
two conditioned kernels used in the successful attack.  Their intersection
has dimension 106.

## Pair core and direct carrier

The two complete kernels are restricted by all pair-incidence conditions.  A
dual certificate selects an 80-dimensional pair core `C` inside their
122-dimensional physical sum.  Its intersection with the common two-anchor
candidate is

```text
N = C intersect common,       dim N = 64.
```

Every ordinary-point local kernel inside `C` has dimension 72 and contains
`N`.  Therefore

```text
C/N                         has dimension 16,
K_j/N                       has dimension 8 for all 87 points,
(K_i/N) + (K_j/N)           has dimension 16 for i != j.
```

This is a binary spread.  `data/pair_core.json` contains exactly the small
coordinate spaces needed to recompute these statements.

## Graph pencil

Choose three frozen local kernels as projective `0`, `infinity`, and `1`.
The first two split the 16-space as `U + V`.  Every other eight-space is the
graph of a binary map `T_j: U -> V`.  Over GF(256), a separating graph map has
eight one-dimensional eigenspaces.  Simultaneous diagonalization of the other
graphs identifies one coherent branch per eigenspace.

For this degree-six construction the graph eigenvalues are fourth powers of
normalized locator cross-ratios.  Since `x -> x^4` is a field automorphism,
two inverse Frobenius rounds recover the locators without ambiguity.  The
eight outputs are one full coefficientwise-Frobenius orbit; any coherent
branch can produce an equivalent binary Goppa key.

## Projective one-pivot finisher

The graph pencil labels 87 ordinary locators.  A public 47-column circuit of
rank 46 contains one omitted label.  For each projective pole outside the
known support and each candidate value of the omitted point, the finisher:

1. recharts the known projective locators to affine coordinates;
2. forms the derivative numerator of the public circuit;
3. uses characteristic two to square-root its degree-46 numerator;
4. factors the degree-23 square root and retains degree-12 factors;
5. checks the restricted public row space and two public semantic families;
6. completes the remaining support with known-polynomial linear algebra; and
7. verifies the complete 96-by-223 public row space.

The successful branch examined 1,727 numerator instances and found 152 degree-12
factor candidates.  All 152 passed the restricted row-space test; the semantic
screens rejected 151, leaving one survivor.  The search stopped at the first
verified key.  Nine completion steps produced the equivalent
key in `data/recovered_key.json`.
