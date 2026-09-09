# Lean proofs used by the TII-254 paper

This Lake project contains the reusable abstract theorems cited in *Two-Anchor Holdout/Hermite: Solving the TII-254 McEliece Key Recovery Challenge*. It is pinned to Lean 4.33.0 and a fixed Mathlib dependency graph. Run the following commands from this `lean/` directory:

```sh
lake exe cache get
lake build
python3 audit.py
```

The modules cover:

- `CoordinateDeflation`: an information set for a known kernel subspace gives a direct coordinate complement and preserves the original equation;
- `ToeplitzStart`: exact Toeplitz indexing, fixed-quotient visibility, transpose symmetry, and the zero-primary identity;
- `ZeroPrimaryReachability`: full-cokernel reachability, the shifted-Hankel upper bound, and equality of a replayed panel with the complete kernel;
- `RelativeAnchor`: completion of a second kernel from an upper bound, a known intersection, and the measured relative rank increment;
- `TwoAnchorPairCore`: the pair-core dual identity and the witness certificate used to prove exact core equality; and
- `Verification`: soundness of literal basis certificates.

These theorems are general linear algebra. Their application to TII-254 additionally requires the exact finite computations described in the paper. The Lean build does not execute or verify the Sage key checker, CUDA or CPU code, PM-basis implementation, or challenge-specific rank computations. The artifact's compact inputs in `../data/` support recovery from the pair core onward, not reconstruction of the complete anchor kernels or their pair-core certificate. No internal repository or local source path is needed to build these proofs.
