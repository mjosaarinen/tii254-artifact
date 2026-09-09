# TII-254 recovery artifact

This repository accompanies the paper:

> Markku-Juhani O. Saarinen.  
> **Two-Anchor Holdout/Hermite: Solving the TII-254 McEliece Key Recovery Challenge**.  
> IACR Cryptology ePrint Archive, Report 2026/XXXX, 2026.  
> <https://eprint.iacr.org/2026/XXXX>

The ePrint report number is pending; `2026/XXXX` is a placeholder.

Local preprint copy: [tii254.pdf](tii254.pdf).

```bibtex
@misc{cryptoeprint:2026/XXXX,
      author = {Markku-Juhani O. Saarinen},
      title = {Two-Anchor Holdout/Hermite: Solving the {TII-254} {McEliece} Key Recovery Challenge},
      howpublished = {Cryptology {ePrint} Archive, Paper 2026/XXXX},
      year = {2026},
      url = {https://eprint.iacr.org/2026/XXXX}
}
```

This repository is a compact, reusable implementation artifact for the
successful recovery of an equivalent binary Goppa key for the TII-254
challenge.  It is organized around the computation rather than the history of
the experiment: there are no Slurm scripts, job records, provenance chains, or
large checkpoints.

The complete expensive calculation is not bundled.  Instead, the artifact
contains:

- the public TII-254 matrix and the recovered equivalent key;
- a compact 200 KB checkpoint at the two-anchor pair-core boundary;
- exact binary quotient and GF(256) graph-pencil code;
- the projective one-pivot finisher and known-polynomial completion code;
- a standalone full-key verifier;
- the CUDA source used for the matrix-free relation supplier and parallel
  reconstruction;
- the CADO-NFS input and relation-extraction adapters;
- the Rust literal-operator implementation used for independent `E(JQ)=0`
  replay; and
- a pinned Lean project proving the reusable linear-algebra theorems.

The main local reproduction starts at the compact pair core.  It recomputes
the locators and key; it does not merely print the stored answer.

## Quick start

Requirements for the compact reproduction are Python 3 and SageMath 10.x.

```sh
make locators
make verify
```

`make locators` performs the exact `80 -> 64 + 16` quotient, checks all 87
eight-dimensional local kernels and all 3,741 pair sums, and recovers eight
Frobenius-conjugate locator branches.  It takes a few seconds.

`make verify` checks the supplied equivalent key from first principles by
reconstructing the 96-dimensional parity-check row space over GF(256).

If the Sage launcher does not support `-python`, run the verifier directly:

```sh
sage verify_key.sage
```

To repeat the actual locator-to-key search and verify its output:

```sh
make reproduce
```

This enumerates projective poles and the omitted circuit point, factors the
degree-46 derivative numerator through its degree-23 square root, applies the
two public semantic screens, completes the support, and checks the full public
row space.  It normally takes under two minutes on a workstation.  Generated
files go under the ignored `build/` directory.

If Sage is installed in an environment rather than as `sage`, override the
commands.  For example:

```sh
make reproduce \
  SAGE_PYTHON='mamba run -n sage python'
```

## What was computed

The successful route was:

```text
public TII-254 parity check
  -> degree-six Tensor--Bier relation operator E
  -> quotient by the explicit L_star nuisance space
  -> two anchor-conditioned 15,527,170-square operators A = R E J
  -> width-512 block Wiedemann sequences (CUDA)
  -> shifted PM basis (CADO-NFS lingen)
  -> parallel relation reconstruction (CUDA)
  -> independent literal E(JQ)=0 replay (Rust)
  -> two complete 121-dimensional anchor kernels K_2 and K_5
  -> certified 80-dimensional pair core
  -> remove its common 64-dimensional nuisance
  -> 16-dimensional spread with 87 local eight-spaces
  -> GF(256) graph pencil and eight locator branches
  -> one-pivot projective finisher
  -> equivalent degree-12 Goppa key
  -> exact equality with the public 96-by-223 row space
```

The equivalent key is not claimed to be the challenge author's original
support and polynomial.  It is a complete decoding key for the same public
code.

The two Krylov sequences used one GH200 each for about 13.6 hours, totalling
27.2 GPU-hours.  GPU reconstruction and CPU processing were additional.
Label-5 reconstruction took 1 h 55 min 23 s of elapsed computing time, using
one GPU first, then seven in parallel.  Label 2 used eight GPUs in parallel,
but its complete reconstruction time was not recorded.  These times exclude
queueing and do not give a total cost for the full recovery.
These resources are not a requirement of the compact reproduction.

## Directory map

- [`tii254.pdf`](tii254.pdf): preprint copy shipped with the artifact.
- `data/public_key.txt`: the challenge parity check and field modulus.
- `data/pair_core.json`: compact input to locator recovery.
- `data/finisher.json`: public circuit and semantic-screen inputs.
- `data/recovered_key.json`: the recovered equivalent key.
- `python/tii254/`: binary linear algebra, graph pencil, and key finisher.
- `scripts/recover_locators.py`: compact pair-core-to-locators driver.
- `scripts/recover_key.py`: Sage locator-to-key driver.
- `verify_key.sage`: standalone public-row-space verification.
- `cuda/`: production-derived CUDA operator, Krylov, and reconstruction code.
- `scripts/sequence_to_cado.c`: constant-memory CADO input converter.
- `scripts/extract_relations.cpp`: shifted-basis relation extractor.
- `rust/`: CPU Tensor--Bier operator and literal replay executable.
- [`lean/`](lean/README.md): pinned Lean proofs and an axiom audit.
- `docs/FORMATS.md`: binary interfaces for large-input reuse.

## Building reusable native components

```sh
make tools
make rust
make cuda CUDA_ARCH=90
```

Use `CUDA_ARCH=60` for a Tesla P100 and `CUDA_ARCH=90` for a GH200.  The CUDA
tree intentionally contains no scheduler logic.  Its target-scale executables
expect the packed operator/codec inputs described in `docs/FORMATS.md`; those
multi-gigabyte inputs are not included.

`make test` runs the compact pair-core regression.  `make cuda-smoke` is not a
top-level target; use `make -C cuda smoke CUDA_ARCH=...` on a CUDA host.

## Lean proofs

The project in `lean/` pins Lean 4.33.0 and its Mathlib dependencies.  From
the artifact root:

```sh
cd lean
lake exe cache get
lake build
python3 audit.py
```

The proofs cover coordinate deflation, the shifted-Hankel completeness
bound, relative anchor completion, the dual pair-core certificate, and
supporting observer and Toeplitz results.  The audit rejects proof holes and
checks that the named theorems use only the standard axioms `propext`,
`Classical.choice`, and `Quot.sound`.

These are abstract linear-algebra theorems.  Lean does not run the recovery
code or verify the challenge-specific large matrices and rank computations.
The compact checkpoint supports replay from the pair core onward; it does
not supply the large panels needed to reconstruct and certify that core
from the public matrix.  See [lean/README.md](lean/README.md).

## Scope

The compact checkpoint makes the downstream scientific result easy to replay.
Recreating it from the public matrix remains an HPC calculation and requires
constructing the degree-six operator and `L_star` codec inputs.  The bundled
native sources are intended to support that work and reuse in related
experiments, but this is not a turnkey cloud/HPC deployment.

The MIT license in `LICENSE` covers the artifact glue, compact Python
package, and Lean sources.  The Rust crates retain their `GPL-2.0-or-later`
declarations and are accompanied by `rust/COPYING`.
