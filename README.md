# TII-254 recovery artifact

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
- the CUDA source used for the matrix-free relation supplier and sharded
  reconstruction;
- the CADO-NFS input and relation-extraction adapters; and
- the Rust literal-operator implementation used for independent `E(JQ)=0`
  replay.

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
  -> sharded relation reconstruction (CUDA)
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

The measured expensive part used one GH200 for each roughly 13.6-hour Krylov
sequence, 64 CPU cores for PM basis, and eight concurrent GH200 reconstruction
shards.  Those figures describe the run, not a requirement of the compact
reproduction.

## Directory map

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

## Scope

The compact checkpoint makes the downstream scientific result easy to replay.
Recreating it from the public matrix remains an HPC calculation and requires
constructing the degree-six operator and `L_star` codec inputs.  The bundled
native sources are intended to support that work and reuse in related
experiments, but this is not a turnkey cloud/HPC deployment.

The MIT license in `LICENSE` covers the artifact glue and compact Python
package.  The Rust crates retain their `GPL-2.0-or-later` declarations and are
accompanied by `rust/COPYING`.
