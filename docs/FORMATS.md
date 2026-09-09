# Large-stage interfaces

This note describes the reusable computation boundaries.  The artifact omits
the large payloads themselves.

## Projected sequence

The CUDA Krylov worker applies

```text
A = R E J : GF(2)^15,527,170 -> GF(2)^15,527,170
```

to a 512-column bit panel.  Rows are packed consecutively in little-endian
64-bit words; one row occupies eight words.  The projected sequence terms are
512-by-512 binary matrices in row-major, little-endian `u64` layout.

The successful calculation used 60,718 terms and restart checkpoints every
4,096 applications.  Label 2 is the default build.  Define
`MCELIECEX_TII254_LSTAR_REJ_ANCHOR5=1` for label 5.

The executable interface is printed by:

```sh
cuda/bin/lstar-krylov-label2
```

Its inputs are:

- the text description of the degree-six literal operator;
- an anchor codec directory;
- the original-to-slice row permutation (`u32` little endian);
- the systematic Toeplitz compressor diagonal;
- the packed left-projection rows;
- an earlier state or `-`; and
- an output directory, step count, and GPU batch size.

The copied CUDA code retains strict shape and request checks from the target
implementation.  For another instance, replace those constants and request
parsers while retaining the operator, codec, cuFFT Toeplitz, and adjoint
kernels.

## CADO-NFS shifted order basis

Compile the adapters with `make tools`.  If `sequence.bin` contains `terms`
projected matrices of shape `source_left_bits` by `source_right_bits`, convert
its selected upper-left panel with:

```sh
build/sequence-to-cado sequence.bin series.single.data \
  TERMS SOURCE_LEFT_BITS SOURCE_RIGHT_BITS LEFT_WIDTH RIGHT_WIDTH
python3 scripts/make_cado_aux.py series.aux \
  --left LEFT_WIDTH --right RIGHT_WIDTH --terms TERMS
```

The converter writes the coefficient stream for `[S(x)^T | I]`.  The auxiliary
file uses the column shift `[0]*right + [1]*left`.  Invoke a CADO-NFS `lingen`
build using its direct coefficient input and the generated auxiliary file.
The resulting `series.aux.pi` is coefficient-major.

Extract a right generator with:

```sh
build/extract-relations series.aux.pi LEFT_WIDTH RIGHT_WIDTH TERMS relations.bin
```

The output is coefficient-major.  Each coefficient is a `right`-by-`right`
binary matrix in row-major little-endian words.  The extractor selects valid
rows in shifted-degree order and reverses the top polynomial so that

```text
Q = sum_j A^j Y F_j
```

is the reconstruction convention.

## Reconstruction

`lstar-reconstruct-label2` and `lstar-reconstruct-label5` consume disjoint
coefficient intervals.  Each shard begins at the matching Krylov checkpoint
and emits one full-size panel contribution.  XORing an exact interval
partition produces `Q`; the combine path then checks `A Q = 0`.

That check is necessary but not sufficient.  The independent Rust program in
`rust/original_ej_replay` scatters `JQ` into literal coordinates, evaluates the
CPU Tensor--Bier transpose, and requires the stronger original-coordinate
identity `E(JQ)=0`.

## Compact pair core

`data/pair_core.json` begins after the two complete anchor kernels have been
constructed and the dual pair-core certificate has selected the 80-space.
Hexadecimal strings encode binary row vectors, least significant bit first in
the named coordinate space:

- 122 physical-sum representatives lift physical coordinates to the two
  anchor source spaces;
- 80 pair-core rows use the 122 physical coordinates;
- 106 common-candidate rows use the same physical coordinates; and
- each of 87 local kernels supplies 72 rows in pair-core coordinates.

`recover_locators.py` intersects the pair core with the common candidate,
obtains dimension 64, selects a deterministic 16-dimensional quotient
section, and maps every local kernel to an eight-space in that quotient.
