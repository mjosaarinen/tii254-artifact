# CUDA relation supplier

These are the production-derived CUDA kernels from the successful TII-254
calculation, separated from scheduler and evidence-record code.

The important executables are:

- `holdout-worker`: generic Tensor--Bier operator and self-tests;
- `lstar-anchor-label2`: label-2 `L_star` anchor codec;
- `lstar-anchor-variable`: variable-anchor codec used for label 5;
- `lstar-krylov-label{2,5}`: restartable width-512 projected sequences; and
- `lstar-reconstruct-label{2,5}`: coefficient-sharded right-generator
  reconstruction and `A Q=0` replay.

Build for a P100 or GH200 with:

```sh
make CUDA_ARCH=60
make clean && make CUDA_ARCH=90
```

The workers need cuFFT and target codec inputs; only their source is bundled.
Run the implementation-level self-tests with:

```sh
make smoke CUDA_ARCH=60
```

The label-2 and label-5 workers differ only in their anchor request policy and
frozen right-start seed.  The sparse operator, `L_star` codec, systematic
Toeplitz range compression, Krylov recurrence, and reconstruction kernels are
shared.

For reuse on another instance, the principal instance-specific portions are
the dimension constants and request readers near the tops of
`tii254_lstar_rej_cufft_worker.cu` and
`tii254_lstar_rej_krylov_worker.cu`.  The packed layouts are described in
`../docs/FORMATS.md`.
