# Bundled inputs

The JSON files here are deliberately computation-oriented.  They contain no
file paths, job metadata, provenance chains, or integrity digests.

- `pair_core.json` is the compact handoff from the expensive two-anchor
  relation calculation to the graph pencil.
- `finisher.json` contains the public circuit and semantic support families
  used by the one-pivot finisher.
- `recovered_key.json` is the equivalent support and Goppa polynomial.
- `public_key.txt` is the 96-by-223 challenge matrix followed by the nine
  coefficients of the GF(256) modulus.

The pair-core rows are hexadecimal encodings of binary vectors.  Their
coordinate spaces are detailed in `../docs/FORMATS.md`.
