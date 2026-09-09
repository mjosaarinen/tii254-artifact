#!/usr/bin/env sage
"""Verify that the supplied equivalent Goppa key reproduces TII-254."""

import argparse
import json
from pathlib import Path
import sys

from sage.all import PolynomialRing, matrix


ROOT = Path.cwd().resolve()
sys.path.insert(0, str(ROOT / "python"))
from tii254.public_key import parse_public_key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, default=ROOT / "data/public_key.txt")
    parser.add_argument("--key", type=Path, default=ROOT / "data/recovered_key.json")
    args = parser.parse_args()

    key = json.loads(args.key.read_text(encoding="ascii"))
    m = int(key["field_degree"])
    coefficients = key["goppa_coefficients"]
    t = len(coefficients) - 1
    support_integers = key["support_integers"]
    H, field, modulus_bits = parse_public_key(
        args.public, m=m, t=t, n=len(support_integers)
    )
    modulus_integer = sum(bit << i for i, bit in enumerate(modulus_bits))
    if modulus_integer != int(key["field_modulus"]):
        raise ArithmeticError("key and public field moduli differ")

    support = [field.from_integer(value) for value in support_integers]
    ring = PolynomialRing(field, "X")
    X = ring.gen()
    goppa = sum(
        field.from_integer(value) * X**i for i, value in enumerate(coefficients)
    )
    if len(set(support)) != len(support):
        raise ArithmeticError("the support is not distinct")
    if goppa.degree() != t or goppa.leading_coefficient() != 1:
        raise ArithmeticError("the Goppa polynomial has the wrong shape")
    if not goppa.is_irreducible() or any(goppa(point) == 0 for point in support):
        raise ArithmeticError("the Goppa polynomial is invalid on the support")

    multipliers = [1 / goppa(point) for point in support]
    reconstructed = matrix(field, m * t, len(support))
    for frobenius in range(m):
        power = 1 << frobenius
        for degree in range(t):
            row = frobenius * t + degree
            reconstructed[row] = [
                (multipliers[column] * support[column] ** degree) ** power
                for column in range(len(support))
            ]
    if reconstructed.rank() != m * t:
        raise ArithmeticError("the reconstructed check matrix lost rank")
    if reconstructed.row_space() != H.change_ring(field).row_space():
        raise ArithmeticError("the equivalent key does not reproduce the public code")
    print(
        "PASS: TII-254 row space reproduced (%dx%d, GF(2^%d), degree %d)"
        % (H.nrows(), H.ncols(), m, t)
    )


main()
