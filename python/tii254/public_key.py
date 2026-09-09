"""Parsing and field construction for the text-form TII public key."""

from __future__ import annotations

from pathlib import Path


def parse_public_key(path: Path, *, m: int = 8, t: int = 12, n: int = 223):
    """Return ``(H, field, modulus_bits)`` using Sage objects."""

    from sage.all import GF, PolynomialRing, matrix

    rows = []
    for number, raw in enumerate(path.read_text(encoding="ascii").splitlines(), 1):
        line = raw.strip()
        if not (line.startswith("[") and line.endswith("]")):
            raise ValueError(f"malformed public-key row {number}")
        row = tuple(int(item) for item in line[1:-1].replace(",", " ").split())
        if any(item not in (0, 1) for item in row):
            raise ValueError(f"nonbinary public-key row {number}")
        rows.append(row)
    if len(rows) != m * t + 1:
        raise ValueError("public key must contain m*t rows followed by a modulus")
    if any(len(row) != n for row in rows[:-1]) or len(rows[-1]) != m + 1:
        raise ValueError("public matrix or modulus has the wrong width")

    binary = GF(2)
    H = matrix(binary, rows[:-1])
    if H.rank() != m * t:
        raise ArithmeticError("public parity-check matrix is not full rank")
    ring = PolynomialRing(binary, "z")
    z = ring.gen()
    modulus = sum(binary(bit) * z**i for i, bit in enumerate(rows[-1]))
    if not modulus.is_irreducible():
        raise ArithmeticError("the supplied field modulus is reducible")
    field = GF(2**m, name="a", modulus=modulus)
    return H, field, rows[-1]
