#!/usr/bin/env python3
"""Write CADO lingen's direct GF(2) auxiliary file for [S(x)^T | I]."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument("--left", type=int, required=True)
    parser.add_argument("--right", type=int, required=True)
    parser.add_argument("--terms", type=int, required=True)
    args = parser.parse_args()
    if min(args.left, args.right, args.terms) <= 0:
        raise ValueError("dimensions and term count must be positive")
    shifts = [0] * args.right + [1] * args.left
    payload = "\n".join(
        [
            "format 4",
            f"{args.left} {args.right}",
            f"0 0 {args.terms} 0",
            str(args.terms),
            "2",
            " " + " ".join(map(str, shifts)),
            " " + " ".join("0" for _ in shifts),
            "0",
            "",
        ]
    )
    args.output.write_text(payload, encoding="ascii")


if __name__ == "__main__":
    main()
