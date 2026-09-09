#!/usr/bin/env python3
"""Run the projective one-pivot finisher and emit an equivalent Goppa key."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from time import perf_counter


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from tii254.public_key import parse_public_key  # noqa: E402
from tii254.tii254_d6_all_ordinary_projective_finisher import (  # noqa: E402
    AllOrdinaryProjectiveFinisherRefusal,
    finish_all_ordinary_projective_support,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--public", type=Path, default=ROOT / "data/public_key.txt")
    parser.add_argument("--locators", type=Path, default=ROOT / "build/locators.json")
    parser.add_argument("--config", type=Path, default=ROOT / "data/finisher.json")
    parser.add_argument("--output", type=Path, default=ROOT / "build/recovered_key.json")
    parser.add_argument("--screen-cap", type=int)
    args = parser.parse_args()

    config = json.loads(args.config.read_text(encoding="ascii"))
    locator_record = json.loads(args.locators.read_text(encoding="ascii"))
    H, field, modulus_bits = parse_public_key(
        args.public, m=config["m"], t=config["t"], n=config["n"]
    )
    modulus_integer = sum(bit << index for index, bit in enumerate(modulus_bits))
    if modulus_integer != config["field_modulus"]:
        raise ArithmeticError("public and finisher field conventions differ")
    if any(int(field.from_integer(i).to_integer()) != i for i in range(256)):
        raise ArithmeticError("Sage changed the polynomial-basis integer encoding")

    labels = tuple(map(int, locator_record["physical_labels"]))
    branches = tuple(
        tuple(tuple(map(int, point)) for point in branch)
        for branch in locator_record["projective_locators_by_branch"]
    )
    started = perf_counter()
    refusals = []
    recovered = None
    selected_branch = None
    for branch_index, branch in enumerate(branches):
        try:
            candidate = finish_all_ordinary_projective_support(
                H,
                labels,
                branch,
                config["active_circuit_labels"],
                config["outside_label"],
                field=field,
                m=config["m"],
                r=config["t"],
                screen_cap=args.screen_cap,
                semantic_support_families=config["semantic_support_families"],
            )
        except AllOrdinaryProjectiveFinisherRefusal as error:
            refusals.append({"branch": branch_index, "gate": error.gate})
            continue
        recovered = candidate
        selected_branch = branch_index
        break
    if recovered is None:
        raise RuntimeError(f"all locator branches refused: {refusals}")

    completion = recovered.completion.completion
    polynomial = completion.polynomial
    output = {
        "field_degree": config["m"],
        "field_modulus": config["field_modulus"],
        "selected_locator_branch": selected_branch,
        "projective_pole": list(recovered.projective_pole),
        "outside_label": config["outside_label"],
        "outside_projective_point": list(recovered.outside_projective_point),
        "support_integers": [int(value.to_integer()) for value in completion.support],
        "goppa_coefficients": [
            int(polynomial[index].to_integer())
            for index in range(int(polynomial.degree()) + 1)
        ],
        "verification": {
            "completion": dict(completion.verification.checks),
            "fresh": dict(recovered.fresh_verification.checks),
        },
        "work": {
            key: value
            for key, value in recovered.audit.items()
            if "sha256" not in key and key != "selected_semantic_profile"
        },
        "wall_seconds": perf_counter() - started,
    }
    if not all(output["verification"]["completion"].values()) or not all(
        output["verification"]["fresh"].values()
    ):
        raise ArithmeticError("the recovered key failed a verification check")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(
        "recovered and verified an equivalent degree-%d key using locator branch %d"
        % (config["t"], selected_branch)
    )


if __name__ == "__main__":
    main()
