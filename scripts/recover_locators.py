#!/usr/bin/env python3
"""Recover the eight Frobenius locator branches from the compact pair core."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))

from tii254.binary_field import BinaryField  # noqa: E402
from tii254.pair_core import recover_direct_pair_core_graph_pencil  # noqa: E402


def hex_rows(values):
    return tuple(int(value, 16) for value in values)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=ROOT / "data/pair_core.json")
    parser.add_argument("--output", type=Path, default=ROOT / "build/locators.json")
    args = parser.parse_args()

    record = json.loads(args.input.read_text(encoding="ascii"))
    recovered = recover_direct_pair_core_graph_pencil(
        observer_labels=record["observer_labels"],
        physical_sum_representatives=hex_rows(
            record["physical_sum_representatives_hex"]
        ),
        pair_core_basis=hex_rows(record["pair_core_basis_hex"]),
        common_physical_basis=hex_rows(record["common_physical_basis_hex"]),
        core_local_kernel_bases=tuple(
            hex_rows(rows) for rows in record["core_local_kernel_bases_hex"]
        ),
        field=BinaryField(record["field_degree"], record["field_modulus"]),
        locator_power_rounds=record["locator_power_rounds"],
    )
    locators = recovered.locators
    output = {
        "physical_labels": list(recovered.observer_labels),
        "anchor_indices": list(recovered.anchor_indices),
        "anchor_labels": list(recovered.anchor_labels),
        "field_degree": locators.field_degree,
        "field_modulus": locators.field_modulus,
        "locator_power_rounds": locators.locator_power_rounds,
        "dimensions": {
            "physical_sum": recovered.physical_sum_dimension,
            "pair_core": recovered.pair_core_dimension,
            "common_nuisance": recovered.common_nuisance_dimension,
            "quotient": recovered.quotient_dimension,
            "local_kernel": sorted(set(recovered.local_kernel_dimensions)),
            "quotient_local_kernel": sorted(
                set(recovered.quotient_local_kernel_dimensions)
            ),
            "pairwise_quotient_sum": sorted(
                set(recovered.pairwise_quotient_sum_dimensions)
            ),
        },
        "branch_count": locators.branch_count,
        "frobenius_permutation": list(locators.frobenius_permutation),
        "projective_locators_by_branch": [
            [list(point) for point in branch]
            for branch in locators.projective_locators_by_branch
        ],
        "checks": {
            "all_kernels_replay": locators.all_kernels_replay,
            "all_locator_powers_replay": locators.all_locator_powers_replay,
            "full_frobenius_orbit": locators.full_frobenius_orbit,
            "all_points_distinct": all(locators.all_points_distinct_by_branch),
        },
    }
    if not all(output["checks"].values()):
        raise ArithmeticError("locator recovery failed an exact replay check")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2) + "\n", encoding="ascii")
    print(
        "recovered %d locator branches from a %d-dimensional quotient"
        % (locators.branch_count, recovered.quotient_dimension)
    )


if __name__ == "__main__":
    main()
