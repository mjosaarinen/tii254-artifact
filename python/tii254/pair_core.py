"""Direct graph-pencil recovery from a certified ``64 + 16`` pair core.

The complete-anchor pair calculation may expose a certified pair core ``C``
whose intersection with the common two-anchor candidate space is a nuisance
space ``N``.  When ``dim(C/N) = 2m`` and every local kernel contains ``N``
with an ``m``-dimensional quotient, the dimension-appropriate consumer is the
ordinary graph pencil.  No ``3m-1`` conic unprojection is involved.

All inputs and outputs here are small coefficient spaces.  Source lifts are
retained so that the enclosing authenticated producer can replay the selected
carrier under the literal relation operator.
"""

from __future__ import annotations

from dataclasses import dataclass

from .binary_field import BinaryField
from .binary_linear import (
    combine as _combine,
    complement_representatives as _complement_representatives,
    coordinates_in_basis as _coordinates_in_independent_basis,
    intersection as _intersection,
    row_basis as _row_basis,
)
from .kernel_pencil import LocatorBranchResult, recover_locator_branches


__all__ = [
    "DirectPairCoreGraphPencil",
    "recover_direct_pair_core_graph_pencil",
]


@dataclass(frozen=True)
class DirectPairCoreGraphPencil:
    physical_sum_dimension: int
    pair_core_dimension: int
    common_nuisance_dimension: int
    quotient_dimension: int
    local_kernel_dimensions: tuple[int, ...]
    quotient_local_kernel_dimensions: tuple[int, ...]
    pairwise_quotient_sum_dimensions: tuple[int, ...]
    observer_labels: tuple[int, ...]
    anchor_indices: tuple[int, int, int]
    anchor_labels: tuple[int, int, int]
    common_nuisance_core_basis: tuple[int, ...]
    common_nuisance_physical_basis: tuple[int, ...]
    quotient_section_core_basis: tuple[int, ...]
    quotient_section_physical_basis: tuple[int, ...]
    quotient_section_source_lifts: tuple[int, ...]
    quotient_local_kernel_bases: tuple[tuple[int, ...], ...]
    quotient_local_kernel_source_lifts: tuple[tuple[int, ...], ...]
    locators: LocatorBranchResult


def _bit_rows(rows: tuple[int, ...], width: int):
    return tuple(
        tuple((value >> coordinate) & 1 for coordinate in range(width))
        for value in rows
    )


def recover_direct_pair_core_graph_pencil(
    *,
    observer_labels,
    physical_sum_representatives,
    pair_core_basis,
    common_physical_basis,
    core_local_kernel_bases,
    field: BinaryField,
    locator_power_rounds: int = 2,
) -> DirectPairCoreGraphPencil:
    """Quotient a certified pair core and run its direct graph pencil.

    ``physical_sum_representatives`` lift physical-sum coordinates to the
    combined source coordinates of the two complete anchor panels.
    ``pair_core_basis`` and ``common_physical_basis`` use physical-sum
    coordinates.  Each local kernel uses coordinates on ``pair_core_basis``.

    The first three observer labels are the frozen projective frame.  There is
    no outcome-dependent search over frames in this consumer.
    """

    labels = tuple(map(int, observer_labels))
    representatives = tuple(map(int, physical_sum_representatives))
    physical_width = len(representatives)
    core = _row_basis(tuple(map(int, pair_core_basis)), physical_width)
    common = _row_basis(tuple(map(int, common_physical_basis)), physical_width)
    m = int(field.degree)
    if (
        len(labels) < 4
        or len(labels) != len(set(labels))
        or len(core) != 10 * m
        or len(common) < 8 * m
        or len(core_local_kernel_bases) != len(labels)
    ):
        raise ValueError("direct pair-core input shape differs")

    core_common_physical = _intersection(core, common, physical_width)
    core_width = len(core)
    core_common = _row_basis(
        tuple(
            _coordinates_in_independent_basis(core, row, physical_width)
            for row in core_common_physical
        ),
        core_width,
    )
    quotient_width = core_width - len(core_common)
    if len(core_common) != 8 * m or quotient_width != 2 * m:
        raise ValueError("pair core does not have the required 8m + 2m split")

    standard = tuple(1 << index for index in range(core_width))
    section_core = _complement_representatives(standard, core_common, core_width)
    if len(section_core) != quotient_width:
        raise ArithmeticError("direct pair-core quotient section differs")
    core_coordinates = core_common + section_core

    def quotient_coordinates(row: int) -> int:
        coordinates = _coordinates_in_independent_basis(
            core_coordinates, int(row), core_width
        )
        return coordinates >> len(core_common)

    def core_to_physical(row: int) -> int:
        return _combine(core, int(row))

    def physical_to_source(row: int) -> int:
        return _combine(representatives, int(row))

    section_physical = tuple(core_to_physical(row) for row in section_core)
    section_source = tuple(physical_to_source(row) for row in section_physical)

    quotient_kernels = []
    quotient_kernel_sources = []
    local_dimensions = []
    for raw_local in core_local_kernel_bases:
        local = _row_basis(tuple(map(int, raw_local)), core_width)
        local_dimensions.append(len(local))
        if len(_row_basis(local + core_common, core_width)) != len(local):
            raise ArithmeticError("common nuisance escaped a local kernel")
        local_section = _complement_representatives(local, core_common, core_width)
        quotient = _row_basis(
            tuple(quotient_coordinates(row) for row in local_section),
            quotient_width,
        )
        if len(local_section) != m or len(quotient) != m:
            raise ValueError("a local kernel does not quotient to dimension m")
        quotient_kernels.append(quotient)
        quotient_kernel_sources.append(
            tuple(
                physical_to_source(core_to_physical(row))
                for row in local_section
            )
        )

    pair_dimensions = []
    for left in range(len(labels)):
        for right in range(left + 1, len(labels)):
            pair_dimensions.append(
                len(
                    _row_basis(
                        quotient_kernels[left] + quotient_kernels[right],
                        quotient_width,
                    )
                )
            )
    if set(pair_dimensions) != {2 * m}:
        raise ValueError("direct pair-core local quotients are not a spread")

    anchors = (0, 1, 2)
    locators = recover_locator_branches(
        tuple(_bit_rows(rows, quotient_width) for rows in quotient_kernels),
        field,
        anchor_indices=anchors,
        locator_power_rounds=locator_power_rounds,
    )
    if (
        locators.branch_count != m
        or locators.point_count != len(labels)
        or not locators.all_kernels_replay
        or not locators.all_locator_powers_replay
        or not locators.full_frobenius_orbit
    ):
        raise ArithmeticError("direct pair-core graph-pencil replay differs")

    return DirectPairCoreGraphPencil(
        physical_sum_dimension=physical_width,
        pair_core_dimension=core_width,
        common_nuisance_dimension=len(core_common),
        quotient_dimension=quotient_width,
        local_kernel_dimensions=tuple(local_dimensions),
        quotient_local_kernel_dimensions=tuple(map(len, quotient_kernels)),
        pairwise_quotient_sum_dimensions=tuple(pair_dimensions),
        observer_labels=labels,
        anchor_indices=anchors,
        anchor_labels=tuple(labels[index] for index in anchors),
        common_nuisance_core_basis=core_common,
        common_nuisance_physical_basis=core_common_physical,
        quotient_section_core_basis=section_core,
        quotient_section_physical_basis=section_physical,
        quotient_section_source_lifts=section_source,
        quotient_local_kernel_bases=tuple(quotient_kernels),
        quotient_local_kernel_source_lifts=tuple(quotient_kernel_sources),
        locators=locators,
    )
