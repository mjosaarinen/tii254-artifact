"""Exact finite-field graph-pencil recovery for GhIsJa+26 local kernels.

The large relation supplier is deliberately absent.  This module begins with
already certified local kernel subspaces in one quotient relation space and
implements the small linear-algebra theorem that turns their graph pencil
into Frobenius branch planes.
"""

from __future__ import annotations

from dataclasses import dataclass
from operator import index as integer_index

from .binary_field import BinaryField

__all__ = [
    "FourAnchorCommonSpanResult",
    "GraphPencilResult",
    "LocatorBranchResult",
    "MultiAnchorCommonSpanResult",
    "frobenius_power_root",
    "frobenius_plane_permutation",
    "normalized_locator_cross_ratio",
    "recover_four_anchor_common_span",
    "recover_graph_pencil",
    "recover_locator_branches",
    "recover_multi_anchor_common_span",
    "synthetic_frobenius_graph_pencil",
    "synthetic_locator_cross_ratio_pencil",
]


@dataclass(frozen=True)
class FourAnchorCommonSpanResult:
    """Fail-closed recovery of a canonical span from four enlarged kernels."""

    ambient_dimension: int
    anchor_dimensions: tuple[int, int, int, int]
    pair_sum_dimensions: tuple[int, int]
    common_dimension: int
    anchor_common_dimensions: tuple[int, int, int, int]
    common_basis: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class MultiAnchorCommonSpanResult:
    """Sequential pair-sum intersection for an even anchor family."""

    ambient_dimension: int
    anchor_dimensions: tuple[int, ...]
    pair_sum_dimensions: tuple[int, ...]
    core_dimensions: tuple[int, ...]
    common_dimension: int
    anchor_common_dimensions: tuple[int, ...]
    common_basis: tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class GraphPencilResult:
    branch_count: int
    ambient_dimension: int
    graph_count: int
    separator_source: tuple[int, ...]
    separator_eigenvalues: tuple[int, ...]
    common_eigenvalues: tuple[tuple[int, ...], ...]
    eigenlines: tuple[tuple[int, ...], ...]
    branch_planes: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...]
    graph_maps: tuple[tuple[tuple[int, ...], ...], ...]
    all_graphs_replay: bool


@dataclass(frozen=True)
class LocatorBranchResult:
    """Projectively normalized locator branches recovered from local kernels.

    Projective points are encoded as pairs ``(numerator, denominator)``.  The
    three selected anchors are normalized to ``0=(0,1)``,
    ``infinity=(1,0)``, and ``1=(1,1)``.  Every other point is represented as
    ``(value,1)``.  The branches are ordered as returned by the certified
    graph-pencil splitting; ``frobenius_permutation`` records coefficientwise
    squaring on that order.
    """

    field_degree: int
    field_modulus: int
    point_count: int
    branch_count: int
    anchor_indices: tuple[int, int, int]
    target_indices: tuple[int, ...]
    locator_power_rounds: int
    projective_locators_by_branch: tuple[
        tuple[tuple[int, int], ...], ...
    ]
    frobenius_permutation: tuple[int, ...]
    all_points_distinct_by_branch: tuple[bool, ...]
    full_frobenius_orbit: bool
    all_locator_powers_replay: bool
    all_kernels_replay: bool


def _shape(matrix) -> tuple[int, int]:
    rows = tuple(tuple(map(int, row)) for row in matrix)
    return len(rows), (len(rows[0]) if rows else 0)


def _normalized_matrix(matrix, columns: int | None = None):
    rows = tuple(tuple(map(int, row)) for row in matrix)
    width = columns if columns is not None else (len(rows[0]) if rows else 0)
    if any(len(row) != width for row in rows):
        raise ValueError("matrix rows have inconsistent widths")
    return rows


def _transpose(matrix):
    rows = _normalized_matrix(matrix)
    if not rows:
        return ()
    return tuple(tuple(rows[row][column] for row in range(len(rows))) for column in range(len(rows[0])))


def _matrix_multiply(left, right, field: BinaryField):
    first = _normalized_matrix(left)
    second = _normalized_matrix(right)
    if not first:
        return ()
    if not second or len(first[0]) != len(second):
        raise ValueError("matrix product dimensions differ")
    columns = len(second[0])
    return tuple(
        tuple(
            _dot(row, tuple(second[index][column] for index in range(len(second))), field)
            for column in range(columns)
        )
        for row in first
    )


def _dot(left, right, field: BinaryField) -> int:
    if len(left) != len(right):
        raise ValueError("dot-product dimensions differ")
    value = 0
    for first, second in zip(left, right, strict=True):
        value ^= field.multiply(int(first), int(second))
    return value


def _row_times_matrix(row, matrix, field: BinaryField):
    rows = _normalized_matrix(matrix)
    if len(row) != len(rows):
        raise ValueError("row/matrix dimensions differ")
    return tuple(
        _dot(row, tuple(rows[index][column] for index in range(len(rows))), field)
        for column in range(len(rows[0]))
    )


def _rref(matrix, field: BinaryField, *, columns: int | None = None):
    rows = [list(row) for row in _normalized_matrix(matrix, columns)]
    width = columns if columns is not None else (len(rows[0]) if rows else 0)
    pivots = []
    pivot_row = 0
    for column in range(width):
        selected = next(
            (row for row in range(pivot_row, len(rows)) if rows[row][column]),
            None,
        )
        if selected is None:
            continue
        rows[pivot_row], rows[selected] = rows[selected], rows[pivot_row]
        inverse = field.inverse(rows[pivot_row][column])
        rows[pivot_row] = [
            field.multiply(inverse, value) for value in rows[pivot_row]
        ]
        for row in range(len(rows)):
            if row == pivot_row or not rows[row][column]:
                continue
            scalar = rows[row][column]
            rows[row] = [
                old ^ field.multiply(scalar, pivot)
                for old, pivot in zip(rows[row], rows[pivot_row], strict=True)
            ]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == len(rows):
            break
    return tuple(tuple(row) for row in rows), tuple(pivots)


def _rank(matrix, field: BinaryField, *, columns: int | None = None) -> int:
    return len(_rref(matrix, field, columns=columns)[1])


def _row_basis(matrix, field: BinaryField, *, columns: int | None = None):
    reduced, pivots = _rref(matrix, field, columns=columns)
    return reduced[: len(pivots)]


def _rowspace_intersection(left, right, field: BinaryField, *, columns: int):
    """Return a basis of the intersection of two row spaces."""

    first = _row_basis(left, field, columns=columns)
    second = _row_basis(right, field, columns=columns)
    dependencies = _right_kernel(
        _transpose(first + second), field, columns=len(first) + len(second)
    )
    rows = tuple(
        tuple(
            _dot(
                coefficients[: len(first)],
                tuple(row[column] for row in first),
                field,
            )
            for column in range(columns)
        )
        for coefficients in dependencies
    )
    return _row_basis(rows, field, columns=columns)


def recover_four_anchor_common_span(
    anchors,
    field: BinaryField,
    *,
    expected_common_dimension: int = 24,
    expected_local_dimension: int = 12,
) -> FourAnchorCommonSpanResult:
    """Recover ``(K0+K1) intersect (K2+K3)`` with exact replay gates.

    Suppose each observed ``Kj`` contains a canonical local kernel ``Cj``
    in one common space ``W``, and each of the pairs ``(C0,C1)`` and
    ``(C2,C3)`` spans ``W``.  Then ``W`` is contained in both observed pair
    sums.  If their measured intersection has ``dim(W)`` dimensions, it is
    exactly ``W`` even when every ``Kj`` has additional directions.  The
    local-intersection checks then discard those directions before the graph
    pencil is invoked.

    This routine proves no canonical containment by itself.  It implements
    the finite, fail-closed linear-algebra conclusion once that containment
    is supplied by the canonical-form premise.
    """

    values = tuple(anchors)
    if len(values) != 4:
        raise ValueError("four-anchor recovery requires exactly four spaces")
    normalized = tuple(_normalized_matrix(value) for value in values)
    widths = {len(row) for value in normalized for row in value}
    if not widths:
        raise ValueError("four-anchor spaces must be nonempty")
    if len(widths) != 1:
        raise ValueError("four-anchor spaces have inconsistent widths")
    ambient = widths.pop()
    bases = tuple(
        _row_basis(value, field, columns=ambient) for value in normalized
    )
    pair_sums = (
        _row_basis(bases[0] + bases[1], field, columns=ambient),
        _row_basis(bases[2] + bases[3], field, columns=ambient),
    )
    common = _rowspace_intersection(
        pair_sums[0], pair_sums[1], field, columns=ambient
    )
    common_dimension = len(common)
    if common_dimension != int(expected_common_dimension):
        raise ArithmeticError(
            "four-anchor pair-sum intersection has the wrong dimension"
        )
    local_dimensions = tuple(
        len(_rowspace_intersection(value, common, field, columns=ambient))
        for value in bases
    )
    if any(value != int(expected_local_dimension) for value in local_dimensions):
        raise ArithmeticError(
            "an enlarged anchor does not cut the recovered common span correctly"
        )
    return FourAnchorCommonSpanResult(
        ambient_dimension=ambient,
        anchor_dimensions=tuple(len(value) for value in bases),
        pair_sum_dimensions=tuple(len(value) for value in pair_sums),
        common_dimension=common_dimension,
        anchor_common_dimensions=local_dimensions,
        common_basis=common,
    )


def recover_multi_anchor_common_span(
    anchors,
    field: BinaryField,
    *,
    expected_common_dimension: int = 24,
    expected_local_dimension: int = 12,
) -> MultiAnchorCommonSpanResult:
    """Intersect successive pair sums until the canonical core is exposed.

    If every declared pair of canonical local kernels spans the same ``W``,
    then every pair sum of the complete observed kernels contains ``W``.
    Successive intersections therefore never delete a canonical direction.
    Reaching measured dimension ``dim(W)`` proves equality; a larger terminal
    core is a refusal, not a guessed projection.
    """

    values = tuple(anchors)
    if len(values) < 4 or len(values) % 2:
        raise ValueError(
            "multi-anchor recovery requires an even number of at least four spaces"
        )
    normalized = tuple(_normalized_matrix(value) for value in values)
    widths = {len(row) for value in normalized for row in value}
    if not widths:
        raise ValueError("multi-anchor spaces must be nonempty")
    if len(widths) != 1:
        raise ValueError("multi-anchor spaces have inconsistent widths")
    ambient = widths.pop()
    bases = tuple(
        _row_basis(value, field, columns=ambient) for value in normalized
    )
    pair_sums = tuple(
        _row_basis(bases[index] + bases[index + 1], field, columns=ambient)
        for index in range(0, len(bases), 2)
    )
    common = pair_sums[0]
    core_dimensions = [len(common)]
    for pair_sum in pair_sums[1:]:
        common = _rowspace_intersection(
            common, pair_sum, field, columns=ambient
        )
        core_dimensions.append(len(common))
    common_dimension = len(common)
    if common_dimension != int(expected_common_dimension):
        raise ArithmeticError(
            "multi-anchor pair-sum core has the wrong terminal dimension"
        )
    local_dimensions = tuple(
        len(_rowspace_intersection(value, common, field, columns=ambient))
        for value in bases
    )
    if any(value != int(expected_local_dimension) for value in local_dimensions):
        raise ArithmeticError(
            "an enlarged anchor does not cut the recovered common core correctly"
        )
    return MultiAnchorCommonSpanResult(
        ambient_dimension=ambient,
        anchor_dimensions=tuple(len(value) for value in bases),
        pair_sum_dimensions=tuple(len(value) for value in pair_sums),
        core_dimensions=tuple(core_dimensions),
        common_dimension=common_dimension,
        anchor_common_dimensions=local_dimensions,
        common_basis=common,
    )


def _inverse(matrix, field: BinaryField):
    value = _normalized_matrix(matrix)
    size = len(value)
    if not size or any(len(row) != size for row in value):
        raise ValueError("matrix inverse requires a nonempty square matrix")
    augmented = tuple(
        row + tuple(1 if index == column else 0 for column in range(size))
        for index, row in enumerate(value)
    )
    reduced, pivots = _rref(augmented, field, columns=2 * size)
    if pivots[:size] != tuple(range(size)):
        raise ArithmeticError("matrix is singular")
    return tuple(row[size:] for row in reduced[:size])


def _right_kernel(matrix, field: BinaryField, *, columns: int | None = None):
    reduced, pivots = _rref(matrix, field, columns=columns)
    width = columns if columns is not None else (len(reduced[0]) if reduced else 0)
    pivot_set = set(pivots)
    free = tuple(column for column in range(width) if column not in pivot_set)
    basis = []
    for free_column in free:
        vector = [0] * width
        vector[free_column] = 1
        for row, pivot in enumerate(pivots):
            vector[pivot] = reduced[row][free_column]
        basis.append(tuple(vector))
    return tuple(basis)


def _matrix_sum(left, right, field: BinaryField, scalar: int = 1):
    first = _normalized_matrix(left)
    second = _normalized_matrix(right)
    if _shape(first) != _shape(second):
        raise ValueError("matrix-sum dimensions differ")
    return tuple(
        tuple(
            value ^ field.multiply(int(scalar), other)
            for value, other in zip(row, second[index], strict=True)
        )
        for index, row in enumerate(first)
    )


def _identity(size: int):
    return tuple(
        tuple(1 if row == column else 0 for column in range(size))
        for row in range(size)
    )


def normalized_locator_cross_ratio(
    point: int,
    first: int,
    second: int,
    unit: int,
    field: BinaryField,
) -> int:
    """Map ``first,second,unit`` to ``0,infinity,1`` projectively.

    All four inputs use a finite affine chart.  The value at ``second`` is
    projective infinity and is deliberately refused rather than represented
    by a sentinel.  In characteristic two the displayed additions are also
    the usual cross-ratio subtractions.
    """

    values = tuple(map(int, (point, first, second, unit)))
    if any(value & ~field.mask for value in values):
        raise ValueError("cross-ratio operand is outside the field")
    if len({first, second, unit}) != 3:
        raise ValueError("cross-ratio base points must be distinct")
    if point == second:
        raise ZeroDivisionError("the second base point maps to infinity")
    numerator = field.multiply(point ^ first, second ^ unit)
    denominator = field.multiply(second ^ point, unit ^ first)
    return field.multiply(numerator, field.inverse(denominator))


def frobenius_power_root(value: int, rounds: int, field: BinaryField) -> int:
    """Invert ``x -> x^(2^rounds)`` in ``GF(2^m)`` exactly."""

    rounds = int(rounds)
    if rounds < 0:
        raise ValueError("Frobenius rounds must be nonnegative")
    return field.power(int(value), 1 << ((-rounds) % field.degree))


def _rowspace_equal(left, right, field: BinaryField, width: int) -> bool:
    first = _normalized_matrix(left, width)
    second = _normalized_matrix(right, width)
    rank_first = _rank(first, field, columns=width)
    rank_second = _rank(second, field, columns=width)
    return (
        rank_first == rank_second
        and _rank(first + second, field, columns=width) == rank_first
    )


def _scalar_multiple(left, right, field: BinaryField) -> int | None:
    """Return ``lambda`` when ``right=lambda*left``, including lambda zero."""

    pivot = next((index for index, value in enumerate(left) if value), None)
    if pivot is None:
        return 0 if not any(right) else None
    scalar = field.multiply(right[pivot], field.inverse(left[pivot]))
    if all(
        second == field.multiply(scalar, first)
        for first, second in zip(left, right, strict=True)
    ):
        return scalar
    return None


def _simple_left_eigenlines(matrix, field: BinaryField):
    value = _normalized_matrix(matrix)
    size = len(value)
    if not size or any(len(row) != size for row in value):
        raise ValueError("eigenline scan requires a square matrix")
    identity = _identity(size)
    lines = []
    for eigenvalue in range(field.order):
        shifted = _matrix_sum(value, identity, field, eigenvalue)
        kernel = _right_kernel(_transpose(shifted), field, columns=size)
        if len(kernel) > 1:
            return None
        if len(kernel) == 1:
            lines.append((eigenvalue, kernel[0]))
    if len(lines) != size or _rank(tuple(line for _value, line in lines), field) != size:
        return None
    return tuple(lines)


def _graph_map(kernel, ambient_inverse, branch_count: int, field: BinaryField):
    m = int(branch_count)
    coordinates = _matrix_multiply(kernel, ambient_inverse, field)
    if len(coordinates) != m or any(len(row) != 2 * m for row in coordinates):
        raise ValueError("graph kernel has the wrong shape")
    source = tuple(row[:m] for row in coordinates)
    target = tuple(row[m:] for row in coordinates)
    return _matrix_multiply(_inverse(source, field), target, field)


def recover_graph_pencil(
    first_kernel,
    second_kernel,
    graph_kernels,
    field: BinaryField,
) -> GraphPencilResult:
    """Recover common branch planes from certified transverse graph kernels.

    ``first_kernel`` and ``second_kernel`` must be disjoint ``m``-spaces.
    Every further kernel must be an ``m``-space in their direct sum and
    transverse to the second kernel.  The first graph must also be transverse
    to the first kernel so that it identifies the two summands.  A separator
    is accepted only when it has ``m`` simple field-rational eigenlines and
    every remaining graph preserves all of them.  Finally every input kernel
    is reconstructed from the recovered planes.
    """

    first = _normalized_matrix(first_kernel)
    second = _normalized_matrix(second_kernel)
    graphs = tuple(_normalized_matrix(value) for value in graph_kernels)
    m = len(first)
    ambient = 2 * m
    if (
        not m
        or len(second) != m
        or len(graphs) < 2
        or any(len(row) != ambient for row in first + second)
        or any(len(value) != m or any(len(row) != ambient for row in value) for value in graphs)
        or _rank(first, field) != m
        or _rank(second, field) != m
        or _rank(first + second, field) != ambient
    ):
        raise ValueError("graph-pencil base kernels are not complementary m-spaces")
    ambient_inverse = _inverse(first + second, field)
    graph_maps = tuple(
        _graph_map(value, ambient_inverse, m, field) for value in graphs
    )
    base_inverse = _inverse(graph_maps[0], field)
    endomorphisms = tuple(
        _matrix_multiply(value, base_inverse, field) for value in graph_maps[1:]
    )

    candidates = []
    for index, value in enumerate(endomorphisms):
        candidates.append(((index, 1), value))
    if len(endomorphisms) >= 2:
        for scalar in range(field.order):
            candidates.append(
                (
                    (0, 1, 1, scalar),
                    _matrix_sum(
                        endomorphisms[0], endomorphisms[1], field, scalar
                    ),
                )
            )

    selected = None
    selected_source = None
    common_values = None
    for source, candidate in candidates:
        split = _simple_left_eigenlines(candidate, field)
        if split is None:
            continue
        values_by_line = []
        valid = True
        for _separator_value, line in split:
            row_values = []
            for endomorphism in endomorphisms:
                image = _row_times_matrix(line, endomorphism, field)
                scalar = _scalar_multiple(line, image, field)
                if scalar is None:
                    valid = False
                    break
                row_values.append(scalar)
            if not valid:
                break
            values_by_line.append(tuple(row_values))
        if valid:
            selected = split
            selected_source = source
            common_values = tuple(values_by_line)
            break
    if selected is None or selected_source is None or common_values is None:
        raise ArithmeticError("graph pencil has no certified simple common splitting")

    eigenlines = tuple(line for _value, line in selected)
    base_graph = graph_maps[0]
    planes = []
    for line in eigenlines:
        first_vector = _row_times_matrix(line, first, field)
        second_coefficients = _row_times_matrix(line, base_graph, field)
        second_vector = _row_times_matrix(second_coefficients, second, field)
        if _rank((first_vector, second_vector), field) != 2:
            raise ArithmeticError("recovered branch plane collapsed")
        planes.append((first_vector, second_vector))

    replay = True
    for kernel, graph in zip(graphs, graph_maps, strict=True):
        reconstructed = tuple(
            tuple(
                first_value ^ second_value
                for first_value, second_value in zip(
                    _row_times_matrix(line, first, field),
                    _row_times_matrix(
                        _row_times_matrix(line, graph, field), second, field
                    ),
                    strict=True,
                )
            )
            for line in eigenlines
        )
        replay &= _rowspace_equal(kernel, reconstructed, field, ambient)
    if not replay:
        raise ArithmeticError("recovered branch planes do not replay every graph")

    return GraphPencilResult(
        branch_count=m,
        ambient_dimension=ambient,
        graph_count=len(graphs),
        separator_source=tuple(selected_source),
        separator_eigenvalues=tuple(value for value, _line in selected),
        common_eigenvalues=common_values,
        eigenlines=eigenlines,
        branch_planes=tuple(planes),
        graph_maps=tuple(
            tuple(tuple(row) for row in value) for value in graph_maps
        ),
        all_graphs_replay=True,
    )


def _frobenius_vector(vector, field: BinaryField):
    return tuple(field.multiply(value, value) for value in vector)


def frobenius_plane_permutation(planes, field: BinaryField):
    """Return the exact permutation induced by coefficientwise Frobenius."""

    values = tuple(
        tuple(tuple(map(int, row)) for row in plane) for plane in planes
    )
    if not values:
        return ()
    width = len(values[0][0])
    permutation = []
    for plane in values:
        image = tuple(_frobenius_vector(row, field) for row in plane)
        matches = [
            index
            for index, candidate in enumerate(values)
            if _rowspace_equal(image, candidate, field, width)
        ]
        if len(matches) != 1:
            raise ArithmeticError("Frobenius does not permute branch planes uniquely")
        permutation.append(matches[0])
    return tuple(permutation)


def recover_locator_branches(
    local_kernels,
    field: BinaryField,
    *,
    anchor_indices=(0, 1, 2),
    locator_power_rounds: int = 2,
) -> LocatorBranchResult:
    """Recover coherent projective locator branches from exact local kernels.

    The input spaces must already be certified local kernels in one common
    quotient-relation coordinate system.  Each must have dimension ``m`` and
    the canonical branch count must equal the extension degree ``m``.  The
    selected anchors are interpreted as ``(zero, infinity, unit)``; all other
    kernels are passed together to :func:`recover_graph_pencil`.

    ``locator_power_rounds`` declares that the graph eigenvalues are the
    ``2^rounds`` powers of normalized locator cross-ratios.  One round is the
    square-locator contract used by the degree-six route; the default two
    rounds preserve the existing fourth-power degree-seven route.  This
    Frobenius power is an automorphism of ``GF(2^m)``, so its exact inverse
    recovers normalized locators without root ambiguity.  Every recovered
    root is raised back to the declared power, every branch must contain
    distinct projective points, and all branches must form one full
    coefficientwise-Frobenius orbit.

    The round count must use the canonical range ``0 <= rounds < m``.  This
    refuses multiple integer encodings of the same periodic field
    automorphism in an authenticated producer/consumer contract.

    This function deliberately does not infer that supplied kernels are the
    canonical ones.  Their construction, quotient membership, and original
    relation replay remain predecessor obligations.
    """

    if isinstance(locator_power_rounds, bool):
        raise ValueError("locator-power rounds must be an integer")
    try:
        rounds = integer_index(locator_power_rounds)
    except TypeError as error:
        raise ValueError("locator-power rounds must be an integer") from error
    if not 0 <= rounds < field.degree:
        raise ValueError(
            "locator-power rounds must be a canonical nonnegative field round count"
        )

    kernels = tuple(_normalized_matrix(value) for value in local_kernels)
    anchors = tuple(map(int, anchor_indices))
    if len(anchors) != 3 or len(set(anchors)) != 3:
        raise ValueError("locator recovery requires three distinct anchors")
    if len(kernels) < 4 or any(index < 0 or index >= len(kernels) for index in anchors):
        raise ValueError("locator recovery has too few kernels or a bad anchor")
    branch_count = len(kernels[anchors[0]])
    if branch_count != field.degree:
        raise ValueError("local-kernel dimension differs from the field degree")
    target_indices = tuple(
        index for index in range(len(kernels)) if index not in anchors
    )
    pencil = recover_graph_pencil(
        kernels[anchors[0]],
        kernels[anchors[1]],
        (kernels[anchors[2]],) + tuple(kernels[index] for index in target_indices),
        field,
    )
    if pencil.branch_count != branch_count:
        raise ArithmeticError("graph-pencil branch count changed")

    locators_by_branch = []
    distinct_by_branch = []
    locator_powers_replay = []
    locator_power = 1 << rounds
    for signature in pencil.common_eigenvalues:
        if len(signature) != len(target_indices):
            raise ArithmeticError("graph-pencil locator signature has the wrong width")
        values = tuple(
            frobenius_power_root(value, rounds, field) for value in signature
        )
        power_replay = tuple(
            field.power(value, locator_power) for value in values
        ) == signature
        if not power_replay:
            raise ArithmeticError("locator roots do not replay the declared power")
        projective = [None] * len(kernels)
        projective[anchors[0]] = (0, 1)
        projective[anchors[1]] = (1, 0)
        projective[anchors[2]] = (1, 1)
        for index, value in zip(target_indices, values, strict=True):
            projective[index] = (int(value), 1)
        branch = tuple(projective)
        distinct = len(set(branch)) == len(branch)
        if not distinct:
            raise ArithmeticError("a recovered locator branch has a collision")
        locators_by_branch.append(branch)
        distinct_by_branch.append(True)
        locator_powers_replay.append(True)

    permutation = frobenius_plane_permutation(pencil.branch_planes, field)
    if len(permutation) != branch_count or set(permutation) != set(
        range(branch_count)
    ):
        raise ArithmeticError("locator branches lack a Frobenius permutation")
    for branch_index, branch in enumerate(locators_by_branch):
        expected = tuple(
            (
                field.multiply(numerator, numerator),
                field.multiply(denominator, denominator),
            )
            for numerator, denominator in branch
        )
        if tuple(locators_by_branch[permutation[branch_index]]) != expected:
            raise ArithmeticError("locator branches do not replay Frobenius")
    orbit = [0]
    for _ in range(1, branch_count):
        orbit.append(permutation[orbit[-1]])
    full_orbit = (
        len(set(orbit)) == branch_count
        and permutation[orbit[-1]] == orbit[0]
    )
    if not full_orbit:
        raise ArithmeticError("locator branches do not form one full Frobenius orbit")

    return LocatorBranchResult(
        field_degree=field.degree,
        field_modulus=field.modulus,
        point_count=len(kernels),
        branch_count=branch_count,
        anchor_indices=anchors,
        target_indices=target_indices,
        locator_power_rounds=rounds,
        projective_locators_by_branch=tuple(locators_by_branch),
        frobenius_permutation=permutation,
        all_points_distinct_by_branch=tuple(distinct_by_branch),
        full_frobenius_orbit=True,
        all_locator_powers_replay=all(locator_powers_replay),
        all_kernels_replay=pencil.all_graphs_replay,
    )


def synthetic_frobenius_graph_pencil(field: BinaryField) -> dict[str, object]:
    """Construct and recover an exact ``m``-branch Frobenius graph pencil."""

    m = field.degree
    ambient = 2 * m
    first = tuple(
        tuple(1 if column == row else 0 for column in range(ambient))
        for row in range(m)
    )
    second = tuple(
        tuple(1 if column == m + row else 0 for column in range(ambient))
        for row in range(m)
    )
    polynomial_basis = tuple(1 << index for index in range(m))
    moore = tuple(
        tuple(field.power(value, 1 << conjugate) for value in polynomial_basis)
        for conjugate in range(m)
    )
    if _rank(moore, field) != m:
        raise ArithmeticError("polynomial basis lost Moore rank")
    primitive = 2
    eigenvalues = tuple(
        field.power(primitive, 1 << conjugate) for conjugate in range(m)
    )
    if len(set(eigenvalues)) != m:
        raise ArithmeticError("chosen element has a short Frobenius orbit")
    diagonal_times_moore = tuple(
        tuple(field.multiply(eigenvalues[row], value) for value in moore[row])
        for row in range(m)
    )
    locator = _matrix_multiply(
        _inverse(moore, field), diagonal_times_moore, field
    )
    if any(field.power(value, 2) != value for row in locator for value in row):
        raise ArithmeticError("Frobenius-covariant locator matrix is not binary")

    identity = _identity(m)

    def graph_kernel(graph):
        return tuple(
            tuple(
                (1 if column == row else 0)
                if column < m
                else graph[row][column - m]
                for column in range(ambient)
            )
            for row in range(m)
        )

    graphs = (graph_kernel(identity), graph_kernel(locator))
    result = recover_graph_pencil(first, second, graphs, field)
    permutation = frobenius_plane_permutation(result.branch_planes, field)
    if set(permutation) != set(range(m)):
        raise ArithmeticError("Frobenius plane map is not a permutation")
    orbit = [0]
    for _ in range(1, m):
        orbit.append(permutation[orbit[-1]])
    if len(set(orbit)) != m or permutation[orbit[-1]] != orbit[0]:
        raise ArithmeticError("branch planes do not form one full Frobenius orbit")
    return {
        "extension_degree": m,
        "ambient_dimension": ambient,
        "graph_count": len(graphs),
        "branch_plane_count": len(result.branch_planes),
        "separator_eigenvalue_count": len(set(result.separator_eigenvalues)),
        "frobenius_permutation": permutation,
        "frobenius_orbit": tuple(orbit),
        "locator_matrix_is_binary": True,
        "all_graphs_replay": result.all_graphs_replay,
    }


def synthetic_locator_cross_ratio_pencil(field: BinaryField) -> dict[str, object]:
    """Replay the locator meaning of a graph pencil at extension degree ``m``.

    For ``s-1=4``, the canonical GhIsJa+26 local annihilator on Frobenius
    branch ``a`` has slope ``alpha_j^(4*2^a)``.  This fixture builds those
    annihilators in a coefficientwise-Frobenius-covariant Moore basis, runs
    the generic graph-pencil extractor, and checks that every output
    eigenvalue is the corresponding normalized locator cross-ratio to the
    fourth power.  It is a semantic calibration, not a public relation
    supplier.
    """

    m = field.degree
    if m < 2:
        raise ValueError("the fourth-power locator calibration needs m >= 2")
    polynomial_basis = tuple(1 << index for index in range(m))
    moore = tuple(
        tuple(field.power(value, 1 << conjugate) for value in polynomial_basis)
        for conjugate in range(m)
    )
    if _rank(moore, field) != m:
        raise ArithmeticError("polynomial basis lost Moore rank")

    def local_kernel(locator: int):
        rows = []
        for conjugate, branch_basis in enumerate(moore):
            slope = field.power(locator, 4 * (1 << conjugate))
            rows.append(
                tuple(branch_basis)
                + tuple(field.multiply(slope, value) for value in branch_basis)
            )
        return tuple(rows)

    first_locator = 0
    second_locator = 1
    unit_locator = 2
    target_locators = (3, 4, 5, 6)
    if field.order <= max(target_locators):
        raise ValueError("field is too small for the fixed locator fixture")

    first = local_kernel(first_locator)
    second = local_kernel(second_locator)
    graphs = tuple(
        local_kernel(locator) for locator in (unit_locator,) + target_locators
    )
    result = recover_graph_pencil(first, second, graphs, field)

    expected_signatures = []
    expected_roots = []
    for conjugate in range(m):
        branch_first = field.power(first_locator, 4 * (1 << conjugate))
        branch_second = field.power(second_locator, 4 * (1 << conjugate))
        branch_unit = field.power(unit_locator, 4 * (1 << conjugate))
        signature = tuple(
            normalized_locator_cross_ratio(
                field.power(locator, 4 * (1 << conjugate)),
                branch_first,
                branch_second,
                branch_unit,
                field,
            )
            for locator in target_locators
        )
        expected_signatures.append(signature)
        expected_roots.append(
            tuple(
                field.power(
                    normalized_locator_cross_ratio(
                        locator,
                        first_locator,
                        second_locator,
                        unit_locator,
                        field,
                    ),
                    1 << conjugate,
                )
                for locator in target_locators
            )
        )

    observed_signatures = tuple(result.common_eigenvalues)
    cross_ratio_replay = set(observed_signatures) == set(expected_signatures)
    observed_roots = tuple(
        tuple(frobenius_power_root(value, 2, field) for value in signature)
        for signature in observed_signatures
    )
    root_replay = set(observed_roots) == set(expected_roots)

    permutation = frobenius_plane_permutation(result.branch_planes, field)
    frobenius_replay = all(
        observed_signatures[permutation[index]]
        == tuple(field.multiply(value, value) for value in signature)
        for index, signature in enumerate(observed_signatures)
    )
    first_target_orbit = tuple(signature[0] for signature in observed_roots)
    full_orbit = len(set(first_target_orbit)) == m
    if not (
        cross_ratio_replay
        and root_replay
        and frobenius_replay
        and full_orbit
        and result.all_graphs_replay
    ):
        raise ArithmeticError("locator cross-ratio graph-pencil replay failed")

    return {
        "extension_degree": m,
        "physical_point_count": 3 + len(target_locators),
        "target_point_count": len(target_locators),
        "branch_plane_count": len(result.branch_planes),
        "graph_count": len(graphs),
        "cross_ratio_signatures_replay": cross_ratio_replay,
        "inverse_fourth_root_replay": root_replay,
        "frobenius_signature_replay": frobenius_replay,
        "first_target_full_frobenius_orbit": full_orbit,
        "all_graphs_replay": result.all_graphs_replay,
    }
