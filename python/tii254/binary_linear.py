"""Small exact GF(2) row-space helpers using Python integers as bit vectors."""

from __future__ import annotations


def validate_vector(value: int, width: int) -> int:
    value = int(value)
    if width < 0 or value < 0 or value.bit_length() > width:
        raise ValueError("binary vector escapes its declared width")
    return value


def row_basis(rows, width: int) -> tuple[int, ...]:
    """Return a deterministic reduced basis, pivoting from low to high bits."""

    values = [validate_vector(value, width) for value in rows if int(value)]
    pivot_row = 0
    for column in range(width):
        selected = next(
            (i for i in range(pivot_row, len(values)) if (values[i] >> column) & 1),
            None,
        )
        if selected is None:
            continue
        values[pivot_row], values[selected] = values[selected], values[pivot_row]
        pivot = values[pivot_row]
        for index in range(len(values)):
            if index != pivot_row and ((values[index] >> column) & 1):
                values[index] ^= pivot
        pivot_row += 1
        if pivot_row == len(values):
            break
    return tuple(values[:pivot_row])


def nullspace(rows, width: int) -> tuple[int, ...]:
    reduced = row_basis(rows, width)
    pivots = tuple((row & -row).bit_length() - 1 for row in reduced)
    pivot_set = set(pivots)
    result = []
    for free in range(width):
        if free in pivot_set:
            continue
        value = 1 << free
        for row, pivot in zip(reduced, pivots, strict=True):
            if (row >> free) & 1:
                value |= 1 << pivot
        result.append(value)
    if any((row & value).bit_count() & 1 for row in reduced for value in result):
        raise ArithmeticError("null-space replay failed")
    return tuple(result)


def intersection(left, right, width: int) -> tuple[int, ...]:
    """Basis of the intersection of two binary row spaces."""

    return nullspace(nullspace(left, width) + nullspace(right, width), width)


def combine(columns: tuple[int, ...], coefficients: int) -> int:
    validate_vector(coefficients, len(columns))
    value = 0
    pending = int(coefficients)
    while pending:
        bit = (pending & -pending).bit_length() - 1
        value ^= int(columns[bit])
        pending &= pending - 1
    return value


def coordinates_in_basis(basis: tuple[int, ...], value: int, width: int) -> int:
    """Express a vector in an ordered independent basis."""

    pivots: dict[int, tuple[int, int]] = {}
    for index, raw in enumerate(basis):
        row = validate_vector(raw, width)
        coefficients = 1 << index
        while row:
            pivot = row.bit_length() - 1
            previous = pivots.get(pivot)
            if previous is None:
                pivots[pivot] = (row, coefficients)
                break
            row ^= previous[0]
            coefficients ^= previous[1]
        if not row:
            raise ValueError("coordinate basis is dependent")

    pending = validate_vector(value, width)
    coordinates = 0
    while pending:
        pivot = pending.bit_length() - 1
        previous = pivots.get(pivot)
        if previous is None:
            raise ValueError("vector is outside the basis span")
        pending ^= previous[0]
        coordinates ^= previous[1]
    if combine(basis, coordinates) != value:
        raise ArithmeticError("coordinate replay failed")
    return coordinates


def complement_representatives(space, subspace, width: int) -> tuple[int, ...]:
    """Choose deterministic rows of ``space`` complementing ``subspace``."""

    space_basis = row_basis(space, width)
    running = list(row_basis(subspace, width))
    if len(intersection(space_basis, tuple(running), width)) != len(running):
        raise ValueError("declared subspace is not contained in the space")
    result = []
    current_rank = len(running)
    for row in space_basis:
        candidate = row_basis((*running, row), width)
        if len(candidate) != current_rank:
            result.append(row)
            running.append(row)
            current_rank += 1
    if current_rank != len(space_basis):
        raise ArithmeticError("quotient complement did not span")
    return tuple(result)
