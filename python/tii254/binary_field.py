"""Exact finite-field helpers for binary descent of Hermite value parities.

The routines here deliberately use the polynomial-basis integer encoding from
the frozen Classic fixture.  They are small enough to replay without Sage and
do not construct the ambient degree-eight holdout operator.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from fractions import Fraction


@dataclass(frozen=True)
class BinaryField:
    """Polynomial-basis arithmetic in ``GF(2^degree)``.

    ``modulus`` includes its leading ``x^degree`` bit.  Irreducibility is an
    input-fixture obligation; this class only replays arithmetic.
    """

    degree: int
    modulus: int

    def __post_init__(self) -> None:
        if self.degree <= 0 or self.modulus.bit_length() != self.degree + 1:
            raise ValueError("modulus degree does not match the field degree")
        if not (self.modulus & 1):
            raise ValueError("field modulus must have nonzero constant term")

    @property
    def order(self) -> int:
        return 1 << self.degree

    @property
    def mask(self) -> int:
        return self.order - 1

    def multiply(self, left: int, right: int) -> int:
        if left & ~self.mask or right & ~self.mask:
            raise ValueError("field operand is outside the polynomial basis")
        value = 0
        while right:
            if right & 1:
                value ^= left
            right >>= 1
            left <<= 1
            if left & self.order:
                left ^= self.modulus
        return value & self.mask

    def power(self, value: int, exponent: int) -> int:
        if exponent < 0:
            return self.power(self.inverse(value), -exponent)
        result = 1
        while exponent:
            if exponent & 1:
                result = self.multiply(result, value)
            value = self.multiply(value, value)
            exponent >>= 1
        return result

    def inverse(self, value: int) -> int:
        if value == 0:
            raise ZeroDivisionError("zero has no finite-field inverse")
        return self.power(value, self.order - 2)

    def product(self, values) -> int:
        result = 1
        for value in values:
            result = self.multiply(result, int(value))
        return result

    def evaluate(self, coefficients, value: int) -> int:
        result = 0
        for coefficient in reversed(tuple(coefficients)):
            result = self.multiply(result, value) ^ int(coefficient)
        return result


def binary_rank(words) -> int:
    """Rank over GF(2) of polynomial-basis words represented as integers."""

    basis: dict[int, int] = {}
    for word in words:
        word = int(word)
        while word:
            pivot = word.bit_length() - 1
            if pivot in basis:
                word ^= basis[pivot]
            else:
                basis[pivot] = word
                break
    return len(basis)


def decode_binary_matrix_rows(record) -> tuple[int, ...]:
    """Decode the repository's frozen row-major little-endian bit matrix."""

    payload = base64.b64decode(record["payload_base64"], validate=True)
    rows = int(record["rows"])
    stride = int(record["stride_bytes"])
    if len(payload) != rows * stride:
        raise ValueError("binary matrix payload length differs")
    return tuple(
        int.from_bytes(payload[index * stride : (index + 1) * stride], "little")
        for index in range(rows)
    )


def prefix_shortening_rref(rows, retained_count: int) -> dict:
    """Shorten on the complement of a coordinate prefix and RREF the result."""

    if retained_count <= 0:
        raise ValueError("retained coordinate count must be positive")
    retained_mask = (1 << retained_count) - 1
    complement_basis: dict[int, int] = {}
    shortened = []
    for original in rows:
        row = int(original)
        complement = row >> retained_count
        while complement:
            pivot = complement.bit_length() - 1
            if pivot in complement_basis:
                row ^= complement_basis[pivot]
                complement = row >> retained_count
            else:
                complement_basis[pivot] = row
                break
        else:
            shortened.append(row & retained_mask)

    pivot_row = 0
    pivots = []
    for column in range(retained_count):
        selected = next(
            (
                row
                for row in range(pivot_row, len(shortened))
                if (shortened[row] >> column) & 1
            ),
            None,
        )
        if selected is None:
            continue
        shortened[pivot_row], shortened[selected] = (
            shortened[selected],
            shortened[pivot_row],
        )
        for row in range(len(shortened)):
            if row != pivot_row and (shortened[row] >> column) & 1:
                shortened[row] ^= shortened[pivot_row]
        pivots.append(column)
        pivot_row += 1
        if pivot_row == len(shortened):
            break
    if pivot_row != len(shortened):
        raise ArithmeticError("shortened rows are not independent")
    pivot_set = set(pivots)
    return {
        "complement_rank": len(complement_basis),
        "shortened_dimension": len(shortened),
        "rows": tuple(shortened),
        "pivots": tuple(pivots),
        "nonpivots": tuple(
            column for column in range(retained_count) if column not in pivot_set
        ),
    }


def expanded_parity_rank(parity_rows, field_degree: int) -> int:
    """Binary rank of extension-field parities restricted to binary values.

    Each input row is a tuple in ``GF(2^m)^h`` in polynomial-basis encoding.
    Expanding its ``m`` coefficient coordinates produces binary parity rows of
    length ``h``.  Their rank is the codimension forced on ``GF(2)^h``.
    """

    binary_rows = []
    for row in parity_rows:
        row = tuple(map(int, row))
        for coordinate in range(field_degree):
            binary_rows.append(
                sum(
                    ((value >> coordinate) & 1) << column
                    for column, value in enumerate(row)
                )
            )
    return binary_rank(binary_rows)


def locator_derivatives(field: BinaryField, support) -> tuple[int, ...]:
    """Return ``L'(alpha_i)=prod_{j != i}(alpha_i-alpha_j)`` in char. two."""

    support = tuple(map(int, support))
    if len(set(support)) != len(support):
        raise ValueError("support values must be distinct")
    return tuple(
        field.product(value ^ other for index, other in enumerate(support) if index != i)
        for i, value in enumerate(support)
    )


def one_parity_coefficients(
    field: BinaryField,
    support,
    polynomial_coefficients,
    held_indices,
    *,
    degree: int,
    order: int,
    derivatives=None,
    polynomial_values=None,
) -> tuple[int, ...]:
    r"""Return the degree-zero Hermite parity after GRS-unit rescaling.

    Let ``L`` be the full chart locator, ``Pi_T`` the held locator, and use the
    binary-Goppa containing-GRS multiplier ``mu_i=g(alpha_i)^2/L'(alpha_i)``.
    If ``beta_i=mu_i^degree A_T(alpha_i)^order``, the first Reed--Solomon dual
    parity has coefficient ``1/(beta_i Pi_T'(alpha_i))``.  Cancellation gives

    ``L'(alpha_i)^(degree-order) Pi_T'(alpha_i)^(order-1)
      / g(alpha_i)^(2 degree)``.

    The formula is used only when the residual degree leaves at least one dual
    parity.  It exposes no claim that the support or divisor are public.
    """

    support = tuple(map(int, support))
    held_indices = tuple(map(int, held_indices))
    if len(set(held_indices)) != len(held_indices):
        raise ValueError("held indices must be distinct")
    if any(index < 0 or index >= len(support) for index in held_indices):
        raise ValueError("held index lies outside the chart support")
    if degree < order or order < 1:
        raise ValueError("this cancellation formula requires degree >= order >= 1")
    if derivatives is None:
        derivatives = locator_derivatives(field, support)
    derivatives = tuple(map(int, derivatives))
    if len(derivatives) != len(support) or any(value == 0 for value in derivatives):
        raise ValueError("locator derivative vector is invalid")
    if polynomial_values is None:
        polynomial_values = tuple(
            field.evaluate(polynomial_coefficients, value) for value in support
        )
    g_values = tuple(map(int, polynomial_values))
    if len(g_values) != len(support):
        raise ValueError("polynomial-value vector has the wrong length")
    if any(value == 0 for value in g_values):
        raise ValueError("Goppa polynomial meets the chart support")

    coefficients = []
    for index in held_indices:
        pi_derivative = field.product(
            support[index] ^ support[other]
            for other in held_indices
            if other != index
        )
        numerator = field.multiply(
            field.power(derivatives[index], degree - order),
            field.power(pi_derivative, order - 1),
        )
        denominator = field.power(g_values[index], 2 * degree)
        coefficients.append(field.multiply(numerator, field.inverse(denominator)))
    return tuple(coefficients)


def direct_one_parity_coefficients(
    field: BinaryField,
    support,
    polynomial_coefficients,
    held_indices,
    *,
    degree: int,
    order: int,
    derivatives=None,
    polynomial_values=None,
) -> tuple[int, ...]:
    """Replay the same coefficients from ``mu``, ``A_T`` and ``Pi_T'``."""

    support = tuple(map(int, support))
    held_indices = tuple(map(int, held_indices))
    if derivatives is None:
        derivatives = locator_derivatives(field, support)
    derivatives = tuple(map(int, derivatives))
    if polynomial_values is None:
        polynomial_values = tuple(
            field.evaluate(polynomial_coefficients, value) for value in support
        )
    g_values = tuple(map(int, polynomial_values))
    if len(g_values) != len(support):
        raise ValueError("polynomial-value vector has the wrong length")
    coefficients = []
    for index in held_indices:
        pi_derivative = field.product(
            support[index] ^ support[other]
            for other in held_indices
            if other != index
        )
        a_value = field.multiply(derivatives[index], field.inverse(pi_derivative))
        multiplier = field.multiply(
            field.power(g_values[index], 2), field.inverse(derivatives[index])
        )
        beta = field.multiply(
            field.power(multiplier, degree), field.power(a_value, order)
        )
        coefficients.append(
            field.inverse(field.multiply(beta, pi_derivative))
        )
    return tuple(coefficients)


def full_binary_column_rank_probability(field_degree: int, column_count: int) -> Fraction:
    """Probability that uniform columns in ``GF(2)^m`` have full column rank."""

    if column_count < 0 or column_count > field_degree:
        return Fraction(0, 1)
    probability = Fraction(1, 1)
    for index in range(column_count):
        probability *= 1 - Fraction(1 << index, 1 << field_degree)
    return probability
