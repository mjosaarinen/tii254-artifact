"""Factorized derivative numerator for one unknown active support root.

Let ``K(X)`` be the product over the already known active roots and let the
remaining root be ``a``.  In characteristic two,

``(K(X)(X-a))' = (X K'(X) + K(X)) + a K'(X)``.

The two coefficient polynomials can be compiled once.  This is the ``q=1``
counterpart of :mod:`hov_mceliece.tii254_d6_two_root_factor`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .goppa_recovery import _balanced_product


__all__ = [
    "OneRootDerivativePencil",
    "direct_one_root_support_product_derivative",
    "evaluate_one_root_derivative_pencil",
    "one_root_derivative_pencil",
]


@dataclass(frozen=True)
class OneRootDerivativePencil:
    known_root_count: int
    known_roots: tuple
    known_product: object
    constant_coefficient: object
    root_coefficient: object


def _validate_known_roots(roots, ring):
    roots = tuple(roots)
    field = ring.base_ring()
    if int(field.characteristic()) != 2:
        raise ValueError("the one-root derivative pencil requires characteristic two")
    if any(value.parent() != field for value in roots):
        raise ValueError("every root must belong to the polynomial base field")
    if len(set(roots)) != len(roots):
        raise ValueError("support roots must be distinct")
    return roots


def one_root_derivative_pencil(known_roots, ring) -> OneRootDerivativePencil:
    """Compile the two coefficient polynomials for one remaining root."""

    roots = _validate_known_roots(known_roots, ring)
    X = ring.gen()
    known_product = _balanced_product(
        [X - value for value in roots], ring.one()
    )
    derivative = known_product.derivative()
    return OneRootDerivativePencil(
        known_root_count=len(roots),
        known_roots=roots,
        known_product=known_product,
        constant_coefficient=X * derivative + known_product,
        root_coefficient=derivative,
    )


def evaluate_one_root_derivative_pencil(pencil: OneRootDerivativePencil, root):
    """Evaluate the compiled pencil on one candidate root."""

    field = pencil.known_product.base_ring()
    if root.parent() != field:
        raise ValueError("the candidate root and pencil use different fields")
    if root in set(pencil.known_roots):
        raise ValueError("the candidate root collides with the known active support")
    return pencil.constant_coefficient + root * pencil.root_coefficient


def direct_one_root_support_product_derivative(known_roots, root, ring):
    """Return the existing balanced-product derivative for a differential."""

    roots = _validate_known_roots((*tuple(known_roots), root), ring)
    X = ring.gen()
    return _balanced_product(
        [X - value for value in roots], ring.one()
    ).derivative()
