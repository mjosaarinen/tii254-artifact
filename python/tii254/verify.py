"""Independent recovered-key verification (requires Sage).

``verify_recovered_key(H, x, g, m, r)`` checks that a candidate support ``x``
and Goppa polynomial ``g`` describe the same code as the public parity check
``H``, *against the independently known parameters* ``(m, r)`` -- not against
values re-derived from ``g`` itself (plan Section 6.12).  It does not reference
any published secret key and compares row spaces over the extension field, so
it is insensitive to the particular echelon representation of ``H``.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class VerificationResult:
    """Result of a conjunction of exact key-verification checks."""

    ok: bool
    checks: dict = field(default_factory=dict)
    reasons: list = field(default_factory=list)

__all__ = ["VerifyResult", "verify_recovered_key"]


VerifyResult = VerificationResult


def verify_recovered_key(H, x, g, *, m, r, field=None) -> VerifyResult:
    """Verify that ``(x, g)`` reconstructs the public parity check ``H``.

    Parameters
    ----------
    H : Sage matrix over GF(2) (the public key).
    x : iterable of support elements of ``GF(2**m)``.
    g : the Goppa polynomial over ``GF(2**m)``.
    m, r : the *expected* extension degree and Goppa degree (required,
        keyword-only).  The attack always knows them, so they are checked, not
        assumed: the field degree must equal ``m`` and ``deg(g)`` must equal
        ``r``.  Passing them independently makes the degree checks meaningful.
    field : ``GF(2**m)``; inferred from ``x`` (or ``g``) when omitted.  When
        supplied it must equal the support elements' actual parent.

    Checks
    ------
    * ``H`` is over GF(2);
    * the working field equals the support elements' parent;
    * ``g``'s base ring equals the support field;
    * field degree ``== m`` and ``deg(g) == r``;
    * ``len(x) == n``, support entries pairwise distinct and in the field;
    * ``g(x_i) != 0`` for all ``i``;
    * ``rank(H) == m*r`` (proper-instance gate);
    * the Frobenius-expanded rows ``(y_i x_i^k)^{2^j}`` (``y_i = 1/g(x_i)``,
      ``0 <= k < deg g``, ``0 <= j < field degree``) span, over the extension,
      exactly the row space of ``H`` embedded into it.
    """
    from sage.all import matrix

    x = list(x)
    checks = {}
    reasons = []

    def _fail(name, cond, message):
        checks[name] = bool(cond)
        if not cond:
            reasons.append(message)

    support_field = x[0].parent() if x else (field if field is not None else g.base_ring())
    if field is None:
        field = support_field
    field_degree = field.degree()
    deg_g = g.degree()
    n = H.ncols()

    # --- field / representation consistency -------------------------------- #
    _fail("public_matrix_over_gf2", H.base_ring().order() == 2,
          f"H is over {H.base_ring()}, expected GF(2)")
    # Compare the working field to the support's actual parent, rather than only
    # testing whether elements coerce into it.
    _fail("field_matches_support", field == support_field,
          "supplied field != support elements' parent")
    _fail("g_base_ring_matches_support", g.base_ring() == support_field,
          "g.base_ring() does not match the support field")
    _fail("field_degree_matches_m", field_degree == m,
          f"field degree {field_degree} != expected m={m}")
    _fail("goppa_degree_matches_r", deg_g == r,
          f"deg(g)={deg_g} != expected r={r}")

    # --- support well-formedness ------------------------------------------- #
    _fail("length_matches", len(x) == n, f"len(x)={len(x)} != n={n}")
    _fail("support_distinct", len(set(x)) == len(x),
          "support entries are not distinct")
    _fail("support_in_field", all(xi in field for xi in x),
          "some support entries are not in the field")
    _fail("g_nonvanishing", all(g(xi) != 0 for xi in x),
          "g vanishes at some support point")

    # --- proper-instance rank gate ----------------------------------------- #
    _fail("public_rank_is_mr", H.rank() == m * r,
          f"rank(H)={H.rank()} != m*r={m * r}")

    # --- row-space reconstruction ------------------------------------------ #
    # Only meaningful once the structural preconditions above hold; build with
    # the *actual* field degree and deg(g) so a parameter mismatch surfaces as
    # a row-space failure rather than an index error.
    if all(checks.values()):
        y = [1 / g(xi) for xi in x]
        Hrec = matrix(field, field_degree * deg_g, n)
        for j in range(field_degree):
            tj = 1 << j
            for k in range(deg_g):
                row = j * deg_g + k
                for l in range(n):
                    Hrec[row, l] = (y[l] * x[l] ** k) ** tj
        Hext = H.change_ring(field)
        _fail("rowspace_matches", Hrec.row_space() == Hext.row_space(),
              "reconstructed row space != public row space")
    else:
        checks["rowspace_matches"] = False
        reasons.append("skipped row-space check (a precondition failed)")

    ok = all(checks.values())
    return VerifyResult(ok=ok, checks=checks, reasons=reasons)
