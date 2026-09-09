"""Complete a binary-Goppa key from one verified partial support.

The completion is the constructive form of the M16 column-extension lemma.
Once a restricted support has yielded a Goppa polynomial, the full-rank public
restriction determines one row transport.  Canonical Goppa columns then index
the unused nonroots, so every omitted public column has to name exactly one
support value.  No saved full key or secret-dependent choice enters this path.
"""

from __future__ import annotations

from dataclasses import dataclass

from .goppa_recovery import GoppaRecoveryResult, recover_goppa_polynomial

__all__ = [
    "PARTIAL_SUPPORT_COMPLETION_VERSION",
    "OmittedColumnMatch",
    "KnownPartialGoppaCompletion",
    "PartialGoppaCompletion",
    "PartialSupportCompletionError",
    "canonical_goppa_column",
    "canonical_goppa_secret_matrix",
    "complete_partial_goppa_support",
    "extend_known_partial_goppa_support",
]


PARTIAL_SUPPORT_COMPLETION_VERSION = "canonical-goppa-column-extension-v1"


class PartialSupportCompletionError(RuntimeError):
    """A partial support failed an exact completion gate."""

    def __init__(self, message: str, *, gate: str):
        super().__init__(message)
        self.gate = str(gate)


@dataclass(frozen=True)
class OmittedColumnMatch:
    column: int
    support_integer: int
    candidate_count: int


@dataclass(frozen=True)
class PartialGoppaCompletion:
    keep: tuple[int, ...]
    restricted_rank: int
    restricted_result: GoppaRecoveryResult
    canonical_restricted_matrix: object
    row_transport: object
    unused_nonroot_count: int
    omitted_matches: tuple[OmittedColumnMatch, ...]
    support: tuple
    polynomial: object
    canonical_full_matrix: object
    verification: object


@dataclass(frozen=True)
class KnownPartialGoppaCompletion:
    """A full key obtained by extending an already verified partial key."""

    keep: tuple[int, ...]
    restricted_rank: int
    canonical_restricted_matrix: object
    row_transport: object
    unused_nonroot_count: int
    omitted_matches: tuple[OmittedColumnMatch, ...]
    support: tuple
    polynomial: object
    canonical_full_matrix: object
    verification: object


def _field_signature(field):
    return (
        int(field.characteristic()),
        int(field.degree()),
        str(field.modulus()),
    )


def _canonical_element(value, field):
    """Rebuild an element in ``field`` after an exact modulus check.

    Sage objects returned by a forked worker can carry a distinct parent
    identity even when the defining field is byte-for-byte the same.  Integer
    transport is valid only after the defining modulus has been checked; a
    different representation must be handled by an explicit root map before
    calling this routine.
    """
    source = value.parent()
    if _field_signature(source) != _field_signature(field):
        raise PartialSupportCompletionError(
            "Goppa worker returned a different field representation; an "
            "explicit recorded root map is required",
            gate="restricted_key",
        )
    return field.from_integer(int(value.to_integer()))


def _canonical_result(result, field):
    from sage.all import PolynomialRing

    support = tuple(_canonical_element(value, field) for value in result.support)
    ring = PolynomialRing(field, "X")
    polynomial = ring([
        _canonical_element(result.polynomial[index], field)
        for index in range(int(result.polynomial.degree()) + 1)
    ])
    return GoppaRecoveryResult(
        support=support,
        polynomial=polynomial,
        transformation=result.transformation,
        gcd_degree=int(result.gcd_degree),
        scheduler=result.scheduler,
    )


def canonical_goppa_column(x, g):
    """Return ``gamma_g(x) = ((x^k/g(x))^(2^j))_(j,k)``.

    Rows use the same ``j``-major, ``k``-minor order as the public verifier and
    :func:`hov_mceliece.toy.goppa_parity_check`.
    """
    from sage.all import vector

    field = x.parent()
    if g.base_ring() != field:
        raise ValueError("x and g must use the same finite-field parent")
    if int(field.characteristic()) != 2:
        raise ValueError("canonical Goppa columns require characteristic two")
    if g(x) == 0:
        raise ValueError("canonical Goppa column is undefined at a root of g")
    r = int(g.degree())
    if r < 1:
        raise ValueError("g must have positive degree")
    inverse = 1 / g(x)
    return vector(field, [
        (x ** k * inverse) ** (1 << j)
        for j in range(int(field.degree()))
        for k in range(r)
    ])


def canonical_goppa_secret_matrix(support, g):
    """Construct the canonical Frobenius-expanded secret matrix by columns."""
    from sage.all import matrix

    support = tuple(support)
    if not support:
        raise ValueError("support must be nonempty")
    columns = [canonical_goppa_column(value, g) for value in support]
    return matrix(g.base_ring(), columns).transpose()


def _column_key(column):
    return tuple(int(value.to_integer()) for value in column)


def extend_known_partial_goppa_support(
    H,
    partial_support,
    keep,
    polynomial,
    *,
    m: int,
    r: int,
):
    """Extend one verified, full-rank partial Goppa key to all columns.

    Unlike :func:`complete_partial_goppa_support`, this entry point assumes
    that the degree-``r`` polynomial has already been recovered together with
    the labelled partial support.  It performs no key-equation search.  The
    exact public restriction determines a unique row transport, after which
    canonical Goppa columns label every omitted public coordinate.
    """
    from .verify import verify_recovered_key

    m, r = int(m), int(r)
    keep = tuple(int(column) for column in keep)
    partial_support = tuple(partial_support)
    if m < 1 or r < 2:
        raise ValueError("partial completion requires m >= 1 and r >= 2")
    if len(keep) != len(partial_support) or not keep:
        raise ValueError("keep and partial_support must have one nonempty entry each")
    if len(set(keep)) != len(keep) or any(
        column < 0 or column >= int(H.ncols()) for column in keep
    ):
        raise ValueError("keep must contain distinct in-range public columns")
    if keep != tuple(sorted(keep)):
        raise ValueError("keep must use canonical increasing public-column order")
    if H.base_ring().order() != 2 or int(H.nrows()) != m * r:
        raise ValueError("H must be a binary m*r-row public parity-check matrix")
    field = partial_support[0].parent()
    if (
        int(field.characteristic()) != 2
        or int(field.degree()) != m
        or polynomial.base_ring() != field
        or int(polynomial.degree()) != r
        or any(value.parent() != field for value in partial_support)
        or len(set(partial_support)) != len(partial_support)
        or any(polynomial(value) == 0 for value in partial_support)
    ):
        raise ValueError("partial key does not match the declared field and degree")

    restricted_public = H[:, keep]
    restricted_rank = int(restricted_public.rank())
    if restricted_rank != m * r:
        raise PartialSupportCompletionError(
            f"restricted public rank {restricted_rank} != m*r={m * r}",
            gate="restricted_key",
        )
    restricted_verification = verify_recovered_key(
        restricted_public,
        partial_support,
        polynomial,
        m=m,
        r=r,
        field=field,
    )
    if not restricted_verification.ok:
        raise PartialSupportCompletionError(
            "supplied partial key failed the restricted public verifier: "
            + "; ".join(restricted_verification.reasons),
            gate="restricted_key",
        )

    canonical_restricted = canonical_goppa_secret_matrix(
        partial_support, polynomial
    )
    public_over_field = restricted_public.change_ring(field)
    if canonical_restricted.row_space() != public_over_field.row_space():
        raise PartialSupportCompletionError(
            "canonical restricted secret and public matrices have different row spaces",
            gate="restricted_key",
        )
    pivots = tuple(int(value) for value in public_over_field.pivots())
    if len(pivots) != m * r:
        raise PartialSupportCompletionError(
            "restricted public matrix did not expose m*r transport pivots",
            gate="row_transport",
        )
    pivot_public = public_over_field[:, pivots]
    row_transport = canonical_restricted[:, pivots] * pivot_public.inverse()
    if not row_transport.is_invertible() or (
        row_transport * public_over_field != canonical_restricted
    ):
        raise PartialSupportCompletionError(
            "unique row transport failed on the restricted public key",
            gate="row_transport",
        )

    used = set(partial_support)
    candidates = {}
    for value in field:
        if value in used or polynomial(value) == 0:
            continue
        key = _column_key(canonical_goppa_column(value, polynomial))
        candidates.setdefault(key, []).append(value)
    if any(len(values) != 1 for values in candidates.values()):
        raise PartialSupportCompletionError(
            "canonical unused-nonroot column index is not injective",
            gate="column_extension",
        )

    keep_positions = {column: index for index, column in enumerate(keep)}
    support = [None] * int(H.ncols())
    for column, position in keep_positions.items():
        support[column] = partial_support[position]
    omitted_matches = []
    full_public = H.change_ring(field)
    for column in range(int(H.ncols())):
        if column in keep_positions:
            continue
        target = row_transport * full_public.column(column)
        matches = candidates.get(_column_key(target), [])
        if len(matches) != 1:
            raise PartialSupportCompletionError(
                f"omitted public column {column} has {len(matches)} canonical matches",
                gate="column_extension",
            )
        value = matches[0]
        support[column] = value
        omitted_matches.append(
            OmittedColumnMatch(
                column=column,
                support_integer=int(value.to_integer()),
                candidate_count=1,
            )
        )

    support = tuple(support)
    if any(value is None for value in support) or len(set(support)) != len(support):
        raise PartialSupportCompletionError(
            "completed support is incomplete or contains repeated points",
            gate="column_extension",
        )
    canonical_full = canonical_goppa_secret_matrix(support, polynomial)
    if row_transport * full_public != canonical_full:
        raise PartialSupportCompletionError(
            "completed canonical matrix does not equal the transported public key",
            gate="full_verification",
        )
    verification = verify_recovered_key(
        H, support, polynomial, m=m, r=r, field=field
    )
    if not verification.ok:
        raise PartialSupportCompletionError(
            "completed key failed independent public verification: "
            + "; ".join(verification.reasons),
            gate="full_verification",
        )
    return KnownPartialGoppaCompletion(
        keep=keep,
        restricted_rank=restricted_rank,
        canonical_restricted_matrix=canonical_restricted,
        row_transport=row_transport,
        unused_nonroot_count=len(candidates),
        omitted_matches=tuple(omitted_matches),
        support=support,
        polynomial=polynomial,
        canonical_full_matrix=canonical_full,
        verification=verification,
    )


def complete_partial_goppa_support(
    H,
    partial_support,
    keep,
    *,
    m: int,
    r: int,
    workers: int = 1,
    worker_peak_bytes: int | None = None,
    memory_budget_bytes: int | None = None,
    reconstruction: str = "key_equation",
    key_equation_seed: int = 0,
    key_equation_max_codewords: int = 32,
):
    """Recover and verify a complete key from a restricted public support.

    ``partial_support[position]`` corresponds to public column
    ``keep[position]``.  All gates are exact: the restricted public matrix must
    have rank ``m*r``; its recovered key must verify; the induced row transport
    must be unique and invertible; every omitted column must have one unused
    nonroot preimage; and the completed key must pass the independent public
    verifier.
    """
    from sage.all import matrix

    from .verify import verify_recovered_key

    m, r = int(m), int(r)
    if m < 1 or r < 2:
        raise ValueError("partial completion requires m >= 1 and r >= 2")
    keep = tuple(int(column) for column in keep)
    partial_support = tuple(partial_support)
    if len(keep) != len(partial_support) or not keep:
        raise ValueError("keep and partial_support must have one nonempty entry each")
    if len(set(keep)) != len(keep) or any(
        column < 0 or column >= int(H.ncols()) for column in keep
    ):
        raise ValueError("keep must contain distinct in-range public columns")
    if keep != tuple(sorted(keep)):
        raise ValueError("keep must use canonical increasing public-column order")
    if H.base_ring().order() != 2:
        raise ValueError("H must be a binary public parity-check matrix")
    if int(H.nrows()) != m * r:
        raise ValueError(
            f"H must have exactly m*r={m * r} rows, got {int(H.nrows())}"
        )
    field = partial_support[0].parent()
    if (
        int(field.characteristic()) != 2
        or int(field.degree()) != m
        or any(value.parent() != field for value in partial_support)
        or len(set(partial_support)) != len(partial_support)
    ):
        raise ValueError("partial support must be distinct in the declared field")

    restricted_public = H[:, keep]
    restricted_rank = int(restricted_public.rank())
    if restricted_rank != m * r:
        raise PartialSupportCompletionError(
            f"restricted public rank {restricted_rank} != m*r={m * r}",
            gate="restricted_key",
        )
    recovered = recover_goppa_polynomial(
        restricted_public,
        partial_support,
        m=m,
        r=r,
        workers=workers,
        worker_peak_bytes=worker_peak_bytes,
        memory_budget_bytes=memory_budget_bytes,
        reconstruction=reconstruction,
        key_equation_seed=key_equation_seed,
        key_equation_max_codewords=key_equation_max_codewords,
    )
    recovered = _canonical_result(recovered, field)
    extended = extend_known_partial_goppa_support(
        H,
        recovered.support,
        keep,
        recovered.polynomial,
        m=m,
        r=r,
    )
    return PartialGoppaCompletion(
        keep=keep,
        restricted_rank=extended.restricted_rank,
        restricted_result=recovered,
        canonical_restricted_matrix=extended.canonical_restricted_matrix,
        row_transport=extended.row_transport,
        unused_nonroot_count=extended.unused_nonroot_count,
        omitted_matches=extended.omitted_matches,
        support=extended.support,
        polynomial=extended.polynomial,
        canonical_full_matrix=extended.canonical_full_matrix,
        verification=extended.verification,
    )
