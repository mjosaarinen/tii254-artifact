"""Finish TII-254 from one 96-point projective locator branch.

The upstream rank-two consumer returns projective points.  One systematic
chart exposes exactly ``m*r = 96`` labelled points, so one fixed omitted
public coordinate makes the restricted code one-dimensional.  This module
enumerates that one missing projective point and the still-unknown affine-chart
pole, applies the identity-chart KiMa23 factor gate, and accepts only a
complete public-key verification.

No candidate score, delayed support, or Goppa polynomial affects the order.
An ordinary factor or extension refusal merely advances the canonical
projective-line enumeration.
"""

from __future__ import annotations

from dataclasses import dataclass

from .goppa_recovery import (
    GoppaRecoveryResult,
    advanced_factor_identity_candidate,
    prepare_advanced_factor_identity,
)
from .partial_support import (
    KnownPartialGoppaCompletion,
    PartialSupportCompletionError,
    extend_known_partial_goppa_support,
)

__all__ = [
    "TII254_ONE_PIVOT_FINISHER_VERSION",
    "OnePivotAttempt",
    "OnePivotFinisherError",
    "OnePivotFinisherResult",
    "finish_one_pivot_projective_support",
    "finish_tii254_locator_branch",
]


TII254_ONE_PIVOT_FINISHER_VERSION = (
    "tii254-projective-one-pivot-advanced-factor-v1"
)


@dataclass(frozen=True)
class OnePivotAttempt:
    candidate_index: int
    projective_candidate: tuple[int, int]
    pole_index: int
    projective_pole: tuple[int, int]
    finite_rechart_pole_integer: int | None
    terminal: str
    advanced_factor_candidate_count: int
    advanced_factor_seconds: float


@dataclass(frozen=True)
class OnePivotFinisherResult:
    branch_index: int | None
    known_labels: tuple[int, ...]
    pivot_label: int
    field_modulus: int
    candidate_count: int
    pole_count_per_candidate: int
    screen_count_ceiling: int
    accepted_candidate_index: int
    accepted_projective_candidate: tuple[int, int]
    accepted_pole_index: int
    accepted_projective_pole: tuple[int, int]
    finite_rechart_pole_integer: int | None
    attempts: tuple[OnePivotAttempt, ...]
    restricted_result: GoppaRecoveryResult
    completion: KnownPartialGoppaCompletion
    fresh_verification: object


class OnePivotFinisherError(RuntimeError):
    """Every admissible missing locator was rejected exactly."""

    def __init__(self, message: str, *, attempts=()):
        super().__init__(message)
        self.attempts = tuple(attempts)


def _field_modulus_integer(field) -> int:
    coefficients = tuple(field.modulus().list())
    return sum(int(value) << index for index, value in enumerate(coefficients))


def _normalized_projective_pair(point, field) -> tuple[int, int]:
    try:
        numerator, denominator = map(int, point)
    except (TypeError, ValueError) as error:
        raise ValueError("projective points must be integer pairs") from error
    order = int(field.order())
    if not (0 <= numerator < order and 0 <= denominator < order):
        raise ValueError("projective coordinate is outside the declared field")
    if numerator == denominator == 0:
        raise ValueError("the zero pair is not a projective point")
    if denominator == 0:
        return (1, 0)
    value = field.from_integer(numerator) / field.from_integer(denominator)
    return (int(value.to_integer()), 1)


def _finite_rechart_at_pole(projective_support, pole, field):
    """Move one projective support away from an explicitly chosen pole."""

    support = tuple(
        _normalized_projective_pair(point, field) for point in projective_support
    )
    selected_pole = _normalized_projective_pair(pole, field)
    if selected_pole in set(support):
        raise ValueError("the affine-chart pole lies on the candidate support")
    if selected_pole == (1, 0):
        if any(point == (1, 0) for point in support):
            raise ArithmeticError("the infinity-pole chart contains infinity")
        return tuple(field.from_integer(point[0]) for point in support), None
    pole_value = field.from_integer(selected_pole[0])
    output = tuple(
        field.zero()
        if point == (1, 0)
        else field.one() / (field.from_integer(point[0]) - pole_value)
        for point in support
    )
    if len(set(output)) != len(output):
        raise ArithmeticError("the explicit finite rechart introduced a collision")
    return output, int(pole_value.to_integer())


def finish_one_pivot_projective_support(
    H,
    known_labels,
    known_projective_support,
    pivot_label: int,
    *,
    field,
    m: int,
    r: int,
    branch_index: int | None = None,
) -> OnePivotFinisherResult:
    """Enumerate one missing point and affine pole; return the first full key.

    ``known_projective_support[i]`` labels public column ``known_labels[i]``.
    The known restriction must already have rank ``m*r`` and contain exactly
    ``m*r`` coordinates.  Consequently adding the fixed pivot label yields a
    one-dimensional shortened code, the setting in which the calibrated
    ``advanced_factor`` step is applicable.  Projective normalization does
    not identify the original point at infinity.  We therefore enumerate all
    support-free poles as a second, bounded projective gauge rather than
    assuming that an arbitrary finite rechart preserves the public code.
    """

    from .verify import verify_recovered_key

    m, r = int(m), int(r)
    labels = tuple(map(int, known_labels))
    support = tuple(known_projective_support)
    pivot = int(pivot_label)
    if H.base_ring().order() != 2 or int(H.nrows()) != m * r:
        raise ValueError("H must be a binary m*r-row public parity-check matrix")
    if int(field.characteristic()) != 2 or int(field.degree()) != m:
        raise ValueError("the finite field does not match m")
    if len(labels) != m * r or len(support) != len(labels):
        raise ValueError("the known chart must contain exactly m*r labelled points")
    if len(set(labels)) != len(labels) or any(
        label < 0 or label >= int(H.ncols()) for label in labels
    ):
        raise ValueError("known labels must be distinct in-range public columns")
    if pivot in set(labels) or not 0 <= pivot < int(H.ncols()):
        raise ValueError("the pivot label must be one distinct public column")

    normalized = tuple(
        _normalized_projective_pair(point, field) for point in support
    )
    if len(set(normalized)) != len(normalized):
        raise ValueError("the known projective support is not distinct")
    canonical_known = tuple(sorted(labels))
    if int(H[:, list(canonical_known)].rank()) != m * r:
        raise ValueError("the known public restriction does not have rank m*r")
    keep = tuple(sorted(labels + (pivot,)))
    if int(H[:, list(keep)].rank()) != m * r:
        raise ValueError("the one-pivot public restriction lost rank m*r")

    support_by_label = dict(zip(labels, normalized, strict=True))
    occupied = set(normalized)
    projective_line = ((1, 0),) + tuple(
        (encoded, 1) for encoded in range(int(field.order()))
    )
    candidates = tuple(point for point in projective_line if point not in occupied)
    expected_count = int(field.order()) + 1 - m * r
    if len(candidates) != expected_count:
        raise ArithmeticError("projective candidate inventory does not close")

    attempts = []
    restricted_public = H[:, list(keep)]
    factor_context = prepare_advanced_factor_identity(
        restricted_public,
        m=m,
        r=r,
    )
    pole_count = int(field.order()) - m * r
    for candidate_index, candidate in enumerate(candidates):
        complete_by_label = {**support_by_label, pivot: candidate}
        complete_projective = tuple(
            complete_by_label[label] for label in keep
        )
        open_poles = tuple(
            point for point in projective_line if point not in set(complete_projective)
        )
        if len(open_poles) != pole_count:
            raise ArithmeticError("affine-chart pole inventory does not close")
        for pole_index, projective_pole in enumerate(open_poles):
            affine_support, pole_integer = _finite_rechart_at_pole(
                complete_projective,
                projective_pole,
                field,
            )
            recovered, audit = advanced_factor_identity_candidate(
                factor_context,
                affine_support,
            )
            factor_seconds = float(audit.get("factor_seconds", 0.0)) + float(
                audit.get("subproduct_tree_seconds", 0.0)
            )
            if recovered is None:
                attempts.append(OnePivotAttempt(
                    candidate_index=candidate_index,
                    projective_candidate=candidate,
                    pole_index=pole_index,
                    projective_pole=projective_pole,
                    finite_rechart_pole_integer=pole_integer,
                    terminal=str(audit["terminal"]),
                    advanced_factor_candidate_count=int(
                        audit.get("candidate_count", 0)
                    ),
                    advanced_factor_seconds=factor_seconds,
                ))
                continue
            try:
                completion = extend_known_partial_goppa_support(
                    H,
                    recovered.support,
                    keep,
                    recovered.polynomial,
                    m=m,
                    r=r,
                )
            except PartialSupportCompletionError as error:
                attempts.append(OnePivotAttempt(
                    candidate_index=candidate_index,
                    projective_candidate=candidate,
                    pole_index=pole_index,
                    projective_pole=projective_pole,
                    finite_rechart_pole_integer=pole_integer,
                    terminal=f"full_extension_refused:{error.gate}",
                    advanced_factor_candidate_count=int(
                        audit.get("candidate_count", 0)
                    ),
                    advanced_factor_seconds=factor_seconds,
                ))
                continue
            fresh = verify_recovered_key(
                H,
                completion.support,
                completion.polynomial,
                m=m,
                r=r,
                field=field,
            )
            if not completion.verification.ok or not fresh.ok:
                raise ArithmeticError("one-pivot completion failed fresh verification")
            attempts.append(OnePivotAttempt(
                candidate_index=candidate_index,
                projective_candidate=candidate,
                pole_index=pole_index,
                projective_pole=projective_pole,
                finite_rechart_pole_integer=pole_integer,
                terminal="verified_full_key",
                advanced_factor_candidate_count=int(audit["candidate_count"]),
                advanced_factor_seconds=factor_seconds,
            ))
            return OnePivotFinisherResult(
                branch_index=branch_index,
                known_labels=labels,
                pivot_label=pivot,
                field_modulus=_field_modulus_integer(field),
                candidate_count=len(candidates),
                pole_count_per_candidate=pole_count,
                screen_count_ceiling=len(candidates) * pole_count,
                accepted_candidate_index=candidate_index,
                accepted_projective_candidate=candidate,
                accepted_pole_index=pole_index,
                accepted_projective_pole=projective_pole,
                finite_rechart_pole_integer=pole_integer,
                attempts=tuple(attempts),
                restricted_result=recovered,
                completion=completion,
                fresh_verification=fresh,
            )
    raise OnePivotFinisherError(
        f"all {len(candidates) * pole_count} candidate/pole screens were rejected",
        attempts=attempts,
    )


def finish_tii254_locator_branch(
    H,
    nonpivot_labels,
    locator_result,
    *,
    branch_index: int,
    pivot_label: int,
    field,
) -> OnePivotFinisherResult:
    """Bind a width-96 graph-pencil branch to the TII-254 finisher."""

    branch = int(branch_index)
    labels = tuple(map(int, nonpivot_labels))
    if tuple(map(int, H.dimensions())) != (96, 223):
        raise ValueError("the TII-254 public matrix must have shape 96 x 223")
    if len(labels) != 96 or locator_result.point_count != 96:
        raise ValueError("the TII-254 locator branch must label 96 nonpivots")
    if not 0 <= branch < int(locator_result.branch_count):
        raise ValueError("locator branch index is out of range")
    modulus = _field_modulus_integer(field)
    if (
        int(locator_result.field_degree) != 8
        or int(locator_result.field_modulus) != modulus
        or int(field.degree()) != 8
    ):
        raise ValueError("locator and Sage GF(256) representations differ")
    if not (
        locator_result.full_frobenius_orbit
        and locator_result.all_kernels_replay
        and locator_result.all_points_distinct_by_branch[branch]
    ):
        raise ValueError("the selected locator branch did not pass its replay gates")
    return finish_one_pivot_projective_support(
        H,
        labels,
        locator_result.projective_locators_by_branch[branch],
        pivot_label,
        field=field,
        m=8,
        r=12,
        branch_index=branch,
    )
