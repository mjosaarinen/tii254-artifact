"""Finish TII-254 from a labelled ordinary-locator projective atlas.

The D6 threshold-panel consumer returns labelled points only up to a common
``PGL(2)`` transformation and Frobenius.  If the atlas contains every known
point of one public all-one circuit except its outside point, the two missing
projective roles are small: an affine-chart pole and that one circuit point.

This module enumerates those roles, factors the circuit derivative, screens
each polynomial against the complete supplied public restriction, and invokes
the established known-polynomial completion only after that exact screen.
Exhaustion is a typed inconclusive refusal; a positive result is accepted only
after complete public-key verification.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
import json
from time import perf_counter

from .goppa_recovery import _polynomial_digest, _square_root_polynomial
from .kima_known_polynomial_completion import (
    KimaCompletionRefusal,
    KimaKnownPolynomialCompletion,
    complete_known_polynomial_support,
    known_polynomial_point_threshold,
    recover_point_set,
)
from .partial_support import (
    PartialSupportCompletionError,
    canonical_goppa_secret_matrix,
)
from .tii254_d6_one_root_factor import (
    evaluate_one_root_derivative_pencil,
    one_root_derivative_pencil,
)
from .tii254_one_pivot_finisher import (
    _finite_rechart_at_pole,
    _normalized_projective_pair,
)
from .verify import verify_recovered_key


TII254_D6_ALL_ORDINARY_PROJECTIVE_FINISHER_VERSION = (
    "tii254-d6-all-ordinary-projective-finisher-v1"
)

__all__ = [
    "TII254_D6_ALL_ORDINARY_PROJECTIVE_FINISHER_VERSION",
    "AllOrdinaryProjectiveCompletion",
    "AllOrdinaryProjectiveFinisherRefusal",
    "finish_all_ordinary_projective_support",
    "projective_line",
]


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("ascii")


def _stream_update(digest, value: object) -> None:
    encoded = _canonical_bytes(value)
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def _field_integer(value) -> int:
    return int(value.to_integer())


def projective_line(field) -> tuple[tuple[int, int], ...]:
    """Return ``P^1(field)`` in the repository's canonical wire order."""

    return ((1, 0),) + tuple(
        (integer, 1) for integer in range(int(field.order()))
    )


def _factor_repeated_degree(numerator, degree: int):
    """Factor the square root of a characteristic-two derivative exactly."""

    root = _square_root_polynomial(numerator)
    if root is None or root**2 != numerator:
        raise ArithmeticError("circuit derivative is not an exact square")
    factorization = root.factor()
    if factorization.prod() != root:
        raise ArithmeticError("square-root factorization failed exact replay")
    candidates = tuple(
        factor.monic()
        for factor, multiplicity in factorization
        if int(factor.degree()) == int(degree) and int(multiplicity) >= 1
    )
    if any(numerator % (candidate**2) for candidate in candidates):
        raise ArithmeticError("a repeated factor does not divide the numerator")
    return root, candidates


def _public_word(H, support):
    from sage.all import vector

    labels = set(map(int, support))
    word = vector(
        H.base_ring(),
        [int(label in labels) for label in range(int(H.ncols()))],
    )
    if not (H * word).is_zero():
        raise ValueError("a semantic support is not a word of the public code")
    return word


def _prepare_semantic_families(H, families, known_domain, degree: int):
    """Bind circuit-equivalent public words and their unknown label sets."""

    prepared = []
    for family_index, family in enumerate(families or ()):
        supports = tuple(tuple(map(int, support)) for support in family)
        if len(supports) < 2 or any(
            not support
            or len(set(support)) != len(support)
            or any(label < 0 or label >= int(H.ncols()) for label in support)
            for support in supports
        ):
            raise ValueError("each semantic family needs at least two valid supports")
        unknown = tuple(
            frozenset(support) - set(known_domain) for support in supports
        )
        if (
            any(value != unknown[0] for value in unknown[1:])
            or not 1 <= len(unknown[0]) <= int(degree)
        ):
            raise ValueError(
                "circuit-equivalent supports must have one common nonempty "
                "unknown label set of size at most r"
            )
        prepared.append(
            {
                "family_index": family_index,
                "supports": supports,
                "words": tuple(_public_word(H, support) for support in supports),
                "unknown_labels": unknown[0],
            }
        )
    return tuple(prepared)


def _semantic_screen(H, polynomial, known, prepared):
    """Require Recover-Point-Set invariance and labelled intersection sizes."""

    recovered_sets = []
    profile = []
    for family in prepared:
        recoveries = tuple(
            recover_point_set(H, polynomial, known, word)
            for word in family["words"]
        )
        sets = tuple(frozenset(item.point_integers) for item in recoveries)
        expected_weight = len(family["unknown_labels"])
        if (
            any(value != sets[0] for value in sets[1:])
            or any(int(item.outside_weight) != expected_weight for item in recoveries)
        ):
            raise KimaCompletionRefusal(
                "circuit-equivalent lifts recover different point sets",
                gate="circuit_lift_invariance",
            )
        recovered_sets.append(sets[0])
        profile.append(
            {
                "family_index": int(family["family_index"]),
                "lift_count": len(recoveries),
                "outside_weight": expected_weight,
                "system_rank": int(recoveries[0].system_rank),
                "point_set_sha256": sha256(
                    _canonical_bytes(sorted(sets[0]))
                ).hexdigest(),
            }
        )
    for left in range(len(prepared)):
        for right in range(left + 1, len(prepared)):
            expected = len(
                prepared[left]["unknown_labels"]
                & prepared[right]["unknown_labels"]
            )
            observed = len(recovered_sets[left] & recovered_sets[right])
            if observed != expected:
                raise KimaCompletionRefusal(
                    f"semantic point-set intersection {observed} != {expected}",
                    gate="labelled_set_intersection",
                )
    return tuple(profile)


@dataclass(frozen=True)
class AllOrdinaryProjectiveCompletion:
    version: str
    known_labels: tuple[int, ...]
    active_labels: tuple[int, ...]
    outside_label: int
    projective_pole: tuple[int, int]
    outside_projective_point: tuple[int, int]
    affine_known_point_integers: tuple[int, ...]
    polynomial: object
    completion: KimaKnownPolynomialCompletion
    fresh_verification: object
    audit: dict[str, object]


class AllOrdinaryProjectiveFinisherRefusal(RuntimeError):
    """A bounded projective inventory produced no fully verified key."""

    def __init__(self, message: str, *, gate: str, audit: dict[str, object]):
        super().__init__(message)
        self.gate = str(gate)
        self.audit = dict(audit)


def finish_all_ordinary_projective_support(
    H,
    known_labels,
    known_projective_support,
    active_labels,
    outside_label: int,
    *,
    field,
    m: int,
    r: int,
    completion_seed: int = 260_904,
    max_attempts_per_point: int = 256,
    screen_cap: int | None = None,
    semantic_support_families=(),
) -> AllOrdinaryProjectiveCompletion:
    """Enumerate ``(pole, outside point)`` and return the first verified key.

    ``known_projective_support[i]`` is the labelled point at
    ``known_labels[i]``.  ``active_labels`` must name a public all-one circuit
    containing exactly one label absent from the atlas, ``outside_label``.

    ``screen_cap`` is only a resource/refusal cap.  Hitting it has no negative
    mathematical meaning.  With no cap, the complete ordered inventory has
    ``(q+1-k)(q-k)`` screens for ``k`` supplied projective points.
    """

    started = perf_counter()
    m, r = int(m), int(r)
    outside = int(outside_label)
    labels = tuple(map(int, known_labels))
    points = tuple(
        _normalized_projective_pair(point, field)
        for point in known_projective_support
    )
    active = tuple(map(int, active_labels))
    columns = int(H.ncols())
    if H.base_ring().order() != 2 or int(H.nrows()) != m * r:
        raise ValueError("H must be a binary m*r-row public parity check")
    if int(field.characteristic()) != 2 or int(field.degree()) != m:
        raise ValueError("field does not match the declared binary-Goppa degree")
    if (
        len(labels) != len(points)
        or len(set(labels)) != len(labels)
        or len(set(points)) != len(points)
        or any(label < 0 or label >= columns for label in labels)
    ):
        raise ValueError("known labels and projective points must be distinct")
    threshold = known_polynomial_point_threshold(m=m, t=r)
    if len(labels) < threshold:
        raise ValueError(
            f"{len(labels)} projective locators are below threshold {threshold}"
        )
    if (
        outside in set(labels)
        or outside < 0
        or outside >= columns
        or len(set(active)) != len(active)
        or outside not in set(active)
        or set(active) - {outside} - set(labels)
        or set(labels).isdisjoint(set(active) - {outside})
    ):
        raise ValueError("the circuit/atlas label contract is inconsistent")
    public_active = H[:, list(active)]
    active_kernel = public_active.right_kernel_matrix()
    if (
        int(public_active.rank()) != len(active) - 1
        or int(active_kernel.nrows()) != 1
        or tuple(map(int, active_kernel.row(0))) != (1,) * len(active)
    ):
        raise ValueError("active labels do not carry one all-one public circuit")

    projective = projective_line(field)
    supplied_by_label = dict(zip(labels, points, strict=True))
    supplied_set = set(points)
    open_points = tuple(point for point in projective if point not in supplied_set)
    expected_screens = len(open_points) * (len(open_points) - 1)
    if len(open_points) != int(field.order()) + 1 - len(points):
        raise ArithmeticError("projective complement inventory did not close")
    if screen_cap is not None and int(screen_cap) < 1:
        raise ValueError("screen_cap must be positive when supplied")

    selected_labels = tuple(sorted((*labels, outside)))
    semantic_families = _prepare_semantic_families(
        H, semantic_support_families, selected_labels, r
    )
    public_restricted = H[:, list(selected_labels)].change_ring(field)
    public_rank = int(public_restricted.rank())
    public_row_space = public_restricted.row_space()
    active_known = tuple(label for label in active if label != outside)
    ring = None
    screens = 0
    numerator_count = 0
    factor_candidate_count = 0
    row_space_passes = 0
    semantic_passes = 0
    completion_attempts = 0
    refusal_histogram: Counter[str] = Counter()
    screen_stream = sha256()

    for pole in open_points:
        nonpole = tuple(point for point in projective if point != pole)
        affine, pole_integer = _finite_rechart_at_pole(nonpole, pole, field)
        affine_by_point = dict(zip(nonpole, affine, strict=True))
        if len(affine_by_point) != int(field.order()) or set(affine) != set(field):
            raise ArithmeticError("P1-minus-pole did not rechart bijectively")
        affine_by_label = {
            label: affine_by_point[point]
            for label, point in supplied_by_label.items()
        }
        if ring is None:
            from sage.all import PolynomialRing

            ring = PolynomialRing(field, "X")
        pencil = one_root_derivative_pencil(
            tuple(affine_by_label[label] for label in active_known), ring
        )

        for root_point in open_points:
            if root_point == pole:
                continue
            if screen_cap is not None and screens >= int(screen_cap):
                audit = {
                    "known_locator_count": len(labels),
                    "open_projective_count": len(open_points),
                    "complete_ordered_inventory": expected_screens,
                    "screens_completed": screens,
                    "screen_cap": int(screen_cap),
                    "screen_stream_sha256": screen_stream.hexdigest(),
                    "wall_seconds": perf_counter() - started,
                }
                raise AllOrdinaryProjectiveFinisherRefusal(
                    "projective screen cap reached before a verified key",
                    gate="screen_cap",
                    audit=audit,
                )
            screens += 1
            root_value = affine_by_point[root_point]
            numerator = evaluate_one_root_derivative_pencil(pencil, root_value)
            numerator_count += 1
            square_root, candidates = _factor_repeated_degree(numerator, r)
            factor_candidate_count += len(candidates)
            candidate_digests = tuple(
                _polynomial_digest(candidate) for candidate in candidates
            )
            _stream_update(
                screen_stream,
                {
                    "screen": screens - 1,
                    "pole": list(pole),
                    "pole_integer": pole_integer,
                    "root": list(root_point),
                    "derivative_sha256": _polynomial_digest(numerator),
                    "square_root_sha256": _polynomial_digest(square_root),
                    "candidate_sha256": list(candidate_digests),
                },
            )
            if not candidates:
                refusal_histogram["no_repeated_degree_r_factor"] += 1
                continue

            known_with_root = {**affine_by_label, outside: root_value}
            ordered_support = tuple(
                known_with_root[label] for label in selected_labels
            )
            for polynomial in candidates:
                if any(polynomial(value) == 0 for value in ordered_support):
                    refusal_histogram["known_goppa_denominator_zero"] += 1
                    continue
                canonical = canonical_goppa_secret_matrix(
                    ordered_support, polynomial
                )
                if (
                    int(canonical.rank()) != public_rank
                    or canonical.row_space() != public_row_space
                ):
                    refusal_histogram["restricted_row_space"] += 1
                    continue
                row_space_passes += 1
                try:
                    semantic_profile = _semantic_screen(
                        H, polynomial, known_with_root, semantic_families
                    )
                except KimaCompletionRefusal as error:
                    refusal_histogram[f"semantic:{error.gate}"] += 1
                    continue
                semantic_passes += 1
                completion_attempts += 1
                try:
                    completed = complete_known_polynomial_support(
                        H,
                        polynomial,
                        known_with_root,
                        m=m,
                        t=r,
                        seed=int(completion_seed),
                        max_attempts_per_point=int(max_attempts_per_point),
                    )
                except (KimaCompletionRefusal, PartialSupportCompletionError) as error:
                    refusal_histogram[
                        f"completion:{getattr(error, 'gate', 'unknown')}"
                    ] += 1
                    continue
                fresh = verify_recovered_key(
                    H,
                    completed.completion.support,
                    completed.completion.polynomial,
                    m=m,
                    r=r,
                    field=field,
                )
                if not completed.completion.verification.ok or not fresh.ok:
                    raise ArithmeticError("projective completion failed fresh verification")
                audit = {
                    "known_locator_count": len(labels),
                    "known_polynomial_threshold": threshold,
                    "known_count_after_circuit_root": len(known_with_root),
                    "restricted_public_rank": public_rank,
                    "open_projective_count": len(open_points),
                    "complete_ordered_inventory": expected_screens,
                    "screens_completed": screens,
                    "numerators_factored": numerator_count,
                    "factor_candidate_count": factor_candidate_count,
                    "restricted_row_space_pass_count": row_space_passes,
                    "semantic_family_count": len(semantic_families),
                    "semantic_screen_pass_count": semantic_passes,
                    "selected_semantic_profile": list(semantic_profile),
                    "completion_attempt_count": completion_attempts,
                    "refusal_histogram": dict(sorted(refusal_histogram.items())),
                    "screen_stream_sha256": screen_stream.hexdigest(),
                    "selected_polynomial_sha256": _polynomial_digest(polynomial),
                    "completion_initial_known_count": int(
                        completed.initial_known_count
                    ),
                    "completion_final_known_count": int(completed.final_known_count),
                    "completion_set_intersection_steps": len(completed.steps),
                    "completion_total_trials": int(completed.attempts),
                    "full_key_verified": True,
                    "exhaustive_inventory_required_for_positive_result": False,
                    "wall_seconds": perf_counter() - started,
                }
                return AllOrdinaryProjectiveCompletion(
                    version=TII254_D6_ALL_ORDINARY_PROJECTIVE_FINISHER_VERSION,
                    known_labels=labels,
                    active_labels=active,
                    outside_label=outside,
                    projective_pole=pole,
                    outside_projective_point=root_point,
                    affine_known_point_integers=tuple(
                        _field_integer(affine_by_label[label]) for label in labels
                    ),
                    polynomial=polynomial,
                    completion=completed,
                    fresh_verification=fresh,
                    audit=audit,
                )

    audit = {
        "known_locator_count": len(labels),
        "known_polynomial_threshold": threshold,
        "open_projective_count": len(open_points),
        "complete_ordered_inventory": expected_screens,
        "screens_completed": screens,
        "numerators_factored": numerator_count,
        "factor_candidate_count": factor_candidate_count,
        "restricted_row_space_pass_count": row_space_passes,
        "semantic_family_count": len(semantic_families),
        "semantic_screen_pass_count": semantic_passes,
        "completion_attempt_count": completion_attempts,
        "refusal_histogram": dict(sorted(refusal_histogram.items())),
        "screen_stream_sha256": screen_stream.hexdigest(),
        "full_key_verified": False,
        "wall_seconds": perf_counter() - started,
    }
    raise AllOrdinaryProjectiveFinisherRefusal(
        "complete projective circuit inventory produced no verified key",
        gate="inventory_exhausted_inconclusive",
        audit=audit,
    )
