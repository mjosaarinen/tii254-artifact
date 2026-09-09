"""Known-polynomial binary-Goppa support completion (KiMa23, Algorithms 3.5--3.6).

The public input is a full-rank binary parity check, a supplied degree-``t``
Goppa polynomial, and coordinate-labelled values at at least
``t * (m - 2) + 1`` public positions.  The randomized set-intersection loop
recovers enough additional labelled values to invoke the exact canonical
column extension in :mod:`hov_mceliece.partial_support`.

The expected running-time statement in KiMa23 treats the public parity check
as random.  Accordingly, exhaustion of the bounded trial budget is a typed
refusal, never a mathematical nonexistence claim.  A returned result has a
stronger acceptance condition: every generated word replays under the public
parity check, every recovered point set satisfies ``p*h = p' (mod g^2)``, and
the completed key passes the independent public-key verifier.
"""

from __future__ import annotations

from dataclasses import dataclass
import random

from .partial_support import (
    KnownPartialGoppaCompletion,
    extend_known_partial_goppa_support,
)


KIMA_KNOWN_POLYNOMIAL_COMPLETION_VERSION = (
    "kima23-known-polynomial-set-intersection-v1"
)


__all__ = [
    "KIMA_KNOWN_POLYNOMIAL_COMPLETION_VERSION",
    "KimaCompletionRefusal",
    "KimaKnownPolynomialCompletion",
    "KimaRecoveredPointSet",
    "KimaSetIntersectionStep",
    "complete_known_polynomial_support",
    "known_polynomial_point_threshold",
    "recover_point_set",
]


class KimaCompletionRefusal(RuntimeError):
    """A fail-closed gate or bounded randomized search refused completion."""

    def __init__(self, message: str, *, gate: str):
        super().__init__(message)
        self.gate = str(gate)


@dataclass(frozen=True)
class KimaRecoveredPointSet:
    outside_weight: int
    known_weight: int
    system_rows: int
    system_columns: int
    system_rank: int
    point_integers: tuple[int, ...]
    locator_polynomial: object


@dataclass(frozen=True)
class KimaSetIntersectionStep:
    step: int
    attempt: int
    new_label: int
    new_point_integer: int
    first_outside_labels: tuple[int, ...]
    second_outside_labels: tuple[int, ...]
    first_recovery_rank: int
    second_recovery_rank: int
    first_known_weight: int
    second_known_weight: int


@dataclass(frozen=True)
class KimaKnownPolynomialCompletion:
    version: str
    m: int
    goppa_degree: int
    threshold: int
    initial_known_count: int
    final_known_count: int
    seed: int
    attempts: int
    steps: tuple[KimaSetIntersectionStep, ...]
    known_labels: tuple[int, ...]
    known_point_integers: tuple[int, ...]
    completion: KnownPartialGoppaCompletion


def known_polynomial_point_threshold(*, m: int, t: int) -> int:
    """Return the KiMa23 known-polynomial threshold ``t*(m-2)+1``."""

    m, t = int(m), int(t)
    if m < 3 or t < 1:
        raise ValueError("known-polynomial completion requires m >= 3 and t >= 1")
    return t * (m - 2) + 1


def _coefficient_vector(polynomial, length: int):
    from sage.all import vector

    field = polynomial.base_ring()
    return vector(field, [polynomial[index] for index in range(int(length))])


def recover_point_set(H, g, known_points, codeword) -> KimaRecoveredPointSet:
    """Implement KiMa23 Algorithm 3.5 for one authenticated codeword.

    ``known_points`` maps public column indices to their supplied field
    elements.  The return value is the *unlabelled* set of values at the
    nonzero positions outside that mapping.
    """

    from sage.all import matrix

    n = int(H.ncols())
    if H.base_ring().order() != 2:
        raise ValueError("H must be binary")
    if len(codeword) != n or codeword.base_ring().order() != 2:
        raise ValueError("codeword must be one binary vector of public length")
    if not (H * codeword).is_zero():
        raise KimaCompletionRefusal(
            "the supplied vector is not in the public code",
            gate="codeword_replay",
        )

    field = g.base_ring()
    t = int(g.degree())
    if int(field.characteristic()) != 2 or t < 1:
        raise ValueError("g must have positive degree over a characteristic-two field")
    known = {int(label): value for label, value in known_points.items()}
    if any(label < 0 or label >= n for label in known):
        raise ValueError("known labels must be distinct in-range public positions")
    if any(value.parent() != field for value in known.values()):
        raise ValueError("known points and g must use the same field parent")

    support = tuple(index for index in range(n) if codeword[index] != 0)
    outside = tuple(index for index in support if index not in known)
    inside = tuple(index for index in support if index in known)
    k = len(outside)
    if k < 1 or k > t:
        raise KimaCompletionRefusal(
            f"outside support weight {k} is not in [1,{t}]",
            gate="outside_weight",
        )

    ring = g.parent()
    x = ring.gen()
    modulus = g**2
    h = ring.zero()
    for index in inside:
        h += (x - known[index]).inverse_mod(modulus)
    h %= modulus

    # Write p=x^k+sum a_j*x^j.  Linearize p*h=p' modulo g^2 by
    # treating the lower k-1 coefficients of p' as independent variables.
    # The coefficient of x^(k-1) in p' is the known field element k mod 2;
    # this matters when k is even.
    columns = []
    for degree in range(k):
        columns.append(_coefficient_vector((x**degree * h) % modulus, 2 * t))
    for degree in range(k - 1):
        columns.append(_coefficient_vector(-(x**degree) % modulus, 2 * t))
    system = matrix(field, columns).transpose()
    leading_derivative = field(k & 1) * x ** (k - 1)
    rhs = _coefficient_vector(
        (leading_derivative - x**k * h) % modulus,
        2 * t,
    )
    rank = int(system.rank())
    if rank != 2 * k - 1:
        raise KimaCompletionRefusal(
            f"Recover-Point-Set system rank {rank} != {2 * k - 1}",
            gate="recover_point_set_unique",
        )
    try:
        solution = system.solve_right(rhs)
    except ValueError as error:
        raise KimaCompletionRefusal(
            "Recover-Point-Set system is inconsistent",
            gate="recover_point_set_consistent",
        ) from error
    if system * solution != rhs:
        raise ArithmeticError("Recover-Point-Set solve did not replay")

    p = x**k + sum(solution[index] * x**index for index in range(k))
    recovered_derivative = leading_derivative + sum(
        solution[k + index] * x**index for index in range(k - 1)
    )
    if p.derivative() != recovered_derivative:
        raise KimaCompletionRefusal(
            "linearized derivative does not equal the recovered p'",
            gate="recover_point_set_derivative",
        )
    if (p * h - p.derivative()) % modulus != 0:
        raise ArithmeticError("recovered locator polynomial failed p*h=p' mod g^2")

    roots_with_multiplicity = tuple(p.roots())
    if (
        len(roots_with_multiplicity) != k
        or any(int(multiplicity) != 1 for _, multiplicity in roots_with_multiplicity)
    ):
        raise KimaCompletionRefusal(
            "recovered locator polynomial does not split into distinct linear factors",
            gate="recover_point_set_split",
        )
    points = tuple(
        sorted((root for root, _ in roots_with_multiplicity), key=lambda value: int(value.to_integer()))
    )
    if len(set(points)) != k or any(point in set(known.values()) for point in points):
        raise KimaCompletionRefusal(
            "recovered outside point set collides with the known support",
            gate="recover_point_set_distinct",
        )
    return KimaRecoveredPointSet(
        outside_weight=k,
        known_weight=len(inside),
        system_rows=2 * t,
        system_columns=2 * k - 1,
        system_rank=rank,
        point_integers=tuple(int(point.to_integer()) for point in points),
        locator_polynomial=p,
    )


def _linear_combinations(basis, *, affine=None, rng=None, sample_cap: int = 4096):
    """Yield exhaustive small-span or bounded seeded large-span combinations."""

    dimension = len(basis)
    if sample_cap < 1:
        raise ValueError("sample_cap must be positive")
    if affine is None:
        start = 1
        zero = basis[0].parent().zero() if basis else None
    else:
        start = 0
        zero = affine
    population = (1 << dimension) - start
    if population <= sample_cap:
        masks = range(start, 1 << dimension)
    else:
        if rng is None:
            raise ValueError("large-span sampling requires a seeded RNG")
        # Try sparse basis choices before seeded generic combinations.  The
        # latter keeps the M8-sized final iterations bounded instead of
        # attempting an exponential enumeration.
        leading = ([] if start else [0]) + [1 << index for index in range(dimension)]
        leading = leading[:sample_cap]
        seen = set(leading)
        while len(leading) < sample_cap:
            mask = rng.randrange(start, 1 << dimension)
            if mask not in seen:
                leading.append(mask)
                seen.add(mask)
        masks = leading
    for mask in masks:
        value = zero
        if value is None:
            value = basis[0].parent().zero()
        else:
            # Dense GF(2) Sage vectors do not expose ``copy()``.  Addition to
            # the parent's zero makes an independent mutable vector.
            value = value + value.parent().zero()
        for index, row in enumerate(basis):
            if (mask >> index) & 1:
                value += row
        yield value


def _embedded_word(binary, n: int, columns, values, *, extra_label=None):
    from sage.all import vector

    word = vector(binary, n)
    for position, label in enumerate(columns):
        word[int(label)] = values[position]
    if extra_label is not None:
        word[int(extra_label)] += 1
    return word


def _valid_first_words(H, known_labels, trial_labels, *, rng):
    columns = tuple(known_labels) + tuple(trial_labels)
    restricted = H[:, list(columns)]
    basis = tuple(restricted.right_kernel().basis())
    for value in _linear_combinations(basis, rng=rng):
        word = _embedded_word(H.base_ring(), int(H.ncols()), columns, value)
        outside = tuple(label for label in trial_labels if word[label] != 0)
        yield word, outside


def _valid_second_words(H, known_labels, trial_labels, target_label, *, rng):
    columns = tuple(known_labels) + tuple(trial_labels)
    restricted = H[:, list(columns)]
    target = H.column(int(target_label))
    try:
        particular = restricted.solve_right(target)
    except ValueError:
        return
    basis = tuple(restricted.right_kernel().basis())
    for value in _linear_combinations(basis, affine=particular, rng=rng):
        word = _embedded_word(
            H.base_ring(),
            int(H.ncols()),
            columns,
            value,
            extra_label=target_label,
        )
        outside = tuple(
            label
            for label in tuple(trial_labels) + (int(target_label),)
            if word[label] != 0
        )
        yield word, outside


def _validate_inputs(H, g, known_points, *, m: int, t: int):
    m, t = int(m), int(t)
    threshold = known_polynomial_point_threshold(m=m, t=t)
    if H.base_ring().order() != 2:
        raise ValueError("H must be binary")
    if int(H.nrows()) != m * t or int(H.rank()) != m * t:
        raise ValueError("H must be a full-rank m*t-row public parity check")
    field = g.base_ring()
    if (
        int(field.characteristic()) != 2
        or int(field.degree()) != m
        or int(g.degree()) != t
    ):
        raise ValueError("g does not match the declared binary-Goppa parameters")
    known = {int(label): value for label, value in known_points.items()}
    if len(known) != len(known_points):
        raise ValueError("known_points contains duplicate public labels")
    if len(known) < threshold:
        raise KimaCompletionRefusal(
            f"{len(known)} labelled points < t*(m-2)+1={threshold}",
            gate="known_point_threshold",
        )
    if any(label < 0 or label >= int(H.ncols()) for label in known):
        raise ValueError("known_points contains an out-of-range public label")
    values = tuple(known.values())
    if (
        any(value.parent() != field for value in values)
        or len(set(values)) != len(values)
        or any(g(value) == 0 for value in values)
    ):
        raise ValueError("known points must be distinct nonroots in g's field")
    return known, threshold


def complete_known_polynomial_support(
    H,
    g,
    known_points,
    *,
    m: int,
    t: int,
    seed: int = 0,
    max_attempts_per_point: int = 4096,
) -> KimaKnownPolynomialCompletion:
    """Run bounded KiMa23 set intersection, then complete and verify the key.

    Search exhaustion is typed ``bounded_search_exhausted`` and has no
    negative mathematical meaning.  The first target inventory is ``m*t+1``
    known points, matching Algorithm 3.6.  If that particular inventory lacks
    the full public rank required by Goppa-Points/canonical column extension,
    set intersection continues until the rank gate passes.  The final
    deterministic extension accepts only after complete public-key
    verification.
    """

    m, t = int(m), int(t)
    known, threshold = _validate_inputs(H, g, known_points, m=m, t=t)
    initial_known_count = len(known)
    target_known_count = m * t + 1
    if target_known_count > int(H.ncols()):
        raise ValueError("public length is smaller than m*t+1")
    if max_attempts_per_point < 1:
        raise ValueError("max_attempts_per_point must be positive")
    rng = random.Random(int(seed))
    steps = []
    total_attempts = 0

    def known_public_rank() -> int:
        return int(H[:, sorted(known)].rank())

    while len(known) < target_known_count or known_public_rank() < m * t:
        accepted = None
        for attempt in range(1, int(max_attempts_per_point) + 1):
            total_attempts += 1
            known_labels = tuple(sorted(known))
            unknown_labels = tuple(
                label for label in range(int(H.ncols())) if label not in known
            )
            if len(unknown_labels) < 2 * t:
                raise KimaCompletionRefusal(
                    "fewer than 2*t unknown public positions remain",
                    gate="unknown_position_inventory",
                )
            first_trials = tuple(sorted(rng.sample(unknown_labels, 2 * t)))
            first_choice = None
            for first_word, first_outside in _valid_first_words(
                H, known_labels, first_trials, rng=rng
            ):
                if not 1 <= len(first_outside) <= t:
                    continue
                try:
                    first_recovery = recover_point_set(H, g, known, first_word)
                except KimaCompletionRefusal:
                    continue
                first_choice = (first_word, first_outside, first_recovery)
                break
            if first_choice is None:
                continue
            first_word, first_outside, first_recovery = first_choice

            target_labels = list(first_outside)
            rng.shuffle(target_labels)
            for target_label in target_labels:
                second_pool = tuple(
                    label
                    for label in unknown_labels
                    if label not in set(first_outside)
                )
                if len(second_pool) < 2 * t - 1:
                    continue
                # Several line-6 choices may be needed.  Charging each choice
                # to the outer bounded attempt keeps refusal accounting simple.
                second_trials = tuple(sorted(rng.sample(second_pool, 2 * t - 1)))
                for second_word, second_outside in _valid_second_words(
                    H, known_labels, second_trials, target_label, rng=rng
                ):
                    if not 1 <= len(second_outside) <= t:
                        continue
                    if set(first_outside).intersection(second_outside) != {target_label}:
                        continue
                    try:
                        second_recovery = recover_point_set(H, g, known, second_word)
                    except KimaCompletionRefusal:
                        continue
                    common = set(first_recovery.point_integers).intersection(
                        second_recovery.point_integers
                    )
                    if len(common) != 1:
                        continue
                    new_integer = int(next(iter(common)))
                    new_point = g.base_ring().from_integer(new_integer)
                    if new_point in set(known.values()):
                        continue
                    if not (H * first_word).is_zero() or not (H * second_word).is_zero():
                        raise ArithmeticError("accepted KiMa codeword failed public replay")
                    accepted = (
                        target_label,
                        new_point,
                        first_outside,
                        second_outside,
                        first_recovery,
                        second_recovery,
                        attempt,
                    )
                    break
                if accepted is not None:
                    break
            if accepted is not None:
                break
        if accepted is None:
            raise KimaCompletionRefusal(
                f"no admitted set intersection after {max_attempts_per_point} trials",
                gate="bounded_search_exhausted",
            )

        (
            new_label,
            new_point,
            first_outside,
            second_outside,
            first_recovery,
            second_recovery,
            accepted_attempt,
        ) = accepted
        known[int(new_label)] = new_point
        steps.append(
            KimaSetIntersectionStep(
                step=len(steps),
                attempt=int(accepted_attempt),
                new_label=int(new_label),
                new_point_integer=int(new_point.to_integer()),
                first_outside_labels=tuple(map(int, first_outside)),
                second_outside_labels=tuple(map(int, second_outside)),
                first_recovery_rank=int(first_recovery.system_rank),
                second_recovery_rank=int(second_recovery.system_rank),
                first_known_weight=int(first_recovery.known_weight),
                second_known_weight=int(second_recovery.known_weight),
            )
        )

    labels = tuple(sorted(known))
    points = tuple(known[label] for label in labels)
    completion = extend_known_partial_goppa_support(
        H,
        points,
        labels,
        g,
        m=m,
        r=t,
    )
    if not completion.verification.ok:
        raise ArithmeticError("KiMa completion returned an unverified full key")
    return KimaKnownPolynomialCompletion(
        version=KIMA_KNOWN_POLYNOMIAL_COMPLETION_VERSION,
        m=m,
        goppa_degree=t,
        threshold=threshold,
        initial_known_count=initial_known_count,
        final_known_count=len(labels),
        seed=int(seed),
        attempts=total_attempts,
        steps=tuple(steps),
        known_labels=labels,
        known_point_integers=tuple(int(point.to_integer()) for point in points),
        completion=completion,
    )
