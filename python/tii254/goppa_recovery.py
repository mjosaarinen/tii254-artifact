"""Recover a Goppa polynomial from an alternant support (H26 Algorithm 2)."""

from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
import hashlib
import json
import multiprocessing
import os
import resource
from time import perf_counter, sleep

__all__ = [
    "GOPPA_ADVANCED_FACTOR_VERSION",
    "GOPPA_KEY_EQUATION_VERSION",
    "GOPPA_MULTIPLIER_COMPLETION_VERSION",
    "GOPPA_SCHEDULER_VERSION",
    "GoppaRecoveryResult",
    "RationalMultiplierCandidate",
    "AdvancedFactorIdentityContext",
    "GoppaRecoveryError",
    "GoppaTransform",
    "advanced_factor_identity_candidate",
    "canonical_goppa_transforms",
    "measure_goppa_worker",
    "prepare_advanced_factor_identity",
    "recover_goppa_from_grs_multiplier",
    "recover_goppa_polynomial",
    "rational_multiplier_candidate",
]


GOPPA_SCHEDULER_VERSION = "legacy-sage-field-order-process-v1"
GOPPA_KEY_EQUATION_VERSION = "sha256-basis-bound-codewords-v1"
GOPPA_ADVANCED_FACTOR_VERSION = "kima23-algorithm-3.2-canonical-basis-v1"
GOPPA_MULTIPLIER_COMPLETION_VERSION = "isolated-edge-multiplier-v1"


class GoppaRecoveryError(RuntimeError):
    pass


@dataclass(frozen=True)
class GoppaTransform:
    index: int
    label: str
    alpha_integer: int | None


@dataclass(frozen=True)
class GoppaRecoveryResult:
    support: tuple
    polynomial: object
    transformation: str
    gcd_degree: int
    scheduler: dict | None = None


@dataclass(frozen=True)
class RationalMultiplierCandidate:
    terminal: str
    solution_dimension: int
    support: tuple | None = None
    polynomial: object | None = None
    transform_label: str | None = None
    alpha_integer: int | None = None


@dataclass(frozen=True)
class AdvancedFactorIdentityContext:
    """One fixed binary restriction and its reusable right-kernel basis."""

    H: object
    code_basis: object
    m: int
    r: int


def rational_multiplier_candidate(structure, *, r: int):
    """Recover the unique degree-``(r,1)`` multiplier interpolation, if any.

    This is a fail-closed accelerator for M13.  Every ambiguity or pole on the
    finite support returns a typed refusal; the caller retains the complete
    canonical inverse-shift enumeration as its acceptance path.
    """
    from sage.all import PolynomialRing, matrix

    support = tuple(structure.support)
    squares = tuple(structure.multiplier)
    if (
        not support
        or len(support) != len(squares)
        or len(set(support)) != len(support)
    ):
        return RationalMultiplierCandidate("invalid_support", 0)
    field = support[0].parent()
    if any(
        value.parent() != field for value in support + squares
    ) or any(value == 0 or not value.is_square() for value in squares):
        return RationalMultiplierCandidate("invalid_multiplier", 0)
    multiplier = tuple(value.sqrt() for value in squares)
    reciprocal = tuple(1 / value for value in multiplier)
    equations = matrix(field, [
        [-(value**degree) for degree in range(r + 1)]
        + [rho, rho * value]
        for value, rho in zip(support, reciprocal)
    ])
    solution = equations.right_kernel().basis_matrix()
    dimension = int(solution.nrows())
    if dimension != 1:
        return RationalMultiplierCandidate(
            "nonunique_rational_interpolation", dimension
        )
    coefficients = solution.row(0)
    ring = PolynomialRing(field, "X")
    numerator = ring(list(coefficients[: r + 1]))
    q0, q1 = coefficients[r + 1], coefficients[r + 2]
    if q0 == 0 and q1 == 0:
        return RationalMultiplierCandidate("zero_denominator", dimension)
    if any(q0 + q1 * value == 0 for value in support):
        return RationalMultiplierCandidate("denominator_hits_support", dimension)
    if any(
        (q0 + q1 * value) * rho != numerator(value)
        for value, rho in zip(support, reciprocal)
    ):
        return RationalMultiplierCandidate("interpolation_identity_failed", dimension)
    if q1 == 0:
        polynomial = (numerator / q0).monic()
        transformed = support
        label = "identity"
        alpha_integer = None
    else:
        alpha = -q0 / q1
        transformed = tuple(1 / (value - alpha) for value in support)
        transformed_multiplier = tuple(
            multiplier[index] * (support[index] - alpha) ** (r - 1)
            for index in range(len(support))
        )
        polynomial = ring.lagrange_polynomial([
            (transformed[index], 1 / transformed_multiplier[index])
            for index in range(r + 1)
        ])
        if any(
            polynomial(transformed[index]) != 1 / transformed_multiplier[index]
            for index in range(len(support))
        ):
            return RationalMultiplierCandidate(
                "transformed_interpolation_failed", dimension
            )
        polynomial = polynomial.monic()
        alpha_integer = int(alpha.to_integer())
        label = f"inverse_shift:{alpha}"
    if polynomial.degree() != r:
        return RationalMultiplierCandidate("wrong_polynomial_degree", dimension)
    return RationalMultiplierCandidate(
        terminal="unique_candidate",
        solution_dimension=dimension,
        support=tuple(transformed),
        polynomial=polynomial,
        transform_label=label,
        alpha_integer=alpha_integer,
    )


@dataclass(frozen=True)
class _TransformOutcome:
    index: int
    label: str
    terminal: str
    record: dict
    support: tuple | None = None
    polynomial: object | None = None
    gcd_degree: int | None = None


def _binary_matrix_digest(value) -> str:
    """Digest a binary matrix without depending on Sage's string format."""
    digest = hashlib.sha256()
    digest.update(f"{value.nrows()}x{value.ncols()}\n".encode())
    packed_bytes = (value.ncols() + 7) // 8
    for row in value.rows():
        packed = bytearray(packed_bytes)
        for index, bit in enumerate(row):
            if bit:
                packed[index // 8] |= 1 << (index % 8)
        digest.update(packed)
    return digest.hexdigest()


def _deterministic_codeword_coefficients(
    dimension: int,
    *,
    seed: int,
    basis_sha256: str,
    index: int,
):
    """Return one public, versioned coefficient vector over ``GF(2)``.

    The stream is bound to the canonical public-code basis and uses SHAKE-256
    rather than Python's evolving pseudo-random-generator implementation.
    """
    from sage.all import GF, vector

    payload = json.dumps({
        "basis_sha256": basis_sha256,
        "index": int(index),
        "seed": int(seed),
        "version": GOPPA_KEY_EQUATION_VERSION,
    }, sort_keys=True, separators=(",", ":")).encode()
    raw = hashlib.shake_256(payload).digest((dimension + 7) // 8)
    return vector(GF(2), [
        (raw[position // 8] >> (position % 8)) & 1
        for position in range(dimension)
    ])


def _balanced_product(factors, one):
    """Multiply polynomial factors through a deterministic subproduct tree."""
    level = list(factors)
    if not level:
        return one
    while len(level) > 1:
        next_level = [
            level[index] * level[index + 1]
            for index in range(0, len(level) - 1, 2)
        ]
        if len(level) % 2:
            next_level.append(level[-1])
        level = next_level
    return level[0]


def _consume_key_equation_polynomials(polynomials, *, r: int, accept):
    """GCD a stream and examine candidates only once degree is at most ``2r``.

    This small generic core is also the semantics fixture for extra-factor and
    more-than-two-codeword tests.  ``polynomials`` yields ``(metadata, value)``
    pairs; ``accept`` returns an accepted object or ``None``.
    """
    current = None
    trace = []
    for metadata, polynomial in polynomials:
        gcd_started = perf_counter()
        current = (
            polynomial.monic()
            if current is None
            else current.gcd(polynomial).monic()
        )
        gcd_seconds = perf_counter() - gcd_started
        degree = int(current.degree())
        examined = degree <= 2 * r
        accepted = accept(current) if examined else None
        trace.append({
            **metadata,
            "gcd_degree": degree,
            "gcd_seconds": gcd_seconds,
            "candidate_examined": examined,
            "candidate_accepted": accepted is not None,
        })
        if accepted is not None:
            return accepted, current, trace
        if examined:
            # The running gcd can only decrease.  Once a degree-at-most-2r
            # candidate has failed the exact square-root/degree/public gate,
            # no later stream element can restore a valid degree-2r square;
            # hand the transform to the exact legacy fallback immediately.
            break
    return None, current, trace


def _candidate_gcd(code_basis, support, ring):
    X = ring.gen()
    product = ring.one()
    for xi in support:
        product *= X - xi
    quotients = [product // (X - xi) for xi in support]
    h = ring.zero()
    for c in code_basis.rows():
        pc = ring.zero()
        for i, bit in enumerate(c):
            if bit:
                pc += quotients[i]
        if not pc:
            continue
        h = pc.monic() if not h else h.gcd(pc).monic()
        if h.degree() == 0:
            break
    return h


def _square_root_polynomial(h):
    """Return the characteristic-two polynomial square root, or ``None``."""
    ring = h.parent()
    X = ring.gen()
    root = ring.zero()
    for exponent, coefficient in h.dict().items():
        if exponent % 2 or not coefficient.is_square():
            return None
        root += coefficient.sqrt() * X ** (exponent // 2)
    return root


def recover_goppa_from_grs_multiplier(
    H,
    square_space,
    structure,
    *,
    m: int,
    r: int,
):
    """Complete a key from an isolated ``GRS_(2r-1)(z, eta^2)`` edge.

    Only the finite transformation family from Proposition M13 is tested:
    identity followed by ``z -> 1/(z-alpha)`` for each unused field element
    in canonical Sage iteration order.  Every candidate transports the GRS
    multiplier exactly and is accepted only by the independent public-key
    verifier.  Failure is explicit so callers can invoke the unchanged M7K
    fallback.
    """
    from sage.all import PolynomialRing, matrix

    from .verify import verify_recovered_key

    support = tuple(structure.support)
    multiplier_square = tuple(structure.multiplier)
    if int(structure.dimension) != 2 * r - 1:
        raise GoppaRecoveryError(
            "isolated-edge GRS dimension must equal 2r-1"
        )
    if (
        not support
        or len(support) != H.ncols()
        or len(multiplier_square) != len(support)
        or len(set(support)) != len(support)
    ):
        raise GoppaRecoveryError(
            "isolated-edge support/multiplier has invalid length or repetitions"
        )
    field = support[0].parent()
    if int(field.characteristic()) != 2 or int(field.degree()) != m:
        raise GoppaRecoveryError(
            "multiplier completion requires the declared characteristic-two field"
        )
    if any(
        value.parent() != field for value in support + multiplier_square
    ):
        raise GoppaRecoveryError("isolated-edge values do not share one field")
    if any(value == 0 or not value.is_square() for value in multiplier_square):
        raise GoppaRecoveryError("isolated-edge multiplier is zero or non-square")

    vandermonde = matrix(field, [
        [multiplier_square[column] * support[column] ** exponent
         for column in range(len(support))]
        for exponent in range(2 * r - 1)
    ])
    if vandermonde.row_space() != square_space.row_space():
        raise GoppaRecoveryError(
            "isolated-edge multiplier failed the exact GRS row-space gate"
        )

    multiplier = tuple(value.sqrt() for value in multiplier_square)
    unused = tuple(value for value in field if value not in set(support))
    transforms = (GoppaTransform(0, "identity", None),) + tuple(
        GoppaTransform(index, f"inverse_shift:{alpha}", _field_integer(alpha))
        for index, alpha in enumerate(unused, start=1)
    )
    expected_count = 1 + int(field.order()) - len(support)
    if len(transforms) != expected_count:
        raise AssertionError("finite multiplier-completion family has wrong size")

    ring = PolynomialRing(field, "X")
    direct = rational_multiplier_candidate(structure, r=r)
    direct_record = {
        "terminal": direct.terminal,
        "solution_dimension": int(direct.solution_dimension),
        "transform_label": direct.transform_label,
        "alpha_integer": direct.alpha_integer,
    }
    if direct.terminal == "unique_candidate":
        verification = verify_recovered_key(
            H,
            direct.support,
            direct.polynomial,
            m=m,
            r=r,
            field=field,
        )
        direct_record["public_verification_ok"] = bool(verification.ok)
        direct_record["verification_reasons"] = list(verification.reasons)
        if verification.ok:
            matching = [
                item for item in transforms
                if item.alpha_integer == direct.alpha_integer
            ]
            if len(matching) != 1:
                raise AssertionError(
                    "direct multiplier pole did not name one canonical transform"
                )
            transform = matching[0]
            scheduler = {
                "scheduler_version": GOPPA_MULTIPLIER_COMPLETION_VERSION,
                "transform_count": len(transforms),
                "expected_transform_count": expected_count,
                "best_verified_index": int(transform.index),
                "attempts": [],
                "direct_interpolation": direct_record,
                "fallback_terminal": "not_run_after_direct_verified_recovery",
                "returned_key_sha256": _result_digest(
                    direct.support, direct.polynomial
                ),
            }
            return GoppaRecoveryResult(
                support=tuple(direct.support),
                polynomial=direct.polynomial,
                transformation=transform.label,
                gcd_degree=2 * r,
                scheduler=scheduler,
            )
    attempts = []
    for transform in transforms:
        if transform.alpha_integer is None:
            candidate_support = support
            candidate_multiplier = multiplier
        else:
            alpha = field.from_integer(transform.alpha_integer)
            candidate_support = tuple(1 / (value - alpha) for value in support)
            candidate_multiplier = tuple(
                multiplier[index] * (support[index] - alpha) ** (r - 1)
                for index in range(len(support))
            )
        record = {
            "index": int(transform.index),
            "label": transform.label,
            "alpha_integer": transform.alpha_integer,
            "transport_exponent": int(r - 1),
        }
        try:
            polynomial = ring.lagrange_polynomial([
                (candidate_support[index], 1 / candidate_multiplier[index])
                for index in range(r + 1)
            ])
        except (ArithmeticError, ValueError, ZeroDivisionError) as exc:
            record.update({
                "terminal": "interpolation_failed",
                "rejection": str(exc),
            })
            attempts.append(record)
            continue
        if polynomial.degree() != r:
            record.update({
                "terminal": "wrong_polynomial_degree",
                "polynomial_degree": int(polynomial.degree()),
            })
            attempts.append(record)
            continue
        if any(
            polynomial(candidate_support[index])
            != 1 / candidate_multiplier[index]
            for index in range(len(support))
        ):
            record.update({
                "terminal": "multiplier_interpolation_rejected",
                "polynomial_degree": int(polynomial.degree()),
            })
            attempts.append(record)
            continue
        polynomial = polynomial.monic()
        verification = verify_recovered_key(
            H, candidate_support, polynomial, m=m, r=r, field=field
        )
        if not verification.ok:
            record.update({
                "terminal": "public_verification_rejected",
                "polynomial_degree": int(polynomial.degree()),
                "reasons": list(verification.reasons),
            })
            attempts.append(record)
            continue
        record.update({
            "terminal": "verified_recovery",
            "polynomial_degree": int(polynomial.degree()),
        })
        attempts.append(record)
        scheduler = {
            "scheduler_version": GOPPA_MULTIPLIER_COMPLETION_VERSION,
            "transform_count": len(transforms),
            "expected_transform_count": expected_count,
            "best_verified_index": int(transform.index),
            "attempts": attempts,
            "direct_interpolation": direct_record,
            "fallback_terminal": "canonical_enumeration_verified_recovery",
            "returned_key_sha256": _result_digest(
                candidate_support, polynomial
            ),
        }
        return GoppaRecoveryResult(
            support=tuple(candidate_support),
            polynomial=polynomial,
            transformation=transform.label,
            gcd_degree=2 * r,
            scheduler=scheduler,
        )
    raise GoppaRecoveryError(
        "isolated-edge multiplier completion exhausted all "
        f"{expected_count} canonical transformations; attempts="
        + json.dumps(attempts, sort_keys=True)
    )


def _key_equation_candidate(
    H,
    code_basis,
    support,
    ring,
    *,
    m: int,
    r: int,
    seed: int,
    max_codewords: int,
):
    """Try deterministic public-codeword key equations on one support.

    Every derivative is built with a balanced subproduct tree.  Candidate
    examination is delayed until the running gcd has degree at most ``2r``;
    acceptance still requires the ordinary independent public-key verifier.
    """
    from .verify import verify_recovered_key

    X = ring.gen()
    basis_sha256 = _binary_matrix_digest(code_basis)
    order_digest = hashlib.sha256()
    product_seconds = 0.0
    generated_count = 0
    usable_count = 0
    skipped = []
    verification_rejections = []

    def polynomial_stream():
        nonlocal product_seconds, generated_count, usable_count
        for index in range(max_codewords):
            coefficients = _deterministic_codeword_coefficients(
                code_basis.nrows(),
                seed=seed,
                basis_sha256=basis_sha256,
                index=index,
            )
            generated_count += 1
            coefficient_bytes = bytes(int(bit) for bit in coefficients)
            coefficient_sha256 = hashlib.sha256(coefficient_bytes).hexdigest()
            order_digest.update(bytes.fromhex(coefficient_sha256))
            if not any(coefficients):
                skipped.append({
                    "index": index,
                    "coefficient_sha256": coefficient_sha256,
                    "terminal": "zero_coefficient_vector",
                })
                continue
            codeword = coefficients * code_basis
            positions = [
                position for position, bit in enumerate(codeword) if bit
            ]
            if len(positions) <= 2 * r:
                skipped.append({
                    "index": index,
                    "coefficient_sha256": coefficient_sha256,
                    "codeword_weight": len(positions),
                    "terminal": "codeword_too_small",
                })
                continue
            product_started = perf_counter()
            product = _balanced_product(
                [X - support[position] for position in positions],
                ring.one(),
            )
            derivative = product.derivative()
            elapsed = perf_counter() - product_started
            product_seconds += elapsed
            if not derivative:
                skipped.append({
                    "index": index,
                    "coefficient_sha256": coefficient_sha256,
                    "codeword_weight": len(positions),
                    "product_seconds": elapsed,
                    "terminal": "zero_derivative",
                })
                continue
            usable_count += 1
            yield ({
                "index": index,
                "coefficient_sha256": coefficient_sha256,
                "codeword_weight": len(positions),
                "derivative_degree": int(derivative.degree()),
                "product_seconds": elapsed,
            }, derivative)

    def accept(current):
        root = _square_root_polynomial(current)
        if root is None:
            verification_rejections.append({
                "gcd_degree": int(current.degree()),
                "terminal": "not_characteristic_two_square",
            })
            return None
        root = root.monic()
        if root.degree() != r:
            verification_rejections.append({
                "gcd_degree": int(current.degree()),
                "root_degree": int(root.degree()),
                "terminal": "wrong_root_degree",
            })
            return None
        verification = verify_recovered_key(
            H, support, root, m=m, r=r
        )
        if not verification.ok:
            verification_rejections.append({
                "gcd_degree": int(current.degree()),
                "root_degree": int(root.degree()),
                "terminal": "public_verification_rejected",
                "reasons": list(verification.reasons),
            })
            return None
        return root

    polynomial, final_gcd, trace = _consume_key_equation_polynomials(
        polynomial_stream(), r=r, accept=accept
    )
    report = {
        "version": GOPPA_KEY_EQUATION_VERSION,
        "seed": int(seed),
        "max_codewords": int(max_codewords),
        "code_basis_sha256": basis_sha256,
        "codeword_order_sha256": order_digest.hexdigest(),
        "generated_codeword_count": generated_count,
        "usable_codeword_count": usable_count,
        "gcd_degree_trace": trace,
        "skipped_codewords": skipped,
        "subproduct_tree_seconds": product_seconds,
        "gcd_seconds": sum(item["gcd_seconds"] for item in trace),
        "final_gcd_degree": (
            None if final_gcd is None else int(final_gcd.degree())
        ),
        "verification_rejections": verification_rejections,
        "terminal": (
            "verified_recovery"
            if polynomial is not None
            else (
                "candidate_gate_rejected"
                if trace and trace[-1]["candidate_examined"]
                else "max_codewords_exhausted"
            )
        ),
    }
    return polynomial, report


def _polynomial_digest(polynomial) -> str:
    payload = [
        int(polynomial[index].to_integer())
        for index in range(int(polynomial.degree()) + 1)
    ]
    encoded = json.dumps(payload, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _advanced_factor_candidate(
    H,
    code_basis,
    support,
    ring,
    *,
    m: int,
    r: int,
):
    """Run the minimal-support factor step of KiMa23, Algorithm 3.2.

    The canonical first right-kernel basis vector supplies the public
    codeword.  Its support polynomial derivative is the numerator in
    Equation (4) of KiMa23.  We fail closed unless there is exactly one
    irreducible degree-``r`` factor occurring with multiplicity at least two,
    and that factor reconstructs the complete restricted public row space.
    """
    from .verify import verify_recovered_key

    basis_sha256 = _binary_matrix_digest(code_basis)
    common = {
        "version": GOPPA_ADVANCED_FACTOR_VERSION,
        "code_basis_sha256": basis_sha256,
        "code_dimension": int(code_basis.nrows()),
        "selected_basis_row": 0 if code_basis.nrows() else None,
    }
    if not code_basis.nrows():
        return None, {
            **common,
            "terminal": "zero_restricted_code_dimension",
            "candidate_count": 0,
        }

    codeword = code_basis.row(0)
    positions = [
        position for position, bit in enumerate(codeword) if bit
    ]
    X = ring.gen()
    product_started = perf_counter()
    product = _balanced_product(
        [X - support[position] for position in positions],
        ring.one(),
    )
    numerator = product.derivative()
    product_seconds = perf_counter() - product_started
    if not numerator:
        return None, {
            **common,
            "terminal": "zero_codeword_numerator",
            "codeword_weight": len(positions),
            "subproduct_tree_seconds": product_seconds,
            "candidate_count": 0,
        }

    factor_started = perf_counter()
    factorization = list(numerator.factor())
    factor_seconds = perf_counter() - factor_started
    factors = [
        {
            "degree": int(factor.degree()),
            "multiplicity": int(multiplicity),
            "polynomial_sha256": _polynomial_digest(factor.monic()),
        }
        for factor, multiplicity in factorization
    ]
    candidates = [
        factor.monic()
        for factor, multiplicity in factorization
        if int(factor.degree()) == r and int(multiplicity) >= 2
    ]
    report = {
        **common,
        "codeword_weight": len(positions),
        "numerator_degree": int(numerator.degree()),
        "subproduct_tree_seconds": product_seconds,
        "factor_seconds": factor_seconds,
        "factorization": factors,
        "candidate_count": len(candidates),
        "candidate_sha256": [
            _polynomial_digest(candidate) for candidate in candidates
        ],
    }
    if len(candidates) != 1:
        return None, {
            **report,
            "terminal": (
                "no_degree_r_multiplicity_two_factor"
                if not candidates
                else "ambiguous_degree_r_multiplicity_two_factors"
            ),
        }

    polynomial = candidates[0]
    verification = verify_recovered_key(
        H, support, polynomial, m=m, r=r
    )
    if not verification.ok:
        return None, {
            **report,
            "terminal": "public_verification_rejected",
            "verification_reasons": list(verification.reasons),
        }
    return polynomial, {
        **report,
        "terminal": "verified_recovery",
        "verification_reasons": [],
    }


def prepare_advanced_factor_identity(
    H,
    *,
    m: int,
    r: int,
) -> AdvancedFactorIdentityContext:
    """Bind the support-independent work for repeated identity screens."""

    m, r = int(m), int(r)
    if H.base_ring().order() != 2 or int(H.nrows()) != m * r:
        raise ValueError("H must be a binary m*r-row public restriction")
    code_basis = H.right_kernel_matrix()
    if int(code_basis.ncols()) != int(H.ncols()) or H * code_basis.transpose():
        raise ArithmeticError("restricted-code basis failed exact replay")
    return AdvancedFactorIdentityContext(H, code_basis, m, r)


def advanced_factor_identity_candidate(
    context: AdvancedFactorIdentityContext,
    support,
):
    """Screen one already-finite support with the KiMa23 factor gate.

    This is the bounded identity-chart form needed by a projective locator
    enumerator.  The caller is responsible for moving its complete candidate
    support into an affine chart first.  In that chart a correct candidate is
    already a valid Goppa support, so trying every inverse-shift transform
    again would duplicate the outer projective enumeration.

    The return value is ``(result, audit)``.  ``result`` is ``None`` for an
    ordinary candidate rejection; a non-``None`` result has already passed
    the exact restricted public-row-space verifier.  This routine deliberately
    has no legacy or key-equation fallback: the one-pivot TII-254 calibration
    established that the minimal-support factor step is the load-bearing
    backend at dimension one.
    """

    from sage.all import PolynomialRing

    if not isinstance(context, AdvancedFactorIdentityContext):
        raise TypeError(
            "advanced-factor identity screening requires a prepared context"
        )
    H, code_basis, m, r = (
        context.H,
        context.code_basis,
        context.m,
        context.r,
    )
    support, field = _validate_inputs(H, support, m, r)
    ring = PolynomialRing(field, "X")
    polynomial, audit = _advanced_factor_candidate(
        H,
        code_basis,
        support,
        ring,
        m=m,
        r=r,
    )
    if polynomial is None:
        return None, audit
    return (
        GoppaRecoveryResult(
            support=tuple(support),
            polynomial=polynomial,
            transformation="identity",
            gcd_degree=2 * r,
            scheduler={
                "scheduler_version": GOPPA_SCHEDULER_VERSION,
                "reconstruction_policy": "advanced_factor_identity_only",
                "advanced_factor_version": GOPPA_ADVANCED_FACTOR_VERSION,
                "field_modulus": str(field.modulus()),
                "attempts": ({
                    "index": 0,
                    "label": "identity",
                    "terminal": audit["terminal"],
                    "advanced_factor": audit,
                },),
                "best_verified_index": 0,
            },
        ),
        audit,
    )


def _field_integer(value) -> int:
    return int(value.to_integer())


def canonical_goppa_transforms(support):
    """Freeze the legacy identity-plus-Sage-field-iteration order.

    Sage's iterator is deterministic for the pinned field modulus but is not
    integer-sorted.  Preserving it is load-bearing: changing this order can
    select a different equivalent verified key.
    """
    support = tuple(support)
    if not support:
        raise ValueError("support must be nonempty")
    field = support[0].parent()
    used = set(support)
    unused = [value for value in field if value not in used]
    transforms = [GoppaTransform(0, "identity", None)]
    for index, alpha in enumerate(unused, start=1):
        transforms.append(GoppaTransform(
            index,
            f"inverse_shift:{alpha}",
            _field_integer(alpha),
        ))
    return tuple(transforms)


def _transform_order_digest(transforms, field) -> str:
    payload = {
        "field_modulus": str(field.modulus()),
        "scheduler_version": GOPPA_SCHEDULER_VERSION,
        "transforms": [
            {
                "index": item.index,
                "label": item.label,
                "alpha_integer": item.alpha_integer,
            }
            for item in transforms
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _result_digest(support, polynomial) -> str:
    payload = {
        "support": [_field_integer(value) for value in support],
        "polynomial": [
            _field_integer(polynomial[index])
            for index in range(polynomial.degree() + 1)
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode()).hexdigest()


def _transformed_support(support, transform):
    if transform.alpha_integer is None:
        return tuple(support)
    field = support[0].parent()
    alpha = field.from_integer(transform.alpha_integer)
    return tuple(1 / (value - alpha) for value in support)


def _evaluate_transform(
    H,
    support,
    code_basis,
    transform,
    *,
    m: int,
    r: int,
    reconstruction: str,
    key_equation_seed: int,
    key_equation_max_codewords: int,
    delay_seconds: float = 0.0,
):
    from sage.all import PolynomialRing

    from .verify import verify_recovered_key

    started = perf_counter()
    if delay_seconds:
        sleep(delay_seconds)
    transformed = _transformed_support(support, transform)
    ring = PolynomialRing(support[0].parent(), "X")
    common = {
        "index": transform.index,
        "label": transform.label,
        "alpha_integer": transform.alpha_integer,
        "pid": os.getpid(),
        "process_group": os.getpgrp(),
        "reconstruction_policy": reconstruction,
    }
    if reconstruction == "advanced_factor":
        polynomial, advanced_factor = _advanced_factor_candidate(
            H,
            code_basis,
            transformed,
            ring,
            m=m,
            r=r,
        )
        record = {
            **common,
            "terminal": advanced_factor["terminal"],
            "reconstruction_method": "advanced_factor",
            "advanced_factor": advanced_factor,
            "legacy_fallback_terminal": "not_run_by_policy",
            "gcd_degree": None,
            "seconds": perf_counter() - started,
            "peak_rss_kib": int(
                resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
            ),
        }
        if polynomial is not None:
            record.update({
                "candidate_branch": "degree_r_multiplicity_two_factor",
                "gcd_degree": 2 * r,
                "result_sha256": _result_digest(transformed, polynomial),
            })
            return _TransformOutcome(
                transform.index,
                transform.label,
                record["terminal"],
                record,
                tuple(transformed),
                polynomial,
                2 * r,
            )
        return _TransformOutcome(
            transform.index,
            transform.label,
            record["terminal"],
            record,
        )

    key_equation = None
    if reconstruction == "key_equation":
        polynomial, key_equation = _key_equation_candidate(
            H,
            code_basis,
            transformed,
            ring,
            m=m,
            r=r,
            seed=key_equation_seed,
            max_codewords=key_equation_max_codewords,
        )
        if polynomial is not None:
            record = {
                **common,
                "terminal": "verified_recovery",
                "reconstruction_method": "key_equation",
                "gcd_degree": key_equation["final_gcd_degree"],
                "candidate_branch": "key_equation_square_root",
                "key_equation": key_equation,
                "legacy_fallback_terminal": "not_run_after_verified_recovery",
                "seconds": perf_counter() - started,
                "peak_rss_kib": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                ),
                "result_sha256": _result_digest(transformed, polynomial),
            }
            return _TransformOutcome(
                transform.index,
                transform.label,
                record["terminal"],
                record,
                tuple(transformed),
                polynomial,
                int(key_equation["final_gcd_degree"]),
            )

    h = _candidate_gcd(code_basis, transformed, ring)
    fallback_common = {
        **common,
        "reconstruction_method": (
            "legacy" if key_equation is None else "legacy_fallback"
        ),
        "key_equation": key_equation,
    }
    if not h or h.degree() <= 0:
        record = {
            **fallback_common,
            "terminal": "no_nonconstant_gcd",
            "legacy_fallback_terminal": "no_nonconstant_gcd",
            "gcd_degree": None if not h else int(h.degree()),
            "seconds": perf_counter() - started,
            "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
        }
        return _TransformOutcome(
            transform.index, transform.label, record["terminal"], record
        )

    root = _square_root_polynomial(h)
    candidates = []
    if root is not None:
        candidates.append(("square_root", root.monic()))
    candidates.append(("gcd", h.monic()))
    degree_r_seen = False
    rejected = []
    for branch, polynomial in candidates:
        if polynomial.degree() != r:
            continue
        degree_r_seen = True
        verification = verify_recovered_key(
            H, transformed, polynomial, m=m, r=r
        )
        if verification.ok:
            record = {
                **fallback_common,
                "terminal": "verified_recovery",
                "legacy_fallback_terminal": "verified_recovery",
                "gcd_degree": int(h.degree()),
                "candidate_branch": branch,
                "seconds": perf_counter() - started,
                "peak_rss_kib": int(
                    resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                ),
                "result_sha256": _result_digest(transformed, polynomial),
            }
            return _TransformOutcome(
                transform.index,
                transform.label,
                record["terminal"],
                record,
                tuple(transformed),
                polynomial,
                int(h.degree()),
            )
        rejected.append({
            "branch": branch,
            "reasons": list(verification.reasons),
        })
    terminal = (
        "public_verification_rejected"
        if degree_r_seen else "no_degree_r_candidate"
    )
    record = {
        **fallback_common,
        "terminal": terminal,
        "legacy_fallback_terminal": terminal,
        "gcd_degree": int(h.degree()),
        "verification_rejections": rejected,
        "seconds": perf_counter() - started,
        "peak_rss_kib": int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss),
    }
    return _TransformOutcome(
        transform.index, transform.label, terminal, record
    )


_WORKER_CONTEXT = None


def _initialize_worker(
    H,
    support,
    code_basis,
    m,
    r,
    delays,
    reconstruction,
    key_equation_seed,
    key_equation_max_codewords,
):
    global _WORKER_CONTEXT
    _WORKER_CONTEXT = (
        H,
        support,
        code_basis,
        m,
        r,
        delays,
        reconstruction,
        key_equation_seed,
        key_equation_max_codewords,
    )


def _evaluate_transform_worker(transform):
    if _WORKER_CONTEXT is None:
        raise RuntimeError("Goppa transform worker was not initialized")
    (
        H,
        support,
        code_basis,
        m,
        r,
        delays,
        reconstruction,
        key_equation_seed,
        key_equation_max_codewords,
    ) = _WORKER_CONTEXT
    return _evaluate_transform(
        H,
        support,
        code_basis,
        transform,
        m=m,
        r=r,
        reconstruction=reconstruction,
        key_equation_seed=key_equation_seed,
        key_equation_max_codewords=key_equation_max_codewords,
        delay_seconds=float(delays.get(transform.index, 0.0)),
    )


def _validate_inputs(H, support, m, r):
    support = tuple(support)
    if len(support) != H.ncols() or len(set(support)) != len(support):
        raise ValueError("support must contain n distinct entries")
    field = support[0].parent()
    if field.degree() != m:
        raise ValueError("support field degree does not match m")
    return support, field


def measure_goppa_worker(
    H,
    support,
    *,
    m: int,
    r: int,
    transform_index=0,
    reconstruction="key_equation",
    key_equation_seed=0,
    key_equation_max_codewords=32,
):
    """Measure one isolated canonical transform without claiming recovery."""
    support, field = _validate_inputs(H, support, m, r)
    transforms = canonical_goppa_transforms(support)
    if not 0 <= transform_index < len(transforms):
        raise ValueError("transform_index is outside the canonical list")
    code_basis = H.right_kernel_matrix()
    context = multiprocessing.get_context("fork")
    parent_group = os.getpgrp()
    with ProcessPoolExecutor(
        max_workers=1,
        mp_context=context,
        initializer=_initialize_worker,
        initargs=(
            H,
            support,
            code_basis,
            m,
            r,
            {},
            reconstruction,
            key_equation_seed,
            key_equation_max_codewords,
        ),
    ) as executor:
        outcome = executor.submit(
            _evaluate_transform_worker, transforms[transform_index]
        ).result()
    if outcome.record["process_group"] != parent_group:
        raise GoppaRecoveryError("Goppa worker escaped the parent process group")
    return {
        "scheduler_version": GOPPA_SCHEDULER_VERSION,
        "process_start_method": context.get_start_method(),
        "field_modulus": str(field.modulus()),
        "transform_order_sha256": _transform_order_digest(transforms, field),
        "transform_count": len(transforms),
        "attempt": outcome.record,
    }


def _scheduler_record(
    transforms,
    field,
    workers,
    outcomes,
    cancelled,
    *,
    reconstruction,
    key_equation_seed,
    key_equation_max_codewords,
):
    records = [outcomes[index].record for index in sorted(outcomes)]
    for index in sorted(cancelled):
        transform = transforms[index]
        records.append({
            "index": index,
            "label": transform.label,
            "alpha_integer": transform.alpha_integer,
            "terminal": "cancelled_above_canonical_winner",
        })
    attempted = set(outcomes) | set(cancelled)
    for transform in transforms:
        if transform.index not in attempted:
            records.append({
                "index": transform.index,
                "label": transform.label,
                "alpha_integer": transform.alpha_integer,
                "terminal": "not_started_after_canonical_winner",
            })
    records.sort(key=lambda record: record["index"])
    winners = [
        outcome for outcome in outcomes.values()
        if outcome.terminal == "verified_recovery"
    ]
    winner = min(winners, key=lambda item: item.index) if winners else None
    return {
        "scheduler_version": GOPPA_SCHEDULER_VERSION,
        "transform_order_sha256": _transform_order_digest(transforms, field),
        "transform_order": [
            {
                "index": item.index,
                "label": item.label,
                "alpha_integer": item.alpha_integer,
            }
            for item in transforms
        ],
        "field_modulus": str(field.modulus()),
        "worker_count": workers,
        "process_start_method": "serial" if workers == 1 else "fork",
        "reconstruction_policy": reconstruction,
        "advanced_factor_version": (
            GOPPA_ADVANCED_FACTOR_VERSION
            if reconstruction == "advanced_factor" else None
        ),
        "key_equation_version": (
            GOPPA_KEY_EQUATION_VERSION
            if reconstruction == "key_equation" else None
        ),
        "key_equation_seed": (
            key_equation_seed if reconstruction == "key_equation" else None
        ),
        "key_equation_max_codewords": (
            key_equation_max_codewords
            if reconstruction == "key_equation" else None
        ),
        "attempts": records,
        "best_verified_index": None if winner is None else winner.index,
        "cancelled_indices": sorted(cancelled),
        "returned_key_sha256": (
            None if winner is None else winner.record["result_sha256"]
        ),
    }, winner


def _run_transform_scheduler(
    H,
    support,
    *,
    m,
    r,
    workers,
    worker_peak_bytes,
    memory_budget_bytes,
    reconstruction="key_equation",
    key_equation_seed=0,
    key_equation_max_codewords=32,
    diagnostic_delays=None,
):
    support, field = _validate_inputs(H, support, m, r)
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    if reconstruction not in {"advanced_factor", "key_equation", "legacy"}:
        raise ValueError(
            "reconstruction must be 'advanced_factor', 'key_equation', or "
            "'legacy'"
        )
    if (
        isinstance(key_equation_seed, bool)
        or not isinstance(key_equation_seed, int)
        or key_equation_seed < 0
    ):
        raise ValueError("key_equation_seed must be a non-negative integer")
    if (
        isinstance(key_equation_max_codewords, bool)
        or not isinstance(key_equation_max_codewords, int)
        or key_equation_max_codewords < 0
    ):
        raise ValueError(
            "key_equation_max_codewords must be a non-negative integer"
        )
    if workers > 1:
        for name, value in (
            ("worker_peak_bytes", worker_peak_bytes),
            ("memory_budget_bytes", memory_budget_bytes),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} is required and must be positive")
        if workers * worker_peak_bytes > memory_budget_bytes:
            raise MemoryError(
                "Goppa process-pool preflight refused "
                f"{workers} * {worker_peak_bytes} > {memory_budget_bytes} bytes"
            )
    delays = {} if diagnostic_delays is None else dict(diagnostic_delays)
    transforms = canonical_goppa_transforms(support)
    code_basis = H.right_kernel_matrix()
    outcomes = {}
    cancelled = set()

    if workers == 1:
        for transform in transforms:
            outcome = _evaluate_transform(
                H,
                support,
                code_basis,
                transform,
                m=m,
                r=r,
                reconstruction=reconstruction,
                key_equation_seed=key_equation_seed,
                key_equation_max_codewords=key_equation_max_codewords,
                delay_seconds=float(delays.get(transform.index, 0.0)),
            )
            outcomes[transform.index] = outcome
            if outcome.terminal == "verified_recovery":
                break
    else:
        parent_group = os.getpgrp()
        context = multiprocessing.get_context("fork")
        executor = ProcessPoolExecutor(
            max_workers=min(workers, len(transforms)),
            mp_context=context,
            initializer=_initialize_worker,
            initargs=(
                H,
                support,
                code_basis,
                m,
                r,
                delays,
                reconstruction,
                key_equation_seed,
                key_equation_max_codewords,
            ),
        )
        futures = {
            executor.submit(_evaluate_transform_worker, transform): transform.index
            for transform in transforms
        }
        best_index = None
        try:
            for future in as_completed(futures):
                index = futures[future]
                if future.cancelled():
                    cancelled.add(index)
                    continue
                outcome = future.result()
                if outcome.record["process_group"] != parent_group:
                    raise GoppaRecoveryError(
                        "Goppa worker escaped the parent process group"
                    )
                outcomes[index] = outcome
                if outcome.terminal == "verified_recovery":
                    best_index = (
                        index if best_index is None else min(best_index, index)
                    )
                    for other_future, other_index in futures.items():
                        if other_index > best_index and other_future.cancel():
                            cancelled.add(other_index)
                if best_index is not None and all(
                    lower in outcomes for lower in range(best_index)
                ):
                    for other_future, other_index in futures.items():
                        if other_index > best_index and other_future.cancel():
                            cancelled.add(other_index)
                    break
        finally:
            executor.shutdown(wait=True, cancel_futures=True)
        # Running tasks above the winner may have completed during shutdown.
        for future, index in futures.items():
            if index in outcomes or index in cancelled:
                continue
            if future.cancelled():
                cancelled.add(index)
            elif future.done():
                outcome = future.result()
                if outcome.record["process_group"] != parent_group:
                    raise GoppaRecoveryError(
                        "Goppa worker escaped the parent process group"
                    )
                outcomes[index] = outcome

    report, winner = _scheduler_record(
        transforms,
        field,
        workers,
        outcomes,
        cancelled,
        reconstruction=reconstruction,
        key_equation_seed=key_equation_seed,
        key_equation_max_codewords=key_equation_max_codewords,
    )
    return report, winner


def recover_goppa_polynomial(
    H,
    support,
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
    """Recover and verify the lowest canonical successful transform.

    The default uses the deterministic key-equation stream and falls back to
    the original systematic-basis numerator gcd whenever the bounded stream
    does not verify.  ``reconstruction="legacy"`` selects that original path
    directly.  ``reconstruction="advanced_factor"`` selects the
    Kirshanova--May minimal-support factorization and refuses ambiguous
    degree-``r`` factors.  Concurrent execution requires an explicit measured
    per-worker peak and aggregate memory budget; completion order can never
    change the returned transform.
    """
    report, winner = _run_transform_scheduler(
        H,
        support,
        m=m,
        r=r,
        workers=workers,
        worker_peak_bytes=worker_peak_bytes,
        memory_budget_bytes=memory_budget_bytes,
        reconstruction=reconstruction,
        key_equation_seed=key_equation_seed,
        key_equation_max_codewords=key_equation_max_codewords,
    )
    if winner is None:
        raise GoppaRecoveryError(
            "no support transform yielded a verified degree-r polynomial"
        )
    return GoppaRecoveryResult(
        winner.support,
        winner.polynomial,
        winner.label,
        winner.gcd_degree,
        report,
    )
