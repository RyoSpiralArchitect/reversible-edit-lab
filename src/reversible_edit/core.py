"""Small numerical experiments on reversible low-rank weight edits.

These functions implement fixed-key, fixed-target matrix updates.  In particular,
``restore_memit`` is an inverse of the stated single-matrix least-squares update,
not an inverse of the full, model-dependent, multilayer MEMIT procedure from an
editing prompt alone.  Every restoration routine operates without an original
weight matrix; ``restore_rome_old_value`` explicitly accepts one old response.

Kernels use float64.  ``precision`` controls rounding of returned weight writes,
not input rounding: quantize an initial matrix explicitly to model storage at a
given precision.  Recorded factors and internal targets are float64 copies.
Their low-rank delta excludes the dense rounding error of the actual write.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


PRECISIONS = ("float64", "float32", "float16", "bfloat16")
_STORAGE_BYTES = {"float64": 8, "float32": 4, "float16": 2, "bfloat16": 2}


def _precision_name(precision: str) -> str:
    if precision not in PRECISIONS:
        raise ValueError(f"precision must be one of {PRECISIONS}; got {precision!r}")
    return precision


def _array(value: Any, name: str, ndim: int | None = None) -> np.ndarray:
    raw = np.asarray(value)
    if np.iscomplexobj(raw):
        raise ValueError(f"{name} must be real-valued")
    try:
        result = np.array(raw, dtype=np.float64, copy=True)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{name} must contain real numerical values") from error
    if ndim is not None and result.ndim != ndim:
        raise ValueError(f"{name} must be {ndim}-dimensional")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} must contain only finite values")
    if ndim is not None and any(size == 0 for size in result.shape):
        raise ValueError(f"{name} must have nonempty dimensions")
    return result


def _readonly(array: np.ndarray) -> np.ndarray:
    result = array.copy()
    result.flags.writeable = False
    return result


def _finite(array: np.ndarray, name: str) -> np.ndarray:
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} overflowed or produced nonfinite values")
    return array


def _covariance(value: Any, size: int) -> tuple[np.ndarray, np.ndarray]:
    covariance = _array(value, "covariance", 2)
    if covariance.shape != (size, size):
        raise ValueError(f"covariance must have shape {(size, size)}")
    if not np.allclose(covariance, covariance.T, rtol=1e-10, atol=1e-12):
        raise ValueError("covariance must be symmetric")
    # Remove only symmetry noise admitted by the check above.
    covariance = 0.5 * covariance + 0.5 * covariance.T
    try:
        cholesky = np.linalg.cholesky(covariance)
    except np.linalg.LinAlgError as error:
        raise ValueError("covariance must be positive definite") from error
    return covariance, cholesky


def _solve_cholesky(cholesky: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    return _finite(
        np.linalg.solve(cholesky.T, np.linalg.solve(cholesky, rhs)),
        "linear solve",
    )


def _alpha(value: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("alpha must be a finite scalar in [0, 1]") from error
    if not np.isfinite(result) or not 0.0 <= result <= 1.0:
        raise ValueError("alpha must be a finite scalar in [0, 1]")
    return result


def _norm(array: np.ndarray) -> float:
    result = float(np.linalg.norm(array))
    if not np.isfinite(result):
        raise ValueError("matrix norm overflowed")
    return result


def _condition(covariance: np.ndarray) -> float | None:
    result = float(np.linalg.cond(covariance))
    return result if np.isfinite(result) else None


def quantize(array: Any, precision: str = "float64") -> np.ndarray:
    """Copy and round finite real values to the requested storage format.

    Bfloat16 is represented by a float32 array whose lower 16 bits are zero.
    Conversion rounds directly from float64 to an 8-bit significand with a
    minimum quantum of 2**-133, using round-to-nearest, ties-to-even.  A prior
    float32 rounding would incorrectly double-round some midpoint neighbors.
    NaN/infinity inputs and finite-to-infinity overflow raise ``ValueError``;
    underflow to signed zero is allowed.  This does not simulate low-precision
    matrix multiplication or accumulated low-precision arithmetic.
    """
    precision = _precision_name(precision)
    values = _array(array, "array")
    if precision == "float64":
        return values
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        if precision != "bfloat16":
            result = values.astype(np.dtype(precision))
        else:
            _, exponents = np.frexp(values)
            quantum_exponents = np.maximum(exponents - 8, -133)
            scaled = np.ldexp(values, -quantum_exponents)
            rounded64 = np.ldexp(np.rint(scaled), quantum_exponents)
            maximum = np.ldexp(np.float64(255.0), 120)
            if np.any(np.abs(rounded64) > maximum) or not np.all(np.isfinite(rounded64)):
                raise ValueError("values overflow bfloat16 storage")
            # Rounded values now belong to bfloat16 and are exactly representable
            # in float32; this final cast introduces no second rounding.
            result = np.asarray(rounded64, dtype=np.float32)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"values overflow {precision} storage")
    return np.array(result, copy=True)


def storage_nbytes(array: Any, precision: str = "float64") -> int:
    """Logical storage bytes, including 2-byte bfloat16 emulated in float32."""
    precision = _precision_name(precision)
    values = _array(array, "array")
    return int(values.size * _STORAGE_BYTES[precision])


@dataclass(frozen=True)
class LowRankDelta:
    """An intended update ``left @ right.T`` stored as read-only float64 factors."""

    left: np.ndarray
    right: np.ndarray

    def __post_init__(self) -> None:
        left = _array(self.left, "left", 2)
        right = _array(self.right, "right", 2)
        if left.shape[1] != right.shape[1]:
            raise ValueError("left and right must have the same factor rank")
        object.__setattr__(self, "left", _readonly(left))
        object.__setattr__(self, "right", _readonly(right))

    def matrix(self) -> np.ndarray:
        return _finite(self.left @ self.right.T, "delta")

    @property
    def nbytes(self) -> int:
        return int(self.left.nbytes + self.right.nbytes)


@dataclass(frozen=True)
class RomeSpec:
    """Fixed ROME key, target, normalized direction q, and write strength.

    ``direction.T @ key == 1``.  This record contains no old weight matrix or
    old response.  The fixed internal target itself can depend on an original
    model in an end-to-end editor; this numerical library does not infer it.
    """

    key: np.ndarray
    value: np.ndarray
    direction: np.ndarray
    alpha: float

    def __post_init__(self) -> None:
        key = _array(self.key, "key", 1)
        value = _array(self.value, "value", 1)
        direction = _array(self.direction, "direction", 1)
        if direction.shape != key.shape:
            raise ValueError("direction and key must have the same shape")
        if not np.any(key != 0.0):
            raise ValueError("key must be nonzero")
        product = float(direction @ key)
        if not np.isfinite(product) or not np.isclose(product, 1.0, rtol=1e-8, atol=1e-10):
            raise ValueError("direction must satisfy direction.T @ key == 1")
        object.__setattr__(self, "key", _readonly(key))
        object.__setattr__(self, "value", _readonly(value))
        object.__setattr__(self, "direction", _readonly(direction))
        object.__setattr__(self, "alpha", _alpha(self.alpha))


@dataclass(frozen=True)
class MemitSpec:
    """Fixed keys, internal targets, and SPD covariance for one matrix update.

    Keys have shape (input_dimension, edit_count), values have shape
    (output_dimension, edit_count).  Full multilayer MEMIT recomputes these
    quantities from model-dependent residuals; they cannot be replaced by edit
    text alone when using the analytical inverse below.
    """

    keys: np.ndarray
    values: np.ndarray
    covariance: np.ndarray

    def __post_init__(self) -> None:
        keys = _array(self.keys, "keys", 2)
        values = _array(self.values, "values", 2)
        if values.shape[1] != keys.shape[1]:
            raise ValueError("keys and values must have the same number of columns")
        if np.any(np.all(keys == 0.0, axis=0)):
            raise ValueError("each key column must be nonzero")
        covariance, _ = _covariance(self.covariance, keys.shape[0])
        object.__setattr__(self, "keys", _readonly(keys))
        object.__setattr__(self, "values", _readonly(values))
        object.__setattr__(self, "covariance", _readonly(covariance))


@dataclass(frozen=True)
class EditResult:
    updated: np.ndarray
    delta: LowRankDelta
    spec: RomeSpec | MemitSpec
    diagnostics: dict[str, Any]

    def __post_init__(self) -> None:
        # Preserve the requested storage dtype while protecting record contents.
        object.__setattr__(self, "updated", _readonly(self.updated))
        object.__setattr__(self, "diagnostics", dict(self.diagnostics))


def rome_edit(
    w: Any,
    key: Any,
    value: Any,
    covariance: Any,
    alpha: float = 1.0,
    precision: str = "float64",
) -> EditResult:
    """Apply W' = W + alpha (v - W k) q.T, q.T k = 1.

    Alpha=1 is the hard equality write; for fixed key/target it is noninjective.
    Alpha<1 admits ``restore_rome`` but becomes ill-conditioned near one.
    """
    _precision_name(precision)
    weights = _array(w, "w", 2)
    key = _array(key, "key", 1)
    value = _array(value, "value", 1)
    if key.shape != (weights.shape[1],) or value.shape != (weights.shape[0],):
        raise ValueError("key/value shapes must match the input/output dimensions of w")
    if not np.any(key != 0.0):
        raise ValueError("key must be nonzero")
    covariance, cholesky = _covariance(covariance, weights.shape[1])
    alpha = _alpha(alpha)
    inverse_key = _solve_cholesky(cholesky, key)
    strength = float(key @ inverse_key)
    if not np.isfinite(strength) or strength <= 0.0:
        raise ValueError("key strength must be finite and positive")
    direction = _finite(inverse_key / strength, "ROME direction")
    spec = RomeSpec(key, value, direction, alpha)
    residual = _finite(value - weights @ key, "ROME residual")
    delta = LowRankDelta((alpha * residual)[:, None], direction[:, None])
    update = delta.matrix()
    updated = quantize(_finite(weights + update, "updated weights"), precision)
    return EditResult(
        updated,
        delta,
        spec,
        {
            "algorithm": "rome_fixed_target",
            "storage_precision": precision,
            "alpha": alpha,
            "key_strength": strength,
            "covariance_condition": _condition(covariance),
            "analytically_invertible": alpha < 1.0,
            "inverse_amplification_whitened": 1.0 / (1.0 - alpha) if alpha < 1.0 else None,
            "delta_rank_bound": 1,
            "update_frobenius_norm": _norm(update),
            "target_residual_norm": _norm(updated.astype(np.float64) @ key - value),
        },
    )


def memit_edit(
    w: Any,
    keys: Any,
    values: Any,
    covariance: Any,
    precision: str = "float64",
) -> EditResult:
    """Apply the fixed-target update (V-WK) K.T (C+K K.T)^-1.

    This is the single-matrix least-squares kernel used by MEMIT, with supplied
    internal keys/targets.  It does not compute language-model target vectors.
    """
    _precision_name(precision)
    weights = _array(w, "w", 2)
    spec = MemitSpec(keys, values, covariance)
    if weights.shape != (spec.values.shape[0], spec.keys.shape[0]):
        raise ValueError("keys/values shapes must match the input/output dimensions of w")
    covariance = spec.covariance
    keys = spec.keys
    values = spec.values
    augmented = _finite(covariance + keys @ keys.T, "augmented covariance")
    try:
        augmented_cholesky = np.linalg.cholesky(augmented)
    except np.linalg.LinAlgError as error:
        raise ValueError("augmented covariance lost positive definiteness numerically") from error
    adjusted_keys = _solve_cholesky(augmented_cholesky, keys)
    residual = _finite(values - weights @ keys, "MEMIT residual")
    delta = LowRankDelta(residual, adjusted_keys)
    update = delta.matrix()
    updated = quantize(_finite(weights + update, "updated weights"), precision)
    whitened_keys = np.linalg.solve(np.linalg.cholesky(covariance), keys)
    max_energy = float(np.linalg.norm(whitened_keys, ord=2) ** 2)
    if not np.isfinite(max_energy):
        raise ValueError("whitened key energy overflowed")
    return EditResult(
        updated,
        delta,
        spec,
        {
            "algorithm": "memit_fixed_target",
            "storage_precision": precision,
            "covariance_condition": _condition(covariance),
            "max_whitened_key_energy": max_energy,
            "inverse_amplification_whitened": 1.0 + max_energy,
            "analytically_invertible": True,
            "delta_rank_bound": min(weights.shape[0], weights.shape[1], keys.shape[1]),
            "update_frobenius_norm": _norm(update),
            "target_residual_norm": _norm(updated.astype(np.float64) @ keys - values),
        },
    )


def restore_delta(
    edited: Any, delta: LowRankDelta, precision: str = "float64"
) -> np.ndarray:
    """Subtract saved intended factors; rounding residue is not in the journal."""
    weights = _array(edited, "edited", 2)
    if not isinstance(delta, LowRankDelta):
        raise ValueError("delta must be a LowRankDelta record")
    if weights.shape != (delta.left.shape[0], delta.right.shape[0]):
        raise ValueError("edited shape does not match delta factors")
    return quantize(_finite(weights - delta.matrix(), "restored weights"), precision)


def _rome_weights(edited: Any, spec: RomeSpec) -> np.ndarray:
    weights = _array(edited, "edited", 2)
    if not isinstance(spec, RomeSpec):
        raise ValueError("spec must be a RomeSpec record")
    if weights.shape != (spec.value.size, spec.key.size):
        raise ValueError("edited shape does not match ROME spec")
    return weights


def restore_rome(
    edited: Any, spec: RomeSpec, precision: str = "float64"
) -> np.ndarray:
    """Invert a fixed-target softened ROME write; reject the singular alpha=1.

    W = W' + alpha/(1-alpha) (W'k-v) q.T.  The inverse amplifies rounding
    errors as alpha approaches one and cannot guarantee bitwise restoration.
    """
    weights = _rome_weights(edited, spec)
    if spec.alpha == 1.0:
        raise ValueError("alpha=1 has no unique inverse; use delta factors or an old value")
    residual = weights @ spec.key - spec.value
    correction = (spec.alpha / (1.0 - spec.alpha)) * np.outer(residual, spec.direction)
    return quantize(_finite(weights + correction, "restored weights"), precision)


def restore_rome_old_value(
    edited: Any,
    spec: RomeSpec,
    old_value: Any,
    precision: str = "float64",
) -> np.ndarray:
    """Undo with the recorded old response m=W_old k, including hard ROME.

    This explicitly uses old-state side information of output_dimension scalars,
    not an original weight matrix.  An incorrect old response cannot be detected
    from the edited weights alone.
    """
    weights = _rome_weights(edited, spec)
    old_value = _array(old_value, "old_value", 1)
    if old_value.shape != spec.value.shape:
        raise ValueError("old_value shape must match the ROME target value")
    correction = spec.alpha * np.outer(old_value - spec.value, spec.direction)
    return quantize(_finite(weights + correction, "restored weights"), precision)


def restore_memit(
    edited: Any, spec: MemitSpec, precision: str = "float64"
) -> np.ndarray:
    """Invert one fixed-key/target MEMIT matrix update using its internal spec.

    W = W' + (W'K - V) K.T C^-1, for C positive definite.  Repeated inversion
    can amplify numerical errors; this does not reconstruct multilayer targets
    or keys from an edit prompt, nor does it invert their model-dependent search.
    """
    weights = _array(edited, "edited", 2)
    if not isinstance(spec, MemitSpec):
        raise ValueError("spec must be a MemitSpec record")
    if weights.shape != (spec.values.shape[0], spec.keys.shape[0]):
        raise ValueError("edited shape does not match MEMIT spec")
    inverse_keys = _solve_cholesky(np.linalg.cholesky(spec.covariance), spec.keys)
    correction = (weights @ spec.keys - spec.values) @ inverse_keys.T
    return quantize(_finite(weights + correction, "restored weights"), precision)


def bottom_rank_restore(
    edited: Any, drop_rank: int, precision: str = "float64"
) -> np.ndarray:
    """Heuristic: remove the largest ``drop_rank`` singular components.

    No original matrix or edit record is used.  This is not an analytical undo:
    original high-energy components are removed along with any edit signal.
    A drop rank of zero returns a quantized copy; dropping every component
    returns zeros.  Rectangular matrices are supported.
    """
    weights = _array(edited, "edited", 2)
    if isinstance(drop_rank, (bool, np.bool_)) or not isinstance(drop_rank, (int, np.integer)):
        raise ValueError("drop_rank must be an integer")
    drop_rank = int(drop_rank)
    maximum = min(weights.shape)
    if not 0 <= drop_rank <= maximum:
        raise ValueError(f"drop_rank must be between 0 and {maximum}")
    if drop_rank == 0:
        return quantize(weights, precision)
    if drop_rank == maximum:
        return quantize(np.zeros_like(weights), precision)
    left, singular_values, right_transpose = np.linalg.svd(weights, full_matrices=False)
    restored = (left[:, drop_rank:] * singular_values[drop_rank:]) @ right_transpose[drop_rank:, :]
    return quantize(_finite(restored, "bottom-rank weights"), precision)
