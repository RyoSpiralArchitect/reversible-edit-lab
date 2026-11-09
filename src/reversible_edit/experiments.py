"""Seeded numerical experiments; reference weights are evaluator-only.

All weights are linear associative memories, not pretrained language models.
The targets and keys are frozen for each write; see METHODS.md for limitations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter
import numpy as np

from .core import (
    bottom_rank_restore, memit_edit, quantize, restore_delta, restore_memit,
    restore_rome, restore_rome_old_value, rome_edit, storage_nbytes,
)
from .metrics import recovery_metrics, relative_frobenius


@dataclass(frozen=True)
class ExperimentConfig:
    seeds: tuple[int, ...] = (7, 23, 61)
    precisions: tuple[str, ...] = ("float64", "float32", "float16", "bfloat16")
    input_dim: int = 24
    output_dim: int = 12
    probe_count: int = 192
    cycle_count: int = 128
    active_rounds: int = 6

    def __post_init__(self):
        if not self.seeds or any(not isinstance(s, int) or s < 0 for s in self.seeds):
            raise ValueError("seeds must be a nonempty sequence of nonnegative integers")
        if not self.precisions or any(p not in ("float64", "float32", "float16", "bfloat16") for p in self.precisions):
            raise ValueError("precisions must be a nonempty sequence of supported names")
        if self.input_dim < 2 or self.output_dim < 2:
            raise ValueError("input_dim and output_dim must be at least 2")
        if any(v < 1 for v in (self.probe_count, self.cycle_count, self.active_rounds)):
            raise ValueError("probe_count, cycle_count, and active_rounds must be positive")


def _fixture(seed: int, config: ExperimentConfig):
    rng = np.random.default_rng(seed)
    n, m = config.input_dim, config.output_dim
    samples = rng.normal(size=(n, 8*n))
    c = samples @ samples.T / samples.shape[1] + 0.1 * np.eye(n)
    w = rng.normal(scale=1/np.sqrt(n), size=(m, n))
    probes = rng.normal(size=(n, config.probe_count))
    probes /= np.linalg.norm(probes, axis=0, keepdims=True)
    return rng, w, c, probes


def _unit(rng: np.random.Generator, dimension: int) -> np.ndarray:
    k = rng.normal(size=dimension)
    return k / np.linalg.norm(k)


def _rome_spec_bytes(spec) -> int:
    return int(spec.key.nbytes + spec.value.nbytes + spec.direction.nbytes + 8)


def _memit_spec_bytes(spec) -> int:
    return int(spec.keys.nbytes + spec.values.nbytes + spec.covariance.nbytes)


def _next_update_metrics(recovered, reference, next_key, next_value, covariance,
                         precision: str) -> dict:
    # Both receive the same future request. No original model is used by undo.
    expected = rome_edit(reference, next_key, next_value, covariance,
                         alpha=0.6, precision=precision).updated
    actual = rome_edit(recovered, next_key, next_value, covariance,
                       alpha=0.6, precision=precision).updated
    return {
        "next_update_weight_relative_error": relative_frobenius(actual, expected),
        "next_update_exact_equal": bool(np.array_equal(actual, expected)),
    }


def run_kernel_sweep(config: ExperimentConfig) -> list[dict]:
    """Apply one write, restore with declared access, then issue a future write."""
    rows = []
    for seed in config.seeds:
        rng, unrounded, covariance, probes = _fixture(seed, config)
        key = _unit(rng, config.input_dim)
        target_offset = rng.normal(scale=0.7, size=config.output_dim)
        next_key = _unit(rng, config.input_dim)
        next_offset = rng.normal(scale=0.4, size=config.output_dim)
        keys = np.column_stack([_unit(rng, config.input_dim) for _ in range(4)])
        batch_offset = rng.normal(scale=0.7, size=(config.output_dim, 4))
        for precision in config.precisions:
            reference = quantize(unrounded, precision)
            value = np.asarray(reference, dtype=np.float64) @ key + target_offset
            next_value = np.asarray(reference, dtype=np.float64) @ next_key + next_offset
            checkpoint_bytes = storage_nbytes(reference, precision)
            for alpha in (0.0, 0.25, 0.75, 0.95, 0.99, 0.999, 1.0):
                result = rome_edit(reference, key, value, covariance,
                                   alpha=alpha, precision=precision)
                before = np.linalg.norm(np.asarray(reference, dtype=np.float64) @ key - value)
                after = np.linalg.norm(np.asarray(result.updated, dtype=np.float64) @ key - value)
                old_response = np.asarray(reference, dtype=np.float64) @ key
                # Only the checkpoint baseline closes over the full reference.
                restorers = {
                    "checkpoint": (lambda: reference.copy(), checkpoint_bytes, "full_layer_snapshot"),
                    "delta_log": (lambda: restore_delta(result.updated, result.delta, precision=precision),
                                  result.delta.nbytes, "low_rank_factors"),
                    "old_response_log": (lambda: restore_rome_old_value(result.updated, result.spec,
                                                                      old_response, precision=precision),
                                         _rome_spec_bytes(result.spec)+old_response.nbytes,
                                         "fixed_write_metadata_and_old_response"),
                    "svd_top1": (lambda: bottom_rank_restore(result.updated, 1, precision=precision),
                                 0, "edited_matrix_only_fixed_drop_rank"),
                    "unchanged": (lambda: result.updated.copy(), 0, "no_undo_control"),
                }
                if alpha < 1.0:
                    restorers["analytic"] = (
                        lambda: restore_rome(result.updated, result.spec, precision=precision),
                        _rome_spec_bytes(result.spec), "fixed_write_metadata_no_old_response")
                for method, (restore, auxiliary_bytes, access) in restorers.items():
                    start = perf_counter()
                    recovered = restore()
                    seconds = perf_counter() - start
                    row = {
                        "experiment": "single_write", "editor": "rome", "seed": seed,
                        "precision": precision, "alpha": alpha, "covariance_scale": None,
                        "method": method, "access": access, "status": "ok",
                        "auxiliary_numeric_bytes": int(auxiliary_bytes),
                        "checkpoint_numeric_bytes": checkpoint_bytes,
                        "write_residual_fraction": float(after/before),
                        "write_fraction": float(1-after/before),
                        "inverse_gain_in_key_direction": None if alpha == 1 else 1/(1-alpha),
                        "restore_seconds": seconds,
                    }
                    row.update(recovery_metrics(recovered, reference, probes))
                    row.update(_next_update_metrics(recovered, reference, next_key,
                                                    next_value, covariance, precision))
                    rows.append(row)
            values = np.asarray(reference, dtype=np.float64) @ keys + batch_offset
            for scale in (0.01, 0.1, 1.0, 10.0):
                c = scale*covariance
                result = memit_edit(reference, keys, values, c, precision=precision)
                before = np.linalg.norm(np.asarray(reference, dtype=np.float64) @ keys-values)
                after = np.linalg.norm(np.asarray(result.updated, dtype=np.float64) @ keys-values)
                restorers = {
                    "checkpoint": (lambda: reference.copy(), checkpoint_bytes, "full_layer_snapshot"),
                    "delta_log": (lambda: restore_delta(result.updated, result.delta, precision=precision),
                                  result.delta.nbytes, "low_rank_factors"),
                    "analytic": (lambda: restore_memit(result.updated, result.spec, precision=precision),
                                 _memit_spec_bytes(result.spec), "fixed_keys_targets_covariance"),
                    "svd_top1": (lambda: bottom_rank_restore(result.updated, 1, precision=precision),
                                 0, "edited_matrix_only_fixed_drop_rank"),
                    "unchanged": (lambda: result.updated.copy(), 0, "no_undo_control"),
                }
                for method, (restore, auxiliary_bytes, access) in restorers.items():
                    start = perf_counter()
                    recovered = restore()
                    seconds = perf_counter()-start
                    row = {
                        "experiment": "single_write", "editor": "memit", "seed": seed,
                        "precision": precision, "alpha": None, "covariance_scale": scale,
                        "method": method, "access": access, "status": "ok",
                        "auxiliary_numeric_bytes": int(auxiliary_bytes),
                        "checkpoint_numeric_bytes": checkpoint_bytes,
                        "write_residual_fraction": float(after/before),
                        "write_fraction": float(1-after/before),
                        "restore_seconds": seconds,
                    }
                    row.update(recovery_metrics(recovered, reference, probes))
                    row.update(_next_update_metrics(recovered, reference, next_key,
                                                    next_value, covariance, precision))
                    rows.append(row)
    return rows


def run_cycle_sweep(config: ExperimentConfig) -> list[dict]:
    """Many edit/undo cycles: target requests are fixed across all methods.

Each journal is made from the immediately preceding live state. Drift relative
to the initial state is measured only by the evaluator. This is not LIFO undo
of one long committed history; it is repeated provisional experimentation.
"""
    rows = []
    for seed in config.seeds:
        rng, unrounded, covariance, probes = _fixture(seed+1000, config)
        keys = [_unit(rng, config.input_dim) for _ in range(config.cycle_count)]
        offsets = rng.normal(scale=0.7, size=(config.cycle_count, config.output_dim))
        next_key = _unit(rng, config.input_dim)
        next_value = unrounded @ next_key + rng.normal(scale=0.3, size=config.output_dim)
        checkpoints = sorted({1, min(8, config.cycle_count), min(32, config.cycle_count),
                              config.cycle_count})
        for precision in config.precisions:
            reference = quantize(unrounded, precision)
            targets = [np.asarray(reference, dtype=np.float64) @ k + offset
                       for k, offset in zip(keys, offsets)]
            for method in ("checkpoint", "delta_log", "old_response_log", "analytic", "svd_top1", "no_rollback"):
                current = reference.copy()
                for index, (key, target) in enumerate(zip(keys, targets), start=1):
                    old_response = np.asarray(current, dtype=np.float64) @ key if method == "old_response_log" else None
                    snapshot = current.copy() if method == "checkpoint" else None
                    result = rome_edit(current, key, target, covariance,
                                       alpha=0.95, precision=precision)
                    if method == "checkpoint":
                        current = snapshot
                    elif method == "delta_log":
                        current = restore_delta(result.updated, result.delta, precision=precision)
                    elif method == "old_response_log":
                        current = restore_rome_old_value(result.updated, result.spec, old_response,
                                                        precision=precision)
                    elif method == "analytic":
                        current = restore_rome(result.updated, result.spec, precision=precision)
                    elif method == "svd_top1":
                        current = bottom_rank_restore(result.updated, 1, precision=precision)
                    else:
                        current = result.updated
                    if index in checkpoints:
                        row = {"experiment": "cycles", "editor": "rome", "seed": seed,
                               "precision": precision, "alpha": 0.95, "cycle": index,
                               "method": method, "status": "ok"}
                        row.update(recovery_metrics(current, reference, probes))
                        row.update(_next_update_metrics(current, reference, next_key,
                                                        next_value, covariance, precision))
                        rows.append(row)
    return rows


def run_selective_undo(seed: int = 7) -> dict:
    """Counterexample: subtracting A's delta does not rebase B onto pre-A W."""
    w = np.zeros((1, 1))
    k, covariance = np.ones(1), np.ones((1, 1))
    a = rome_edit(w, k, np.array([1.0]), covariance)
    b = rome_edit(a.updated, k, np.array([2.0]), covariance)
    naive = restore_delta(b.updated, a.delta)
    without_b = restore_delta(b.updated, b.delta)
    without_ba = restore_delta(without_b, a.delta)
    replay_b = rome_edit(without_ba, k, np.array([2.0]), covariance).updated
    counterfactual_b = rome_edit(w, k, np.array([2.0]), covariance).updated
    config = ExperimentConfig(input_dim=8, output_dim=4, probe_count=64)
    rng, w0, c, probes = _fixture(seed+2000, config)
    ka = _unit(rng, 8)
    kb = ka + 0.25*_unit(rng, 8)
    kb /= np.linalg.norm(kb)
    va, vb = w0 @ ka + rng.normal(size=4), w0 @ kb + rng.normal(size=4)
    ra = rome_edit(w0, ka, va, c)
    rb = rome_edit(ra.updated, kb, vb, c)
    expected = rome_edit(w0, kb, vb, c).updated
    naive_matrix = restore_delta(rb.updated, ra.delta)
    reverted = restore_delta(restore_delta(rb.updated, rb.delta), ra.delta)
    replayed = rome_edit(reverted, kb, vb, c).updated
    return {
        "description": "Undo B then A, replay B using its fixed request; compare naive subtract A.",
        "scalar": {"initial": float(w[0, 0]), "after_a_b": float(b.updated[0, 0]),
                   "subtract_a_only": float(naive[0, 0]),
                   "undo_b_a_then_replay_b": float(replay_b[0, 0]),
                   "b_without_a": float(counterfactual_b[0, 0])},
        "matrix": {"naive": recovery_metrics(naive_matrix, expected, probes),
                   "replayed": recovery_metrics(replayed, expected, probes)},
        "scope": "Fixed requests only. Real editors may require recomputing target optimization and keys.",
    }


def run_all(config: ExperimentConfig, include_active: bool = True) -> dict:
    start = perf_counter()
    kernel = run_kernel_sweep(config)
    cycles = run_cycle_sweep(config)
    active = []
    if include_active:
        from .active import run_active_experiment
        for seed in config.seeds:
            for precision in config.precisions:
                active.append(run_active_experiment(seed=seed, precision=precision,
                                                     n_rounds=config.active_rounds))
    return {
        "schema_version": 1, "package_version": "0.1.0", "config": asdict(config),
        "scope": "Synthetic fixed-target linear write kernels and a toy acquisition loop, not pretrained LLM editing.",
        "arithmetic": "float64 work and journal arrays; each live weight write rounded to the storage precision; bf16 emulated.",
        "access_policy": "Restoration functions receive only declared metadata. Forward generators make targets/journals from pre-edit states; full originals are used for evaluator scores and the explicit checkpoint baseline.",
        "storage_policy": "Numeric payload bytes only; metadata includes all arrays required by that method. Python overhead excluded.",
        "svd_policy": "Drop top 1 singular component, preselected without reference-based hyperparameter tuning.",
        "kernel_rows": kernel, "cycle_rows": cycles, "active_runs": active,
        "selective_undo": run_selective_undo(config.seeds[0]),
        "runtime_seconds": perf_counter()-start,
    }
