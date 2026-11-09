"""Evaluation-only metrics. These functions may inspect a reference model.

No restoration routine calls this module; references belong to the evaluator.
"""
from __future__ import annotations

import numpy as np


def relative_frobenius(actual: np.ndarray, reference: np.ndarray) -> float:
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    if a.shape != b.shape or not np.isfinite(a).all() or not np.isfinite(b).all():
        raise ValueError("Metrics require finite arrays of identical shape")
    denominator = max(float(np.linalg.norm(b)), np.finfo(np.float64).tiny)
    return float(np.linalg.norm(a - b) / denominator)


def log_softmax(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError("Expected finite (examples, classes) logits")
    centered = x - x.max(axis=1, keepdims=True)
    return centered - np.log(np.exp(centered).sum(axis=1, keepdims=True))


def mean_kl(reference_logits: np.ndarray, actual_logits: np.ndarray) -> float:
    p_log = log_softmax(reference_logits)
    q_log = log_softmax(actual_logits)
    if p_log.shape != q_log.shape:
        raise ValueError("Logit shapes must agree")
    # Tiny negatives from rounding are not meaningful KL divergences.
    return max(0.0, float(np.mean(np.sum(np.exp(p_log) * (p_log - q_log), axis=1))))


def recovery_metrics(actual: np.ndarray, reference: np.ndarray,
                     probes: np.ndarray) -> dict[str, float | bool]:
    """Probe columns are inputs; weight rows are synthetic output classes."""
    a = np.asarray(actual, dtype=np.float64)
    b = np.asarray(reference, dtype=np.float64)
    x = np.asarray(probes, dtype=np.float64)
    pa, pb = (a @ x).T, (b @ x).T
    return {
        "weight_relative_error": relative_frobenius(a, b),
        "weight_max_absolute_error": float(np.max(np.abs(a - b))),
        "weight_exact_equal": bool(np.array_equal(a, b)),
        "probe_mean_kl": mean_kl(pb, pa),
        "probe_top1_agreement": float(np.mean(pa.argmax(axis=1) == pb.argmax(axis=1))),
        "probe_logit_rmse": float(np.sqrt(np.mean((pa - pb) ** 2))),
    }

