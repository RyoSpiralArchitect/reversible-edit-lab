"""Synthetic active acquisition with edit/score/undo look-ahead.

This is a linear categorical classifier, not a pretrained language model.  The
acquisition proxy is expected entropy reduction on *unlabelled* probe inputs.
Hypothetical class probabilities come from the current model.  A separate oracle
is queried only after selecting an input.  In particular, becoming more confident
is not assumed to mean becoming more correct.

An evaluator retains a reference state to measure drift.  Only the explicitly
named ``checkpoint`` protocol restores that full reference. Corrected protocols
use the pre-edit state while making an explicit write-time roundoff record;
their restoration stage receives only a predicted state and that record.
The ``no_rollback`` protocol is a cumulative-contamination control.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from typing import Any, Iterable

import numpy as np

from .core import bottom_rank_restore, quantize, restore_delta, restore_rome, rome_edit, storage_nbytes


PROTOCOLS = ("checkpoint", "factor", "analytic", "bottom_rank", "no_rollback")
SUPPORTED_PROTOCOLS = PROTOCOLS + ("factor_exact", "analytic_exact", "factor_sparse")


def _state_digest(weights: np.ndarray) -> str:
    """Hash dtype, shape, and stored bytes, including signed zero."""
    contiguous = np.ascontiguousarray(weights)
    header = json.dumps({"shape": list(contiguous.shape), "dtype": contiguous.dtype.str},
                        sort_keys=True).encode("ascii")
    return hashlib.sha256(header + contiguous.tobytes()).hexdigest()


def _base_numeric_bytes(edited, protocol: str, precision: str) -> int:
    if protocol == "checkpoint":
        return storage_nbytes(edited.updated, precision)
    if protocol in ("factor", "factor_exact", "factor_sparse"):
        return edited.delta.nbytes
    if protocol in ("analytic", "analytic_exact"):
        spec = edited.spec
        return int(spec.key.nbytes + spec.value.nbytes + spec.direction.nbytes + 8)
    return 0


def _probabilities(weights: np.ndarray, keys: np.ndarray) -> np.ndarray:
    logits = np.asarray(keys, dtype=np.float64) @ np.asarray(weights, dtype=np.float64).T
    logits -= np.max(logits, axis=1, keepdims=True)
    probabilities = np.exp(logits)
    return probabilities / np.sum(probabilities, axis=1, keepdims=True)


def _entropy(probabilities: np.ndarray) -> float:
    return float(-np.mean(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)), axis=1)))


def _relative_error(actual: np.ndarray, reference: np.ndarray) -> float:
    numerator = np.linalg.norm(np.asarray(actual, dtype=np.float64) - np.asarray(reference, dtype=np.float64))
    denominator = max(float(np.linalg.norm(reference)), np.finfo(np.float64).tiny)
    return float(numerator / denominator)


def _class_value(label: int, n_classes: int, target_logit: float) -> np.ndarray:
    value = np.full(n_classes, -target_logit / (n_classes - 1), dtype=np.float64)
    value[label] = target_logit
    return value


def _rank(scores: np.ndarray) -> list[int]:
    # Explicit index tie-breaking keeps the reference independent of iteration order.
    return sorted(range(len(scores)), key=lambda index: (-float(scores[index]), index))


@dataclass
class AcquisitionResult:
    """An audit result; ``final_weights`` is the sole operational model state."""

    scores: np.ndarray
    ranking: list[int]
    final_weights: np.ndarray
    diagnostics: dict[str, Any]

    @property
    def selected_index(self) -> int:
        return self.ranking[0]

    def as_record(self) -> dict[str, Any]:
        """JSON-ready report that deliberately excludes weight arrays."""
        return {
            "scores": [float(value) for value in self.scores],
            "ranking": [int(index) for index in self.ranking],
            "selected_index": int(self.selected_index),
            **self.diagnostics,
        }


def evaluate_acquisition_candidates(
    weights: np.ndarray,
    candidates: np.ndarray,
    probes: np.ndarray,
    covariance: np.ndarray,
    *,
    protocol: str = "checkpoint",
    precision: str = "float32",
    alpha: float = 0.5,
    target_logit: float = 2.0,
    order: str = "forward",
    patch_budget_fraction: float = 0.25,
) -> AcquisitionResult:
    """Score hypothetical labels without labels or any access to a teacher.

    All hypotheses use priors and reference probe entropy from the same starting
    state.  One mutable working state is edited, scored, and restored repeatedly.
    Inexact restoration therefore contaminates later hypotheses; this is measured
    rather than silently repaired.  The returned state is used for the actual
    selected update by ``run_active_experiment``.

    Stored update factors are allowed for ``factor``.  The internal write spec is
    allowed for ``analytic``.  ``bottom_rank`` always removes exactly one leading
    singular component, without choosing that number using original weights.
    ``factor_exact`` and ``analytic_exact`` additionally record compressed
    roundoff bits at write time. ``factor_sparse`` records a limited correction
    payload; its metadata overhead is counted separately from that budget.
    """
    if protocol not in SUPPORTED_PROTOCOLS:
        raise ValueError(f"Unknown protocol: {protocol}")
    if order not in ("forward", "reverse"):
        raise ValueError("order must be 'forward' or 'reverse'")
    if not 0.0 < alpha < 1.0:
        raise ValueError("The acquisition experiment requires 0 < alpha < 1")
    if not np.isfinite(target_logit) or target_logit <= 0:
        raise ValueError("target_logit must be finite and positive")
    if not np.isfinite(patch_budget_fraction) or patch_budget_fraction < 0:
        raise ValueError("patch_budget_fraction must be finite and nonnegative")

    reference = np.array(quantize(np.asarray(weights), precision), copy=True)
    keys = np.asarray(candidates, dtype=np.float64)
    probe_keys = np.asarray(probes, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    if reference.ndim != 2 or reference.shape[0] < 2:
        raise ValueError("weights must have at least two classifier rows")
    n_classes, n_features = reference.shape
    if keys.ndim != 2 or keys.shape[1] != n_features or len(keys) == 0:
        raise ValueError("candidates must be a nonempty [candidate, feature] array")
    if probe_keys.ndim != 2 or probe_keys.shape[1] != n_features or len(probe_keys) == 0:
        raise ValueError("probes must be a nonempty [probe, feature] array")
    if cov.shape != (n_features, n_features):
        raise ValueError("covariance shape does not match the feature dimension")
    if not all(np.all(np.isfinite(array)) for array in (reference, keys, probe_keys, cov)):
        raise ValueError("All experiment inputs must be finite")

    working = np.array(reference, copy=True)
    priors = _probabilities(reference, keys)
    reference_probabilities = _probabilities(reference, probe_keys)
    reference_entropy = _entropy(reference_probabilities)
    scores = np.empty(len(keys), dtype=np.float64)
    candidate_order: Iterable[int] = range(len(keys))
    if order == "reverse":
        candidate_order = reversed(range(len(keys)))
    maximum_single_restore_error = 0.0
    trial_count = 0
    initial_digest = _state_digest(reference)
    all_trials_bitwise_restored = True
    base_numeric_bytes_total = 0
    correction_payload_bytes_total = 0
    correction_packet_bytes_total = 0
    max_correction_payload_bytes = 0
    max_correction_packet_bytes = 0
    accounted_record_bytes_total = 0
    max_accounted_record_bytes = 0
    corrected_trials = 0
    correction_patched_elements_total = 0
    logical_weight_bytes = storage_nbytes(reference, precision)
    correction_payload_budget = int(np.floor(patch_budget_fraction * logical_weight_bytes))
    for candidate_index in candidate_order:
        expected_entropy = 0.0
        for label in range(n_classes):
            # This copy belongs to the evaluator, to the checkpoint baseline,
            # and (only for corrected protocols) to write-time record creation.
            # Correction application itself is never passed this full state.
            before_trial = np.array(working, copy=True)
            edited = rome_edit(
                working,
                keys[candidate_index],
                _class_value(label, n_classes, target_logit),
                cov,
                alpha=alpha,
                precision=precision,
            )
            expected_entropy += float(priors[candidate_index, label]) * _entropy(
                _probabilities(edited.updated, probe_keys)
            )
            base_bytes = _base_numeric_bytes(edited, protocol, precision)
            patch_payload_bytes = 0
            patch_packet_bytes = 0
            if protocol == "checkpoint":
                working = np.array(before_trial, copy=True)
            elif protocol == "factor":
                working = restore_delta(edited.updated, edited.delta, precision=precision)
            elif protocol == "analytic":
                working = restore_rome(edited.updated, edited.spec, precision=precision)
            elif protocol in ("factor_exact", "analytic_exact", "factor_sparse"):
                # A compressed old-state correction is part of the declared
                # receipt. This is not weights-only or information-free undo.
                from .roundoff import apply_roundoff_patch, make_roundoff_patch
                if protocol == "analytic_exact":
                    predicted = restore_rome(edited.updated, edited.spec, precision=precision)
                else:
                    predicted = restore_delta(edited.updated, edited.delta, precision=precision)
                correction = make_roundoff_patch(
                    before_trial, predicted, precision,
                    strategy="sparse" if protocol == "factor_sparse" else "xor_zlib",
                    budget_bytes=correction_payload_budget if protocol == "factor_sparse" else None,
                )
                # This call has no original matrix or evaluator correction.
                working = apply_roundoff_patch(predicted, correction)
                patch_payload_bytes = correction.payload_nbytes
                patch_packet_bytes = correction.nbytes
                correction_patched_elements_total += correction.patched_count
                corrected_trials += 1
            elif protocol == "bottom_rank":
                working = bottom_rank_restore(edited.updated, drop_rank=1, precision=precision)
            else:
                working = np.array(edited.updated, copy=True)
            maximum_single_restore_error = max(
                maximum_single_restore_error, _relative_error(working, before_trial)
            )
            all_trials_bitwise_restored = all_trials_bitwise_restored and (
                _state_digest(working) == _state_digest(before_trial)
            )
            base_numeric_bytes_total += base_bytes
            correction_payload_bytes_total += patch_payload_bytes
            correction_packet_bytes_total += patch_packet_bytes
            max_correction_payload_bytes = max(max_correction_payload_bytes, patch_payload_bytes)
            max_correction_packet_bytes = max(max_correction_packet_bytes, patch_packet_bytes)
            accounted_bytes = base_bytes + patch_packet_bytes
            accounted_record_bytes_total += accounted_bytes
            max_accounted_record_bytes = max(max_accounted_record_bytes, accounted_bytes)
            trial_count += 1
        scores[candidate_index] = reference_entropy - expected_entropy

    final_probabilities = _probabilities(working, probe_keys)
    probe_kl = float(
        np.mean(
            np.sum(
                reference_probabilities
                * (
                    np.log(np.maximum(reference_probabilities, 1e-300))
                    - np.log(np.maximum(final_probabilities, 1e-300))
                ),
                axis=1,
            )
        )
    )
    if not np.all(np.isfinite(scores)):
        raise FloatingPointError("Nonfinite acquisition score")
    return AcquisitionResult(
        scores=scores,
        ranking=_rank(scores),
        final_weights=np.array(working, copy=True),
        diagnostics={
            "protocol": protocol,
            "candidate_order": order,
            "n_candidates": int(len(keys)),
            "n_classes": int(n_classes),
            "hypothetical_edits": int(trial_count),
            "restore_calls": int(trial_count if protocol != "no_rollback" else 0),
            "trial_probe_forward_passes": int(trial_count),
            "trial_probe_examples": int(trial_count * len(probe_keys)),
            "teacher_queries_for_acquisition": 0,
            "heldout_labels_used_for_acquisition": False,
            "final_relative_weight_drift": _relative_error(working, reference),
            "max_single_restore_relative_error": float(maximum_single_restore_error),
            "all_trials_bitwise_restored": bool(all_trials_bitwise_restored),
            "initial_state_sha256": initial_digest,
            "final_state_sha256": _state_digest(working),
            "final_state_bitwise_equal_to_start": bool(_state_digest(working) == initial_digest),
            "logical_weight_storage_bytes": int(logical_weight_bytes),
            "base_numeric_bytes_total": int(base_numeric_bytes_total),
            "correction_payload_bytes_total": int(correction_payload_bytes_total),
            "correction_packet_bytes_total": int(correction_packet_bytes_total),
            "max_correction_payload_bytes": int(max_correction_payload_bytes),
            "max_correction_packet_bytes": int(max_correction_packet_bytes),
            "accounted_record_bytes_total": int(accounted_record_bytes_total),
            "mean_accounted_record_bytes_per_trial": float(accounted_record_bytes_total / trial_count),
            "max_accounted_record_bytes_per_trial": int(max_accounted_record_bytes),
            "corrected_trials": int(corrected_trials),
            "correction_patched_elements_total": int(correction_patched_elements_total),
            "correction_payload_budget_bytes_per_trial": correction_payload_budget if protocol == "factor_sparse" else None,
            "record_accounting": "base numeric arrays/scalars plus serialized correction packet; excludes base container overhead and evaluator copies; totals are generated over trials, not simultaneously retained",
            "final_probe_kl_from_start": max(0.0, probe_kl),
            "final_max_probe_probability_drift": float(np.max(np.abs(final_probabilities - reference_probabilities))),
            "reference_probe_entropy": float(reference_entropy),
            "fixed_bottom_rank_drop": 1 if protocol == "bottom_rank" else None,
        },
    )


class _ExternalLabelOracle:
    """Synthetic environment; no model or acquisition state is available here."""

    def __init__(self, teacher_weights: np.ndarray):
        self._weights = np.array(teacher_weights, copy=True)
        self.query_count = 0

    def query(self, selected_key: np.ndarray) -> int:
        self.query_count += 1
        return int(np.argmax(self._weights @ np.asarray(selected_key, dtype=np.float64)))


def _normalised_keys(rng: np.random.Generator, n_keys: int, n_features: int) -> np.ndarray:
    keys = rng.normal(size=(n_keys, n_features))
    return keys / np.maximum(np.linalg.norm(keys, axis=1, keepdims=True), 1e-12)


def _accuracy(weights: np.ndarray, keys: np.ndarray, labels: np.ndarray) -> float:
    predictions = np.argmax(_probabilities(weights, keys), axis=1)
    return float(np.mean(predictions == labels))


def run_active_experiment(
    seed: int,
    precision: str = "float32",
    n_rounds: int = 6,
    *,
    n_candidates: int = 8,
    n_features: int = 8,
    n_classes: int = 3,
    n_probes: int = 24,
    n_test: int = 128,
    alpha: float = 0.5,
    target_logit: float = 2.0,
    protocols: Iterable[str] = PROTOCOLS,
    patch_budget_fraction: float = 0.25,
) -> dict[str, Any]:
    """Run shared-start audits and actual acquisition paths; return JSON data.

    Every path gets the same initial model, candidate pool, probes, and held-out
    environment.  It gets exactly one external class label per selected query.
    The evaluator alone uses held-out labels and references to score performance.
    Candidate availability follows a fixed pool order and selected points are
    removed.  Reversing *evaluation* order does not change candidate availability.
    """
    if any(value < 1 for value in (n_rounds, n_candidates, n_probes, n_test)):
        raise ValueError("Round and sample counts must be positive")
    if n_features < 2 or n_classes < 2:
        raise ValueError("At least two features and two classes are required")
    chosen_protocols = tuple(protocols)
    if not chosen_protocols or len(set(chosen_protocols)) != len(chosen_protocols):
        raise ValueError("protocols must be nonempty and unique")
    if any(protocol not in SUPPORTED_PROTOCOLS for protocol in chosen_protocols):
        raise ValueError("Unsupported protocol")
    rng = np.random.default_rng(seed)
    teacher_weights = rng.normal(size=(n_classes, n_features))
    initial_weights = 0.55 * teacher_weights + 0.65 * rng.normal(size=(n_classes, n_features))
    initial_weights -= np.mean(initial_weights, axis=0, keepdims=True)
    initial_weights = np.array(quantize(initial_weights, precision), copy=True)
    candidate_pool = _normalised_keys(rng, n_candidates + n_rounds, n_features)
    probes = _normalised_keys(rng, n_probes, n_features)
    test_keys = _normalised_keys(rng, n_test, n_features)
    # Held-out teacher labels are evaluator-only, never arguments to acquisition.
    test_labels = np.argmax(test_keys @ teacher_weights.T, axis=1)
    covariance = probes.T @ probes / len(probes) + 0.1 * np.eye(n_features)
    initial_accuracy = _accuracy(initial_weights, test_keys, test_labels)

    shared_start_audits: list[dict[str, Any]] = []
    common_candidates = candidate_pool[:n_candidates]
    checkpoint = evaluate_acquisition_candidates(
        initial_weights, common_candidates, probes, covariance,
        protocol="checkpoint", precision=precision, alpha=alpha,
        target_logit=target_logit, order="forward",
        patch_budget_fraction=patch_budget_fraction,
    )
    for protocol in chosen_protocols:
        for order in ("forward", "reverse"):
            audit = evaluate_acquisition_candidates(
                initial_weights, common_candidates, probes, covariance,
                protocol=protocol, precision=precision, alpha=alpha,
                target_logit=target_logit, order=order,
                patch_budget_fraction=patch_budget_fraction,
            )
            record = audit.as_record()
            record.update({
                "selected_agrees_with_checkpoint": bool(audit.selected_index == checkpoint.selected_index),
                "ranking_position_agreement_with_checkpoint": float(np.mean(np.asarray(audit.ranking) == np.asarray(checkpoint.ranking))),
                "max_score_error_from_checkpoint": float(np.max(np.abs(audit.scores - checkpoint.scores))),
                "scores_exactly_equal_to_checkpoint": bool(np.array_equal(audit.scores, checkpoint.scores)),
            })
            shared_start_audits.append(record)

    active_paths: list[dict[str, Any]] = []
    for protocol in chosen_protocols:
        for order in ("forward", "reverse"):
            working = np.array(initial_weights, copy=True)
            oracle = _ExternalLabelOracle(teacher_weights)
            remaining = list(range(len(candidate_pool)))
            history: list[dict[str, Any]] = []
            total_trials = 0
            total_restores = 0
            total_base_numeric_bytes = 0
            total_correction_payload_bytes = 0
            total_correction_packet_bytes = 0
            total_accounted_record_bytes = 0
            max_accounted_record_bytes = 0
            all_trial_states_bitwise_restored = True
            for round_index in range(n_rounds):
                pool_ids = remaining[:n_candidates]
                candidate_keys = candidate_pool[pool_ids]
                audit = evaluate_acquisition_candidates(
                    working, candidate_keys, probes, covariance,
                    protocol=protocol, precision=precision, alpha=alpha,
                    target_logit=target_logit, order=order,
                    patch_budget_fraction=patch_budget_fraction,
                )
                selected_pool_id = pool_ids[audit.selected_index]
                selected_key = candidate_pool[selected_pool_id]
                # The first and only access to the real label for this round.
                real_label = oracle.query(selected_key)
                prediction_before_trials = int(np.argmax(_probabilities(working, selected_key[None, :])[0]))
                retained = rome_edit(
                    audit.final_weights, selected_key,
                    _class_value(real_label, n_classes, target_logit), covariance,
                    alpha=alpha, precision=precision,
                )
                working = np.array(retained.updated, copy=True)
                remaining.remove(selected_pool_id)
                total_trials += audit.diagnostics["hypothetical_edits"]
                total_restores += audit.diagnostics["restore_calls"]
                total_base_numeric_bytes += audit.diagnostics["base_numeric_bytes_total"]
                total_correction_payload_bytes += audit.diagnostics["correction_payload_bytes_total"]
                total_correction_packet_bytes += audit.diagnostics["correction_packet_bytes_total"]
                total_accounted_record_bytes += audit.diagnostics["accounted_record_bytes_total"]
                max_accounted_record_bytes = max(max_accounted_record_bytes,
                                                audit.diagnostics["max_accounted_record_bytes_per_trial"])
                all_trial_states_bitwise_restored = (
                    all_trial_states_bitwise_restored and audit.diagnostics["all_trials_bitwise_restored"])
                history.append({
                    "round": int(round_index + 1),
                    "selected_pool_id": int(selected_pool_id),
                    "acquired_label": int(real_label),
                    "prediction_before_trials": int(prediction_before_trials),
                    "selected_was_correct_before_trials": bool(prediction_before_trials == real_label),
                    "selected_score": float(audit.scores[audit.selected_index]),
                    "trial_state_relative_drift": float(audit.diagnostics["final_relative_weight_drift"]),
                    "trial_probe_kl": float(audit.diagnostics["final_probe_kl_from_start"]),
                    "trial_state_bitwise_restored": bool(audit.diagnostics["all_trials_bitwise_restored"]),
                    "trial_correction_packet_bytes": int(audit.diagnostics["correction_packet_bytes_total"]),
                    "trial_correction_payload_bytes": int(audit.diagnostics["correction_payload_bytes_total"]),
                    "trial_max_accounted_record_bytes": int(audit.diagnostics["max_accounted_record_bytes_per_trial"]),
                    "test_accuracy_after_retained_update": _accuracy(working, test_keys, test_labels),
                    "teacher_queries_so_far": int(oracle.query_count),
                })
            final_accuracy = _accuracy(working, test_keys, test_labels)
            active_paths.append({
                "protocol": protocol,
                "candidate_order": order,
                "teacher_label_queries": int(oracle.query_count),
                "teacher_queries_for_acquisition": 0,
                "hypothetical_edits": int(total_trials),
                "restore_calls": int(total_restores),
                "trial_probe_forward_passes": int(total_trials),
                "trial_probe_examples": int(total_trials * n_probes),
                "retained_real_label_edits": int(n_rounds),
                "all_trial_states_bitwise_restored": bool(all_trial_states_bitwise_restored),
                "initial_state_sha256": _state_digest(initial_weights),
                "final_state_sha256": _state_digest(working),
                "base_numeric_bytes_total": int(total_base_numeric_bytes),
                "correction_payload_bytes_total": int(total_correction_payload_bytes),
                "correction_packet_bytes_total": int(total_correction_packet_bytes),
                "accounted_record_bytes_total": int(total_accounted_record_bytes),
                "mean_accounted_record_bytes_per_trial": float(total_accounted_record_bytes / total_trials),
                "max_accounted_record_bytes_per_trial": int(max_accounted_record_bytes),
                "initial_test_accuracy": float(initial_accuracy),
                "final_test_accuracy": float(final_accuracy),
                "test_accuracy_gain": float(final_accuracy - initial_accuracy),
                "selected_pool_ids": [entry["selected_pool_id"] for entry in history],
                "history": history,
            })

    query_baselines: list[dict[str, Any]] = []
    for query_policy in ("random", "uncertainty"):
        working = np.array(initial_weights, copy=True)
        oracle = _ExternalLabelOracle(teacher_weights)
        remaining = list(range(len(candidate_pool)))
        query_rng = np.random.default_rng(int(seed) ^ 0xA57F1)
        history = []
        for round_index in range(n_rounds):
            pool_ids = remaining[:n_candidates]
            if query_policy == "random":
                selected_index = int(query_rng.integers(len(pool_ids)))
                selected_score = None
            else:
                probabilities = _probabilities(working, candidate_pool[pool_ids])
                entropies = -np.sum(
                    probabilities * np.log(np.maximum(probabilities, 1e-300)), axis=1
                )
                selected_index = _rank(entropies)[0]
                selected_score = float(entropies[selected_index])
            selected_pool_id = pool_ids[selected_index]
            selected_key = candidate_pool[selected_pool_id]
            real_label = oracle.query(selected_key)
            retained = rome_edit(
                working, selected_key,
                _class_value(real_label, n_classes, target_logit), covariance,
                alpha=alpha, precision=precision,
            )
            working = np.array(retained.updated, copy=True)
            remaining.remove(selected_pool_id)
            history.append({
                "round": int(round_index + 1),
                "selected_pool_id": int(selected_pool_id),
                "acquired_label": int(real_label),
                "selected_score": selected_score,
                "test_accuracy_after_retained_update": _accuracy(working, test_keys, test_labels),
                "teacher_queries_so_far": int(oracle.query_count),
            })
        final_accuracy = _accuracy(working, test_keys, test_labels)
        query_baselines.append({
            "query_policy": query_policy,
            "teacher_label_queries": int(oracle.query_count),
            "teacher_queries_for_acquisition": 0,
            "hypothetical_edits": 0,
            "restore_calls": 0,
            "trial_probe_forward_passes": 0,
            "query_scoring_forward_passes": int(n_rounds if query_policy == "uncertainty" else 0),
            "retained_real_label_edits": int(n_rounds),
            "final_state_sha256": _state_digest(working),
            "initial_test_accuracy": float(initial_accuracy),
            "final_test_accuracy": float(final_accuracy),
            "test_accuracy_gain": float(final_accuracy - initial_accuracy),
            "selected_pool_ids": [entry["selected_pool_id"] for entry in history],
            "history": history,
        })

    checkpoint_paths = {path["candidate_order"]: path for path in active_paths
                        if path["protocol"] == "checkpoint"}
    for path in active_paths:
        reference_path = checkpoint_paths.get(path["candidate_order"])
        path["selection_sequence_matches_checkpoint"] = (
            bool(path["selected_pool_ids"] == reference_path["selected_pool_ids"])
            if reference_path is not None else None
        )
        path["final_state_bitwise_matches_checkpoint"] = (
            bool(path["final_state_sha256"] == reference_path["final_state_sha256"])
            if reference_path is not None else None
        )

    order_sensitivity = []
    for protocol in chosen_protocols:
        forward, reverse = [record for record in shared_start_audits if record["protocol"] == protocol]
        forward_path, reverse_path = [record for record in active_paths if record["protocol"] == protocol]
        order_sensitivity.append({
            "protocol": protocol,
            "shared_start_selected_agrees_across_order": bool(forward["selected_index"] == reverse["selected_index"]),
            "shared_start_max_score_difference_across_order": float(np.max(np.abs(np.asarray(forward["scores"]) - np.asarray(reverse["scores"])))),
            "active_selection_sequence_agrees_across_order": bool(forward_path["selected_pool_ids"] == reverse_path["selected_pool_ids"]),
            "active_final_state_bitwise_agrees_across_order": bool(forward_path["final_state_sha256"] == reverse_path["final_state_sha256"]),
            "active_final_accuracy_difference_forward_minus_reverse": float(forward_path["final_test_accuracy"] - reverse_path["final_test_accuracy"]),
        })

    return {
        "schema_version": 2,
        "experiment": "synthetic_linear_active_acquisition",
        "config": {
            "seed": int(seed), "precision": precision, "n_rounds": int(n_rounds),
            "n_candidates_per_round": int(n_candidates), "n_features": int(n_features),
            "n_classes": int(n_classes), "n_probes": int(n_probes), "n_test": int(n_test),
            "alpha": float(alpha), "target_logit": float(target_logit),
            "patch_budget_fraction": float(patch_budget_fraction),
            "protocols": list(chosen_protocols),
        },
        "scope": {
            "model": "synthetic linear categorical classifier; not a pretrained LLM",
            "acquisition": "expected unlabelled-probe entropy reduction under starting-model class probabilities",
            "claim": "confidence-based acquisition proxy; no guaranteed causal or accuracy gain",
            "oracle": "synthetic teacher supplies one real label only after each selected query",
            "reference_use": "evaluator drift measurements and explicit checkpoint baseline; corrected protocols additionally access pre-edit state only when making the declared write-time patch",
            "roundoff_information": "exact/sparse correction packets contain old-state information; correction application receives only predicted state and packet",
            "roundoff_budget": "fraction of logical weight bytes bounds sparse correction payload only; serialized metadata overhead is separately counted",
            "record_accounting": "base numeric arrays/scalars plus serialized correction packet; NPZ wrapper overhead excluded; totals are generated records, not peak memory",
            "memory_measurement": "evaluator retains reference/per-trial copies; protocol payload counts are not measured process peak memory",
            "query_baseline_comparison": "random/uncertainty share label budget and retained edit rule; compute is not matched",
            "heldout_labels_used_for_acquisition": False,
            "bottom_rank_selection": "fixed drop_rank=1; not tuned against original weights or heldout labels",
            "no_rollback_role": "cumulative-contamination control",
            "precision": "initial and updated weights use requested precision; scoring uses float64 arithmetic",
            "candidate_order": "forward/reverse evaluation of the same available candidates; class order fixed",
        },
        "initial_test_accuracy": float(initial_accuracy),
        "shared_start_audits": shared_start_audits,
        "active_paths": active_paths,
        "query_baselines": query_baselines,
        "order_sensitivity": order_sensitivity,
        "total_teacher_label_queries": int(
            sum(path["teacher_label_queries"] for path in active_paths)
            + sum(path["teacher_label_queries"] for path in query_baselines)
        ),
    }
