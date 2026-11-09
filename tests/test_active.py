"""Behavioral checks for the synthetic acquisition experiment."""

import json
import unittest
from unittest.mock import patch

import numpy as np

from reversible_edit.active import PROTOCOLS, SUPPORTED_PROTOCOLS, evaluate_acquisition_candidates, run_active_experiment
from reversible_edit.core import quantize, storage_nbytes


class ActiveExperimentTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(23)
        self.weights = rng.normal(size=(3, 6))
        self.candidates = rng.normal(size=(5, 6))
        self.candidates /= np.linalg.norm(self.candidates, axis=1, keepdims=True)
        self.probes = rng.normal(size=(12, 6))
        self.covariance = np.eye(6)

    def audit(self, protocol, order="forward"):
        return evaluate_acquisition_candidates(
            self.weights, self.candidates, self.probes, self.covariance,
            protocol=protocol, order=order, precision="float64",
        )

    def test_exact_restorers_preserve_reference_query_ranking(self):
        checkpoint = self.audit("checkpoint")
        for protocol in ("factor", "analytic"):
            for order in ("forward", "reverse"):
                with self.subTest(protocol=protocol, order=order):
                    actual = self.audit(protocol, order)
                    self.assertEqual(actual.ranking, checkpoint.ranking)
                    np.testing.assert_allclose(actual.scores, checkpoint.scores, rtol=1e-11, atol=1e-12)
                    np.testing.assert_allclose(actual.final_weights, self.weights, rtol=1e-11, atol=1e-12)

    def test_acquisition_never_queries_external_teacher(self):
        with patch("reversible_edit.active._ExternalLabelOracle.query", side_effect=AssertionError("teacher leaked")):
            for protocol in SUPPORTED_PROTOCOLS:
                with self.subTest(protocol=protocol):
                    result = self.audit(protocol)
                    self.assertEqual(result.diagnostics["teacher_queries_for_acquisition"], 0)
                    self.assertFalse(result.diagnostics["heldout_labels_used_for_acquisition"])

    def test_acquisition_does_not_mutate_caller_arrays(self):
        originals = [array.copy() for array in (self.weights, self.candidates, self.probes, self.covariance)]
        for protocol in SUPPORTED_PROTOCOLS:
            self.audit(protocol)
        for actual, expected in zip((self.weights, self.candidates, self.probes, self.covariance), originals):
            np.testing.assert_array_equal(actual, expected)

    def test_full_experiment_is_deterministic_and_json_serializable(self):
        options = dict(seed=7, precision="float64", n_rounds=2, n_candidates=4, n_probes=8, n_test=24)
        first = run_active_experiment(**options)
        second = run_active_experiment(**options)
        self.assertEqual(first, second)
        json.dumps(first, allow_nan=False)
        self.assertEqual(first["total_teacher_label_queries"], 2 * (len(PROTOCOLS) * 2 + 2))
        for path in first["active_paths"]:
            self.assertEqual(path["teacher_label_queries"], 2)
            self.assertEqual(path["teacher_queries_for_acquisition"], 0)
            self.assertEqual(len(set(path["selected_pool_ids"])), 2)
        self.assertEqual({path["query_policy"] for path in first["query_baselines"]}, {"random", "uncertainty"})
        for path in first["query_baselines"]:
            self.assertEqual(path["teacher_label_queries"], 2)
            self.assertEqual(path["hypothetical_edits"], 0)
            self.assertEqual(path["restore_calls"], 0)

    def test_contamination_controls_expose_drift_without_hidden_reset(self):
        for protocol in ("bottom_rank", "no_rollback"):
            with self.subTest(protocol=protocol):
                result = self.audit(protocol)
                self.assertGreater(result.diagnostics["final_relative_weight_drift"], 1e-3)
                self.assertFalse(np.allclose(result.final_weights, self.weights))

    def test_legacy_default_protocols_are_preserved(self):
        self.assertEqual(PROTOCOLS, ("checkpoint", "factor", "analytic", "bottom_rank", "no_rollback"))
        self.assertEqual(set(SUPPORTED_PROTOCOLS) - set(PROTOCOLS),
                         {"factor_exact", "analytic_exact", "factor_sparse"})

    def test_exact_corrections_preserve_bits_scores_and_order_in_all_precisions(self):
        for precision in ("float64", "float32", "float16", "bfloat16"):
            checkpoint = evaluate_acquisition_candidates(
                self.weights, self.candidates, self.probes, self.covariance,
                protocol="checkpoint", precision=precision,
            )
            expected = quantize(self.weights, precision)
            for protocol in ("factor_exact", "analytic_exact"):
                for order in ("forward", "reverse"):
                    with self.subTest(precision=precision, protocol=protocol, order=order):
                        actual = evaluate_acquisition_candidates(
                            self.weights, self.candidates, self.probes, self.covariance,
                            protocol=protocol, precision=precision, order=order,
                        )
                        self.assertEqual(actual.final_weights.dtype, expected.dtype)
                        self.assertEqual(actual.final_weights.tobytes(), expected.tobytes())
                        np.testing.assert_array_equal(actual.scores, checkpoint.scores)
                        self.assertEqual(actual.ranking, checkpoint.ranking)
                        self.assertTrue(actual.diagnostics["all_trials_bitwise_restored"])
                        self.assertTrue(actual.diagnostics["final_state_bitwise_equal_to_start"])
                        self.assertEqual(actual.diagnostics["final_relative_weight_drift"], 0.0)
                        self.assertGreater(actual.diagnostics["correction_packet_bytes_total"], 0)
                        self.assertGreaterEqual(actual.diagnostics["correction_packet_bytes_total"],
                                                actual.diagnostics["correction_payload_bytes_total"])

    def test_sparse_budget_is_payload_only_and_metadata_is_counted(self):
        for precision in ("float32", "bfloat16"):
            logical_bytes = storage_nbytes(self.weights, precision)
            for fraction in (0.0, 0.25, 1.0):
                with self.subTest(precision=precision, fraction=fraction):
                    actual = evaluate_acquisition_candidates(
                        self.weights, self.candidates, self.probes, self.covariance,
                        protocol="factor_sparse", precision=precision,
                        patch_budget_fraction=fraction,
                    )
                    diagnostics = actual.diagnostics
                    budget = int(np.floor(fraction * logical_bytes))
                    self.assertEqual(diagnostics["correction_payload_budget_bytes_per_trial"], budget)
                    self.assertLessEqual(diagnostics["max_correction_payload_bytes"], budget)
                    self.assertGreater(diagnostics["correction_packet_bytes_total"],
                                       diagnostics["correction_payload_bytes_total"])
                    self.assertEqual(diagnostics["accounted_record_bytes_total"],
                                     diagnostics["base_numeric_bytes_total"] + diagnostics["correction_packet_bytes_total"])
                    if fraction == 0:
                        baseline = evaluate_acquisition_candidates(
                            self.weights, self.candidates, self.probes, self.covariance,
                            protocol="factor", precision=precision,
                        )
                        self.assertEqual(actual.final_weights.tobytes(), baseline.final_weights.tobytes())

    def test_exact_corrected_learning_paths_match_checkpoint_after_six_queries(self):
        for precision in ("float64", "float32", "float16", "bfloat16"):
            with self.subTest(precision=precision):
                result = run_active_experiment(
                    seed=7, precision=precision, n_rounds=6,
                    protocols=("checkpoint", "factor_exact", "analytic_exact"),
                )
                expected = next(path for path in result["active_paths"]
                                if path["protocol"] == "checkpoint" and path["candidate_order"] == "forward")
                for actual in result["active_paths"]:
                    self.assertEqual(actual["selected_pool_ids"], expected["selected_pool_ids"])
                    self.assertEqual(actual["final_state_sha256"], expected["final_state_sha256"])
                    self.assertEqual(actual["final_test_accuracy"], expected["final_test_accuracy"])
                    self.assertEqual(actual["teacher_label_queries"], 6)
                    self.assertTrue(actual["all_trial_states_bitwise_restored"])
                    self.assertTrue(actual["final_state_bitwise_matches_checkpoint"])
                json.dumps(result, allow_nan=False)

    def test_patch_budget_must_be_finite_and_nonnegative(self):
        for budget in (-1, float("nan"), float("inf")):
            with self.subTest(budget=budget), self.assertRaises(ValueError):
                evaluate_acquisition_candidates(
                    self.weights, self.candidates, self.probes, self.covariance,
                    protocol="factor_sparse", patch_budget_fraction=budget,
                )


if __name__ == "__main__":
    unittest.main()
