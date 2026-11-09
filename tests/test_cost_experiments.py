import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from reversible_edit.core import quantize, rome_edit
from reversible_edit.cost_experiments import (
    CostConfig, _run_worker, checkpoint_record, identity_rome_fixture,
    restore_checkpoint, run_cost_experiment, summarize_costs,
)
from reversible_edit.receipts import tensor_digest


class CostExperimentTests(unittest.TestCase):
    def test_precomputed_identity_fixture_matches_rome_write(self):
        for precision in ("float64", "float32", "float16", "bfloat16"):
            for alpha in (0.0, 0.95, 1.0):
                original, edit, meta = identity_rome_fixture(7, (12, 24), precision, alpha)
                reference = rome_edit(original, edit.spec.key, edit.spec.value, np.eye(24),
                                      alpha=alpha, precision=precision)
                self.assertEqual(tensor_digest(reference.updated), tensor_digest(edit.updated))
                np.testing.assert_array_equal(reference.delta.left, edit.delta.left)
                np.testing.assert_array_equal(reference.delta.right, edit.delta.right)
                self.assertEqual(meta["original_tensor_sha256"], tensor_digest(original))

    def test_checkpoint_signed_zeros_logical_bfloat16_and_binding(self):
        for precision in ("float64", "float32", "float16", "bfloat16"):
            original = quantize([[0., -0.], [0.25, -2.]], precision)
            edited = quantize(original + 1, precision)
            record = checkpoint_record(original, edited, precision)
            self.assertEqual(tensor_digest(restore_checkpoint(edited, record)), tensor_digest(original))
            if precision == "bfloat16":
                self.assertEqual(record["weight"].dtype, np.dtype("uint16"))
                self.assertEqual(record["weight"].nbytes, original.size * 2)
            with self.assertRaisesRegex(ValueError, "edited tensor"):
                restore_checkpoint(edited + 1, record)
            record["weight"] = record["weight"].copy()
            record["weight"].flat[0] = 1
            with self.assertRaisesRegex(ValueError, "digest mismatch"):
                restore_checkpoint(edited, record)

    def test_separate_undo_workers_restore_after_original_file_is_deleted(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            original, edit, meta = identity_rome_fixture(23, (12, 24), "bfloat16", 1.0)
            inputs, edited_path = root / "inputs.npz", root / "edited.npz"
            np.savez(inputs, original=original, edited=edit.updated,
                     left=edit.delta.left, right=edit.delta.right,
                     key=edit.spec.key, value=edit.spec.value, direction=edit.spec.direction)
            np.savez(edited_path, weight=edit.updated)
            for method in ("checkpoint", "delta_exact"):
                record = root / f"{method}.npz"
                result = _run_worker(["build", "--inputs", str(inputs), "--record", str(record),
                                      "--method", method, "--precision", "bfloat16", "--alpha", "1"], 30)
                self.assertEqual(result["status"], "ok", result)
                self.assertEqual(record.stat().st_size, result["serialized_bytes"])
                self.assertGreaterEqual(result["process_peak_rss_bytes"], result["baseline_peak_rss_bytes"])
            inputs.unlink()
            del original, edit
            for method in ("checkpoint", "delta_exact"):
                result = _run_worker(["undo", "--edited", str(edited_path),
                                      "--record", str(root / f"{method}.npz"), "--method", method], 30)
                self.assertEqual(result["status"], "ok", result)
                self.assertEqual(result["restored_tensor_sha256"], meta["original_tensor_sha256"])

    def test_failures_are_preserved_and_existing_evidence_is_not_overwritten(self):
        missing = _run_worker(["undo", "--edited", "no-such-input.npz",
                               "--record", "no-such-record.npz", "--method", "delta_exact"], 30)
        self.assertEqual(missing["status"], "error")
        self.assertIn("FileNotFoundError", missing["stderr"])
        rows = [{"fixture_id": 0, "repeat": 0, "shape": "12x24", "precision": "bfloat16",
                 "method": "delta_exact", "status": "worker_failure", "build": missing,
                 "bitwise_restored": False}]
        summary = summarize_costs(rows)
        self.assertEqual(summary["failed_observations"], 1)
        self.assertEqual(summary["exact_observations"], 1)
        self.assertEqual(summary["exact_bitwise_restored"], 0)
        json.dumps(summary, allow_nan=False)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            evidence = root / "observations.jsonl"
            evidence.write_text("keep me")
            with self.assertRaises(FileExistsError):
                run_cost_experiment(root, CostConfig())
            self.assertEqual(evidence.read_text(), "keep me")

    def test_invalid_configurations_are_rejected_before_measurement(self):
        for values in ({"seeds": ()}, {"seeds": (7, 7)}, {"shapes": ((1, 10),)},
                       {"shapes": ((10, 20, 30),)}, {"precisions": ("int8",)},
                       {"alpha": float("nan")}, {"repeats": 0}, {"repeats": True},
                       {"worker_timeout_seconds": float("inf")}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                CostConfig(**values)


if __name__ == "__main__":
    unittest.main()
