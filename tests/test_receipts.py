from pathlib import Path
from tempfile import TemporaryDirectory
import io
import json
import unittest
import numpy as np

from reversible_edit.core import memit_edit, quantize, restore_delta, rome_edit
from reversible_edit.receipts import (edit_npz, make_receipt, receipt_metadata,
                                      restore_from_receipt, serialize_receipt,
                                      tensor_digest, undo_npz)
from reversible_edit.experiments import ExperimentConfig, run_selective_undo


class ReceiptTests(unittest.TestCase):
    def test_portable_rome_receipts_have_only_declared_information(self):
        rng = np.random.default_rng(44)
        w, k, v = rng.normal(size=(4, 7)), rng.normal(size=7), rng.normal(size=4)
        for recovery in ("delta", "analytic", "old_response"):
            with self.subTest(recovery=recovery), TemporaryDirectory() as temp:
                base = Path(temp)
                np.savez(base/"request.npz", weight=w, covariance=np.eye(7), key=k, value=v)
                meta = edit_npz(base/"request.npz", base/"edited.npz", base/"receipt.npz",
                                recovery=recovery, alpha=0.75, precision="float64")
                # Remove the source before restoration: no original dependency.
                (base/"request.npz").unlink()
                with np.load(base/"receipt.npz", allow_pickle=False) as record:
                    self.assertNotIn("weight", record.files)
                    self.assertEqual(meta["numeric_payload_bytes"], sum(record[key].nbytes for key in record.files if key != "__meta__"))
                    if recovery == "delta":
                        self.assertEqual(set(record.files), {"__meta__", "left", "right"})
                undo_npz(base/"edited.npz", base/"receipt.npz", base/"restored.npz")
                with np.load(base/"restored.npz") as restored:
                    np.testing.assert_allclose(restored["weight"], w, rtol=2e-13, atol=2e-13)

    def test_memit_portable_inverse_counts_covariance(self):
        rng = np.random.default_rng(12)
        w, keys, values = rng.normal(size=(3, 6)), rng.normal(size=(6, 2)), rng.normal(size=(3, 2))
        with TemporaryDirectory() as temp:
            p=Path(temp)
            np.savez(p/"request.npz", weight=w, keys=keys, values=values, covariance=np.eye(6))
            meta=edit_npz(p/"request.npz", p/"edited.npz", p/"receipt.npz", editor="memit", recovery="analytic", precision="float64")
            self.assertEqual(meta["numeric_payload_bytes"], (12+6+36)*8)
            (p/"request.npz").unlink()
            undo_npz(p/"edited.npz",p/"receipt.npz",p/"restored.npz")
            with np.load(p/"restored.npz") as archive:
                np.testing.assert_allclose(archive["weight"], w, rtol=2e-13, atol=2e-13)

    def test_wrong_or_stale_tensor_is_rejected(self):
        with TemporaryDirectory() as temp:
            p=Path(temp)
            np.savez(p/"request.npz", weight=np.eye(3), covariance=np.eye(3), key=np.ones(3), value=np.zeros(3))
            edit_npz(p/"request.npz",p/"edited.npz",p/"receipt.npz")
            np.savez(p/"wrong.npz",weight=np.eye(3))
            with self.assertRaisesRegex(ValueError,"does not match"):
                undo_npz(p/"wrong.npz",p/"receipt.npz",p/"out.npz")
            self.assertFalse((p/"out.npz").exists())

    def test_hard_rome_analytic_rejection_creates_no_outputs(self):
        with TemporaryDirectory() as temp:
            p=Path(temp)
            np.savez(p/"request.npz", weight=np.eye(2), covariance=np.eye(2), key=np.ones(2), value=np.zeros(2))
            with self.assertRaisesRegex(ValueError,"noninjective"):
                edit_npz(p/"request.npz",p/"edit.npz",p/"receipt.npz",alpha=1,recovery="analytic")
            self.assertFalse((p/"edit.npz").exists())
            self.assertFalse((p/"receipt.npz").exists())

    def test_unknown_editor_is_rejected_even_for_delta_receipt(self):
        with TemporaryDirectory() as temp:
            p=Path(temp)
            np.savez(p/"request.npz",weight=np.eye(3),covariance=np.eye(3),key=np.ones(3),value=np.zeros(3))
            edit_npz(p/"request.npz",p/"edited.npz",p/"receipt.npz")
            with np.load(p/"receipt.npz") as record:
                fields={key:np.array(record[key]) for key in record.files}
            meta=json.loads(str(fields["__meta__"].item()))
            meta["editor"]="unknown"
            fields["__meta__"]=np.array(json.dumps(meta))
            np.savez(p/"bad-receipt.npz",**fields)
            with self.assertRaisesRegex(ValueError,"Unsupported"):
                undo_npz(p/"edited.npz",p/"bad-receipt.npz",p/"out.npz")
            self.assertFalse((p/"out.npz").exists())

    def test_existing_output_is_preserved(self):
        with TemporaryDirectory() as temp:
            p=Path(temp)
            (p/"edit.npz").write_bytes(b"preserve this")
            with self.assertRaises(FileExistsError):
                edit_npz(p/"missing.npz",p/"edit.npz",p/"receipt.npz")
            self.assertEqual((p/"edit.npz").read_bytes(),b"preserve this")

    def test_svd_receipt_is_weights_only(self):
        with TemporaryDirectory() as temp:
            p=Path(temp)
            np.savez(p/"request.npz",weight=np.eye(3),covariance=np.eye(3),key=np.ones(3),value=np.zeros(3))
            meta=edit_npz(p/"request.npz",p/"edited.npz",p/"receipt.npz",recovery="svd")
            self.assertEqual(meta["numeric_payload_bytes"],0)
            with np.load(p/"receipt.npz") as record:
                self.assertEqual(record.files,["__meta__"])
            undo_npz(p/"edited.npz",p/"receipt.npz",p/"restored.npz")

    def test_selective_undo_requires_replay(self):
        result=run_selective_undo()
        self.assertEqual(result["scalar"]["subtract_a_only"],1.0)
        self.assertEqual(result["scalar"]["b_without_a"],2.0)
        self.assertEqual(result["scalar"]["undo_b_a_then_replay_b"],2.0)
        self.assertGreater(result["matrix"]["naive"]["weight_relative_error"],0.01)
        self.assertLess(result["matrix"]["replayed"]["weight_relative_error"],1e-12)

    def test_invalid_run_configuration_is_rejected(self):
        for options in ({"seeds":()}, {"precisions":()}, {"cycle_count":0}, {"input_dim":1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                ExperimentConfig(**options)

    def test_corrected_roundtrips_are_bit_exact_for_both_editors_all_precisions(self):
        rng = np.random.default_rng(48)
        unrounded = rng.normal(size=(4, 9))
        unrounded[0, 0] = -0.0
        covariance = np.eye(9)
        for precision in ("float64", "float32", "float16", "bfloat16"):
            original = quantize(unrounded, precision)
            key, value = rng.normal(size=9), rng.normal(size=4)
            keys, values = rng.normal(size=(9, 2)), rng.normal(size=(4, 2))
            edits = {
                "rome": rome_edit(original, key, value, covariance, alpha=0.95, precision=precision),
                "memit": memit_edit(original, keys, values, covariance, precision=precision),
            }
            for editor, edit in edits.items():
                for recovery in ("delta_exact", "analytic_exact"):
                    with self.subTest(precision=precision, editor=editor, recovery=recovery):
                        payload = make_receipt(original, edit, editor=editor, recovery=recovery, precision=precision)
                        meta = receipt_metadata(payload)
                        packet = serialize_receipt(payload)
                        self.assertTrue(meta["recovery_exact"])
                        self.assertNotIn("weight", payload)
                        self.assertEqual(meta["numeric_payload_bytes"],
                                         sum(a.nbytes for name, a in payload.items() if name != "__meta__"))
                        self.assertEqual(meta["numeric_payload_bytes"],
                                         meta["base_numeric_payload_bytes"] + meta["correction_packet_bytes"])
                        self.assertEqual(meta["correction_packet_bytes"],
                                         meta["correction_payload_bytes"] + meta["correction_metadata_bytes"])
                        with np.load(io.BytesIO(packet), allow_pickle=False) as archive:
                            loaded = {name: np.array(archive[name]) for name in archive.files}
                        restored = restore_from_receipt(edit.updated, loaded)
                        self.assertEqual(tensor_digest(restored), tensor_digest(original))
                        self.assertEqual(restored.dtype, original.dtype)
                        self.assertTrue(np.signbit(restored[0, 0]))

    def test_exact_file_receipts_chain_back_without_source_archives(self):
        rng = np.random.default_rng(32)
        unrounded = rng.normal(size=(5, 8))
        unrounded[0, 0] = -0.0
        key_a, key_b = rng.normal(size=8), rng.normal(size=8)
        value_a, value_b = rng.normal(size=5), rng.normal(size=5)
        for precision in ("float64", "float32", "float16", "bfloat16"):
            for recovery in ("delta_exact", "analytic_exact"):
                with self.subTest(precision=precision, recovery=recovery), TemporaryDirectory() as temp:
                    p = Path(temp)
                    original = quantize(unrounded, precision)
                    np.savez(p/"request-a.npz", weight=original, covariance=np.eye(8), key=key_a, value=value_a)
                    meta_a = edit_npz(p/"request-a.npz", p/"edited-a.npz", p/"receipt-a.npz",
                                      recovery=recovery, alpha=0.95, precision=precision)
                    with np.load(p/"edited-a.npz") as archive:
                        after_a = np.array(archive["weight"])
                    np.savez(p/"request-b.npz", weight=after_a, covariance=np.eye(8), key=key_b, value=value_b)
                    edit_npz(p/"request-b.npz", p/"edited-b.npz", p/"receipt-b.npz",
                              recovery=recovery, alpha=0.95, precision=precision)
                    self.assertEqual(meta_a["serialized_receipt_bytes"], (p/"receipt-a.npz").stat().st_size)
                    for file in ("request-a.npz", "request-b.npz", "edited-a.npz"):
                        (p/file).unlink()
                    with self.assertRaisesRegex(ValueError, "does not match"):
                        undo_npz(p/"edited-b.npz", p/"receipt-a.npz", p/"wrong-order.npz")
                    undo_npz(p/"edited-b.npz", p/"receipt-b.npz", p/"restored-a.npz")
                    with np.load(p/"restored-a.npz") as archive:
                        self.assertEqual(tensor_digest(archive["weight"]), tensor_digest(after_a))
                    undo_npz(p/"restored-a.npz", p/"receipt-a.npz", p/"restored-original.npz")
                    with np.load(p/"restored-original.npz") as archive:
                        self.assertEqual(tensor_digest(archive["weight"]), tensor_digest(original))

    def test_sparse_receipts_obey_data_budget_and_reduce_error(self):
        rng = np.random.default_rng(71)
        original = quantize(rng.normal(size=(8, 16)), "bfloat16")
        edit = rome_edit(original, rng.normal(size=16), rng.normal(size=8), np.eye(16),
                         alpha=0.95, precision="bfloat16")
        uncorrected = restore_delta(edit.updated, edit.delta, precision="bfloat16")
        base_error = np.linalg.norm(uncorrected.astype(float) - original.astype(float))
        for budget in (0, 12, 64, 4096):
            with self.subTest(budget=budget):
                payload = make_receipt(original, edit, editor="rome", recovery="delta_sparse",
                                       precision="bfloat16", patch_budget_bytes=budget)
                meta = receipt_metadata(payload)
                self.assertLessEqual(meta["correction_payload_bytes"], budget)
                self.assertGreater(meta["correction_metadata_bytes"], 0)
                restored = restore_from_receipt(edit.updated, payload)
                error = np.linalg.norm(restored.astype(float) - original.astype(float))
                self.assertLessEqual(error, base_error)
                if budget == 0:
                    self.assertEqual(tensor_digest(restored), tensor_digest(uncorrected))
                if budget == 4096:
                    self.assertTrue(meta["recovery_exact"])
                    self.assertEqual(tensor_digest(restored), tensor_digest(original))

    def test_corrected_modes_reject_unstored_input_without_writing(self):
        with TemporaryDirectory() as temp:
            p = Path(temp)
            np.savez(p/"request.npz", weight=np.eye(3, dtype=np.float64), covariance=np.eye(3),
                     key=np.ones(3), value=np.zeros(3))
            with self.assertRaisesRegex(ValueError, "canonical"):
                edit_npz(p/"request.npz", p/"edited.npz", p/"receipt.npz",
                          recovery="delta_exact", precision="float32")
            self.assertFalse((p/"edited.npz").exists())
            self.assertFalse((p/"receipt.npz").exists())
            bad_bfloat = np.eye(3, dtype=np.float32)
            bad_bfloat[0, 0] = np.float32(1.001)
            np.savez(p/"request-bf16.npz", weight=bad_bfloat, covariance=np.eye(3),
                     key=np.ones(3), value=np.zeros(3))
            with self.assertRaisesRegex(ValueError, "representable"):
                edit_npz(p/"request-bf16.npz", p/"edited.npz", p/"receipt.npz",
                          recovery="delta_exact", precision="bfloat16")
            self.assertFalse((p/"edited.npz").exists())

    def test_v1_receipts_decode_and_unknown_schema_or_extra_state_is_rejected(self):
        original = np.eye(3)
        edit = rome_edit(original, np.ones(3), np.zeros(3), np.eye(3), alpha=0.75)
        payload = make_receipt(original, edit, editor="rome", recovery="delta", precision="float64")
        meta = receipt_metadata(payload)
        meta["schema_version"] = 1
        payload["__meta__"] = np.array(json.dumps(meta))
        np.testing.assert_allclose(restore_from_receipt(edit.updated, payload), original, atol=1e-15)
        meta["schema_version"] = 91
        payload["__meta__"] = np.array(json.dumps(meta))
        with self.assertRaisesRegex(ValueError, "schema"):
            restore_from_receipt(edit.updated, payload)
        meta["schema_version"] = 2
        payload["__meta__"] = np.array(json.dumps(meta))
        payload["hidden_original"] = original
        with self.assertRaisesRegex(ValueError, "fields"):
            restore_from_receipt(edit.updated, payload)

    def test_corrupt_inverse_factors_are_caught_by_prediction_binding(self):
        original = quantize(np.arange(12).reshape(3, 4)/7, "float32")
        edit = rome_edit(original, np.ones(4), np.zeros(3), np.eye(4), alpha=0.95, precision="float32")
        payload = make_receipt(original, edit, editor="rome", recovery="delta_exact", precision="float32")
        payload["left"][0, 0] += 0.125
        with self.assertRaises(ValueError):
            restore_from_receipt(edit.updated, payload)


if __name__ == "__main__":
    unittest.main()
