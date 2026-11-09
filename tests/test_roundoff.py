import dataclasses
import struct
import unittest
import zlib

import numpy as np

from reversible_edit.core import quantize, restore_delta, restore_rome, rome_edit
from reversible_edit.roundoff import (RoundoffPatch, apply_roundoff_patch,
                                     make_roundoff_patch, storage_bit_digest)


class RoundoffPatchTests(unittest.TestCase):
    def test_corrects_actual_rollback_bits_in_every_precision(self):
        rng = np.random.default_rng(31)
        for precision in ("float64", "float32", "float16", "bfloat16"):
            with self.subTest(precision=precision):
                original = quantize(rng.normal(size=(9, 17)), precision)
                key, target = rng.normal(size=17), rng.normal(size=9)
                edited = rome_edit(original, key, target, np.eye(17), alpha=0.95, precision=precision)
                for predicted in (restore_delta(edited.updated, edited.delta, precision=precision),
                                  restore_rome(edited.updated, edited.spec, precision=precision)):
                    patch = make_roundoff_patch(original, predicted, precision)
                    self.assertTrue(patch.exact)
                    restored = apply_roundoff_patch(predicted, patch)
                    self.assertEqual(restored.dtype, original.dtype)
                    self.assertEqual(restored.tobytes(), original.tobytes())
                    self.assertEqual(patch.nbytes, len(patch.to_bytes()))
                    self.assertEqual(patch.nbytes, patch.metadata_nbytes+patch.payload_nbytes)

    def test_sparse_budget_limits_data_and_reduces_error(self):
        original = np.zeros((2, 5), dtype=np.float32)
        predicted = np.arange(1, 11, dtype=np.float32).reshape(2, 5)
        errors = []
        for budget in (0, 4, 5, 10, 25, 50):
            patch = make_roundoff_patch(original, predicted, "float32", strategy="sparse", budget_bytes=budget)
            restored = apply_roundoff_patch(predicted, patch)
            errors.append(float(np.linalg.norm(restored-original)))
            self.assertLessEqual(patch.payload_nbytes, budget)
            self.assertEqual(patch.patched_count, min(10,budget//5))
            self.assertGreater(patch.metadata_nbytes, 0)
            self.assertFalse(any(isinstance(field, np.ndarray) for field in dataclasses.astuple(patch)))
        self.assertTrue(all(b <= a for a,b in zip(errors, errors[1:])))
        self.assertEqual(errors[-1], 0)

    def test_sparse_uses_stable_error_ranking(self):
        original = np.zeros(4,dtype=np.float16)
        predicted = np.array([2., 5., -5., 1.],dtype=np.float16)
        patch=make_roundoff_patch(original,predicted,"float16",strategy="sparse",budget_bytes=3)
        restored=apply_roundoff_patch(predicted,patch)
        np.testing.assert_array_equal(restored,np.array([2.,0.,-5.,1.],dtype=np.float16))

    def test_signed_zeros_subnormals_and_scalar(self):
        for precision, tiny in (("float64",np.nextafter(0.,1.)),("float32",2.**-149),
                                ("float16",2.**-24),("bfloat16",2.**-133)):
            original=quantize(np.array([0.,-0.,tiny,-tiny]),precision)
            predicted=quantize(np.array([-0.,0.,0.,-0.]),precision)
            patch=make_roundoff_patch(original,predicted,precision)
            self.assertEqual(apply_roundoff_patch(predicted,patch).tobytes(),original.tobytes())
        original=np.array(-0.,dtype=np.float32)
        predicted=np.array(0.,dtype=np.float32)
        restored=apply_roundoff_patch(predicted,make_roundoff_patch(original,predicted,"float32"))
        self.assertEqual(restored.shape,())
        self.assertTrue(np.signbit(restored))

    def test_roundtrip_and_immutable_record(self):
        original=np.array([[1.,2.],[-0.,4.]],dtype=np.float32)
        predicted=np.array([[1.,2.01],[0.,3.99]],dtype=np.float32)
        for strategy,budget in (("xor_zlib",None),("sparse",5),("sparse",100)):
            patch=make_roundoff_patch(original,predicted,"float32",strategy=strategy,budget_bytes=budget)
            recovered_record=RoundoffPatch.from_bytes(patch.to_bytes())
            self.assertEqual(patch,recovered_record)
            old=original.copy(); before=predicted.copy()
            apply_roundoff_patch(predicted,recovered_record)
            self.assertEqual(original.tobytes(),old.tobytes())
            self.assertEqual(predicted.tobytes(),before.tobytes())
            with self.assertRaises(dataclasses.FrozenInstanceError):
                patch.payload=b"bad"

    def test_identity_counts_metadata_even_with_zero_data(self):
        original=np.ones((2,3),dtype=np.float16)
        for strategy,budget in (("xor_zlib",None),("sparse",0)):
            patch=make_roundoff_patch(original,original,"float16",strategy=strategy,budget_bytes=budget)
            self.assertEqual(patch.encoding,"identity")
            self.assertEqual(patch.payload_nbytes,0)
            self.assertGreater(patch.nbytes,0)
            self.assertTrue(patch.exact)
            self.assertEqual(apply_roundoff_patch(original,patch).tobytes(),original.tobytes())

    def test_wrong_predicted_state_is_rejected(self):
        original=np.ones(5,dtype=np.float32)
        predicted=np.zeros(5,dtype=np.float32)
        patch=make_roundoff_patch(original,predicted,"float32")
        with self.assertRaisesRegex(ValueError,"state digest"):
            apply_roundoff_patch(np.full(5,2.,dtype=np.float32),patch)
        with self.assertRaisesRegex(ValueError,"state digest"):
            apply_roundoff_patch(predicted.reshape(1,5),patch)

    def test_nonrepresentable_values_and_bad_budgets_are_rejected(self):
        for precision,value in (("float32",1+2.**-30),("float16",1+2.**-20),("bfloat16",1+2.**-10)):
            with self.assertRaisesRegex(ValueError,"representable"):
                make_roundoff_patch(np.array([value]),np.array([1.]),precision)
        for budget in (-1,1.5,True,None):
            with self.assertRaises(ValueError):
                make_roundoff_patch(np.ones(1),np.zeros(1),"float64",strategy="sparse",budget_bytes=budget)
        for array in (np.array([np.nan]),np.array([np.inf]),np.array([1+0j]),np.array([],dtype=float)):
            with self.assertRaises(ValueError):
                make_roundoff_patch(array,array,"float64")

    def test_packet_corruption_and_trailing_compressed_data_are_rejected(self):
        original=np.ones(8,dtype=np.float32)
        predicted=np.zeros(8,dtype=np.float32)
        patch=make_roundoff_patch(original,predicted,"float32")
        packet=patch.to_bytes()
        for corrupt in (packet[:-1],packet+b"extra",b"XXXX"+packet[4:],packet[:4]+struct.pack("<I",100000)+packet[8:]):
            with self.assertRaises(ValueError):
                RoundoffPatch.from_bytes(corrupt)
        for payload in (b"invalid",patch.payload+b"extra",zlib.compress(b"a"*100000)):
            bad=dataclasses.replace(patch,payload=payload)
            with self.assertRaises(ValueError):
                apply_roundoff_patch(predicted,bad)
        sparse=make_roundoff_patch(original,predicted,"float32",strategy="sparse",budget_bytes=5)
        with self.assertRaises(ValueError):
            dataclasses.replace(sparse,index_width=1.0)

    def test_canonical_endianness_and_bfloat16_logical_storage(self):
        original=np.array([1.,2.,3.],dtype=">f4")
        predicted=np.array([1.,2.,2.],dtype="<f4")
        restored=apply_roundoff_patch(predicted,make_roundoff_patch(original,predicted,"float32"))
        np.testing.assert_array_equal(restored,original)
        self.assertEqual(storage_bit_digest(original,"float32"),storage_bit_digest(original.astype("<f4"),"float32"))
        bf=quantize(np.array([1.,2.,-0.]),"bfloat16")
        patch=make_roundoff_patch(bf,bf,"bfloat16")
        self.assertEqual(apply_roundoff_patch(bf,patch).dtype,np.dtype("float32"))


if __name__ == "__main__":
    unittest.main()
