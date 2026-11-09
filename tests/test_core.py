"""Algebraic invariants and numerical failure cases for the editing kernels."""

import unittest

import numpy as np

from reversible_edit.core import (
    LowRankDelta,
    MemitSpec,
    RomeSpec,
    bottom_rank_restore,
    memit_edit,
    quantize,
    restore_delta,
    restore_memit,
    restore_rome,
    restore_rome_old_value,
    rome_edit,
    storage_nbytes,
)


class CoreTests(unittest.TestCase):
    def setUp(self):
        self.rng = np.random.default_rng(20261110)
        self.w = self.rng.normal(size=(5, 7))
        a = self.rng.normal(size=(7, 7))
        self.c = a @ a.T + 2.0 * np.eye(7)
        self.k = self.rng.normal(size=7)
        self.v = self.rng.normal(size=5)

    def test_hard_rome_is_noninjective_but_old_response_is_sufficient(self):
        result = rome_edit(self.w, self.k, self.v, self.c)
        np.testing.assert_allclose(result.updated @ self.k, self.v, atol=2e-14)
        alternative = self.w + np.outer(self.rng.normal(size=5), result.spec.direction)
        self.assertGreater(np.linalg.norm(alternative - self.w), 0.1)
        alternative_result = rome_edit(alternative, self.k, self.v, self.c)
        np.testing.assert_allclose(alternative_result.updated, result.updated, atol=2e-14)
        with self.assertRaisesRegex(ValueError, "no unique inverse"):
            restore_rome(result.updated, result.spec)
        for before, written in ((self.w, result), (alternative, alternative_result)):
            restored = restore_rome_old_value(written.updated, written.spec, before @ self.k)
            np.testing.assert_allclose(restored, before, atol=2e-14)

    def test_soft_rome_inverse_and_recorded_delta(self):
        for alpha in (0.0, 0.1, 0.75, 0.99):
            with self.subTest(alpha=alpha):
                result = rome_edit(self.w, self.k, self.v, self.c, alpha=alpha)
                np.testing.assert_allclose(restore_rome(result.updated, result.spec), self.w, atol=8e-14)
                np.testing.assert_allclose(restore_delta(result.updated, result.delta), self.w, atol=2e-14)
                np.testing.assert_allclose(
                    result.updated @ self.k,
                    (1.0 - alpha) * (self.w @ self.k) + alpha * self.v,
                    atol=2e-14,
                )

    def test_memit_inverse_for_rectangular_and_dependent_keys(self):
        for edit_count in (1, 3, 10):
            keys = self.rng.normal(size=(7, edit_count))
            if edit_count > 1:
                keys[:, -1] = keys[:, 0]
            values = self.rng.normal(size=(5, edit_count))
            with self.subTest(edit_count=edit_count):
                result = memit_edit(self.w, keys, values, self.c)
                np.testing.assert_allclose(restore_memit(result.updated, result.spec), self.w, atol=3e-14)
                np.testing.assert_allclose(restore_delta(result.updated, result.delta), self.w, atol=2e-14)
                self.assertGreater(result.diagnostics["inverse_amplification_whitened"], 1.0)

    def test_single_key_memit_matches_softened_rome(self):
        strength = self.k @ np.linalg.solve(self.c, self.k)
        alpha = strength / (1.0 + strength)
        soft = rome_edit(self.w, self.k, self.v, self.c, alpha=alpha)
        ridge = memit_edit(self.w, self.k[:, None], self.v[:, None], self.c)
        np.testing.assert_allclose(soft.updated, ridge.updated, atol=2e-14)

    def test_low_precision_write_is_not_a_bit_exact_undo(self):
        w = np.array([[1.0]], dtype=np.float32)
        result = rome_edit(w, [1.0], [8.1], [[1.0]], precision="float32")
        restored = restore_delta(result.updated, result.delta, precision="float32")
        self.assertEqual(restored.dtype, np.float32)
        self.assertFalse(np.array_equal(w, restored))
        self.assertLess(float(np.max(np.abs(w - restored))), 1e-6)

    def test_records_copy_inputs_and_edit_does_not_mutate(self):
        originals = [array.copy() for array in (self.w, self.k, self.v, self.c)]
        result = rome_edit(self.w, self.k, self.v, self.c)
        for original, current in zip(originals, (self.w, self.k, self.v, self.c)):
            np.testing.assert_array_equal(original, current)
        saved_key = result.spec.key.copy()
        self.k[:] = 0.0
        np.testing.assert_array_equal(result.spec.key, saved_key)
        with self.assertRaises(ValueError):
            result.delta.left[0, 0] = 0.0
        with self.assertRaises(ValueError):
            result.spec.value[0] = 0.0
        self.assertEqual(result.delta.nbytes, (5 + 7) * 8)

    def test_memit_spec_owns_its_arrays(self):
        keys = self.k[:, None].copy()
        values = self.v[:, None].copy()
        covariance = self.c.copy()
        spec = MemitSpec(keys, values, covariance)
        keys[:] = 0.0
        values[:] = 0.0
        covariance[:] = 0.0
        self.assertGreater(np.linalg.norm(spec.keys), 0.0)
        self.assertGreater(np.linalg.norm(spec.values), 0.0)
        self.assertGreater(np.linalg.norm(spec.covariance), 0.0)

    def test_bfloat16_rounds_ties_to_even_including_subnormals(self):
        source_bits = np.array(
            [0x3F808000, 0x3F818000, 0xBF808000, 0xBF818000, 0x00008000, 0x00018000],
            dtype=np.uint32,
        )
        expected_bits = np.array(
            [0x3F800000, 0x3F820000, 0xBF800000, 0xBF820000, 0x00000000, 0x00020000],
            dtype=np.uint32,
        )
        rounded = quantize(source_bits.view(np.float32), "bfloat16")
        self.assertEqual(rounded.dtype, np.float32)
        np.testing.assert_array_equal(rounded.view(np.uint32), expected_bits)
        np.testing.assert_array_equal(quantize(rounded, "bfloat16"), rounded)
        self.assertEqual(storage_nbytes(rounded, "bfloat16"), 12)
        self.assertEqual(storage_nbytes(rounded, "float64"), 48)

    def test_quantize_formats_signed_zero_and_overflow(self):
        for precision, dtype in (("float64", np.float64), ("float32", np.float32),
                                 ("float16", np.float16), ("bfloat16", np.float32)):
            rounded = quantize([0.0, -0.0, 1.5], precision)
            self.assertEqual(rounded.dtype, dtype)
            self.assertTrue(np.signbit(rounded[1]))
            self.assertEqual(rounded[2], 1.5)
            scalar = quantize(np.float64(1.5), precision)
            self.assertIsInstance(scalar, np.ndarray)
            self.assertEqual(scalar.shape, ())
        for values, precision in (([70000.0], "float16"), ([1e300], "float32"),
                                  ([np.finfo(np.float32).max], "bfloat16")):
            with self.subTest(values=values, precision=precision):
                with self.assertRaisesRegex(ValueError, "overflow"):
                    quantize(values, precision)
        for bad in ([np.nan], [np.inf], [-np.inf], [1.0 + 1.0j]):
            with self.assertRaises(ValueError):
                quantize(bad)

    def test_bfloat16_rounds_float64_midpoint_neighbors_directly(self):
        midpoint = np.float64(1.0 + 2.0**-8)
        inputs = np.array([np.nextafter(midpoint, -np.inf), midpoint,
                           np.nextafter(midpoint, np.inf)])
        expected = np.array([1.0, 1.0, 1.0 + 2.0**-7], dtype=np.float32)
        np.testing.assert_array_equal(quantize(inputs, "bfloat16"), expected)
        np.testing.assert_array_equal(quantize(-inputs, "bfloat16"), -expected)
        odd_midpoint = np.float64(1.0 + 3.0 * 2.0**-8)
        odd_inputs = [np.nextafter(odd_midpoint, -np.inf), odd_midpoint,
                      np.nextafter(odd_midpoint, np.inf)]
        np.testing.assert_array_equal(
            quantize(odd_inputs, "bfloat16"),
            np.array([1.0 + 2.0**-7, 1.0 + 2.0**-6, 1.0 + 2.0**-6], dtype=np.float32),
        )
        subnormal_midpoint = np.float64(2.0**-134)
        subnormal_inputs = [np.nextafter(subnormal_midpoint, 0.0), subnormal_midpoint,
                            np.nextafter(subnormal_midpoint, np.inf)]
        np.testing.assert_array_equal(
            quantize(subnormal_inputs, "bfloat16"),
            np.array([0.0, 0.0, 2.0**-133], dtype=np.float32),
        )
        maximum = np.ldexp(np.float64(255.0), 120)
        overflow_midpoint = maximum + np.ldexp(np.float64(1.0), 119)
        self.assertEqual(quantize(np.nextafter(overflow_midpoint, 0.0), "bfloat16"), maximum)
        for value in (overflow_midpoint, np.nextafter(overflow_midpoint, np.inf),
                      -overflow_midpoint):
            with self.assertRaisesRegex(ValueError, "overflow"):
                quantize(value, "bfloat16")

    def test_bottom_rank_is_explicitly_a_spectral_heuristic(self):
        w = np.zeros((3, 4))
        w[:3, :3] = np.diag([10.0, 3.0, 1.0])
        expected = w.copy()
        expected[0, 0] = 0.0
        np.testing.assert_allclose(bottom_rank_restore(w, 1), expected, atol=1e-14)
        np.testing.assert_array_equal(bottom_rank_restore(w, 0), w)
        np.testing.assert_array_equal(bottom_rank_restore(w, 3), np.zeros_like(w))
        for invalid in (-1, 4, 1.0, True):
            with self.assertRaises(ValueError):
                bottom_rank_restore(w, invalid)

    def test_shape_and_information_contracts_are_checked(self):
        invalid_calls = (
            lambda: rome_edit(self.w, np.zeros(7), self.v, self.c),
            lambda: rome_edit(self.w, self.k[:-1], self.v, self.c),
            lambda: rome_edit(self.w, self.k, self.v[:-1], self.c),
            lambda: rome_edit(self.w, self.k, self.v, np.zeros((7, 7))),
            lambda: rome_edit(self.w, self.k, self.v, -np.eye(7)),
            lambda: rome_edit(self.w, self.k, self.v, np.triu(self.c)),
            lambda: rome_edit(self.w, self.k, self.v, self.c, alpha=-0.1),
            lambda: rome_edit(self.w, self.k, self.v, self.c, alpha=1.01),
            lambda: rome_edit(self.w, self.k, self.v, self.c, alpha=np.nan),
            lambda: rome_edit(self.w, self.k, self.v, self.c, precision="float8"),
            lambda: memit_edit(self.w, np.zeros((7, 1)), self.v[:, None], self.c),
            lambda: memit_edit(self.w, self.k[:, None], np.zeros((5, 2)), self.c),
            lambda: LowRankDelta(np.ones((5, 2)), np.ones((7, 3))),
            lambda: RomeSpec(self.k, self.v, np.zeros(7), 0.5),
        )
        for call in invalid_calls:
            with self.subTest(call=call):
                with self.assertRaises(ValueError):
                    call()
        result = rome_edit(self.w, self.k, self.v, self.c, alpha=0.5)
        with self.assertRaises(ValueError):
            restore_delta(self.w[:, :-1], result.delta)
        with self.assertRaises(ValueError):
            restore_rome_old_value(result.updated, result.spec, self.v[:-1])


if __name__ == "__main__":
    unittest.main()
