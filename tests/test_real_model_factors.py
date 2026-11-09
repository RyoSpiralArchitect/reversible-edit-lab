"""Dependency-free checks of the bridge used by the optional real-model pilot."""
import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('real_pilot', Path(__file__).resolve().parents[1] / 'scripts/real_model_costs.py')
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


class GradientBridgeTests(unittest.TestCase):
    def test_factors_are_normalized_negative_sgd_and_valid_rome_transport(self):
        weight = np.array([[1., 2., -3.], [4., -2., 1.]], dtype=np.float32)
        key = np.array([2., -1., 3.])
        gradient = np.array([0.7, -0.2])
        left, right, value, rate = pilot.gradient_factors(weight, key, gradient)
        delta = left @ right.T
        np.testing.assert_allclose(delta, -rate * np.outer(gradient, key), rtol=1e-14)
        self.assertAlmostEqual(np.linalg.norm(delta)/np.linalg.norm(weight.astype(float)), .001)
        self.assertAlmostEqual(float(right[:, 0] @ key), 1.)
        np.testing.assert_allclose(value-weight@key, left[:, 0], atol=1e-14)

    def test_zero_or_nonfinite_gradient_is_rejected(self):
        for gradient in (np.zeros(2), np.array([np.nan, 1.])):
            with self.assertRaises(ValueError):
                pilot.gradient_factors(np.ones((2, 3)), np.ones(3), gradient)
