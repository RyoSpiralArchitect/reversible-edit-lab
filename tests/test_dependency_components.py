import importlib.util
from pathlib import Path
import unittest

import numpy as np
from reversible_edit.receipts import tensor_digest


def script(name):
    spec = importlib.util.spec_from_file_location(name, Path(__file__).resolve().parents[1] / 'scripts' / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


components = script('dependency_components')
runner = script('dependency_replay')


class DependencyComponentsTests(unittest.TestCase):
    def test_prefix_includes_upstream_and_same_layer_input_only(self):
        for name in ['transformer.wte.weight', 'transformer.wpe.weight',
                     'transformer.h.4.mlp.c_proj.weight', 'transformer.h.5.ln_1.weight',
                     'transformer.h.5.attn.bias', 'transformer.h.5.ln_2.bias',
                     'transformer.h.5.mlp.c_fc.weight']:
            self.assertTrue(components.prefix_dependency(name, 5), name)
        for name in ['transformer.h.5.mlp.c_proj.weight', 'transformer.h.5.mlp.c_proj.bias',
                     'transformer.h.6.ln_1.weight', 'transformer.ln_f.weight', 'lm_head.weight']:
            self.assertFalse(components.prefix_dependency(name, 5), name)
        with self.assertRaises(ValueError):
            components.prefix_dependency('transformer.h.5.unrecognized.weight', 5)
        with self.assertRaises(ValueError):
            components.prefix_dependency('unknown.weight', 5)

    def test_changed_contract_rejects_and_corrupt_values_raise(self):
        value = np.array([1., -0.], dtype=np.float32)
        entry = {'dependency': runner.json_digest({'request': 'A', 'cov': 'fixed'}),
                 'value': value, 'value_digest': tensor_digest(value)}
        self.assertTrue(runner.valid_hit(entry, entry['dependency']))
        self.assertFalse(runner.valid_hit(entry, runner.json_digest({'request': 'B', 'cov': 'fixed'})))
        self.assertFalse(runner.valid_hit(entry, runner.json_digest({'request': 'A', 'cov': 'changed'})))
        value[1] = 0.  # Exact bytes, including signed zero, are part of the contract.
        with self.assertRaises(ValueError):
            runner.valid_hit(entry, entry['dependency'])
        self.assertFalse(runner.valid_hit(None, 'anything'))

    def test_rome_adapter_recomputes_current_output_and_normalization(self):
        source = '''def compute_v(layer, output, key):
    target_init = 2.
    delta = 3.
    target = target_init + delta
    right_vector = (target - output) / key
    print(target_init)
    return right_vector
'''
        fn = components.adapt_rome(source, {}, lambda layer, compute: compute())
        self.assertEqual(fn(5, 1., 2.), 2.)
        stale = components.adapt_rome(source, {}, lambda layer, compute: 10.)
        self.assertEqual(stale(5, 4., 3.), 2.)
        with self.assertRaises(ValueError):
            components.adapt_rome(source.replace('target =', 'other ='), {}, None)

    def test_memit_adapter_preserves_solve_arguments_and_layer_order(self):
        source = '''def execute_memit():
    result = []
    for layer in [4, 5, 6]:
        layer_ks, cov = layer + 1, 2
        adj_k = torch.linalg.solve(cov + layer_ks * layer_ks, layer_ks)
        result.append(adj_k)
    return result
'''
        class Linalg:
            @staticmethod
            def solve(a, b):
                return b / a
        class Torch:
            linalg = Linalg
        calls = []
        def capture(layer, key, cov, compute):
            calls.append((layer, key, cov))
            return compute()
        fn = components.adapt_memit(source, {'torch': Torch}, None, None, capture)
        self.assertEqual(fn(), [5/27, 6/38, 7/51])
        self.assertEqual(calls, [(4, 5, 2), (5, 6, 2), (6, 7, 2)])
        with self.assertRaises(ValueError):
            components.adapt_memit(source.replace('torch.linalg.solve', 'solve'), {}, None, None, capture)
