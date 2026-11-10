import importlib.util
from pathlib import Path
import unittest
import io
import numpy as np
from reversible_edit.receipts import tensor_digest

spec=importlib.util.spec_from_file_location('factual',Path(__file__).resolve().parents[1]/'scripts/factual_edit_chain.py')
factual=importlib.util.module_from_spec(spec);spec.loader.exec_module(factual)

class FactualReceiptTests(unittest.TestCase):
    def test_multirank_actual_float32_write_restores_exactly(self):
        rng=np.random.default_rng(23)
        original=rng.normal(size=(12,24)).astype(np.float32)
        left=rng.normal(size=(12,3)).astype(np.float32)
        right=rng.normal(size=(24,3)).astype(np.float32)
        edited=original+left@right.T
        for method in ['checkpoint','delta','delta_exact']:
            blob=factual.pack_record(original,edited,left,right,method)
            payload=factual.arrays(io.BytesIO(blob))
            result=factual.restore(edited,payload)
            if method!='delta':self.assertEqual(tensor_digest(result),tensor_digest(original))
            if method!='checkpoint':self.assertNotIn('weight',payload)
            with self.assertRaises(ValueError):factual.restore(edited+1,payload)

    def test_exact_record_rejects_changed_factor_prediction(self):
        rng=np.random.default_rng(7)
        original=rng.normal(size=(12,24)).astype(np.float32)
        left=rng.normal(size=(12,1));right=rng.normal(size=(24,1))
        edited=original+(left@right.T).astype(np.float32)
        record=factual.arrays(io.BytesIO(factual.pack_record(original,edited,left,right,'delta_exact')))
        record['left']=record['left']+1
        with self.assertRaises(ValueError):factual.restore(edited,record)
