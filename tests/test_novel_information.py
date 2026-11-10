import copy
import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("novel", Path(__file__).resolve().parents[1] / "scripts/novel_information.py")
novel = importlib.util.module_from_spec(spec)
spec.loader.exec_module(novel)


class NovelInformationTests(unittest.TestCase):
    def test_retrieval_requires_complete_prefix(self):
        self.assertTrue(novel.target_prefix([4, 5, 6], [4, 5]))
        self.assertFalse(novel.target_prefix([4], [4, 5]))
        self.assertFalse(novel.target_prefix([3, 4, 5], [4, 5]))
        self.assertFalse(novel.target_prefix([4], []))

    def test_failure_to_answer_does_not_prove_restoration(self):
        before = {"state_digest": "original", "logits_digest": "base",
                  "probes": [{"id": "a/rewrite", "target_prefix_match": False, "generated_ids": [1]}]}
        edited = copy.deepcopy(before)
        edited["probes"][0]["target_prefix_match"] = True
        restored = copy.deepcopy(before)
        restored["state_digest"] = "residual-changes"
        restored["logits_digest"] = "different"
        result = novel.assess_case(before, edited, restored, "a")
        self.assertTrue(result["behavioral_cycle"])
        self.assertFalse(result["full_state_restored"])
        self.assertFalse(result["probe_logits_restored"])

    def test_control_retention_excludes_baseline_failures(self):
        before = {"probes": [{"kind": "control", "target_prefix_match": x, "generated_ids": [i]} for i, x in enumerate([True, False, True])]}
        after = copy.deepcopy(before)
        after["probes"][0]["target_prefix_match"] = False
        after["probes"][1]["target_prefix_match"] = True
        result = novel.control_comparison(before, after)
        self.assertEqual(result["baseline_target_matches"], 2)
        self.assertEqual(result["baseline_matches_retained"], 1)
        self.assertEqual(result["target_matches"], 2)
