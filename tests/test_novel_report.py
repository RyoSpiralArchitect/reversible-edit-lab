import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("novel_report", Path(__file__).resolve().parents[1] / "scripts/report_novel_information.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class NovelReportTests(unittest.TestCase):
    def test_optional_audits_are_not_claimed_when_skipped(self):
        audit = {"binary_files_checked": 149, "model_files_checked": 0,
                 "upstream_files_checked": 0, "errors": []}
        skipped = report.audit_description(audit)
        self.assertIn("モデルファイル: 未実施", skipped)
        self.assertIn("著者コード: 未実施", skipped)
        checked = report.audit_description(dict(audit, model_files_checked=6, upstream_files_checked=21))
        self.assertIn("モデルファイル: 6件を確認", checked)
        self.assertIn("著者コード: 21件を確認", checked)
        self.assertNotIn("未実施", checked)

    def test_word_extension_is_not_exact_phrase_retrieval(self):
        self.assertFalse(report.phrase_prefix(" frozen sparrows fly", " frozen sparrow"))
        self.assertFalse(report.phrase_prefix("some frozen sparrow", " frozen sparrow"))
        self.assertTrue(report.phrase_prefix(" frozen sparrow.", " frozen sparrow"))
        self.assertTrue(report.phrase_prefix(" frozen sparrow flies", " frozen sparrow"))
        self.assertTrue(report.phrase_prefix(" frozen sparrow", " frozen sparrow"))
