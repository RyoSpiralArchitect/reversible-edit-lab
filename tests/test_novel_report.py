import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location("novel_report", Path(__file__).resolve().parents[1] / "scripts/report_novel_information.py")
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class NovelReportTests(unittest.TestCase):
    def test_word_extension_is_not_exact_phrase_retrieval(self):
        self.assertFalse(report.phrase_prefix(" frozen sparrows fly", " frozen sparrow"))
        self.assertFalse(report.phrase_prefix("some frozen sparrow", " frozen sparrow"))
        self.assertTrue(report.phrase_prefix(" frozen sparrow.", " frozen sparrow"))
        self.assertTrue(report.phrase_prefix(" frozen sparrow flies", " frozen sparrow"))
        self.assertTrue(report.phrase_prefix(" frozen sparrow", " frozen sparrow"))
