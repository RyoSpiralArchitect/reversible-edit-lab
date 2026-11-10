import importlib.util
from pathlib import Path
import unittest

spec = importlib.util.spec_from_file_location('report_dependency', Path(__file__).resolve().parents[1] / 'scripts/report_dependency_replay.py')
report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(report)


class DependencyReportTests(unittest.TestCase):
    def test_last_retained_request_wins_after_selective_deletion(self):
        history = [dict(prompt='{} capital', subject='Spain', target_new={'str': ' Oslo'}),
                   dict(prompt='{} capital', subject='Spain', target_new={'str': ' Tokyo'})]
        observation = {'probes': [{'prompt': 'Spain capital', 'targets': {
            ' Oslo': {'word_prefix': False, 'probability': .1},
            ' Tokyo': {'word_prefix': True, 'probability': .9}}}]}
        both = report.latest_scores(history, observation)
        self.assertEqual(len(both), 1)
        self.assertEqual(both['Spain capital']['target'], ' Tokyo')
        self.assertTrue(both['Spain capital']['word_prefix'])
        deleted = report.latest_scores(history[:1], observation)
        self.assertEqual(deleted['Spain capital']['target'], ' Oslo')
        self.assertFalse(deleted['Spain capital']['word_prefix'])

    def test_controls_are_not_counted_as_edit_requests(self):
        observation = {'probes': [
            {'id': 'control_hot', 'targets': {' cold': {'word_prefix': True}}},
            {'id': 'spain', 'targets': {' Oslo': {'word_prefix': False}}}]}
        self.assertEqual(report.controls(observation), {'control_hot': True})
