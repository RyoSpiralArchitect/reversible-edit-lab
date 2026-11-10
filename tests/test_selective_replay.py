import importlib.util
from pathlib import Path
import unittest
import tempfile
import numpy as np

spec = importlib.util.spec_from_file_location("selective", Path(__file__).resolve().parents[1] / "scripts/selective_replay.py")
selective = importlib.util.module_from_spec(spec)
spec.loader.exec_module(selective)


class SelectiveReplayTests(unittest.TestCase):
    def test_artifacts_share_only_identical_bytes_and_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = selective.ArtifactStore()
            first, second = root / 'a.npz', root / 'b.npz'
            store.save(first, weight=np.array([1., 2.], dtype=np.float32))
            store.save(second, weight=np.array([1., 2.], dtype=np.float32))
            self.assertEqual(first.stat().st_ino, second.stat().st_ino)
            with self.assertRaises(FileExistsError):
                store.save(first, weight=np.array([3.]))
            third = root / 'c.npz'
            third.write_bytes(first.read_bytes())
            store.adopt(third)
            self.assertEqual(first.stat().st_ino, third.stat().st_ino)
            fresh = selective.ArtifactStore([root])
            fresh.save(root / 'd.npz', weight=np.array([1., 2.], dtype=np.float32))
            self.assertEqual(first.stat().st_ino, (root / 'd.npz').stat().st_ino)

    def test_direct_xor_is_exact_and_state_bound(self):
        rng = np.random.default_rng(101)
        original = rng.normal(size=(8, 12)).astype(np.float32)
        original[0, 0] = -0.0
        edited = original + rng.normal(size=original.shape).astype(np.float32) * .001
        blob = selective.direct_xor_record(original, edited)
        restored = selective.restore_direct_xor(edited, blob)
        self.assertEqual(original.tobytes(), restored.tobytes())
        with self.assertRaises(ValueError):
            selective.restore_direct_xor(edited + 1, blob)

    def test_receipts_reject_wrong_history_and_rewind_in_reverse_order(self):
        import io
        base = np.array([[1.0]], dtype=np.float32)
        one = np.ones_like(base)
        update_a = one * 2
        after_a = base + update_a
        update_b = after_a * 2
        after_ab = after_a + update_b
        pack = lambda old, new, delta: selective.chain.arrays(io.BytesIO(selective.chain.pack_record(old, new, delta, one, 'delta_exact')))
        record_a = pack(base, after_a, update_a)
        record_b = pack(after_a, after_ab, update_b)
        with self.assertRaises(ValueError):
            selective.chain.restore(after_ab, record_a)
        rewind = selective.chain.restore(selective.chain.restore(after_ab, record_b), record_a)
        self.assertEqual(rewind.tobytes(), base.tobytes())
        stale = rewind + update_b
        fresh = base + base * 2
        self.assertEqual(float(stale[0, 0]), 7.0)
        self.assertEqual(float(fresh[0, 0]), 3.0)

    def test_answer_requires_word_boundary(self):
        self.assertTrue(selective.phrase_prefix(" Rome is", " Rome"))
        self.assertFalse(selective.phrase_prefix(" Romanian", " Rome"))
        self.assertFalse(selective.phrase_prefix(" Paris then Rome", " Rome"))
