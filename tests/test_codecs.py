import importlib.util
import unittest
import numpy as np
from reversible_edit.codecs import compress, decompress
from reversible_edit.roundoff import make_roundoff_patch, apply_roundoff_patch, RoundoffPatch

class CodecTests(unittest.TestCase):
    def test_frames_and_corrections(self):
        for codec,module in [('none',None),('zlib',None),('zstd','zstandard'),('lz4','lz4')]:
            if module and importlib.util.find_spec(module) is None:continue
            with self.subTest(codec=codec):
                raw=b'abc'*1000;encoded=compress(raw,codec)
                self.assertEqual(decompress(encoded,codec,len(raw)),raw)
                for bad in [encoded[:-1],encoded+b'x']:
                    with self.assertRaises(ValueError):decompress(bad,codec,len(raw))
                with self.assertRaises(ValueError):decompress(encoded,codec,len(raw)-1)
                x=np.array([0.,-0.,1.,-2.],dtype=np.float32);y=x.copy();y[2]+=np.float32(1e-6)
                patch=RoundoffPatch.from_bytes(make_roundoff_patch(x,y,'float32',strategy='xor_'+codec).to_bytes())
                self.assertEqual(apply_roundoff_patch(y,patch).tobytes(),x.tobytes())
                with self.assertRaises(ValueError):apply_roundoff_patch(x,patch)
                blob=patch.to_bytes()
                with self.assertRaises(ValueError):RoundoffPatch.from_bytes(blob[:-1])
                with self.assertRaises(ValueError):RoundoffPatch.from_bytes(blob[:-1]+bytes([blob[-1]^1]))

    def test_empty_frames_require_complete_single_frame(self):
        self.assertEqual(decompress(b'', 'none', 0), b'')
        for codec, module in [('zlib', None), ('zstd', 'zstandard'), ('lz4', 'lz4')]:
            if module and importlib.util.find_spec(module) is None:
                continue
            encoded = compress(b'', codec)
            with self.subTest(codec=codec, case='roundtrip'):
                self.assertEqual(decompress(encoded, codec, 0), b'')
            invalid = [encoded[:end] for end in range(len(encoded))]
            invalid.extend([
                encoded + b'x',
                encoded + encoded,
                encoded + compress(b'nonempty', codec),
                encoded[:-1] + bytes([encoded[-1] ^ 1]),
            ])
            for index, payload in enumerate(invalid):
                with self.subTest(codec=codec, case=index), self.assertRaises(ValueError):
                    decompress(payload, codec, 0)

    @unittest.skipUnless(importlib.util.find_spec('zstandard'), 'zstandard is optional')
    def test_zstd_rejects_nonempty_frame_claiming_zero_size(self):
        import zstandard

        payload = bytearray(compress(b'x' * 32, 'zstd'))
        # A small single-segment frame has a one-byte content size after its descriptor.
        self.assertTrue(payload[4] & 0x20)
        self.assertEqual(zstandard.frame_content_size(payload), 32)
        payload[5] = 0
        self.assertEqual(zstandard.frame_content_size(payload), 0)
        with self.assertRaises(ValueError):
            decompress(bytes(payload), 'zstd', 0)

if __name__=='__main__':unittest.main()
