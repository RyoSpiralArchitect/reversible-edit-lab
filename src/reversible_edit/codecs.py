"""Lossless codecs with bounded, single-frame decoding. Extras are optional."""
import zlib

CODECS = ('none', 'zlib', 'zstd', 'lz4')


def compress(raw: bytes, codec: str, level: int | None = None) -> bytes:
    if codec == 'none':
        return raw
    if codec == 'zlib':
        return zlib.compress(raw, 9 if level is None else level)
    if codec == 'zstd':
        import zstandard
        return zstandard.ZstdCompressor(level=3 if level is None else level,
                                       write_checksum=True).compress(raw)
    if codec == 'lz4':
        import lz4.frame
        return lz4.frame.compress(raw, compression_level=0 if level is None else level,
                                  content_checksum=True, store_size=True)
    raise ValueError('Unknown codec')


def decompress(payload: bytes, codec: str, expected: int) -> bytes:
    if type(expected) is not int or expected < 0:
        raise ValueError('Invalid decoded size')
    try:
        if codec == 'none':
            raw = payload
        elif codec == 'zlib':
            obj = zlib.decompressobj()
            raw = obj.decompress(payload, expected + 1)
            if not obj.eof or obj.unused_data or obj.unconsumed_tail:
                raise ValueError('Incomplete frame or trailing data')
        elif codec == 'zstd':
            import zstandard
            if zstandard.frame_content_size(payload) != expected:
                raise ValueError('Unexpected frame size')
            raw = zstandard.ZstdDecompressor().decompress(
                payload, max_output_size=expected + 1, allow_extra_data=False)
        elif codec == 'lz4':
            import lz4.frame
            if lz4.frame.get_frame_info(payload)['content_size'] != expected:
                raise ValueError('Unexpected frame size')
            obj = lz4.frame.LZ4FrameDecompressor()
            raw = obj.decompress(payload, max_length=expected + 1)
            if not obj.eof or obj.unused_data:
                raise ValueError('Incomplete frame or trailing data')
        else:
            raise ValueError('Unknown codec')
    except ImportError:
        raise
    except Exception as error:
        raise ValueError('Invalid compressed payload') from error
    if len(raw) != expected:
        raise ValueError('Incorrect decoded size')
    return raw
