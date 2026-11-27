"""Write-time correction records for finite-precision rollback.

The original state is inspected while a record is made, never while applying
it. This is explicit auxiliary old-state information, not blind restoration.
An exact record XORs the expected rollback's logical storage bits with the
original bits and compresses the XOR using zlib. Compression is not guaranteed
to beat a checkpoint. Sparse records retain only selected corrections.

``budget_bytes`` limits sparse *data payload* bytes. The self-describing header
and its state digests are additional and are included in ``patch.nbytes``.
All inputs must already be representable in the declared storage format.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import struct
from .codecs import compress, decompress

import numpy as np

from .core import PRECISIONS, quantize


_MAGIC = b"RRP1"
_HEADER_LIMIT = 65536
_WORD_BYTES = {"float64": 8, "float32": 4, "float16": 2, "bfloat16": 2}
_FLOAT_TYPES = {"float64": "<f8", "float32": "<f4", "float16": "<f2"}


def _logical_bits(array: np.ndarray, precision: str) -> tuple[np.ndarray, tuple[int, ...]]:
    if precision not in PRECISIONS:
        raise ValueError(f"Unsupported storage precision: {precision!r}")
    raw = np.asarray(array)
    if raw.dtype.kind != "f" or raw.dtype.itemsize > 8:
        raise ValueError("Storage correction requires real floating arrays of at most 64 bits")
    if raw.ndim > 32 or raw.size == 0:
        raise ValueError("Storage correction requires a nonempty array with at most 32 axes")
    rounded = quantize(raw, precision)
    original64 = raw.astype(np.float64)
    rounded64 = rounded.astype(np.float64)
    if not np.array_equal(original64, rounded64) or not np.array_equal(np.signbit(original64), np.signbit(rounded64)):
        raise ValueError("Input is not exactly representable in declared storage precision; quantize first")
    shape = tuple(int(size) for size in raw.shape)
    if precision == "bfloat16":
        words = np.asarray(rounded, dtype="<f4").reshape(-1).view("<u4")
        bits = (words >> np.uint32(16)).astype("<u2")
    else:
        values = np.array(rounded, dtype=_FLOAT_TYPES[precision], copy=True, order="C")
        bits = values.reshape(-1).view(f"<u{_WORD_BYTES[precision]}").copy()
    return bits, shape


def _from_bits(bits: np.ndarray, precision: str, shape: tuple[int, ...]) -> np.ndarray:
    if precision == "bfloat16":
        words = bits.astype("<u4") << np.uint32(16)
        result = words.view("<f4").reshape(shape).copy()
    else:
        result = bits.astype(f"<u{_WORD_BYTES[precision]}", copy=False).view(_FLOAT_TYPES[precision]).reshape(shape).copy()
    if not np.isfinite(result).all():
        raise ValueError("Correction produced a nonfinite stored value")
    return result


def _digest(bits: np.ndarray, precision: str, shape: tuple[int, ...]) -> str:
    layout = json.dumps([precision, list(shape)], separators=(",", ":")).encode("ascii")
    return hashlib.sha256(layout + b"\0" + bits.tobytes()).hexdigest()


def storage_bit_digest(array: np.ndarray, precision: str) -> str:
    """Digest canonical logical storage bits, including shape and signed zeros."""
    bits, shape = _logical_bits(array, precision)
    return _digest(bits, precision, shape)


def _index_width(elements: int) -> int:
    for width in (1, 2, 4, 8):
        if elements <= 1 << (8 * width):
            return width
    raise ValueError("Array is too large for the correction index format")


def _nonnegative_integer(value, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def _valid_digest(value: str) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


@dataclass(frozen=True)
class RoundoffPatch:
    """Immutable bytes and metadata, with no retained original-state arrays.

    ``exact`` means every write-time mismatch was included. ``apply`` verifies
    the target digest in this case. It also always verifies the predicted state;
    a different BLAS result or changed tensor is rejected rather than silently
    applying a correction constructed for another bit pattern.
    """

    precision: str
    shape: tuple[int, ...]
    encoding: str
    payload: bytes
    mismatch_count: int
    patched_count: int
    predicted_digest: str
    target_digest: str
    index_width: int = 0
    budget_bytes: int | None = None

    def __post_init__(self) -> None:
        if self.precision not in PRECISIONS:
            raise ValueError("Unsupported correction precision")
        if not isinstance(self.shape, tuple) or len(self.shape) > 32:
            raise ValueError("shape must be a tuple with at most 32 axes")
        if any(type(n) is not int or n < 1 for n in self.shape):
            raise ValueError("shape dimensions must be positive integers")
        elements = math.prod(self.shape)
        if elements > 2**63 - 1:
            raise ValueError("Correction shape is too large")
        _nonnegative_integer(self.mismatch_count, "mismatch_count")
        _nonnegative_integer(self.patched_count, "patched_count")
        _nonnegative_integer(self.index_width, "index_width")
        if not 0 <= self.patched_count <= self.mismatch_count <= elements:
            raise ValueError("Invalid mismatch/patch counts")
        if not isinstance(self.payload, bytes):
            raise ValueError("Correction payload must be immutable bytes")
        if not _valid_digest(self.predicted_digest) or not _valid_digest(self.target_digest):
            raise ValueError("Invalid correction state digest")
        if self.budget_bytes is not None:
            _nonnegative_integer(self.budget_bytes, "budget_bytes")
        if self.encoding == "identity":
            if self.payload or self.index_width or self.mismatch_count or self.patched_count:
                raise ValueError("Identity correction must be empty")
            if self.predicted_digest != self.target_digest:
                raise ValueError("Identity correction digests differ")
        elif self.encoding in ("xor_zlib", "xor_zstd", "xor_lz4", "xor_none"):
            if self.index_width != 0 or not self.exact or self.budget_bytes is not None:
                raise ValueError("Compressed XOR correction must be complete and unbudgeted")
        elif self.encoding == "sparse":
            if self.index_width != _index_width(elements) or self.budget_bytes is None:
                raise ValueError("Sparse correction requires canonical indices and a payload budget")
            expected = self.patched_count * (self.index_width + _WORD_BYTES[self.precision])
            if len(self.payload) != expected or len(self.payload) > self.budget_bytes:
                raise ValueError("Sparse payload size or budget is invalid")
        else:
            raise ValueError("Unsupported correction encoding")

    @property
    def strategy(self) -> str:
        return self.encoding

    @property
    def exact(self) -> bool:
        return self.patched_count == self.mismatch_count

    @property
    def payload_nbytes(self) -> int:
        return len(self.payload)

    def to_bytes(self) -> bytes:
        header = {
            "precision": self.precision, "shape": list(self.shape),
            "encoding": self.encoding, "mismatch_count": self.mismatch_count,
            "patched_count": self.patched_count, "predicted_digest": self.predicted_digest,
            "target_digest": self.target_digest, "index_width": self.index_width,
            "budget_bytes": self.budget_bytes, "payload_bytes": len(self.payload),
            "payload_sha256": hashlib.sha256(self.payload).hexdigest(),
        }
        encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode("ascii")
        return _MAGIC + struct.pack("<I", len(encoded)) + encoded + self.payload

    @property
    def nbytes(self) -> int:
        """Entire serialized patch size, including metadata and digests."""
        return len(self.to_bytes())

    @property
    def metadata_nbytes(self) -> int:
        return self.nbytes - self.payload_nbytes

    @classmethod
    def from_bytes(cls, data: bytes) -> "RoundoffPatch":
        if not isinstance(data, bytes) or len(data) < 8 or data[:4] != _MAGIC:
            raise ValueError("Invalid correction packet magic or length")
        header_size = struct.unpack("<I", data[4:8])[0]
        if header_size > _HEADER_LIMIT or len(data) < 8 + header_size:
            raise ValueError("Invalid correction header length")
        try:
            header = json.loads(data[8:8+header_size].decode("ascii"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError("Malformed correction header") from error
        required = {"precision", "shape", "encoding", "mismatch_count", "patched_count",
                    "predicted_digest", "target_digest", "index_width", "budget_bytes",
                    "payload_bytes", "payload_sha256"}
        if not isinstance(header, dict) or set(header) != required or not isinstance(header["shape"], list):
            raise ValueError("Unexpected correction header fields")
        payload_size = _nonnegative_integer(header["payload_bytes"], "payload_bytes")
        payload = data[8+header_size:]
        if len(payload) != payload_size or hashlib.sha256(payload).hexdigest() != header["payload_sha256"]:
            raise ValueError("Correction payload integrity or length mismatch")
        return cls(precision=header["precision"], shape=tuple(header["shape"]),
                   encoding=header["encoding"], payload=payload,
                   mismatch_count=header["mismatch_count"], patched_count=header["patched_count"],
                   predicted_digest=header["predicted_digest"], target_digest=header["target_digest"],
                   index_width=header["index_width"], budget_bytes=header["budget_bytes"])


def make_roundoff_patch(original: np.ndarray, predicted_restore: np.ndarray,
                       precision: str, *, strategy: str = "xor_zlib",
                       budget_bytes: int | None = None, level: int | None = None) -> RoundoffPatch:
    """Create an explicit old-state record at write time.

    Sparse selection keeps the largest absolute numerical errors, breaking ties
    by flattened index. A zero payload budget yields an empty-data correction;
    its metadata cost is still exposed by ``nbytes``. No budget is silently
    allocated to the dense inverse factors or other edit metadata.
    """
    if strategy not in ("xor_zlib", "xor_zstd", "xor_lz4", "xor_none", "sparse"):
        raise ValueError("Unsupported correction strategy")
    if strategy == "sparse":
        _nonnegative_integer(budget_bytes, "budget_bytes")
    elif budget_bytes is not None:
        raise ValueError("budget_bytes applies only to sparse corrections")
    original_bits, shape = _logical_bits(original, precision)
    predicted_bits, predicted_shape = _logical_bits(predicted_restore, precision)
    if shape != predicted_shape:
        raise ValueError("Original and predicted restore must have the same shape")
    different = np.flatnonzero(original_bits != predicted_bits)
    count = int(len(different))
    source_digest = _digest(predicted_bits, precision, shape)
    target_digest = _digest(original_bits, precision, shape)
    if count == 0:
        return RoundoffPatch(precision, shape, "identity", b"", 0, 0,
                             source_digest, target_digest, budget_bytes=budget_bytes)
    xor = np.bitwise_xor(original_bits, predicted_bits)
    if strategy.startswith("xor_"):
        return RoundoffPatch(precision, shape, strategy, compress(xor.tobytes(), strategy[4:], level),
                             count, count, source_digest, target_digest)
    index_width = _index_width(original_bits.size)
    per_element = index_width + _WORD_BYTES[precision]
    keep = min(count, budget_bytes // per_element)
    with np.errstate(over="ignore", invalid="ignore"):
        errors = np.abs(np.asarray(original, dtype=np.float64).ravel()[different]
                        - np.asarray(predicted_restore, dtype=np.float64).ravel()[different])
    order = np.lexsort((different, -errors))
    selected = np.sort(different[order[:keep]])
    indices = selected.astype(f"<u{index_width}")
    values = xor[selected].astype(f"<u{_WORD_BYTES[precision]}", copy=False)
    payload = indices.tobytes() + values.tobytes()
    return RoundoffPatch(precision, shape, "sparse", payload, count, int(keep),
                         source_digest, target_digest, index_width, budget_bytes)


def apply_roundoff_patch(predicted_restore: np.ndarray, patch: RoundoffPatch) -> np.ndarray:
    """Apply a patch without the original state; reject a different prediction.

    Exactness is conditional on recomputing the same predicted storage bits.
    This function detects arithmetic/environment changes through the digest;
    it does not claim cross-platform numerical reproducibility of the inverse.
    """
    if not isinstance(patch, RoundoffPatch):
        raise ValueError("patch must be a RoundoffPatch")
    bits, shape = _logical_bits(predicted_restore, patch.precision)
    if shape != patch.shape or _digest(bits, patch.precision, shape) != patch.predicted_digest:
        raise ValueError("Predicted restore does not match the correction's state digest")
    if patch.encoding.startswith("xor_"):
        raw = decompress(patch.payload, patch.encoding[4:], bits.nbytes)
        xor = np.frombuffer(raw, dtype=f"<u{_WORD_BYTES[patch.precision]}")
        bits ^= xor
    elif patch.encoding == "sparse":
        index_bytes = patch.patched_count * patch.index_width
        indices = np.frombuffer(patch.payload[:index_bytes], dtype=f"<u{patch.index_width}")
        xor = np.frombuffer(patch.payload[index_bytes:], dtype=f"<u{_WORD_BYTES[patch.precision]}")
        if len(indices) and (np.any(indices >= bits.size) or np.any(indices[1:] <= indices[:-1]) or np.any(xor == 0)):
            raise ValueError("Sparse indices or corrections are not canonical")
        bits[indices.astype(np.intp)] ^= xor
    if patch.exact and _digest(bits, patch.precision, shape) != patch.target_digest:
        raise ValueError("Corrected bits do not match the expected original-state digest")
    return _from_bits(bits, patch.precision, shape)
