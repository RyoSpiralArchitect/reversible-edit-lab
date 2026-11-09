"""Portable recovery receipts with declared old-state information.

No generated receipt stores a dense original weight array. Corrected modes do
store information about the old state: compressed XOR roundoff data or selected
old-state corrections. Their full serialized size must be counted. This is not
weights-only recovery, nor a promise that a receipt is smaller than a snapshot.

Exact correction validates both the edited tensor and the reconstructed tensor.
It enables reverse-order receipt traversal when the predictor used for rollback
is bitwise reproducible. A different prediction is rejected by the patch digest;
this is deliberately fail-closed rather than cross-platform numerical guessing.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .core import (EditResult, LowRankDelta, MemitSpec, PRECISIONS, RomeSpec,
                   bottom_rank_restore, memit_edit, quantize, restore_delta,
                   restore_memit, restore_rome, restore_rome_old_value, rome_edit)


RECOVERY_MODES = ("delta", "analytic", "old_response", "svd", "delta_exact",
                  "analytic_exact", "delta_sparse")
_LEGACY_MODES = ("delta", "analytic", "old_response", "svd")
_CORRECTED_MODES = ("delta_exact", "analytic_exact", "delta_sparse")
_EXACT_MODES = ("delta_exact", "analytic_exact")


def tensor_digest(weight: np.ndarray) -> str:
    """Bind shape, physical dtype, and exact tensor bytes, including signed zero."""
    w = np.ascontiguousarray(weight)
    header = json.dumps({"shape": list(w.shape), "dtype": w.dtype.str}, sort_keys=True).encode()
    return hashlib.sha256(header + w.tobytes()).hexdigest()


def _validate_mode(editor: str, recovery: str, precision: str) -> None:
    if editor not in ("rome", "memit") or recovery not in RECOVERY_MODES:
        raise ValueError("Unsupported receipt editor/recovery combination")
    if precision not in PRECISIONS:
        raise ValueError("Unsupported receipt precision")
    if recovery == "old_response" and editor != "rome":
        raise ValueError("old_response receipts require editor=rome")


def _canonical_stored(weight: np.ndarray, precision: str, name: str) -> np.ndarray:
    """Corrected receipts recover an already stored state, without changing it."""
    w = np.asarray(weight)
    dtype = np.dtype("float32" if precision == "bfloat16" else precision)
    if w.dtype != dtype:
        raise ValueError(f"{name} must already use canonical {precision} storage dtype {dtype}")
    rounded = quantize(w, precision)
    if tensor_digest(w) != tensor_digest(rounded):
        raise ValueError(f"{name} must already be representable in {precision}; quantize explicitly before editing")
    return w


def _meta(payload: Mapping[str, Any]) -> dict:
    try:
        scalar = np.asarray(payload["__meta__"])
        if scalar.ndim != 0 or scalar.dtype.kind not in ("U", "S"):
            raise ValueError("Receipt metadata must be a JSON string scalar")
        meta = json.loads(scalar.item())
    except (KeyError, TypeError, json.JSONDecodeError) as error:
        raise ValueError("Invalid receipt metadata") from error
    if not isinstance(meta, dict):
        raise ValueError("Receipt metadata must be a JSON object")
    if type(meta.get("schema_version")) is not int or meta["schema_version"] not in (1, 2):
        raise ValueError("Unsupported receipt schema")
    editor, recovery, precision = meta.get("editor"), meta.get("recovery"), meta.get("precision")
    _validate_mode(editor, recovery, precision)
    if meta["schema_version"] == 1 and recovery not in _LEGACY_MODES:
        raise ValueError("Corrected recovery requires receipt schema 2")
    return meta


def receipt_metadata(payload: Mapping[str, Any]) -> dict:
    """Return validated metadata for reporting; no model state is returned."""
    return _meta(payload)


def _base_mode(recovery: str) -> str:
    if recovery in ("delta_exact", "delta_sparse"):
        return "delta"
    if recovery == "analytic_exact":
        return "analytic"
    return recovery


def _required_fields(editor: str, recovery: str) -> set[str]:
    mode = _base_mode(recovery)
    fields = {"__meta__"}
    if mode == "delta":
        fields.update(("left", "right"))
    elif mode in ("analytic", "old_response") and editor == "rome":
        fields.update(("key", "value", "direction"))
        if mode == "old_response":
            fields.add("old_response")
    elif mode == "analytic":
        fields.update(("keys", "values", "covariance"))
    if recovery in _CORRECTED_MODES:
        fields.add("roundoff_patch")
    return fields


def _base_restore(edited: np.ndarray, payload: Mapping[str, Any], meta: dict) -> np.ndarray:
    mode, editor, precision = _base_mode(meta["recovery"]), meta["editor"], meta["precision"]
    if mode == "delta":
        return restore_delta(edited, LowRankDelta(payload["left"], payload["right"]), precision=precision)
    if mode in ("analytic", "old_response") and editor == "rome":
        spec = RomeSpec(payload["key"], payload["value"], payload["direction"], meta["alpha"])
        if mode == "analytic":
            return restore_rome(edited, spec, precision=precision)
        return restore_rome_old_value(edited, spec, payload["old_response"], precision=precision)
    if mode == "analytic" and editor == "memit":
        spec = MemitSpec(payload["keys"], payload["values"], payload["covariance"])
        return restore_memit(edited, spec, precision=precision)
    if mode == "svd":
        return bottom_rank_restore(edited, meta["drop_rank"], precision=precision)
    raise ValueError("Unsupported receipt editor/recovery combination")


def make_receipt(original: np.ndarray, edit_result: EditResult, *, editor: str,
                 recovery: str, precision: str, patch_budget_bytes: int | None = None,
                 drop_rank: int = 1) -> dict[str, np.ndarray]:
    """Build declared write-time metadata, discarding the full original afterward.

    Corrected modes require original to have the physical storage dtype and
    values implied by precision (bf16 uses a quantized fp32 carrier). Sparse
    patch_budget_bytes bounds correction DATA bytes, not packet metadata, base
    factors, or NPZ overhead. Count serialize_receipt bytes for actual storage.
    No test labels or evaluator-global state are used.
    """
    _validate_mode(editor, recovery, precision)
    if not isinstance(edit_result, EditResult):
        raise ValueError("edit_result must be an EditResult")
    expected_spec = RomeSpec if editor == "rome" else MemitSpec
    if not isinstance(edit_result.spec, expected_spec):
        raise ValueError("editor does not match the supplied EditResult specification")
    w = np.asarray(original)
    if w.ndim != 2 or w.shape != edit_result.updated.shape or not np.isfinite(w).all():
        raise ValueError("original must be a finite matrix matching the edited tensor")
    if recovery in _CORRECTED_MODES:
        _canonical_stored(w, precision, "original")
        _canonical_stored(edit_result.updated, precision, "edited tensor")
    spec = edit_result.spec
    mode = _base_mode(recovery)
    if mode == "analytic" and editor == "rome" and spec.alpha == 1.0:
        raise ValueError("Hard ROME is noninjective; use delta or old_response")
    if recovery == "delta_sparse":
        if isinstance(patch_budget_bytes, (bool, np.bool_)) or not isinstance(patch_budget_bytes, (int, np.integer)) or patch_budget_bytes < 0:
            raise ValueError("delta_sparse requires a nonnegative integer patch_budget_bytes")
        patch_budget_bytes = int(patch_budget_bytes)
    elif patch_budget_bytes is not None:
        raise ValueError("patch_budget_bytes is only supported for delta_sparse")
    payload: dict[str, np.ndarray] = {}
    if mode == "delta":
        payload = {"left": np.array(edit_result.delta.left), "right": np.array(edit_result.delta.right)}
    elif mode in ("analytic", "old_response") and editor == "rome":
        payload = {"key": np.array(spec.key), "value": np.array(spec.value), "direction": np.array(spec.direction)}
        if mode == "old_response":
            payload["old_response"] = np.asarray(w, dtype=np.float64) @ spec.key
    elif mode == "analytic":
        payload = {"keys": np.array(spec.keys), "values": np.array(spec.values), "covariance": np.array(spec.covariance)}
    else:
        if isinstance(drop_rank, (bool, np.bool_)) or not isinstance(drop_rank, (int, np.integer)) or not 0 <= drop_rank <= min(w.shape):
            raise ValueError("drop_rank must be an integer in [0, min(weight.shape)]")
        drop_rank = int(drop_rank)
    base_bytes = int(sum(array.nbytes for array in payload.values()))
    meta = {
        "schema_version": 2, "editor": editor, "recovery": recovery,
        "precision": precision, "alpha": float(spec.alpha) if editor == "rome" else None,
        "drop_rank": drop_rank if mode == "svd" else None,
        "edited_tensor_sha256": tensor_digest(edit_result.updated),
        "base_numeric_payload_bytes": base_bytes,
        "correction_packet_bytes": 0, "correction_payload_bytes": 0,
        "correction_metadata_bytes": 0, "numeric_payload_bytes": base_bytes,
        "patch_budget_bytes": patch_budget_bytes,
        "recovery_exact": False,
        "contains_original_weight": False,
        "contains_old_state_correction": recovery in _CORRECTED_MODES,
        "scope": "Fixed-target matrix write; full originals are not retained, declared old-state information is.",
    }
    if recovery in _CORRECTED_MODES:
        from .roundoff import make_roundoff_patch
        predicted = _base_restore(edit_result.updated, payload, meta)
        patch = make_roundoff_patch(w, predicted, precision,
                                    strategy="sparse" if recovery == "delta_sparse" else "xor_zlib",
                                    budget_bytes=patch_budget_bytes)
        if recovery in _EXACT_MODES and not patch.exact:
            raise ValueError("Exact recovery mode produced an inexact correction")
        packet = patch.to_bytes()
        payload["roundoff_patch"] = np.frombuffer(packet, dtype=np.uint8).copy()
        meta.update({
            "correction_packet_bytes": len(packet),
            "correction_payload_bytes": patch.payload_nbytes,
            "correction_metadata_bytes": patch.metadata_nbytes,
            "numeric_payload_bytes": base_bytes + len(packet),
            "correction_mismatch_count": patch.mismatch_count,
            "correction_patched_count": patch.patched_count,
            "recovery_exact": bool(patch.exact),
        })
        if patch.exact:
            meta["original_tensor_sha256"] = tensor_digest(w)
    payload["__meta__"] = np.array(json.dumps(meta, sort_keys=True, allow_nan=False))
    return payload


def _serialize_npz(payload: Mapping[str, Any]) -> bytes:
    arrays = {name: np.asarray(array) for name, array in payload.items()}
    if any(array.dtype.hasobject for array in arrays.values()):
        raise ValueError("Object arrays are not allowed in NPZ output")
    with io.BytesIO() as stream:
        np.savez_compressed(stream, **arrays)
        return stream.getvalue()


def serialize_receipt(payload: Mapping[str, Any]) -> bytes:
    """Actual compressed NPZ bytes, including arrays, correction, and headers."""
    meta = _meta(payload)
    if set(payload) != _required_fields(meta["editor"], meta["recovery"]):
        raise ValueError("Receipt fields do not match declared recovery information")
    return _serialize_npz(payload)


def restore_from_receipt(edited: np.ndarray, payload: Mapping[str, Any]) -> np.ndarray:
    """Restore from the paired edited state and declared receipt, without W_old.

    Exact corrected receipts verify the output digest and can therefore be
    traversed in reverse write order. Approximate legacy/sparse receipts retain
    strict input binding but may not reconstruct the next receipt's input bits.
    """
    meta = _meta(payload)
    if set(payload) != _required_fields(meta["editor"], meta["recovery"]):
        raise ValueError("Receipt fields do not match declared recovery information")
    if tensor_digest(edited) != meta.get("edited_tensor_sha256"):
        raise ValueError("Receipt does not match this edited tensor (stale or wrong input)")
    predicted = _base_restore(edited, payload, meta)
    if meta["recovery"] not in _CORRECTED_MODES:
        return predicted
    from .roundoff import RoundoffPatch, apply_roundoff_patch
    packet_array = np.asarray(payload["roundoff_patch"])
    if packet_array.ndim != 1 or packet_array.dtype != np.dtype(np.uint8):
        raise ValueError("roundoff_patch must be a one-dimensional uint8 packet")
    patch = RoundoffPatch.from_bytes(packet_array.tobytes())
    if patch.precision != meta["precision"] or tuple(patch.shape) != tuple(np.asarray(edited).shape):
        raise ValueError("Roundoff patch precision/shape does not match the receipt")
    if type(meta.get("recovery_exact")) is not bool or meta["recovery_exact"] != bool(patch.exact):
        raise ValueError("Receipt exactness does not match its roundoff patch")
    if meta["recovery"] in _EXACT_MODES and not patch.exact:
        raise ValueError("Exact recovery requires an exact roundoff patch")
    restored = apply_roundoff_patch(predicted, patch)
    if patch.exact and tensor_digest(restored) != meta.get("original_tensor_sha256"):
        raise ValueError("Exact receipt did not reconstruct the original tensor digest")
    return restored


def _save_new_npz(path: str | Path, arrays: Mapping[str, Any]) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Serialize first so validation errors cannot leave a partially published file.
    data = _serialize_npz(arrays)
    with destination.open("xb") as stream:
        try:
            stream.write(data)
        except BaseException:
            stream.close()
            destination.unlink(missing_ok=True)
            raise


def edit_npz(input_path: str | Path, output_path: str | Path, receipt_path: str | Path,
             *, editor: str = "rome", recovery: str = "delta", alpha: float = 0.75,
             precision: str = "float32", drop_rank: int = 1,
             patch_budget_bytes: int | None = None) -> dict:
    """Edit an NPZ request and save its selected recovery information.

    Inputs contain weight, covariance, and key/value or keys/values. Corrected
    modes require an already quantized canonical stored input; legacy modes
    preserve v1's interpretation of the supplied current state. Outputs are new.
    """
    _validate_mode(editor, recovery, precision)
    if _base_mode(recovery) == "analytic" and editor == "rome" and alpha == 1:
        raise ValueError("Hard ROME is noninjective; use delta or old_response")
    if Path(output_path).exists() or Path(receipt_path).exists():
        raise FileExistsError("Output and receipt paths must be new")
    if Path(output_path).resolve() == Path(receipt_path).resolve():
        raise ValueError("Weight and receipt output paths must be distinct")
    with np.load(input_path, allow_pickle=False) as archive:
        w, c = np.array(archive["weight"]), np.array(archive["covariance"])
        if recovery in _CORRECTED_MODES:
            _canonical_stored(w, precision, "original")
        if editor == "rome":
            result = rome_edit(w, archive["key"], archive["value"], c,
                               alpha=alpha, precision=precision)
        else:
            result = memit_edit(w, archive["keys"], archive["values"], c, precision=precision)
    receipt = make_receipt(w, result, editor=editor, recovery=recovery,
                           precision=precision, patch_budget_bytes=patch_budget_bytes,
                           drop_rank=drop_rank)
    _save_new_npz(output_path, {"weight": result.updated})
    try:
        _save_new_npz(receipt_path, receipt)
    except BaseException:
        Path(output_path).unlink(missing_ok=True)
        raise
    meta = receipt_metadata(receipt)
    # This field is a return-only observation: embedding its own byte size would
    # change the NPZ whose length it purports to describe.
    return {**meta, "serialized_receipt_bytes": Path(receipt_path).stat().st_size}


def undo_npz(edited_path: str | Path, receipt_path: str | Path, output_path: str | Path) -> dict:
    """Restore only from the edited file and its paired receipt.

    Exact corrected receipts support strict reverse-order chains, provided the
    numerical predictor reproduces its recorded bits. Approximate receipts may
    still fail the preceding receipt's hash. Wrong/stale inputs are never forced.
    """
    if Path(output_path).exists():
        raise FileExistsError("Restored output path must be new")
    with np.load(edited_path, allow_pickle=False) as archive:
        edited = np.array(archive["weight"])
    with np.load(receipt_path, allow_pickle=False) as archive:
        payload = {key: np.array(archive[key]) for key in archive.files}
    restored = restore_from_receipt(edited, payload)
    meta = receipt_metadata(payload)
    _save_new_npz(output_path, {"weight": restored})
    return {"recovery": meta["recovery"], "output": str(output_path),
            "precision": meta["precision"], "recovery_exact": meta.get("recovery_exact", False),
            "restored_tensor_sha256": tensor_digest(restored)}

