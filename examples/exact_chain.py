"""Two hard-ROME writes and exact reverse-order BF16 file restoration.

Run: python examples/exact_chain.py --out demo_chain
The output directory must be new or empty. Original request archives and the
intermediate edited-A file are removed before undo; only their expected digests
remain for validation. Correction receipts contain declared old-state data.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from reversible_edit.core import quantize
from reversible_edit.receipts import edit_npz, tensor_digest, undo_npz


def _new_request(path: Path, **arrays: np.ndarray) -> None:
    with path.open("xb") as stream:
        np.savez_compressed(stream, **arrays)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    output = args.out
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        parser.error("--out must be a new or empty directory; existing files are preserved")
    output.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    original = quantize(rng.normal(size=(16, 32)), "bfloat16")
    original[0, 0] = -0.0
    expected_original = tensor_digest(original)
    covariance = np.eye(32)
    request_a, request_b = output/"request_a.npz", output/"request_b.npz"
    edited_a, edited_b = output/"edited_a.npz", output/"edited_b.npz"
    receipt_a, receipt_b = output/"receipt_a.npz", output/"receipt_b.npz"

    _new_request(request_a, weight=original, covariance=covariance,
                 key=rng.normal(size=32), value=rng.normal(size=16))
    meta_a = edit_npz(request_a, edited_a, receipt_a, editor="rome",
                       recovery="delta_exact", alpha=1.0, precision="bfloat16")
    with np.load(edited_a, allow_pickle=False) as archive:
        after_a = np.array(archive["weight"])
    expected_after_a = tensor_digest(after_a)
    _new_request(request_b, weight=after_a, covariance=covariance,
                 key=rng.normal(size=32), value=rng.normal(size=16))
    meta_b = edit_npz(request_b, edited_b, receipt_b, editor="rome",
                       recovery="delta_exact", alpha=1.0, precision="bfloat16")

    # These are exactly the three files created by this example above. Never
    # delete arbitrary existing contents or follow a directory-wide glob.
    removed = (request_a, request_b, edited_a)
    for path in removed:
        path.unlink()
    del original, after_a
    if any(path.exists() for path in removed):
        raise RuntimeError("An original/intermediate archive unexpectedly remains")

    restored_a = output/"restored_a.npz"
    restored_original = output/"restored_original.npz"
    undo_b = undo_npz(edited_b, receipt_b, restored_a)
    undo_a = undo_npz(restored_a, receipt_a, restored_original)
    after_a_exact = undo_b["restored_tensor_sha256"] == expected_after_a
    original_exact = undo_a["restored_tensor_sha256"] == expected_original
    if not after_a_exact or not original_exact:
        raise AssertionError("Exact receipt chain failed tensor digest validation")

    evidence = {
        "precision": "bfloat16", "editor": "rome", "alpha": 1.0,
        "shape": [16, 32], "recovery": "delta_exact",
        "forward_order": ["A", "B"], "undo_order": ["B", "A"],
        "removed_before_undo": [path.name for path in removed],
        "expected_after_a_sha256": expected_after_a,
        "restored_after_a_sha256": undo_b["restored_tensor_sha256"],
        "expected_original_sha256": expected_original,
        "restored_original_sha256": undo_a["restored_tensor_sha256"],
        "after_a_bit_exact": after_a_exact,
        "original_bit_exact": original_exact,
        "receipt_a_file_bytes": meta_a["serialized_receipt_bytes"],
        "receipt_b_file_bytes": meta_b["serialized_receipt_bytes"],
        "old_state_information": "FP64 update factors plus compressed roundoff correction; no dense original array in the receipt",
        "arithmetic_requirement": "The rollback predictor must reproduce its recorded logical bits; mismatches are rejected",
    }
    with (output/"evidence.json").open("x", encoding="utf-8") as stream:
        json.dump(evidence, stream, ensure_ascii=False, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(evidence, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
