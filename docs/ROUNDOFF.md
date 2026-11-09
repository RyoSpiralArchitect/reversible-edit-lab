# v0.2: exact stored-state correction

## Goal and information contract

The v0.1 implementation showed numerical drift after repeated edit/undo cycles,
even when the real-arithmetic update was known. Version 0.2 records the finite
storage information needed to correct a predicted rollback. It compares this
record with a compressed checkpoint, including all serialization overhead.

The writer can inspect the original state when making the record. The undo
procedure receives only the edited state and that record. The patch explicitly
contains information from the old state. This is useful when instrumenting
future edits; it cannot reconstruct an unknown unlogged editing history.

These experiments use synthetic fixed-target ROME/MEMIT matrix writes. The
algebraic scope and the nonidentifiability of hard ROME without old-state
information remain as stated in [METHODS.md](METHODS.md).

## Predictor plus lossless correction

Let `Q_p` round to the declared storage precision, and `B_p` return its canonical
stored bit words. For a saved low-rank update `U @ V.T`, compute:

```python
edited = Q_p(original + U @ V.T)
predicted = Q_p(edited - U @ V.T)
correction = compress(B_p(original) ^ B_p(predicted))
```

During undo, recompute `predicted` from the edited weight and the saved factors.
Then:

```python
restored_bits = B_p(predicted) ^ decompress(correction)
restored = B_p_inverse(restored_bits)
```

The same construction applies when `predicted` comes from the analytic inverse.
The correction then covers both numerical inverse error and weight-write
rounding. Exactness follows from XOR cancellation, conditional on reproducing
the prediction bits and decoding the patch intact. It does not depend on a
favorable compression ratio.

The implementation checks the prediction digest before applying a patch and
checks the restored target digest for exact patches. Wrong inputs or changed
arithmetic fail explicitly. A hash binds values and checks equality; it is not
an authenticity signature or an error-correction mechanism by itself.

## Representation

- float64, float32, and float16 use canonical little-endian stored words.
- bfloat16 uses the high 16 bits of its exactly representable float32 carrier.
- Signed zeros are preserved and compared as bits.
- Nonfinite values and nonrepresentable declared inputs are rejected.
- Zero mismatch uses an identity patch, whose metadata still costs space.
- Full correction uses zlib level 9 on the wordwise XOR bytes.
- Patches retain immutable bytes and scalar metadata, not original arrays.

The record contains shape, precision, format, counts, prediction and target
digests, and the correction data. The `RRP1` packet header makes its serialized
size explicit. Decompression is bounded to the expected shape; malformed,
truncated, or trailing compressed data is rejected.

## Sparse budget control

The partial alternative selects elements with the largest absolute numerical
rollback errors, with flattened index order breaking ties. It saves the index
and original/predicted XOR word for each selected element. Unselected elements
remain at their predicted values.

An element is either corrected completely or left unchanged. Its correction
data cost includes the smallest supported index width and one stored word.
The user-supplied budget bounds this data payload; it does not bound metadata,
saved factors, or the final NPZ container. Signed-zero mismatches can be fixed
when budget permits even though their numerical error is zero.

Large numerical errors can be sparse while bit errors are frequent and highly
compressible. Therefore sparse correction need not beat compressing all XOR
words. Both methods are measured, rather than assuming a smaller number of
stored entries implies a smaller complete record.

## Receipts and exact reverse chains

New modes are `delta_exact`, `analytic_exact`, and `delta_sparse`. Schema-v1
receipts still decode; new receipts use schema v2. Each receipt contains the
declared base inverse inputs and, when applicable, one correction packet.
Undeclared arrays are rejected. The receipt does not contain a dense original
matrix.

`delta_exact` supports hard ROME because its saved update factors retain the
lost response information. `analytic_exact` retains the base inverse's scope:
soft ROME with `alpha < 1`, or a fixed-target single-matrix MEMIT update. It does
not make a nonexistent hard-ROME analytic inverse identifiable.

For updates A then B, undo B with B's receipt, then undo A with A's receipt. Each
exactly restored intermediate weight satisfies the preceding receipt's input
digest. A change of numerical environment that changes prediction bits may
prevent this chain, even if the analytical formula is equivalent. Cross-BLAS or
GPU reproducibility is not established by the included CPU experiment.

Selective deletion of A while retaining the consequences of B is a different
operation. If B's inputs depend on A, replay B after restoring the appropriate
state. Undoing weights does not automatically undo optimizer, RNG, cache, or
external-memory state in a larger learning system.

## Cost accounting

The single-write sweep records:

| Quantity | Definition |
| --- | --- |
| `base_numeric_payload_bytes` | Saved factors or analytic metadata arrays/scalars |
| `correction_payload_bytes` | Compressed XOR data or sparse index/word data |
| `correction_metadata_bytes` | Serialized patch fields and header |
| `correction_packet_bytes` | Complete patch packet |
| `receipt_npz_bytes` | Full validated, compressed NPZ receipt |
| `checkpoint_raw_storage_bytes` | Original shape times logical precision width |
| `checkpoint_npz_bytes` | Full compressed checkpoint with precision, shape, and digest |

The checkpoint comparator stores BF16 as actual uint16 words, avoiding an
inflated float32-carrier baseline. It uses the same NumPy compressed NPZ
container family. There is no assumed free reuse or amortization of a dense
MEMIT covariance. Container field counts and semantic validation metadata
differ because the two formats represent different objects; their actual
serialized bytes are the reported quantity.

The current analytic MEMIT receipt retains the full covariance. Its large size
is a property of this representation, not a lower bound for analytic recovery.
For fixed keys, a future compact inverse could precompute and save `solve(C,K)`
along with `K` and `V`. Changing the predictor would require making and testing
new correction packets. The measured delta receipt already avoids storing `C`.

Active-loop byte counters use base numeric fields plus the serialized patch,
without the base NPZ wrapper. They are not directly interchangeable with the
single-write sweep's complete NPZ sizes. Cycle totals count all records ever
generated; the loop keeps one undo record at a time and discards it after use.

Evaluation copies and compression workspaces are present. This implementation
does not measure or claim a process-memory advantage or a speedup. A random,
high-entropy XOR correction can approach dense storage and exceed a checkpoint
once metadata is included. The observed size advantage is conditional.

## Reproducible comparisons

`refine` uses three seeds, float32/float16/BF16 storage, and 12×24 and 64×128
matrices by default. It compares ROME strengths 0, .5, .95, .999, and 1, and two
MEMIT preservation scales. The main capacity comparison excludes the no-write
`alpha=0` control.

The repeated experiment runs 512 edit/undo pairs at `alpha=.95`. Every receipt
is made from that method's current live state. Approximate drift remains live;
the evaluator never resets it with the original. Both exact stepwise recovery
and a subsequent identical update are checked.

The active experiment keeps the v0.1 expected-entropy acquisition objective and
label budget. It compares candidate scores, rankings, full query sequences,
and final-state digests against a checkpoint control. A changed restoration
method is not itself evidence of a better acquisition objective. Random and
uncertainty baselines remain available in the raw results.

## Relationship to prior work

The local writes follow ROME and MEMIT; see the primary references in
[references.bib](../references.bib). Retaining information discarded by finite
arithmetic to make learning reversible has prior work, including
[Maclaurin, Duvenaud, and Adams (2015)](https://proceedings.mlr.press/v37/maclaurin15.html).
The XOR-plus-compression construction is an engineering baseline here. No claim
of novelty for reversible learning, information conservation, or XOR encoding
is made. The measured questions are its cost and restoration fidelity for the
declared editing and subsequent-learning procedures.
