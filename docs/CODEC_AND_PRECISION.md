# Codec and native precision experiments

Optional install: `python -m pip install -e '.[compression]'`.
`make_roundoff_patch(..., strategy='xor_zstd', level=3)` and `xor_lz4`,
`xor_none`, `xor_zlib` select lossless correction payload encodings. The
existing zlib level 9 default and old RRP1 decoding remain unchanged. A reader
without optional codec packages can still read old records; reading a new
encoding requires its package. Older releases reject unknown new encodings.
Every serialized patch contains its encoding, precision, dimensions, lengths,
payload checksum and predicted/target state digests. Frames with trailing data,
wrong sizes or truncation are rejected; decode is bounded by the tensor size.

The benchmark uses a separate experimental version-1 outer envelope with codec,
level, uncompressed size and payload digest. It applies one outer codec to the
entire uncompressed NPZ record for each method, with an uncompressed inner
correction. It is not a silent replacement of existing factual-chain receipts.
Do not use the experimental envelope as an untrusted-input service endpoint.

Run from the repository with single-thread BLAS settings:

```
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
  .venv/bin/python -I scripts/codec_benchmark.py --out results/NEW_RUN
```

Copy the frozen PLAN.md to NEW_RUN first. Four local binary input fixtures are
required at the paths in run.json. Their SHA256 values are verified in the
published run; weights are not redistributed. Timings exclude disk I/O and
worker startup. RSS is combined process high-water, not isolated codec memory.
There is no throughput or end-to-end service latency claim.

The CUDA probe runs stored real CPU-author factors on actual FP32/FP16/BF16
storage, with FP32 factor products and accumulation before storage rounding.
It restores the pre-edit stored bits, not discarded pre-cast FP32 information.
It uses zlib as the established correction control, not the newly benchmarked
outer envelope. The CPU bridge and GPU copies are included in its reported
build/restore timings, but factor generation and forward passes are absent.
One pass per cell establishes a correctness probe, not a timing benchmark.

## Ordered continuation and gates

1. Codec comparison: complete, 252 observations, 756 negative checks. Keep
   defaults unchanged pending broader data; zstd 3 is the next experiment's
   write-latency candidate.
2. Native precision: storage/update preflight complete, 12 cells. Full live
   mixed-precision ROME/MEMIT optimization and edit/restore/next-edit forward
   evaluation remain to implement and run. Freeze cases before execution;
   retain all failed edits and test unrelated knowledge and general tasks.
3. Distributed: not executed. First use two ranks with explicit history version,
   staged writes and all-rank commit; inject interruption before/after commit,
   stale receipt and retries. Compare reconstructed state to fresh retained
   history in the same precision/backend. One GPU is available on the inspected
   host; two-rank CPU simulation must be labeled separately from multi-GPU work.
4. Larger models: not executed. Start 1–3B, then 7–8B and another architecture.
   Advance only after full native-editor correctness/quality results exist.
   Freeze editing histories, locality/control tasks and all cost measures.
   Compare changed-layer checkpoints, direct XOR, exact fresh replay and checked
   reuse under identical codecs. Report failures rather than filtering them.

The existing dependency-validation slowdown excluded serialization: these codec
results do not establish that checked reuse is now faster. Exact reconstruction
of a degraded edited model does not establish editing quality or general ability.
