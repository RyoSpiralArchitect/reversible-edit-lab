# Frozen codec comparison

Before execution: four existing factual-chain A edit tensors (ROME layer 5;
MEMIT layers 4,5,6), all retained regardless of editing efficacy. Three repeats,
seed 20261110 shuffled trial order; no warmup discard. Codecs: none, zlib 1/6/9,
zstd 1/3, LZ4 0. Methods: changed-layer checkpoint, direct XOR, factors plus
exact roundoff correction. Same uncompressed NPZ envelope then ONE outer codec
for all methods. Factors correction uses xor_none, avoiding double compression.
Existing default format stays unchanged; this is a new experimental envelope.

Build includes predictor, hashes, envelope serialization and compression. Restore
includes decompression, envelope parsing, state validation, predictor and correction.
Input loading, worker startup and file I/O are excluded and must not be described
as service end-to-end latency. Fresh process per observation; combined build/restore
process peak RSS includes inputs and interpreter. Exact stored bits required in every
trial; wrong edited state, truncated packet, and a flipped byte must be rejected.
No new optimizer calls or factual efficacy claims. All codec versions, source and
input hashes are recorded. Results cannot explain the previous dependency-check
slowdown whose measurement excluded serialization.

Next gates: native GPU low-precision arithmetic and model quality, then coordinated
shards and interrupted commits, then larger model/full inference. Matrix-only probes
must not be presented as full ROME/MEMIT or distributed service benchmarks.
