# Temporary access to newly assigned information

Fixed panel: three fresh OS-random 48-bit project names, each assigned two words
drawn from declared 16 × 16 adjective/noun pools. The new information is the
arbitrary assignment, not novel vocabulary. These are public experimental
strings, not real credentials or high-entropy secrets. Freeze the panel and
protocol in Git before any model query. No case selection by success.

Use the same pinned GPT-2 FP32 and patched authors' ROME/MEMIT source as
[the factual pilot](FACTUAL_CHAIN.md). Each algorithm edits each case singly,
with the previous 20-step optimizer, layers, contexts and covariance settings
unchanged. Reuse hash-verified original-model covariance when available, or
recompute it from the declared 16 sentences. This is a small-model adaptation,
not a reproduction of the original benchmarks.

For every case ask the identical cloze question, with no answer in the prompt:
`The access phrase for Project <nonce> is`. Decode greedily for eight new tokens
(or target length if longer). Retrieval requires that the generated token prefix
equal the complete target phrase. Save the entire continuation and token IDs;
mentioning the phrase later does not count. A separate teacher-forced sequence
probability is diagnostic and does not establish retrieval. Never supply earlier
answers, demonstrations, caches, retrieval, or inference-time hooks to decoding.

Stages: original → actual weight edit → restore → identical question again.
Perform a no-edit repeat first. All three target questions, three unoptimized
paraphrases, and 12 fixed arithmetic/sequence/antonym/completion/factual controls
are asked at every stage. Score controls against declared targets, retain baseline
failures, and report retention only among baseline successes. Also compare full
continuations and first-token distributions; this small panel cannot establish
that all intelligence is normal. Questions about the other two project names
expose collateral effects on similar entities.

Compare checkpoint, plain factor subtraction, and exact corrected subtraction,
starting each arm from the same edited state. NumPy-only restoration workers
receive only the edited layer and the declared record. The exact record contains
old-state information as a compressed roundoff correction. The parent evaluator
retains baseline weights for record construction and measurement but never uses
them to reset the model during undo. Following exact undo, proceed to the next
case directly, with a fresh optimizer. A failed exact state check stops the chain.

Keep two conclusions separate:

1. Behavioral cycle: target absent before, retrieved while edited, absent after.
2. Restoration: full model-state bytes, all probe first-token logits and greedy
   continuations equal their baseline. Failure to answer alone proves neither
   restoration nor absence of residual information, as the plain-delta arm tests.

Do not tune on these cases or replace failures. Any follow-up tuning needs a
separately labeled diagnostic run. Retain source/panel/model/upstream hashes,
optimizer traces, per-stage metrics, all factors and correction records. Large
binary arrays stay in ignored `work/`; publish their hashes and reproducible code.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -I scripts/novel_information.py \
  --model-dir work/models/gpt2 --upstream work/upstream-memit-cpu \
  --out results/my_novel_information
```

Requires Python 3.11+, Torch, Transformers and the optional factual-edit runtime.
Prepared source and checkpoint instructions are in the preceding pilot docs.

Primary method sources: [ROME](https://rome.baulab.info/) and
[MEMIT](https://memit.baulab.info/); source revision and compatibility changes
are recorded in `provenance/factual-edit/` and the run manifest.

## Recorded run and later scoring diagnostic

[The fixed run](../results/20261110_novel_information/RESULTS.md) is complete.
After inspecting outputs, a stricter phrase-boundary diagnostic was added:
`frozen sparrows` passes the predeclared token-prefix test for `frozen sparrow`,
but fails the complete-word test. Both scores and all outputs are retained.
This is a post-run diagnostic; no hyperparameters or cases were retuned.
Generate the report and hash audit with Python 3.11+:

```bash
python3 -I -S scripts/report_novel_information.py results/20261110_novel_information \
  --model-dir work/models/gpt2 --upstream work/upstream-memit-cpu
```
