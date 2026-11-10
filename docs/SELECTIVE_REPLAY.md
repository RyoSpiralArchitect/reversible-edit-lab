# Removing A after A → B: fixed development protocol

Freeze this document, runner and panel before the first query. This is an
exploratory dependency diagnostic, not a confirmatory benchmark or a test of
general intelligence. Same pinned CPU FP32 GPT-2, patched authors' ROME/MEMIT,
20-step optimization settings and frozen base-model covariance as the factual
pilot. Each request is a single edit for both algorithms. No hyperparameter
tuning or filtering by edit success.

Three new pairs, each tested with ROME and MEMIT:

1. Different subjects, same relation: Spain capital → Oslo; Canada capital → Rome.
2. Conflicting targets, same subject: Spain capital → Oslo; Spain capital → Tokyo.
3. Different relations: Charles Dickens native language → French; Canada capital → Rome.

These counterfactual targets are experimental instructions, not assertions
about the world. Baseline target knowledge is measured without success filtering.
Pairs are deliberately developmental; no population-level conclusion from six.

Compute fresh B-only from W0 as a reference. Reset the evaluator to base, then
compute A and compute B freshly from WA, retaining both write-time receipts.
Evaluate the final WAB and the following withdrawal arms:

- `subtract_A`: subtract the A factors from WAB without the binding of an exact
  receipt. This is explicitly an unsafe arithmetic baseline; attempting to apply
  the actual A exact receipt to the wrong edited state must be rejected.
- `exact_rewind_stale_B`: undo B, then A with exact receipts; verify full W0;
  reapply the actual stored B update learned on WA, including author dtypes.
- `exact_rewind_recompute_B`: same rewind; discard old B targets/factors and
  rerun the authors' optimizer with seed 7 on W0.
- `checkpoint_recompute_B`: rewind B then A using checkpoint records; fresh B.

All arms start from the same WAB. Independent NumPy undo workers receive only
the live edited layer and its bound record. The evaluator keeps a base model
copy for reference comparisons and separate experiment resets, never to repair
an exact undo. Fresh optimizers are recreated for each E call. Context templates,
tokenizer and base-model covariance are declared fixed inputs; this pilot does
not cover adaptive covariance caches or persistent optimizer state.

Primary measurements: full-state digest equality to B-only, per-layer relative
weight error, first-token maximum logit error and mean KL(reference || arm),
all greedy continuations (six new tokens maximum), target first-token probability/rank and correct-word
prefix for eight fixed probes. The target rule requires a word boundary after
the desired answer; token-prefix-only success is not used. Four additional
fixed control prompts expose other changes. Full B-only equivalence and
editing efficacy are separate: a method may reproduce an ineffective edit.

Measure the norm of delta_B(WA) - delta_B(W0) separately from rollback error.
Record actual serialized bytes of factors + exact correction, compressed layer
checkpoints and direct XOR + zlib of old/edited layer bits (no factor predictor).
Verify the direct XOR inverse too. This is a matched-compressor ablation,
**not a reproduction of ZipNN/ULPack or a state-of-the-art claim**.
Time undo and fresh optimization separately; six single runs cannot establish
speed advantages. Retain logs, hashes and failed cells without overwriting.

Interpretation: exact rewind + fresh B should match B-only. A nonzero difference
for stale B after a verified rewind isolates dependence on the prior edit.
Naive subtraction mixes that effect with finite-precision error. If all stale
updates match, retain this negative result rather than changing the pairs.

Run after preparing the same optional model environment and author-code copy:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -I scripts/selective_replay.py --model-dir work/models/gpt2 \
  --upstream work/upstream-memit-cpu --out results/my_selective_replay
```

Python 3.11+, Torch and Transformers are optional dependencies. Base covariance
is reused only after checking the previous run's hash, or recomputed from its
declared neutral sentences if local binaries are absent.

## Storage-only amendment after an incomplete run

The first run at `eb143ba` completed five chains and two arms of the sixth,
then failed with `OSError: [Errno 28] No space left on device` while saving the
MEMIT different-relations fresh-B factors. Its metrics, optimizer logs, failure
log and binary hashes remain in `results/20261110_selective_replay/`; it is
explicitly marked incomplete, including the partially written artifact.

To finish without deleting evidence, verified byte-identical generated NPZ
files now share hard links. Numerical arrays are computed again and serialized
in memory before hashing; a match only avoids storing duplicate bytes. Existing
paths are never overwritten by the runner. No hyperparameter, pair, score or
withdrawal calculation changes. All six chains are rerun into a new directory,
with optional `--reuse-artifacts <previous work dirs>` supplying duplicate-file
indices. The restart is a storage fix, not a successful-case filter.

Report accounting also displays the single W0 checkpoint size: a dedicated
withdrawal implementation could reset straight to that checkpoint rather than
traverse two checkpoint records. Journal totals must not be mistaken for the
best possible checkpoint baseline or for total process memory.

The [completed rerun](../results/20261110_selective_replay_rerun/RESULTS.md)
retains all six chains. All 69 comparable measurement groups from the incomplete
run match exactly (timings excluded); all 365 binary hashes verify. The report
can be regenerated with:

```bash
python3 -I -S scripts/report_selective_replay.py results/20261110_selective_replay_rerun \
  --model-dir work/models/gpt2 --upstream work/upstream-memit-cpu \
  --prior-incomplete results/20261110_selective_replay
```
