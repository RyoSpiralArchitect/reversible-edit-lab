# Experiment sequence

1. **Imported baseline:** preserve v0.1/v0.2 evidence; reproduce numerical tests
   on macOS and Linux. The original source remains in the import commit.
2. **Resource pilot:** measure serialization, restoration time, and process peak
   memory at larger matrix sizes using [COSTS.md](COSTS.md). Keep the three
   outcomes separate and record negative results.
3. **Real-weight pilot (bounded run completed):** GPT-2 final projection with
   actual single-token loss gradients; see [results](../results/20261110_gpt2/RESULTS.md).
   The implemented procedure pins a small accessible model revision,
   verify projection orientation against forward computation, then measure
   exact rollback on its stored tensor. Distinguish synthetic writes on real
   weights from actual optimized factual edits.
4. **Actual edit/rollback/next-edit comparison (bounded run completed):**
   [Author ROME/MEMIT pilot](../results/20261110_factual_chain/RESULTS.md) preserves
   exact rollback and next-edit equality but shows major locality failures.
   Retain key/value
   generation, update factors, prompts, and forward outputs. Compare target,
   paraphrase, locality, and next-edit behavior with a full-state reference.
5. **Fresh arbitrary information (bounded run completed):**
   [Novel association pilot](../results/20261110_novel_information/RESULTS.md)
   compares unprompted retrieval before/during/after temporary weight edits.
   Exact rollback restores state in all six cases; actual target retrieval is
   mixed, and ordinary subtraction also hides the answer despite residual weight
   differences. Retain word-extension scoring failures and locality spillover.
6. **Selective withdrawal (bounded dependency diagnostic completed):**
   [Six actual-editor chains](../results/20261110_selective_replay_rerun/RESULTS.md)
   separate rounding from later-update dependence. Exact rewind with stale B
   matches the B-only state in 0/6; fresh B recomputation matches in 6/6.
   Retained B target answers alone fail to expose that distinction.
   [Prior-work positioning](PRIOR_WORK_20261110.md) explicitly credits reversible
   learning, selective spectral reversal, SISA and lossless delta compression.
7. **Verified partial replay and learning policy (planned):** test whether
   necessary recomputation can be reduced with a checkable dependency condition;
   evaluate acquisition-policy value independently. This is a research hypothesis,
   not an established new algorithm or a qualified performance result.

No real-model factual-editing, training-quality, or cross-device transport result
is established by steps 1–2. Experimental changes are proposed in pull requests
with raw evidence and validation results.
