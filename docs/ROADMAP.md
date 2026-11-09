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
4. **Actual edit/rollback/next-edit comparison (planned):** retain key/value
   generation, update factors, prompts, and forward outputs. Compare target,
   paraphrase, locality, and next-edit behavior with a full-state reference.
5. **Selective withdrawal and learning policy (planned):** evaluate dependent
   update replay and acquisition-policy value independently of exact rollback.

No real-model factual-editing, training-quality, or cross-device transport result
is established by steps 1–2. Experimental changes are proposed in pull requests
with raw evidence and validation results.
