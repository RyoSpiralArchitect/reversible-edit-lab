# GPT-2 actual-gradient rollback pilot

PR #1 was reviewed and merged at `32af8bc`. This run used preregistered numerical source `687c712ac70e06eb49d6a964258d6a9414133007` and the protocol in [GPT2_PILOT.md](../../docs/GPT2_PILOT.md).

81 observations (9 fixtures × 3 methods × 3 timing repeats), zero worker failures. Exact correction: 27/27 bitwise restores and 9/9 restored next-token logit vectors identical. Checkpoint: same counts. Delta: 0/27 bitwise restores and 0/9 identical logit vectors. Timing repeats are not additional independent inputs.

| Precision | Exact bytes / checkpoint | Build time ratio | Undo time ratio | Undo peak RSS ratio |
| --- | ---: | ---: | ---: | ---: |
| float32 | 0.50% | 0.42× | 1.42× | 1.64× |
| float16 | 0.96% | 0.82× | 2.58× | 2.16× |
| bfloat16 | 1.15% | 0.74× | 3.88× | 2.65× |

Ratios are paired medians across three prompts and three repeats on one macOS arm64 host. All phases use the unchanged NumPy resource workers from PR #1. Process RSS includes imports/input preparation; model forward/backward is outside timed workers.

The target is GPT-2 final MLP projection (768×3072), pinned revision `607a30d783dfa663caf39e06633721c8d4cfcd7e`. All six local model/tokenizer files were checked against upstream Git blob or LFS hashes. Autograd outer-product relative errors were below 2.6e-8; projection forward absolute errors below 8.4e-5.

Updates use real input activations and actual next-token loss gradients, normalized to a 0.001 relative Frobenius step before storage rounding. Target probabilities increased on these three directly optimized prompts; this is not evidence of generalization or factual-editing quality. FP16/BF16 affect only projection storage; the full model forward runs in float32.

Compared with the prior synthetic pilot, update magnitude and source distribution both changed. The much smaller correction records cannot be attributed to pretrained weights alone. Restoration still takes longer and more process memory than checkpoint loading. This is not complete ROME/MEMIT value optimization, a multi-layer edit, or a next-training-step equivalence test.

Raw measurements: `observations.jsonl`; gradient/orientation and target diagnostics: `fixtures.json`; restoration logits: `behavior.json`; complete groups/min/max: `summary.json`. Actual factors, records and forward outputs remain locally under ignored `work/`; their hashes are in [the audit](../../validation/gpt2-pilot-audit.json). Model weights remain in ignored `work/models/gpt2` under the upstream MIT license.

A later guard checks the pinned weight SHA before starting future runs; the retained numerical run predates that guard and was independently audited against the same upstream hash.
