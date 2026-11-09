# Author-algorithm factual edit chain pilot

PR #2 reviewed and merged: `2f8b529`. Numerical source/protocol frozen at `21f6aeb9213b9f9dbc283002dd815798c9d33a2e`. Author implementation: kmeng01/memit `80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b`; see [protocol](../../docs/FACTUAL_CHAIN.md) and [compatibility patch](../../provenance/factual-edit/compatibility.patch).

Both algorithms executed latent target optimization for counterfactual factual requests. MEMIT additionally distributed residuals across three layers. ROME A changes France → Rome; MEMIT A jointly changes France → Rome and Japan → Paris. B changes Italy → Tokyo, optimized again from each restored model and compared with B freshly computed from the original model.

| Algorithm | Recovery | Full model restored | Next B model identical | Next B logits max error | Total record bytes |
| --- | --- | --- | --- | ---: | ---: |
| rome | checkpoint | True | True | 0 | 8770096 |
| rome | delta | False | False | 0.000267029 | 15047 |
| rome | delta_exact | True | True | 0 | 303838 |
| memit | checkpoint | True | True | 0 | 26308211 |
| memit | delta | False | False | 0.000114441 | 164372 |
| memit | delta_exact | True | True | 0 | 846614 |

Checkpoint and exact correction passed both full-state digest and all eight probe-logit checks before and after the freshly recomputed next edit. Ordinary delta retained nonzero state/logit differences. This is one bounded sequence per algorithm, not a statistical replication or a large editing benchmark.

## Editing quality is separate from rollback

| Algorithm / stage | Probe | Target probability | Target rank | Top token |
| --- | --- | ---: | ---: | --- |
| rome / after_A | rewrite_A | 0.999671 | 1 | Rome |
| rome / after_A | paraphrase_A | 0.240472 | 1 | Rome |
| rome / after_A | rewrite_A2 | 0.000000 | 6050 | Rome |
| rome / after_A | rewrite_B | 0.000001 | 362 | Rome |
| rome / after_A | paraphrase_B | 0.009433 | 16 | Rome |
| rome / after_A | locality | 0.000019 | 10 | Rome |
| rome / reference_B | rewrite_A | 0.000000 | 485 | Tokyo |
| rome / reference_B | paraphrase_A | 0.000535 | 111 | Tokyo |
| rome / reference_B | rewrite_A2 | 0.000000 | 1178 | Tokyo |
| rome / reference_B | rewrite_B | 0.992789 | 1 | Tokyo |
| rome / reference_B | paraphrase_B | 0.484956 | 1 | Tokyo |
| rome / reference_B | locality | 0.000209 | 9 | Tokyo |
| memit / after_A | rewrite_A | 0.994743 | 1 | Rome |
| memit / after_A | paraphrase_A | 0.051824 | 3 | Paris |
| memit / after_A | rewrite_A2 | 0.990872 | 1 | Paris |
| memit / after_A | rewrite_B | 0.000000 | 698 | Rome |
| memit / after_A | paraphrase_B | 0.001553 | 79 | Paris |
| memit / after_A | locality | 0.292230 | 2 | Rome |
| memit / reference_B | rewrite_A | 0.000001 | 404 | Tokyo |
| memit / reference_B | paraphrase_A | 0.001260 | 68 | Tokyo |
| memit / reference_B | rewrite_A2 | 0.000002 | 889 | Tokyo |
| memit / reference_B | rewrite_B | 0.974036 | 1 | Tokyo |
| memit / reference_B | paraphrase_B | 0.496170 | 1 | Tokyo |
| memit / reference_B | locality | 0.000033 | 58 | Tokyo |

**Locality failed qualitatively in this setup.** ROME A pushed Germany and other countries toward Rome; reference B pushed other countries toward Tokyo for both algorithms. MEMIT A also made Rome top-ranked on the Germany probe. The targeted rewrite prompts reached their requested top token, but this does not qualify a useful factual editor. MEMIT A also missed top-1 on the France paraphrase. Baseline outputs and full probe metrics are retained rather than filtered for initially correct facts.

Base→edited KL on the Germany locality probe:
- rome-A: 8.642910 nats.
- rome-reference-B: 7.170739 nats.
- memit-A: 4.505677 nats.
- memit-reference-B: 7.114450 nats.

## Scope and evidence

This uses FP32 GPT-2 with an explicit small-model adaptation: ROME layer 5; MEMIT layers 4–6; fixed contexts; only 16 calibration sentences with ridge and a preservation multiplier of 1000. These are not the authors’ large-model/data settings. The poor locality must not be generalized to the original published methods. No tuning was performed on these results.

Actual upstream update arithmetic is preserved. Undo runs in a separate NumPy-only process, with only edited weights and the record. `results.json` records layer bytes, parent-process record construction times, undo times and worker RSS. There is only one timing sample per layer/method; parent build memory includes a live model and is not compared with prior isolated-worker build RSS.

Every optimization trace is retained as `*-optimization.txt`. Local binary artifacts include actual per-layer factors, edited/original layers, covariance and every stage’s logits; `summary.json` stores their hashes. All 76 binary hashes and 21 patched upstream Python hashes were verified. The author code is reproducibly reconstructed with `scripts/prepare_factual_upstream.py`; only its patch/license are published here.

Next scientific step: qualify preservation/calibration and known-fact/paraphrase panels under a separately declared design before interpreting edit quality. The present result establishes reversible execution of these particular optimized edits and repeatability of the next edit, alongside substantial editing side effects.
