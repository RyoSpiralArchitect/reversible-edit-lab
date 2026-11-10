# Dependency decomposition and longer counterfactual replay

Freeze this protocol and runner in git before the first model query. This is a
development study, using the previous pinned CPU FP32 GPT-2, author ROME/MEMIT,
seed 7, fixed contexts/covariance and 20-step settings. No efficacy filtering.
PR #5 remains the prerequisite; this work does not silently merge it.

## 1. Decompose B after A

Reuse the three pre-existing A/B pairs, both algorithms. Capture B's optimized
latent target separately from its key-side calculation. In ROME the final value
factor is **not** a latent target: subtract the current output and normalize by
the current key dot product every time. In MEMIT recompute live residuals and
distribute them over layers in the author's original order.

From the same AB tip, exactly rewind B then A using bound receipts before each
arm. Compare to freshly computed B-only: fresh target/key; old target/fresh key;
fresh target/old key; old target/old key; old whole update; dependency-checked
reuse. Stale arms are explicitly unsafe diagnostics. Key-side reuse includes
MEMIT's adjusted-key solve. Measure latent/key/solve differences individually.
Run unadapted author code too: no-cache instrumentation must match it bitwise.

## 2. Check dependencies and measure total cost

The experimental cache binds request, hparams, contexts, covariance, tokenizer,
model configuration, source files, runtime versions and seed. Target optimization
depends conservatively on every model tensor. GPT-2 c_proj inputs depend on
embeddings, all earlier blocks, and the same block through c_fc, excluding its
c_proj. Include model buffers. Key-side solves depend on their actual key and
covariance arrays. Hash live bytes at each lookup; no output-similarity heuristic.
Unknown tensor names fail closed. This argument is specific to the pinned eager
GPT-2 graph, eval mode, single writer and immutable non-tensor configuration.

Verify reuse against fresh computation; negative controls perturb a dependency,
request, covariance or cached value. An internal certificate is not an adversarial
signature or a cross-platform guarantee. Store value digests and reject corruption.
Three paired timing repeats per six cells, alternating order, compare unadapted
fresh B with checked reuse. Include dependency hashing and lookup in edit latency;
measure undo separately. Exclude measurement-only hashing, probe evaluation and
artifact I/O from both arms. Report cache creation and payload bytes separately;
do not call it faster unless measured including the check. Retain negative results.

## 3. Replay longer histories

Predeclared 16 requests (all experimental counterfactuals): Spain→Oslo,
Canada→Rome, Dickens language→French, Japan→Paris, Spain→Tokyo, Germany→Rome,
Italy→Berlin, France→Oslo, Canada→Paris, Japan→Rome, Dickens language→Italian,
Italy→Tokyo, France→Berlin, Germany→Paris, Spain→Rome, Canada→Tokyo.
Use capital relation except the two explicit language edits. Lengths 8 and 16;
delete indices 0, length//2, length-2 (zero based), separately for each algorithm.
No selection based on prior success. These 12 withdrawals also bound the cost
diagnostic if stage 2 finds no speed gain; scaling is not conditional on a desired
positive result and no larger-model run is implied by this protocol.

Reference: start from W0 and freshly execute the retained requests in order.
Candidate: start from the original history tip, reverse receipts through the
deleted edit, check the exact prefix state, then replay only its surviving suffix
using old per-request component caches under the dependency check. Negative
control: same rewind, blindly reapply old whole updates. All starts/rewinds must
have the same full state; prefix checkpoints are evaluator-only references.

### Amendment before the first long-history query

The initial run at `352fb0f` was interrupted during the fifth decomposition cell;
four complete cells and the partial logs/artifacts remain in
`results/20261110_dependency_replay/`. Inspection found that unsafe diagnostic
arms labelled copied old values with the live dependency in their saved metadata.
Those diagnostic outputs were never admitted to the checked arms; numerical
measurements are unaffected, but that metadata would be unsafe to reuse later.
Keep each old value's original dependency and add a regression test for subsequent
checked lookup. Start a new full run and compare all retained complete-cell
observations, excluding wall times. No panel/hyperparameter changes.

Also add an unadapted-author `fresh_suffix` arm for all 12 long-history withdrawals,
starting from the same exact rewind as checked reuse. This separates skipping an
unchanged prefix from savings due to cached components. Long-history timings are
single observations with fresh_suffix first, not a statistical speed benchmark.
Byte-identical artifacts may share hard links, without reusing numeric computation.

Report full tensor-state equality, logits, greedy generations, target word-prefix
success and locality/control behavior separately. Equality to a degraded reference
does not establish editing efficacy, general ability or forgetting/privacy. Keep
all failed edits and withdrawals. Save hashes, small component arrays and exact
receipts locally; publish logs/metrics/manifests, not pretrained weights. Account
for journal plus component cache, and distinguish measurement artifacts/shared
base-model resources. No SOTA or established novelty claim.
