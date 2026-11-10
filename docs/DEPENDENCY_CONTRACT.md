# What the experimental reuse check means

The check is a sufficient dependency condition for this pinned execution graph,
not a semantic similarity score. It does not infer that two edits are independent
because their subjects or relations differ. The runtime must remain the declared
CPU FP32 eager GPT-2, with eval mode, fixed tokenization, contexts and covariance,
fresh seed-7 target optimizers and a single writer. The adapter retains the
author's expression order, output subtraction and normalization.

## Dependency boundaries

For the input to GPT-2 block `l`'s `mlp.c_proj`, all earlier blocks, embeddings,
and block `l` through `mlp.c_fc` are ancestors. Its own `c_proj` and later blocks
are not ancestors. Changing `c_proj` at the same layer therefore does not change
these key inputs, **provided the input tokens and contexts stay fixed**. This
does not apply to contexts regenerated from the edited model.

The ROME layer-5 key can consequently be reused after earlier layer-5 edits.
In MEMIT's layer-4/5/6 execution, the layer-4 key has the same boundary. A change
to layer 4 can affect layer-5 and layer-6 keys. Those keys must be checked against
the current model at their actual point of use, after preceding within-edit
writes. The adjusted-key linear solve depends on the key array, frozen covariance
and hparams; it can be reused when those inputs match. The latent target optimizer
is conservatively bound to every model tensor because its objective runs through
the model. The current output/residual is always recomputed.

`prefix_dependency` covers named parameters and buffers, rejecting unknown
tensor names. The contract also binds requests and declared non-tensor inputs.
Every lookup hashes current dependency bytes; a hit verifies the stored value
digest. SHA-256 comparisons assume collision resistance. This is not a machine-
checked proof, an adversarial signature, or protection against a changed runtime
outside the declared model. Immutable settings are a harness assumption.

## Why exact replay is a useful reference

After withdrawing request A, let a fresh run of the retained request sequence be
the reference. Exact receipt traversal recovers the original prefix before A.
That prefix matches the fresh reference's prefix because A has not occurred yet.
If each reused subcomputation has identical inputs and all others are recomputed
with the same deterministic arithmetic, the next weight update is the same.
Repeating that argument over the retained suffix gives the expected exact state.
The experiment tests this expectation against actual full-state digests and
retained arrays, rather than treating the argument alone as a successful result.

The reference can itself give bad answers. Reproducing it does not establish
healthy general abilities, high-quality editing, privacy, removal of pretraining
information, or invisibility of the withdrawn answer. Retained edits may cause
the same answer through their own effects.

## Accounting and prior work

Hashing large prefixes can cost more than recomputing a short forward pass.
Check costs belong in the timing comparison, and cached components belong in
storage totals. The 12 long-history cases additionally compare with an unadapted
fresh suffix from the same rewind, so skipping a prefix is not attributed to
component caching. Timings are local development observations, not a benchmark
against other editing or unlearning systems.

Dependency-guided reuse is established in
[self-adjusting computation](https://arxiv.org/abs/1106.0478), and the
[ROME author code](https://github.com/kmeng01/memit/blob/80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b/rome/rome_main.py)
already recognizes that prior-layer edits affect later-layer caches. The present
contribution is a bounded experimental application and accounting contract. Its
methodological novelty and practical advantage remain unestablished; see the
[broader positioning](PRIOR_WORK_20261110.md).

## Reproduction

Use Python 3.11+ and the optional Torch/Transformers environment and local pinned
model/upstream prepared for [the factual pilot](FACTUAL_CHAIN.md). The runner
checks model, author-source and covariance hashes before querying. A new output
directory is required. The checked cache is an experiment harness, not exported
as a general editing API.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -I scripts/dependency_replay.py --model-dir work/models/gpt2 \
  --upstream work/upstream-memit-cpu --out results/my_dependency_replay

OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 \
.venv/bin/python -I scripts/audit_dependency_arrays.py results/my_dependency_replay

python3 -I -S scripts/report_dependency_replay.py results/my_dependency_replay \
  --model-dir work/models/gpt2 --upstream work/upstream-memit-cpu

.venv/bin/python -I scripts/plot_dependency_replay.py results/my_dependency_replay \
  --out work/dependency-overview.png
```

The plot additionally requires the optional `plots` dependencies. The array
auditor uses NumPy without loading the model; it rechecks saved comparisons and
replays actual history receipts, not fresh model optimization or generation.
