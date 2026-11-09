# Reversible edit sandbox: methods and interpretation

## Scope

This sandbox studies recovery of a **linear weight matrix** after an edit.
It is a NumPy experiment, not an implementation of a complete language-model
editor. The formulas come from the local write operations in ROME and MEMIT;
the data, controlled matrices, recovery comparisons, and learning loops are
experimental constructions for this project.

The useful question is: **what information, precision, and update strength are
needed to make trying an update and withdrawing it dependable?** An improvement
on these synthetic tasks does not establish factual editing, language-model
generalization, or a benefit for deployed test-time training.

## What “without the original” means

These are different access conditions and must be reported separately:

| Condition | Recovery procedure may access |
| --- | --- |
| Full-checkpoint oracle | The original matrix; evaluation/control only |
| Update journal | The edited matrix and saved update factors or old responses |
| Corrected journal/inverse | The edited matrix, base inverse inputs, and a compressed old-state bit correction |
| Fixed-target inverse | The edited matrix, keys, targets, and covariance metadata |
| Weights only | The edited matrix and a policy fixed before evaluation |

“No full checkpoint” does not mean “no auxiliary information.” In particular,
keys, targets, an old response, covariance statistics, a learned recovery model,
or an update log are all auxiliary information. The recovery method's access
condition is part of the result, not an implementation detail to omit.

## Notation

Let `W` have shape `(output_dim, input_dim)`. Column vectors `k` and `q` have
length `input_dim`; `v` has length `output_dim`. For a batch of `b` edits, `K`
has shape `(input_dim, b)` and `V` has shape `(output_dim, b)`. Transpose is
written `T`. All formulas below initially refer to exact real arithmetic.

`C` is a symmetric positive-definite preservation matrix. Its scale includes
any preservation coefficient; it is not an implicitly free identity matrix.
Solves should be used in code instead of explicitly forming matrix inverses.

## Hard ROME: the write loses a response direction

For fixed `k`, `v`, and `C`, define

```text
q = solve(C, k) / (k.T @ solve(C, k))
W_edit = W + (v - W @ k) @ q.T
q.T @ k = 1
```

The outer product notation assumes column vectors. Equivalently,
`W_edit = W @ (I - k @ q.T) + v @ q.T`. The matrix
`I - k @ q.T` is singular. For any output vector `a`, matrices `W` and
`W + a @ q.T` produce the same edited matrix under this fixed write.
Consequently, exact recovery of an arbitrary original `W` from the edited
matrix and this fixed metadata alone is not uniquely identifiable.

Saving `h_old = W @ k` supplies exactly the missing response in this model:

```text
W_restore = W_edit + (h_old - v) @ q.T
```

Conditional on the fixed metadata, the lost family has `output_dim` real
degrees of freedom. This is a dimensional statement, not a universal byte-level
compression theorem. In finite precision, the write may additionally discard
rounding information throughout the matrix.

A rank-one journal can instead save `u = v - W @ k` and `q`, and subtract
`u @ q.T` later. It does not need the full original checkpoint, but it is
explicitly a journal and its factors must be counted.

## Soft ROME: recoverability depends on write strength

For a fixed strength `alpha`, write

```text
W_edit = W + alpha * (v - W @ k) @ q.T
```

When `alpha != 1`, the local inverse is

```text
W_restore = W_edit + alpha / (1 - alpha) * (W_edit @ k - v) @ q.T
```

This follows from `W_edit @ k = (1 - alpha) * (W @ k) + alpha * v`.
As `alpha` approaches one, the old response is increasingly compressed and
its reconstruction amplifies numerical error. With Euclidean normalization
`q = k / (k.T @ k)` and `0 <= alpha < 1`, the linear map's condition number is
`1 / (1 - alpha)`. With general covariance, the projection is oblique and its
singular-value conditioning can be worse; eigenvalues alone do not describe it.

An almost-zero update is easy to undo because it has barely changed anything.
Recovery must therefore be reported together with achieved edit strength or
target error. The identity operation is a useful control, not a useful edit.

## Fixed-target, single-layer MEMIT inverse

Freeze `K`, `V`, and positive-definite `C` and consider the local least-squares
write from MEMIT:

```text
B = C + K @ K.T
W_edit = W + (V - W @ K) @ K.T @ inverse(B)
```

Multiplying by `B` gives `W_edit @ B = W @ C + V @ K.T`. Hence

```text
W_restore = (W_edit @ B - V @ K.T) @ inverse(C)
          = W_edit + (W_edit @ K - V) @ K.T @ inverse(C)
```

This is an algebraic inverse under the stated fixed-metadata assumptions.
Unlike hard ROME, finite positive-definite preservation leaves a recoverable
linear contribution from the previous matrix. Weak preservation can make that
contribution tiny. In whitened coordinates, the relevant write contraction is
controlled by `I + C^(-1/2) @ K @ K.T @ C^(-1/2)`; large eigenvalues imply strong
inverse amplification. Ordinary coordinate conditioning also depends on `C`.

Actual MEMIT computes representations and distributes residuals across several
layers. Those keys and targets can depend on the original or partially edited
network. Recomputing them from the final edited model is not guaranteed to
reproduce the metadata used by the forward write. This sandbox's inverse is
therefore a result about a local fixed-target update, not a demonstrated inverse
of the entire original MEMIT algorithm.

## Arithmetic and storage precision

The controlled arithmetic model performs linear algebra in `float64`, then
rounds a materialized weight matrix to the selected storage format. Emulated
bfloat16 rounding means rounding the stored values to bfloat16 precision; it
does not emulate bfloat16 accelerator kernels, accumulation order, or hardware
performance. Float16 storage likewise does not imply native float16 execution.

Round after the forward write and after a materialized rollback when measuring
repeated storage cycles. State precisely which metadata remains in float64.
Otherwise an apparently low-precision result can hide a full-precision journal.
Exact inverse formulas need not yield bitwise restoration after lossy storage.
Repeating an addition and subtraction can accumulate error even when the same
real-valued update is known. In v0.2, a lossless bit-correction log is implemented
and its full serialized cost measured; see [ROUNDOFF.md](ROUNDOFF.md). Making
that log requires the old state at write time. Undo uses only the declared
record, and exactness requires reproducing the recorded rollback prediction.

## Baselines and information budgets

The full-checkpoint control establishes the best recovery available to the
evaluator. Journal subtraction establishes what a compact saved update buys.
An SVD baseline removes a preselected number of leading singular components
from the edited matrix. This weights-only baseline is inspired by Youssef et al.;
it is a heuristic about edit structure, not an inverse of arbitrary low-rank
addition. Original weight structure can overlap the removed components.

For a blind SVD result, fix the dropped rank before looking at the held-out
original matrix, labels, or recovery scores. An oracle rank sweep may be shown
as a separately labeled upper comparison, never as an original-free algorithm.

Persistent recovery storage and peak computational workspace are different
budgets. Report both when available, and make shared metadata explicit:

- A dense checkpoint costs `output_dim * input_dim` stored numbers.
- A rank-`r` update journal costs `r * (output_dim + input_dim)` numbers before
  metadata, indexing, compression, or dtype overhead.
- An old-response journal costs `output_dim` numbers **in addition to** whatever
  keys, write directions, targets, and configuration are needed for recovery.
- A dense `C` costs `input_dim**2` numbers. Storing `K`, `V`, or a factorization
  is also a cost; describing a value as cached does not make it free.
- Shared metadata can be amortized across edits only when the experiment
  actually reuses it. A seed can regenerate a deliberately seeded synthetic
  matrix, but that shortcut is not evidence of a cheap real-model covariance.

The instrumented acquisition harness retains full reference matrices to score
drift, including a pre-trial snapshot. Corrected protocols also use the pre-trial
state to construct their declared bit-correction record at write time. Neither
the base restoration routine nor patch application receives the original.
Consequently, a recovery-payload comparison is
an access and storage-accounting result; this harness does not demonstrate a
reduction in measured process peak memory.

## Evaluation isolation and repeated learning

Keep the original checkpoint on the evaluator side. Recovery functions should
receive only the fields allowed by their declared condition. Oracle originals
and held-out targets may score a run but must not select recovery rank, strength,
stopping time, or candidate updates for that same held-out run.

The fixture generator deliberately creates some internal targets as an original
response plus a random offset. The old-response journal also deliberately saves
an original response. These are declared inputs to the corresponding methods;
they must not be described as wholly free of original-state information. The
isolation claim concerns what the recovery routine receives, not whether the
simulation ever held an original matrix.

Measure parameter error and held-out function error separately. A useful
relative parameter metric is `norm(W_restore - W_original) / norm(W_original)`
with an explicit zero-norm convention. A single restored answer is weaker than
restoration of a distribution of outputs or of the weight matrix.

For active learning, count each requested oracle label. Candidate uncertainty
can select a query; the answer to an unqueried test item cannot. Hold test data
apart from query acquisition and update acceptance. Compare at equal acquired
label counts and report performance as well as rollback error. For test-time
adaptation, separately identify the adaptation signal and the held-out scoring
signal; this toy experiment does not supply an unsupervised language-model loss.

To test whether restoration preserves later learning, apply the same subsequent
update to the restored and reference states. Match data order, seeds, precision,
and optimizer state. Numerical closeness now does not guarantee identical later
learning, and one matching update does not prove an entire trajectory matches.

## Sequential edits and selective undo

Subtracting saved deltas in reverse chronological order restores their recorded
arithmetic path, subject to rounding. V0.2's exact correction receipts also
restore the recorded intermediate storage bits under reproducible predictor
arithmetic, enabling strict digest-checked reverse chains. Removing an earlier delta while retaining
later recorded deltas is a different operation from rerunning later edits on a
history in which the earlier edit never occurred. Later keys, targets, gradients,
and optimizer states may depend on that earlier edit. Evaluate selective undo
against replay if the intended claim is counterfactual history restoration.

## Provenance and claims

`references.bib` records the primary sources. ROME and MEMIT motivate the local
writes. The SVD-reversal paper and the shared-mask paper demonstrate forms of
behavioral recovery; neither establishes exact arbitrary-weight reconstruction.
Shared-mask training uses original fact labels and original-model distributions.
Caddeo et al. study reverse factual editing, which supplies the old fact again.

This sandbox can support claims about the measured local algebra, numerical
drift, information budgets, and controlled learning loops. Language-model edits,
multilayer inversion, learned recovery without original information, and broad
active-learning or test-time-training benefits require additional experiments.
In particular, comparisons between rollback protocols test restoration fidelity
and evaluation-order dependence. Random-query and uncertainty-query controls
share the acquired-label budget and retained update rule, but use less compute
than hypothetical edit/score/undo look-ahead. Report those compute differences
alongside accuracy. The entropy proxy rewards confidence, which can increase
while accuracy worsens. SVD on these small synthetic matrices is a control, not
a replication of the language-model reversal paper's reported success rates.
