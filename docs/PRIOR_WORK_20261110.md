# Prior work and the next falsifiable question

Focused primary-source search, 2026-11-10. This is a research-positioning note,
not a systematic review or a claim of priority. The original project already
cites reversible learning and behavioral reversal; this note adds newer
selective-reversal and compression work and uses it to choose a narrower test.

| Work | What it already establishes / studies | Consequence for this project |
| --- | --- | --- |
| [Maclaurin et al., ICML 2015, §§2.2–2.3](https://proceedings.mlr.press/v37/maclaurin15.pdf) | Exactly reverses learning dynamics by retaining information lost to finite precision. | Saving discarded information to reverse learning is not new. Our floating-point edit receipts are an application, not a new reversibility principle. |
| [Caddeo et al., Applied Sciences 2026, §3.4.3](https://www.mdpi.com/2076-3417/16/13/6567) | Evaluates ROME/MEMIT edit–revert behavior using an inverse factual edit, without restoring saved weights. | A before/edit/after behavioral cycle is already studied; distinguish inverse editing from exact state restoration. |
| [Youssef et al., ICLR 2026, §5](https://proceedings.iclr.cc/paper_files/paper/2026/file/28bfea825dc61f7adc9f1e2eaf5bdd2b-Paper-Conference.pdf) | Infers edited objects and reverses edits from modified weights, using dominant singular components. | Blind reversal has a harder information constraint. We retain write-time information and cannot claim superiority by achieving exact restoration with it. |
| [Jiang et al., arXiv:2609.02091v1, §§3, 6](https://arxiv.org/html/2609.02091v1) | Selectively reverses targeted facts while keeping remaining edits; original model and full history are unavailable. Targets are pretrained behavior on reversed facts and edited behavior on retained facts. | Selective reversal itself is not new. Our proposed reference instead reruns retained edits from a history that excludes the withdrawn edit, with an explicit journal available. These are different contracts. |
| [Bourtoule et al., SISA, IEEE S&P 2021](https://arxiv.org/abs/1912.03817) | Structures training to reduce the recomputation required for unlearning. | Rewind-and-recompute is not new either. Do not present deterministic replay as a novel unlearning algorithm or claim removal of pretraining data. |
| [ZipNN, CLOUD 2025 / author documentation](https://github.com/zipnn/zipnn/blob/main/docs/UTH.md) | Lossless model compression and XOR deltas between checkpoints. | Compare against direct XOR state differences, not only a full checkpoint. A small correction payload alone does not establish a new compression result. |
| [ULPack, arXiv:2610.14184v1](https://arxiv.org/html/2610.14184v1) | Reference-conditioned lossless checkpoint compression using ordered floating-point residuals. | Codec competition needs direct, matched implementations and total bytes/time. Our zlib ablation is not a benchmark against ULPack or ZipNN. |
| [Acar, Blume & Donham, ESOP 2007 / full version 2011](https://arxiv.org/abs/1106.0478) | Combines memoization and dependency-driven change propagation, with consistency/correctness theorems for self-adjusting computation. | Dependency-aware cache invalidation and recomputation are not new principles. Our bounded editor-specific dependency map and empirical contract do not reproduce their formal system or proof. |
| [Pinned ROME author implementation](https://github.com/kmeng01/memit/blob/80426fd9316cf9a50c5ba15e0912f2c2c5bfe84b/rome/rome_main.py) | Already caches update factors and explicitly limits cache use to the first layer because prior-layer writes affect later layers. | Neither caching editor components nor noticing cross-layer dependence is a new observation. Test bindings to the current computation inputs and retained-history references rather than claim to invent this dependency. |
| [Gupta, Rao & Anumanchipalli, ACL Findings 2024](https://arxiv.org/abs/2401.07453) | Studies forgetting of edited facts and downstream-task ability under sequential ROME/MEMIT edits. | A long-history reference can itself be degraded. Reproducing it exactly is not evidence that retained facts or other abilities are healthy. |

## Candidate contribution, not established novelty

Measure **the difference between removing a stored update and removing its
influence on subsequent optimization**, with finite-precision error separated.
For a deterministic editor E and base weights W0:

```
WA  = E(W0, A)
WAB = E(WA, B) = float32(WA + delta_B(WA))
counterfactual reference = E(W0, B)
stale replay            = float32(W0 + delta_B(WA))
```

Even a perfect rewind to W0 does not make an old B update equal to a freshly
optimized B update. Compare naive subtraction of A, exact rewind plus stale B,
exact rewind plus freshly optimized B, and checkpoint rewind plus fresh B.
The stale-replay arm removes rounding as an explanation for dependency error.
Record full-state equality, output distributions, factual behavior and cost.

The opportunity is a reproducible diagnostic and accounting contract for
**history-aware exact withdrawal of actual ROME/MEMIT edits**. It is narrower
than broad reversibility or selective unlearning. Search found no directly
matching evaluation in the sources inspected, but that does not establish
absence elsewhere or a publishable methodological advance.

An exact counterfactual reference need not restore the withdrawn fact's original
answer: the retained edit may independently affect that answer. Likewise,
preserving the originally edited answer for B does not imply equality to the
counterfactual model. Report both rather than conflating their objectives.

## Next evidence required

The fixed six-chain diagnostic in [SELECTIVE_REPLAY.md](SELECTIVE_REPLAY.md) and
the [component/long-history extension](../results/20261110_dependency_replay_rerun/RESULTS.md)
are now complete. The latter identifies reusable key-side components and matches
the retained-history reference, but the cost of byte-hash checks outweighs the
saved computation in its paired timing cells. Reference editing quality also
degrades. These are limitations to resolve, not reasons to claim novelty from
exact replay alone.

A next candidate is enforcing write-time dependency maintenance cheaply enough
to beat fresh suffix computation, while detecting mutations outside that writer.
Its verification/maintenance costs and information access must be matched.
Stronger models and better-qualified editing settings are still unrun; they must
not be presumed to fix quality or cost. Run literature implementations under
matched conditions before claiming performance against them.
