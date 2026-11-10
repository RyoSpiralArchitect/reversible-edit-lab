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

Start with the fixed six-chain development diagnostic in
[SELECTIVE_REPLAY.md](SELECTIVE_REPLAY.md). Keep every failure. If dependency
error survives exact rewind, extend to longer histories and stronger models,
then test whether the dependency structure permits safe partial recomputation.
Compare against history-aware baselines under the same information access.
Run literature implementations on matched settings before making performance
claims against them. Exact replay is the control, not by itself the new method.
