# GPT-2 gradient-receipt pilot protocol

Pin openai-community/gpt2 revision 607a30d783dfa663caf39e06633721c8d4cfcd7e
(MIT model license). Use local safetensors only, CPU eager float32 forward,
eval mode, no KV cache, deterministic algorithms, and one Torch/BLAS thread.
Target transformer.h.11.mlp.c_proj: transpose GPT-2 Conv1D weights to our
(output,input) convention, verified against the actual projection output.

Predeclared prompts/targets: The capital of France is → Paris; Japan → Tokyo;
Italy → Rome (targets include the leading space). Each must tokenize to one
next token. These three probes are not a factual editing benchmark.

For each prompt and FP32/FP16/emulated-BF16 target projection storage, reset the
projection from the original checkpoint and obtain the real single-token
negative-log-likelihood gradient. Check that output-gradient × input-activation
matches the projection's autograd gradient (relative error <=1e-5); require
projection forward maximum absolute error <=1e-4. The final block ensures this
single-token loss has a rank-one projection gradient. Apply one negative-gradient
step normalized to 0.001 times the original matrix Frobenius norm, then round
the updated projection to declared storage precision. All other parameters stay
fixed. FP16/BF16 are storage emulation, not full low-precision model execution.

Transport uses the algebraically equivalent hard-ROME representation:
q=k/(k.T k), u=-learning_rate*(k.T k)*output_gradient, v=Wk+u.
The saved delta is u q.T. This is a gradient-derived update, not ROME value
optimization or MEMIT residual distribution. Do not describe it as complete
ROME/MEMIT factual editing.

Use the unchanged PR #1 build/undo workers with checkpoint/delta/delta_exact,
three timing repeats, rotating method order: 9 fixtures, 81 observations.
Torch model evaluation stays in the parent; measured workers import only NumPy.
Retain all worker failures, raw metrics, model file/source hashes, gradient
checks, before/edited target probabilities, and restored next-token logits.
Reapply restored tensors and compare logits once per fixture/method outside
resource timing. Exact/checkpoint bit or logit mismatches fail the run. Target
probability changes are diagnostics, with no performance gate or tuning.

Model binaries, actual update factors, records and logits stay locally in ignored
work directories. Public evidence contains their digests and scalar observations.
The repository Apache-2.0 license does not relicense upstream model weights.

Run (Python 3.11+ for this optional script):

```bash
.venv/bin/python -m pip install torch transformers safetensors
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 \
  .venv/bin/python -I scripts/real_model_costs.py \
  --model-dir work/models/gpt2 --out results/my_gpt2
```

The model directory must contain files downloaded from the pinned revision.
The run records their SHA-256 hashes for independent verification.
