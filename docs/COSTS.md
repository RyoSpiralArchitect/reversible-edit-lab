# Resource-cost pilot protocol

This protocol is committed before the retained measurement run. The question is:
**When exact correction saves serialized bytes, what time and process memory
does constructing and restoring the record actually require?**

## Fixed design

- Synthetic fixed-target ROME, identity covariance, alpha 0.95.
- Shapes: 64×128, 256×512, 1024×2048 (output × input).
- Seeds: 7, 23, 61. Storage: FP32, FP16, emulated BF16.
- Methods: compressed checkpoint, delta receipt, delta receipt with exact XOR correction.
- Three timing repeats for each seed/shape/precision/method: 243 paired
  observations, each with independent build and undo processes (486 workers).
- Method order rotates across fixtures and repeats. No optimization or parameter
  selection based on the retained run. Additional configurations get new directories.
- Report all failures and exact mismatches. There is no speed or memory success
  threshold. Medians and min/max describe this host, not a population estimate.

The identity-covariance direction is computed directly as `q=k/(k.T@k)`.
The fixture is checked against `rome_edit(..., covariance=I)` in tests. This
avoids an irrelevant dense solve while constructing large fixtures; forward
editing, covariance estimation, and ROME target optimization are not benchmarked.

## Timed regions and memory

Every phase launches `python -I -m reversible_edit.cost_experiments` with BLAS
thread environment settings fixed to one. Use an isolated installed virtual
environment so user-site patches do not change NumPy. Process startup/imports
and input disk reads occur before timing. Each process measures one operation
without an in-process warmup. OS scheduling/cache effects remain possible.

| Phase | Inputs present at start | Timed operations |
| --- | --- | --- |
| Build | Original, edited matrix, factors, fixed ROME spec | Construct record and integrity hashes; serialize compressed NPZ |
| Undo | Edited matrix and serialized record bytes | Decode NPZ; restore; integrity checks required by the method |

Build writes the resulting bytes to disk after the measured region. Undo returns
a digest to the evaluator; that benchmark-only final digest is also outside the
region. Checkpoint and exact restoration include their own integrity checks
inside the timed region. The ordinary delta implementation checks its edited
input but does not verify an original digest, and that difference is retained.

Checkpoints use the same compressed logical-storage representation as the v0.2
capacity experiment, plus an edited-state digest. BF16 occupies two bytes per
element in the checkpoint and four bytes per element in NumPy's in-memory
emulation. Delta factors are float64. Whole NPZ bytes include container and
metadata costs. Checkpoint ratios are paired by fixture and repeat.

`resource.getrusage(RUSAGE_SELF).ru_maxrss` records the **whole process lifetime
high-water RSS** immediately before and after the operation. macOS reports bytes;
Linux reports KiB, converted to bytes. The values include Python, NumPy, input
loading, allocator retention, and temporaries. The difference between high-water
marks is not the operation's allocation count or isolated peak. Zero growth can
occur when input preparation established a higher peak. No inference about GPU,
live-model integration, or peak-memory savings follows from serialized size.

## Reproduce and inspect

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -I -m reversible_edit.cli costs --out results/my_costs
```

Small diagnostic run:

```bash
.venv/bin/python -I -m reversible_edit.cli costs --out work/cost-smoke \
  --shapes 16x32 --seeds 7 --precisions bfloat16 --repeats 1
```

`run.json` is written before workers launch and records source hashes,
configuration, NumPy/BLAS build, Python, OS, and thread settings. `fixtures.json`
records input hashes, original/edited digests, physical/logical weight bytes,
factor bytes, and achieved write residual. `observations.jsonl` is flushed after
each paired observation, including worker errors/timeouts. A completed run has
`summary.json` with its status; an interrupted run must not be labeled complete.
CLI exit status is nonzero on worker failures or checkpoint/exact mismatches.

Binary fixtures and actual NPZ records remain under the run's ignored `work/`
directory. They are synthetic and reproducible from the configuration. Public
evidence retains their hashes, serialized lengths, raw timing/RSS observations,
and source versions. Original arrays are used at record creation, never by undo.

Follow-up work should first profile the measured costs. A real-model matrix
pilot must record model revision, tensor orientation and dtype, and actual
update factors separately from any synthetic fixed-target write. That next step
has not been performed by this resource pilot.
