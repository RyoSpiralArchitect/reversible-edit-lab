# Rollback resource-cost pilot

Synthetic identity-covariance ROME; alpha=0.95. 3 seeds × 3 timing repeats per condition. All figures below are medians; raw observations and min/max are retained.

Completed observations: 243; worker failures: 0. Checkpoint/exact bitwise restores: 162/162.

| Shape | Precision | Method | Record KiB | /checkpoint | Build ms | Undo ms | Build peak MiB | Undo peak MiB | Bitwise |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024x2048 | bfloat16 | checkpoint | 3257.9 | 100.0% | 229.89 | 18.23 | 90.8 | 93.3 | 9/9 |
| 1024x2048 | bfloat16 | delta | 24.1 | 0.7% | 5.25 | 30.86 | 75.9 | 215.3 | 0/9 |
| 1024x2048 | bfloat16 | delta_exact | 590.2 | 18.1% | 2141.36 | 70.11 | 269.9 | 271.5 | 9/9 |
| 1024x2048 | float16 | checkpoint | 3777.2 | 100.0% | 129.91 | 16.49 | 65.3 | 70.8 | 9/9 |
| 1024x2048 | float16 | delta | 24.1 | 0.6% | 5.53 | 15.08 | 56.1 | 108.2 | 0/9 |
| 1024x2048 | float16 | delta_exact | 584.7 | 15.5% | 2610.89 | 41.25 | 160.0 | 140.2 | 9/9 |
| 1024x2048 | float32 | checkpoint | 7607.4 | 100.0% | 292.04 | 31.85 | 94.8 | 95.8 | 9/9 |
| 1024x2048 | float32 | delta | 24.1 | 0.3% | 6.14 | 18.11 | 84.6 | 133.8 | 0/9 |
| 1024x2048 | float32 | delta_exact | 659.6 | 8.7% | 2797.98 | 50.14 | 204.6 | 176.8 | 9/9 |
| 256x512 | bfloat16 | checkpoint | 203.1 | 100.0% | 18.64 | 1.68 | 37.1 | 36.9 | 9/9 |
| 256x512 | bfloat16 | delta | 6.9 | 3.4% | 0.80 | 2.64 | 36.3 | 42.5 | 0/9 |
| 256x512 | bfloat16 | delta_exact | 42.8 | 21.1% | 165.96 | 6.16 | 53.4 | 45.3 | 9/9 |
| 256x512 | float16 | checkpoint | 236.8 | 100.0% | 7.81 | 1.14 | 35.1 | 35.1 | 9/9 |
| 256x512 | float16 | delta | 6.9 | 2.9% | 0.78 | 1.20 | 34.6 | 37.2 | 0/9 |
| 256x512 | float16 | delta_exact | 42.6 | 18.0% | 147.20 | 3.43 | 39.0 | 37.9 | 9/9 |
| 256x512 | float32 | checkpoint | 475.3 | 100.0% | 14.27 | 1.98 | 36.1 | 36.0 | 9/9 |
| 256x512 | float32 | delta | 6.9 | 1.4% | 0.64 | 1.20 | 35.3 | 37.6 | 0/9 |
| 256x512 | float32 | delta_exact | 47.3 | 9.9% | 132.08 | 3.06 | 42.0 | 38.3 | 9/9 |
| 64x128 | bfloat16 | checkpoint | 13.4 | 100.0% | 0.69 | 0.32 | 33.2 | 32.8 | 9/9 |
| 64x128 | bfloat16 | delta | 2.6 | 19.3% | 0.36 | 0.44 | 33.0 | 33.1 | 0/9 |
| 64x128 | bfloat16 | delta_exact | 5.6 | 41.6% | 8.42 | 1.27 | 33.5 | 33.2 | 9/9 |
| 64x128 | float16 | checkpoint | 15.4 | 100.0% | 0.44 | 0.27 | 33.1 | 32.7 | 9/9 |
| 64x128 | float16 | delta | 2.6 | 16.8% | 0.38 | 0.39 | 33.1 | 33.0 | 0/9 |
| 64x128 | float16 | delta_exact | 5.6 | 36.4% | 7.44 | 1.12 | 33.0 | 33.1 | 9/9 |
| 64x128 | float32 | checkpoint | 30.4 | 100.0% | 0.83 | 0.32 | 33.1 | 33.0 | 9/9 |
| 64x128 | float32 | delta | 2.6 | 8.5% | 0.35 | 0.36 | 33.2 | 33.1 | 0/9 |
| 64x128 | float32 | delta_exact | 5.9 | 19.3% | 8.73 | 1.17 | 33.2 | 33.1 | 9/9 |

## Measurement boundary

Each build and undo uses a fresh process, one numerical operation, and single-thread BLAS settings. Process startup, input file loading, fixture generation, forward editing, and record file writing are excluded from time. Build includes record construction, integrity hashes, and NPZ encoding; undo includes NPZ decoding and restoration checks.

RSS is the whole process lifetime high-water mark at the end of the measured region, including imports and input preparation. The pre-operation high-water mark and its growth are also recorded. Growth is not an allocation count or an isolated operation peak; a zero growth does not mean zero allocations. BF16 weights occupy float32 arrays in memory but uint16 words in checkpoint storage.

The checkpoint is compressed and bound to the edited state and original digest. Undo processes load only that edited state and the selected record; the evaluator compares returned digests. Delta records may not restore bits. Exact records retain old-state information. These measurements do not establish native GPU speed, end-to-end LLM memory savings, or factual editing quality.

`observations.jsonl` preserves every completed observation, including errors. `summary.json` contains min/median/max. `run.json` records configuration, environment, and source hashes; `fixtures.json` records input hashes and edit strength. Generated binary fixtures/records remain in ignored `work/` and can be regenerated from the configuration.
