# Frozen real-edit codec results

252/252 exact restorations; 756/756 negative checks rejected.
Four fixed A-edit tensors, three repeats per method/codec. Entries below are
medians pooled across the four fixtures (12 observations per row).
No efficacy-based filtering. Build and restore exclude disk I/O and process startup.

| Method | Codec | MiB | Build ms | Restore ms | Process peak MiB |
|---|---|---:|---:|---:|---:|
| checkpoint | none  | 9.001 | 11.284 | 12.085 | 96.469 |
| checkpoint | zlib 1 | 8.387 | 201.807 | 29.842 | 116.641 |
| checkpoint | zlib 6 | 8.364 | 239.531 | 29.734 | 117.844 |
| checkpoint | zlib 9 | 8.364 | 243.994 | 31.515 | 118.758 |
| checkpoint | zstd 1 | 8.349 | 19.667 | 18.567 | 105.609 |
| checkpoint | zstd 3 | 8.357 | 23.187 | 19.683 | 107.039 |
| checkpoint | lz4 0 | 9.002 | 17.342 | 14.321 | 105.430 |
| direct_xor | none  | 9.001 | 12.472 | 12.140 | 96.477 |
| direct_xor | zlib 1 | 7.607 | 170.289 | 32.431 | 117.195 |
| direct_xor | zlib 6 | 7.495 | 310.362 | 30.686 | 116.883 |
| direct_xor | zlib 9 | 7.495 | 308.991 | 29.960 | 115.828 |
| direct_xor | zstd 1 | 7.352 | 23.198 | 17.738 | 103.312 |
| direct_xor | zstd 3 | 7.356 | 25.537 | 17.918 | 103.352 |
| direct_xor | lz4 0 | 9.002 | 18.051 | 14.304 | 105.922 |
| factors_correction | none  | 9.061 | 47.745 | 38.545 | 177.547 |
| factors_correction | zlib 1 | 0.469 | 57.515 | 39.935 | 175.031 |
| factors_correction | zlib 6 | 0.327 | 89.613 | 38.759 | 173.141 |
| factors_correction | zlib 9 | 0.277 | 650.379 | 37.969 | 174.406 |
| factors_correction | zstd 1 | 0.327 | 52.442 | 40.296 | 160.812 |
| factors_correction | zstd 3 | 0.325 | 51.033 | 38.803 | 158.836 |
| factors_correction | lz4 0 | 0.671 | 50.729 | 39.804 | 179.305 |

For factors+correction, zstd 3 reduces median build time from 650.38 ms
(zlib 9) to 51.03 ms; bytes increase from 0.277 to 0.325 MiB. Restore time
is similar (37.97 vs 38.80 ms). These are ratios of pooled medians, not a
claim that every edit is accelerated by the same factor. LZ4 is 0.671 MiB
and does not materially improve this method's restore time.
No default codec is changed; no comparison to old double-compressed timings
is implied. All three methods use the same new outer envelope.

## CUDA native-storage preflight

RTX 5090 / Torch 2.9.1+cu130: four real stored CPU-author factor pairs ×
FP32/FP16/BF16 = 12 cells. Exact correction: 12/12. Ordinary subtraction:
0/12. Wrong predictor rejected: 12/12. No forward/model-quality evaluation,
live optimizer, or distributed run occurred. Peak CUDA allocation and single
timings are diagnostic only; objects from prior cells can remain live.
The first attempt stopped before numeric work on AppleDouble file counting;
its failure is retained separately.

See [scope and continuation](../../docs/CODEC_AND_PRECISION.md).
