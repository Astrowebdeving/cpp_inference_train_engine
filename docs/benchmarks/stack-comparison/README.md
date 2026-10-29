# Matched training records

Recorded 2026-10-07 on Apple M2 Max, 64 GiB, macOS 26.7.1, Apple Clang 17, native Release build, Python 3.14.7 and PyTorch 2.10.0. See [the comparison guide](../../stack_comparisons.md) for the workload and interpretation.

- `fp32-batch2`: FP32, microbatch 2, accumulation 4.
- `fp16-batch2`: FP16, microbatch 2, accumulation 4.
- `fp16-batch8`: FP16, microbatch 8, accumulation 1.

All use length 128, submission window 4, three warmups and thirty measured updates. Each has separate initial-loss/every-gradient/one-update correctness runs, then separate native and PyTorch performance processes. The CPU correctness oracle and tensor export are excluded from the native performance process. PyTorch variants are explicit attention and SDPA, each eager and compiled forward/loss with generated backward. Clipping and foreach AdamW remain eager in the compiled variants. CPU fallback and MPS fast math are disabled. CPU thread limits are one.

Each directory contains `comparison.json`, exact `commands.json`, individual timing/parity reports, compiler logs, and `input-manifest.json`. `sources` preserves the benchmark source versions named by the result checksums. The current runner differs from this snapshot only in a metadata clarification: native per-microbatch staging is timed, while PyTorch IDs/targets are preloaded on-device. The combined reports retain the original scope text and an amendment; measurements and checksums are unchanged.

The 429,166,080-byte tensor binary contains initial FP32 weights, initial gradients, and final weights. The runner regenerates it from native seed 42 and records its SHA256. These temporary binaries and compiler caches are not copied into documentation. Reproduction commands in the records refer to their original temporary output directories; use a new directory when rerunning.

Peak process RSS and PyTorch MPS driver allocations are recorded separately. RSS does not uniformly include private GPU allocations, and the driver figure is sampled after the run rather than at its peak. These records do not establish an end-to-end GPU memory advantage.
