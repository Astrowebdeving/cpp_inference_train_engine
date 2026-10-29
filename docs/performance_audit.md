# Performance audit

The [throughput follow-up](throughput_followup.md) continues this work. It changes two defaults described here: new Metal Muon runs use the `composed` order instead of `gram`, and accumulated microbatches are merged into larger device passes. Runs saved with the settings below keep them.

This implementation is optimized for measured Mac workloads. A universal maximum is not established: context length, batch size, model shape, data and software versions all change the best choice. The audit retains comparison paths and checks correctness before accepting speed gains.

## Changes

**Muon multiplication order.** Once `A = X Xᵀ` is formed, compute the Newton–Schulz polynomial using `A A` and one multiplication by `X`. For wide matrices, `A` is much smaller than `X`; the previous `A (A X)` order repeats a larger multiplication. New Metal runs default to `--muon-ns-order gram`. `chain` retains the previous order, and old checkpoints continue using it. Momentum, iteration count, parameter routing and FP32 optimizer state are unchanged. See the [Muon guide](muon_on_metal.md).

**Gemma packed-cache reuse.** Keep K/V arrays across single-token steps for batches of at least two sessions with unchanged ordered membership. Track each row's position, valid length and ring cursor. Aligned rows append together; unequal rows scatter the new K/V into their own slots, preserving padding and masks. Reordering, admission or removal rebuilds the batch. The first decode after an oversized sliding prefill also normalizes that cache through the reference path. Single requests retain independent caches because the trials found no consistent reuse benefit. `--independent-kv` disables reuse for comparison. Global history still grows by concatenation; this is not a paged cache or a preallocated memory pool.

The default `auto` policy retains aligned-row reuse above batch one. Unequal-row scatter runs at batch eight and above, or batch four and above once every row reaches the 512-token sliding window. Smaller, short-context unequal batches retain independent caches. Controlled trials found scatter slower by about 6–7% at short-context batch two and 3–6% at batch four; batch-four long contexts saved about 2–4%. Batch-two long contexts showed no consistent benefit. This shape selection is based on the tested M2 Max and checkpoint, not a universal hardware rule. Use `--all-batch-kv` to force all multi-request cases on another Mac, and compare with `--independent-kv`. `/health` reports `batch_kv_policy` and actual reused-step counts.

**One response per serving step.** Embed engine status in prefill, decode and release responses. The scheduler no longer sends a separate `info` command after every decode. The explicit status command remains supported, including compatibility with older binaries.

The baseline process-boundary profile put approximately 98% of a short-context, eight-request decode cycle inside the native engine. Its 63 decode calls took 1,977.8 ms inside the engine and 1,992.5 ms wall time. Token handling took 16.2 ms, and separate status calls took 4.9 ms. This is a diagnostic sample rather than a kernel-level GPU trace or a repeated performance result. It supports prioritizing cache and matrix work; a larger HTTP rewrite has limited potential on this measured workload.

## Muon results

Apple M2 Max, 64 GiB, full 35,763,840-parameter MLA model, FP16 forward/backward, FP32 optimizer, sequence 128, microbatch eight, one accumulation, five Newton–Schulz iterations, batched matrices, submission window four. Both timers include gradient reset, forward, backward and optimizer completion. Each trial has three warmups and 30 updates; the repeat reverses order.

| Multiplication order | Median update, first / repeat | Training tokens/s, first / repeat |
| --- | ---: | ---: |
| Chain | 97.16 / 97.17 ms | 10,539 / 10,539 |
| Gram | 90.06 / 89.30 ms | 11,370 / 11,466 |

Gram saves 7.3–8.1% of update time, or provides 7.9–8.8% more tokens/s, in this workload. It retains two graph submissions and one host wait per update. The full-model first-update comparison checked all 132 parameters: initial weights and gradients were bitwise equal, the maximum per-parameter update relative L2 difference was `1.625e-5`, the minimum update cosine was `0.999999999868`, and the maximum absolute weight difference was `7.45e-9`. This verifies small local arithmetic drift, not identical future training trajectories or a quality advantage over AdamW.

Reproduce, then replace `gram` with `chain`:

```sh
build/bin/mini_mla_bench --kind train --config config/mla36m-v1.json \
  --backend mps --precision fp16 --attention reference --optimizer muon \
  --muon-layout batched --muon-ns-order gram \
  --sequence-length 128 --microbatch 8 --grad-accum 1 \
  --submission-window 4 --warmup 3 --repetitions 30 --time-zero-grad
```

## Serving measurement

The audit compares the preserved earlier binary and server, the current server, and the current independent-cache path. It uses identical pinned Gemma weights and pretokenized prompts, serial server processes, BF16 activations, temperature zero, fixed output lengths and an MLX-LM greedy reference. Timed work includes scheduling, GPU completion, sampling, output detokenization and HTTP/SSE. Startup and input tokenization are excluded. The same runner also tests actual vLLM-Metal with its decode pipeline. Its optional compiled-MLP flag does not wrap Gemma in the installed 0.30.0 policy; that variant is another Gemma trial, not a stronger compiled Gemma implementation.

```sh
.venv-serving/bin/python tools/compare_serving.py \
  --model artifacts/gemma-4-E2B-it-4bit --output artifacts/new-cache-comparison \
  --concurrency 1 4 8 --prompt-length 128 --max-tokens 64 \
  --warmup 1 --repetitions 5 --independent-kv-baseline --vllm-compiled-mlp
```

Repeat with `--prompt-length 1024`, and with `--mixed-lengths`. Use `--native-only` for native ablations. Prompt lengths and token outputs are recorded alongside throughput and first-token/request latency. Throughput gains must be read with these latency and numerical records.

The final automatic-policy trial uses a 512-token vLLM prefill budget, five bursts per concurrency, and vLLM first. Both services generate 64 tokens from 128-token prompts:

| Concurrent requests | Native output tokens/s | Actual vLLM-Metal | Native median / p95 first-token latency | vLLM median / p95 first-token latency |
| --- | ---: | ---: | ---: | ---: |
| 1 | 109.65 | 88.37 | 34.96 / 35.28 ms | 54.06 / 54.36 ms |
| 4 | 200.38 | 183.12 | 109.96 / 132.71 ms | 173.75 / 177.31 ms |
| 8 | 225.13 | 216.18 | 231.39 / 237.19 ms | 303.61 / 325.92 ms |

The earlier 512-token-budget vLLM trial reached 91.91/186.53/222.18 tokens/s; a 1,024-token-budget trial reached 88.47/182.63/217.80. Increasing its budget did not improve throughput in those trials. These are specific scheduler configurations, not an exhaustive tuning search. The margin at eight requests remains small and varies between runs. Native throughput also varies with admission grouping: an ungated short-context trial reached 111.93/209.80/224.79 tokens/s. The preserved earlier server reached 109.49/206.36/224.37. The full HTTP results do not establish a reliable large improvement over that earlier native implementation, despite the isolated cache savings below.

All eight native 64-token short-context reference outputs match MLX-LM. vLLM's outputs diverge on several prompts; matching prefixes and token positions are recorded, and the broader quality impact is unestablished. At 1,024-token contexts, even the preserved native implementation differs from the MLX-LM greedy reference on seven prompts; the aligned-cache follow-up preserved all eight earlier native sequences. Numerical parity and exact greedy agreement are separate checks.

An isolated engine ablation of aligned-row reuse uses the same fixed decode input tokens, with one warm prefill/eight warm decode steps followed by 64 measured steps. Each option runs twice, in reversed order. At batch eight, total native decode time changed from 1,964/1,979 ms to 1,944/1,941 ms for 128-token contexts, and from 2,178/2,163 ms to 2,082/2,072 ms for 1,024-token contexts. That is roughly 1–2% and 4% less decode time respectively. These synthetic cache timings exclude prefill and HTTP, and do not describe end-to-end serving throughput. Real HTTP admission frequently leaves rows at unequal positions; the subsequent scatter path addresses that missed reuse. The preliminary single-request trials were variable, so that path is excluded from the final reuse default.

The unequal-row ablation uses eight prompts of lengths 128–121 or 1,024–1,017, with fixed decode inputs and reversed option order. Independent versus packed total native decode times were 2,097/2,085 versus 2,065/2,075 ms at short contexts, and 2,246/2,255 versus 2,168/2,129 ms at long contexts. Long-context decode saves approximately 3.5–5.6% in these trials. HTTP cache-reuse counts rose from 197/1,647 decode batches in the aligned-only short-context trial to 747/1,647 with ungated unequal-row support; single-request batches are deliberately excluded. The later automatic policy avoids the measured small-batch regressions.

The automatic-policy confirmation measured short-context batch-two total decode times of 707/706 ms versus 719/728 ms with forced scatter, and batch-four times of 1,124/1,140 versus 1,192/1,154 ms. At long-context batch four, both options selected the same packed path. The final HTTP run used the automatic policy and recorded 502 reused batches; it correctly released every session. Logical `kv_bytes` describes valid per-session data, while `mlx_active_bytes` and `mlx_peak_bytes` include backing allocations and other model tensors; they are different memory scopes.

Three-burst mixed-length HTTP trials use prompts 128/112/96/80 repeated across eight requests. Before the guard, forced scatter reached 190.57/218.68 tokens/s at concurrency four/eight versus independent caches at 196.12/182.71. These admission-sensitive results support testing several shapes rather than enabling every new path globally. All eight native/independent greedy sequences agreed in the long- and mixed-context comparisons, although some differed from the MLX-LM reference.

Packing each MLP's gate/up projection into one larger quantized matrix was also tried on two real layer shapes and several batches. It did not give a consistent kernel-time reduction and changed BF16 output at one decode shape, so it was not added to the runtime. The finite experiment does not prove packing can never help other shapes.

## Choosing settings and further work

| Workload | Current choice | Next useful measurement |
| --- | --- | --- |
| MLA AdamW training | Fused Metal update, lazy gradient reset, FP32 state, bounded submissions. | Raise microbatch while holding tokens per update fixed; measure memory and held-out quality. |
| MLA Muon training | Batched FP32 Gram order; saved execution settings. | Tune optimizer learning rates separately and compare time to held-out loss across seeds. |
| MLA attention | `auto` currently selects reference, faster on the tested M2 Max shapes. Explicit Flash saves attention memory. | Compare long sequences and peak memory; do not assume a custom Flash kernel is faster. |
| Gemma stable batches | Shape-selected packed-cache reuse and reduced-query prefill enabled. | Compare `auto`, `--all-batch-kv` and `--independent-kv` on your Mac; also measure allocator memory and latency under sustained arrivals. |
| Gemma mixed lengths | Shape-selected packed updates with valid-length masks; independent fallback for other shapes. | Length buckets, preallocated/paged caches and interleaved prefill to reduce copying and long-prompt stalls. |
| Repeated prompts | No prefix cache; comparison disables it on vLLM too. | Implement invalidation and reference-counted reuse, then compare realistic hit rates. |

The earlier strongest PyTorch 2.10 compiled/fused AdamW comparison is approximately tied with our AdamW training path; this Muon change does not establish an AdamW or general training-stack advantage. Gemma remains text-only inference using MLX kernels. Speculative decoding, prefix caching, Gemma training, distributed execution and a vLLM adapter for the custom MLA architecture are separate work.

Verification passed 67 C++ cases, nine Python cases, 51 Gemma engine checks, nine HTTP checks and six CPU/tools UBSan targets. The engine checks require bitwise equality with independent caches through sliding wrap, reordering, shrinking batches, mixed partial/full caches and different ring cursors. They also exercise the automatic small-batch guard. An 80-step FP16 Metal Muon run with Gram order reduced loss from 5.6082 to 0.1313 and learned `Say hi.` → `Hi!`; this is a tiny learning/replay/export check, not general language quality. [Verification records](verification/performance-audit/results.json) and [benchmark records with exact source snapshots](benchmarks/performance-audit/README.md) are preserved.

The subsequent [throughput follow-up](throughput_followup.md) adds pipelined updates, coalesced accumulation, the composed Muon order and scheduler improvements. Measure one change at a time and retain the reference path.
