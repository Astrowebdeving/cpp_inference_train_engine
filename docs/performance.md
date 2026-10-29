# Performance and verification

The latest follow-up measurements are in the [performance audit](performance_audit.md); the broader training and Gemma/vLLM-Metal comparison is in the [Mac guide](mac_training_and_serving.md). This page preserves earlier measurements and their original timing scopes.

Measured on this project's target Mac: Apple M2 Max, 64 GiB unified memory, macOS 26.7.1, Apple Clang 17, CMake Release build. The full model has 35,763,840 parameters. The original baseline runs below were serial, with three warmups and twenty measured samples. The newer attention/submission comparison has its own settings and records. Raw results and exact commands are in [benchmarks](benchmarks); the hardware record is [environment.json](benchmarks/environment.json).

## Original full-model baseline

| Operation | Backend / precision | Workload | Median | p95 | Tokens/s | Peak process RSS |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| Forward + backward + AdamW | MPS / FP32 | Batch 2, length 128 | 30.764 ms | 31.149 ms | 8,321 | 1,968 MiB |
| Forward + backward + AdamW | MPS / FP16 | Batch 2, length 128 | 28.189 ms | 28.519 ms | 9,081 | 1,925 MiB |
| Prefill | MPS / FP32 | 128 tokens | 4.990 ms | 5.160 ms | 25,651 | 444 MiB |
| Prefill | MPS / FP16 | 128 tokens | 5.053 ms | 5.535 ms | 25,334 | 417 MiB |
| One cached decode token | CPU / FP32 | 256-token prompt | 4.613 ms | 4.704 ms | 217 | 422 MiB |
| One cached decode token | MPS / FP32 | 256-token prompt | 3.207 ms | 3.761 ms | 312 | 625 MiB |
| One cached decode token | MPS / FP16 | 256-token prompt | 3.458 ms | 3.722 ms | 289 | 603 MiB |

Training timing includes forward, backward, and the device optimizer update; it excludes `zero_grad` preparation, validation, checkpoint writes, and initial compilation. It is one microbatch per update, rather than the default 16-batch accumulation setting. Prefill and decode benchmarks use newly initialized weights and seeded inputs, so these numbers measure execution speed rather than language quality. Decode excludes its preparatory prefill.

Peak RSS is the process high-water mark and includes the CPU correctness reference and graph compilation. It is not a dedicated measurement of steady-state server memory or all driver allocations. FP16 retains FP32 master/gradient/moment buffers, so its training memory reduction is modest.

The benchmark compares the CPU absorbed-cache decode against its key/value reconstruction fallback: **1.05×** at this context length. Cache compression's clearest benefit is memory. The logical maximum-context cache is 11.25 MiB in FP32 and 5.625 MiB in FP16, compared with 80 MiB for expanded FP32 keys and values. Additional combined projections and GPU double buffers also consume memory.

The GEMM benchmark (`M=256, N=1360, K=512`) measured 0.150 ms for Accelerate versus 12.808 ms for the scalar reference, with zero observed output difference: **85.3× for that matrix operation**. This is not an end-to-end engine speedup.

FP16 improved training throughput by about 9% here. FP32 cached decode was faster than FP16. Use workload measurements to choose precision; memory, conversion overhead, and dispatch costs affect the result.

## First-token latency

Warm 128-token prefill takes approximately 5 ms. A fresh process generating from the trained full-model FP16 export measured **409 ms** to its first token using FP32 GPU execution and **824 ms** using FP16. Those engine timings include graph compilation and sampling but exclude executable startup and model-file loading. The 16-token test prompt generated EOS immediately, so it provides first-token timing rather than a sustained decode measurement. See [cold-generation.json](benchmarks/cold-generation.json).

New graph shapes have a cold-start cost. The server retains compiled programs, but sufficiently many different shapes can evict an LRU entry. Its metrics expose queue, prefill, decode, and first-token time sums; they do not provide a latency histogram. Measure wall-clock request latency and p95 under the intended concurrency separately.

## Learning and correctness

The original engine verification passed **7/7 CTest targets**, containing **56 C++ cases and 7 Python cases**, with actual Metal and loopback access. The attention/submission update has a separate current record below. The independent PyTorch oracle is installed and was exercised. [Release log](verification/final_release_tests.log), [GPU log](verification/final_mps_tests.log), [server log](verification/final_server_tests.log), and [machine-readable results](verification/results.json) preserve the evidence.

| Check | Observed result |
| --- | --- |
| Full-model FP32 GPU pilot, 12 updates | Training NLL 10.2566 → 5.4029; held-out NLL 6.0096 at step 6 → 5.3077 at step 12 |
| Tiny CPU SFT, 80 updates | NLL 5.6082 → 0.02754; greedy `Say hi.` → `Hi!` |
| Tiny FP16 GPU SFT, 80 updates | NLL 5.6082 → 0.02743; greedy `Say hi.` → `Hi!` |
| Tiny FP16 GPU DPO, 8 updates | Preference loss 0.69315 → 0.23454; reference frozen and reload verified |
| CPU GRPO, fixed held-out prompt/seeds | Reward 0.0625 at step 20 → 0.5 at step 80; final sampled validation KL 0.1694 |
| FP16 GPU GRPO smoke run | Online rollouts, multiple gradient epochs, finite updates, reference/checkpoint reload verified |
| CPU SFT/DPO/GRPO replay | Recomputed weights and optimizer moments match byte-for-byte; reference and RNG/data behavior preserved |
| Full-model FP32 GPU replay, steps 7–12 | Data/RNG/options match exactly; maximum weight difference `1.19e-7`, moment difference `3.73e-9` |
| Native FP32 GPU versus PyTorch | Maximum logit error `9.54e-7`; every parameter gradient and one clipped AdamW update pass |

The full-model pilot is a numerical/learning check on the provided small corpus, rather than a pretrained general language model. The GRPO evaluator is a deterministic toy task, with sixteen fixed-seed held-out samples at each evaluation. Its improvement is evidence that the online objective works, not a general capability result. The longer unconstrained reward experiment is not used as the acceptance test; the checked run uses `kl_coefficient=0.05` and an 80-update schedule.

GPU replay is **not bit-for-bit guaranteed**. Checkpoint values restore exactly, but later GPU arithmetic can differ in the last bits. The measured comparison passes `atol=2e-6, rtol=1e-5`; its actual errors are much smaller. Existing checkpoint files are only reused after an exact comparison. Move later outputs aside when replaying a GPU run from an older checkpoint. [Replay comparison](verification/full_replay_report.json).

The CPU UndefinedBehaviorSanitizer build passes all six enabled CPU/tools targets, including the loopback integration check. Metal is disabled for that build and is verified separately in Release. [UBSan log](verification/final_ubsan_tests.log).

AddressSanitizer compilation succeeds, but its runtime hangs before `main` on this machine, including a standalone one-line program. ASan execution remains **unverified**. Similar initialization hangs are documented in primary [LLVM](https://github.com/llvm/llvm-project/issues/200447) and [CPython](https://github.com/python/cpython/issues/145199) reports; those reports do not establish the cause of this machine's hang. The `MINI_MLA_ENABLE_ASAN` build option remains available.

## FP16 drift

The trained full-model comparison uses 32 held-out next-token positions. CPU FP32 master-weight evaluation is the baseline, with NLL 5.3077056 and perplexity 201.8865.

| Path | Maximum absolute logit error | NLL difference | Relative perplexity difference |
| --- | ---: | ---: | ---: |
| FP32 MPS with master weights | `3.81e-6` | `+2.48e-7` | `+0.000025%` |
| FP16 exported weights, CPU FP32 arithmetic | `6.44e-4` | `-3.17e-5` | `-0.003174%` |
| FP16 MPS computation from master weights | `1.24e-3` | `+2.63e-6` | `+0.000263%` |

These are observed errors on a short pilot, rather than universal bounds. Recheck a representative validation set after larger training runs or architecture changes. [Raw precision comparison](verification/fp16-drift.json).

## Optimization methods

The [matched PyTorch comparison](stack_comparisons.md) checks identical weights, inputs, gradients, and a clipped AdamW update before comparing native training against eager and compiled MPS execution with explicit attention and SDPA. It runs performance separately from the CPU oracle and tensor export. Its scope and memory accounting differ from the original native-only benchmarks above.

Already implemented: Accelerate GEMM, reused CPU activation/backward storage, device-resident GPU gradients and optimizer state, bounded backward/update submission, tiled Metal attention forward/backward, local projection/FFN recomputation, compiled graph reuse, FP16 matrix computation with FP32 updates, shared RoPE tables, compressed MLA caches, query/value projection absorption, active-context decode buckets, native heap BPE, mmap shards, and bounded input prefetch. Each optimization retains a CPU or numerical comparison path.

For the next round of work:

1. **Measure the actual bottleneck.** Benchmark the intended context, batch, precision, and request mix. Separate compilation, loading, prefill, decode, and queue time. Compare repeated runs and retain correctness results with every timing result.
2. **Tune useful batch size.** Increase microbatch size until memory or latency becomes limiting, then use gradient accumulation to reach the desired effective batch. Track supervised positions as well as padded positions. Training metrics count padded processed tokens; GRPO gradient-epoch timing excludes the preceding rollout/evaluation cost.
3. **Profile the remaining synchronization and casts.** Backward/update submission now waits once per bounded window. DPO/GRPO probability evaluation and token sampling still require host results. Move those computations on-device only with objective parity checks, Inference FP16 weights are now converted once when loading.
4. **Tune the optional Flash path.** It avoids full attention matrices and recomputes local intermediates, but is slower in the measured workloads. Profile matrix tiles, memory occupancy, and stage dispatches before changing the default. Compare every gradient, causal tails, and total memory alongside time.
5. **Improve variable-length workloads.** Bucket SFT examples by length and compile padded prefill buckets with explicit valid-token masks. Preserve the current loss normalization and exact consumed-data state. Check both shorter prompt replacement and bucket growth.
6. **Improve server throughput.** The MLA server now shares immutable weights/graphs across private request caches. Batched MLA decode remains future work; Gemma has its own continuous batch scheduler. Further changes must retain bounded queueing, per-request seeds, deadlines, stop handling, and disconnect cancellation.
7. **Evaluate algorithm changes on quality per second.** Compare AdamW and hybrid Muon with the same data split and effective batch, tune their learning rates separately, and record wall-clock time to a fixed validation threshold. Quantization needs its own precision/quality gate. Neither optimizer choice nor lower precision is an automatic speed or quality improvement.

Quantization and continuous batching are implemented for the optional Gemma text path. MLA quantization, general configurable activation checkpointing, distributed training and MLA continuous batching remain extensions. Flash attention with local recomputation and bounded submission are implemented and verified separately below.

The verification directories retain logs, run options, tensor manifests, state summaries, and comparisons. Large temporary pilot weights were removed after verification. The two small trained SFT demos remain in `artifacts`; `config/pilot.json` and `examples/pilot-data` reproduce the full-model pilot.

## Attention and submission measurements

The current update passes **7/7 Release targets (61 C++ cases and 7 Python cases)** with actual Metal and loopback access, plus all seven GPU cases with Metal resource/shader validation enabled. The CPU UBSan build passes six enabled targets. The [new verification record](verification/attention-sync/results.json) is separate from the original baseline.

The FP32 Flash path passes the independent PyTorch comparison for logits, every parameter gradient, and a clipped AdamW update. Maximum logit error is `2.15e-6`; maximum parameter-gradient error is `2.38e-6`. Tests cover FP16, head widths 16/64/128, unequal head widths, partial causal tiles, signed coefficients, both optimizers, mixed-shape accumulation, invalid requests, and prefill/cache replacement. [Oracle report](verification/attention-sync/flash-oracle-report.json), [Metal validation log](verification/attention-sync/final-metal-validation.log).

Actual Flash learning checks include the full 35.8M FP32 model's four-update pilot (training NLL 10.2566 → 7.2750, held-out NLL 7.3204 → 6.6736), tiny FP16 SFT (5.6082 → 0.02750, greedy `Hi!`), eight DPO updates (0.69315 → 0.22896), and eight online GRPO updates with two gradient epochs per group. SFT/DPO/GRPO replay preserved all weights/moments and RNG/data state in these runs; both preference references remained unchanged. These are numerical and toy-learning checks. [Replay report](verification/attention-sync/flash-replay-report.json).

The following final runs use the same full model, FP16 computation, three warmups, and thirty measured samples, executed individually. Timing includes forward/backward/update and excludes initial compilation, gradient zeroing, validation, and checkpoints. The benchmark includes a CPU correctness reference. Raw JSON, exact commands, and environment/binary metadata are in [attention-sync benchmarks](benchmarks/attention-sync).

| Attention | Batch × length | Accumulation | Window | Host waits/update | Median | p95 | Peak RSS |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| reference | 2 × 128 | 4 | 1 | 5 | 87.470 ms | 87.865 ms | 1925 MiB |
| reference | 2 × 128 | 4 | 4 | 1 | 80.312 ms | 85.507 ms | 1941 MiB |
| reference | 1 × 512 | 1 | 4 | 1 | 34.886 ms | 35.109 ms | 2109 MiB |
| flash | 1 × 512 | 1 | 4 | 1 | 58.981 ms | 59.360 ms | 2134 MiB |
| reference | 1 × 1024 | 1 | 4 | 1 | 56.957 ms | 57.916 ms | 2588 MiB |
| flash | 1 × 1024 | 1 | 4 | 1 | 136.766 ms | 137.421 ms | 2632 MiB |

At length 128 with four accumulated microbatches, window four reduced five host waits to one and improved median update throughput by about **8.9%** in this run. Reference attention submits five graphs in either case: the change removes intermediate CPU fences. With 16 microbatches and window four, the implementation uses four waits instead of seventeen, verified by the windowing design and partial-window tests.

The custom Flash path is **slower here**: about 59 ms versus 35 ms at length 512, and 137 ms versus 57 ms at length 1,024. Its matrix tiles improved the first scalar tiled implementation, but graph segmentation, recomputation, and custom kernel execution still cost more than the optimized reference path. It submits 44 graphs per full-model single-microbatch update versus two for reference attention, as well as the custom Metal attention kernels. Therefore `auto` retains reference attention; Flash is available explicitly for study and further tuning.

Flash avoids full token-to-token score/probability allocations, but **total peak process RSS did not fall** at these lengths. That high-water mark also includes CPU reference activations, compiled programs, logits, weights, and optimizer state; it does not isolate attention allocation or GPU driver memory. Further memory claims require a GPU allocation capture at the intended workload. The implementation provides an algorithmic storage reduction, not an established end-to-end memory or speed advantage on this device.

Use the [attention/synchronization guide](attention_and_synchronization.md) for flags, equations, error boundaries, and reproduction commands.
