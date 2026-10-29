# Attention and CPU/GPU synchronization

The engine has two training attention paths and a bounded submission window. Both use the same MLA projections, causal mask, loss coefficients, optimizer, and checkpoint format.

## Choose a path

| Option | Behavior |
| --- | --- |
| `--attention auto` | Current measured default: MPSGraph reference attention. |
| `--attention reference` | Explicit score matrix, softmax, and value product, with whole-model MPSGraph autodiff. Also the CPU path. |
| `--attention flash` | Custom Metal tiled attention, including explicit backward. Requires Apple GPU family 7 or later and query/key and value head widths no larger than 128. CPU execution rejects it. |
| `--submission-window 4` | Submit up to four backward passes before the CPU waits; include the optimizer in the final window. New runs default to four. |
| `--submission-window 1` | Serial comparison path: wait for each backward pass, then the optimizer. |

`auto` currently selects the faster measured path on the target M2 Max. Flash remains explicit because its lower attention storage does not guarantee lower total process memory or faster training. [Measurements](performance.md#attention-and-submission-measurements) include that tradeoff. There is no per-step autotuning.

These settings can be fields in a training JSON configuration. Resume restores the saved choices and rejects incompatible overrides. Older checkpoints with neither field retain reference attention and serial submission, preserving their original options identity.

```sh
cd /path/to/your/checkout
python3 tools/smoke.py --backend mps --precision fp16 \
  --attention flash --submission-window 4 --output artifacts/flash-demo

build/bin/mini_mla_train --config config/pilot.json \
  --attention flash --submission-window 4 --output runs/pilot-flash

build/bin/mini_mla_generate --model artifacts/flash-demo/run/model-latest \
  --backend mps --precision fp16 --attention flash \
  --prompt 'Say hi.' --chat --temperature 0 --max-new-tokens 12 --json
```

The option affects full-sequence training, probability evaluation, and prompt prefill. Single-token decode continues to use the compressed MLA cache: there is only one query, so allocating a training-style attention tile would add overhead.

## What Flash changes

Ordinary attention writes all token-to-token scores, then all probabilities. Their size grows as `batch × heads × length²`. Flash computes small query/key tiles, consumes their probabilities immediately, and keeps each row's running maximum, normalizer, and weighted value sum. When a later tile raises the maximum, it rescales the old sum before adding the new tile. This is the same softmax with a different evaluation order.

The shader saves the output and one log-sum-exp value per query. Backward recomputes probabilities tile by tile instead of storing the full matrix. It uses `dS = P × (dP − dot(dO, O))`, with the attention scale included when producing query/key gradients. Separate query and key/value passes give each output gradient one writer, avoiding atomic reductions.

This is a Metal implementation of the [FlashAttention algorithm](https://arxiv.org/abs/2205.14135), rather than the CUDA `flash-attn` package. Scores and reductions are FP32; Q/K/V can be FP16 or FP32. Head widths divisible by eight and at most 64 use SIMD-group matrix operations on 8-query/32-key tiles. Other supported widths use SIMD reductions with four queries per group. Both handle causal boundaries and partial tiles.

MPSGraph owns projection, normalization, FFN, loss, and parameter derivatives. The Flash path splits those into graph stages and joins their activation gradients through the custom attention backward. Projection and FFN intermediates are recomputed during backward. Attention storage is linear in sequence length; computation remains quadratic, and vocabulary logits and other activations still consume memory.

The installed stable MPSGraph fused SDPA entry point was tested before building this path. It rejects unequal query/value head widths, and this SDK could not produce the required training gradients through it. A future SDK must pass the same gradient tests before replacing this implementation. Apple's [SDPA presentation](https://developer.apple.com/videos/play/wwdc2024/10218/) describes the fused operation; it does not establish autodiff support on this installation.

## What fewer host waits changes

The CPU encodes work and the GPU executes it later. Waiting for every microbatch loss repeatedly interrupts that pipeline. The new `Execution::step` submits a bounded window and reads its scalar losses only after the window completes. Parameters, gradients, and moments stay on the device.

For `A` accumulated microbatches, window one uses `A + 1` waits. A window `W > 1` uses `ceil(A/W)` waits, including the optimizer in the last one. With 16 microbatches and window four, that is four waits instead of seventeen. This changes submission timing, not the effective batch or loss normalization.

Each in-flight microbatch has separate token, target, coefficient, and loss staging. GPU scratch can be reused in queue order; the CPU must wait before rewriting staging or zeroing gradients. Command batches retain buffers and graph executables until completion, check errors, and drain outstanding work on failure. Windows are limited to 1–16 to bound staging and queued work. This follows Apple's [CPU/GPU synchronization model](https://developer.apple.com/documentation/metal/synchronizing-cpu-and-gpu-work).

`step`, `accumulate` and `update` still return completed results. The trainer's pretraining and SFT loops instead use `begin_step` and `finish_step`, which encode the next update while the device executes the current one and report each result one update late; see the [throughput follow-up](throughput_followup.md#pipelined-updates). Updates that are validated or checkpointed land first, so both still happen at an explicit boundary. Checkpoint export copies parameters and moments at an explicit boundary. DPO/GRPO still wait when host code needs policy/reference probabilities or a generated token; their backward passes use the window, but rollout generation and objective calculations have additional waits. Training metrics `gpu_host_waits` and `graph_submissions` count the backward/update portion only.

## Compare and debug

Keep configuration, length, precision, effective batch, and warmups identical. For synchronization, compare reference attention with window one and four. For attention, keep window four and compare reference against flash.

```sh
build/bin/mini_mla_bench --kind train --config config/mla36m-v1.json \
  --backend mps --precision fp16 --sequence-length 128 --microbatch 2 \
  --grad-accum 4 --attention reference --submission-window 4 \
  --warmup 3 --repetitions 30 --output artifacts/window-benchmark.json

ctest --test-dir build --output-on-failure
MTL_DEBUG_LAYER=1 MTL_SHADER_VALIDATION=1 build/bin/test_mps

build/bin/mini_mla_inspect --model artifacts/demo-cpu/run/model-latest \
  --backend mps --precision fp32 --attention flash \
  --tokens 1,3,5 --targets 3,5,7 --mask 0,1,1 \
  --debug-output artifacts/flash-oracle --debug-update
python3 tools/reference_model.py --compare artifacts/flash-oracle
```

The benchmark reports actual attention selection, loss error, graph submissions, host waits, median/p95, and peak RSS. RSS includes the CPU reference and compilation, so it cannot isolate attention memory. Use a GPU capture for allocation lifetime and occupancy analysis. Extend the parity tests before changing tiles, precision, graph boundaries, or buffer reuse.

## Source map

| File | What to inspect |
| --- | --- |
| `src/backends/flash_attention.metal` | Online softmax, matrix tiles, recomputed backward, causal tails. |
| `src/backends/mps_runtime.hpp` | Shader compilation, pipeline selection, dispatch, retained command resources. |
| `src/backends/mps_backend.mm` | Stage graphs, activation buffers, gradient accumulation, bounded submission. |
| `include/mini_mla/execution.hpp` | Borrowed gradient batches, execution options and counters. |
| `src/train/trainer.cpp` | Pretraining/SFT/DPO/GRPO coefficient lifetimes and step calls. |
| `tests/test_mps.cpp` | Every-gradient comparisons, signed coefficients, widths, precision, cache replacement. |
