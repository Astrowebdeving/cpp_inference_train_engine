# Compare training stacks

The native engine has a matched PyTorch benchmark. The [new Mac guide](mac_training_and_serving.md) records the fused optimizer update, stronger PyTorch baselines and actual Gemma/vLLM-Metal serving comparison. The results table on this page is the historical baseline. Use it to answer two questions separately: how efficiently an implementation executes the same update, and how quickly a training run reaches useful validation quality. The benchmark answers the first. A held-out training experiment is needed for the second.

## Run the comparison

From `CPP_Completed`:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
python3 tools/compare_training.py --device mps --precision fp32 \
  --sequence-length 128 --microbatch 2 --grad-accum 4 \
  --warmup 3 --repetitions 30 --torch-compile --torch-fused \
  --output artifacts/stack-fp32
```

Choose a new output directory for each run. Repeat with `--precision fp16`, or use `--device cpu --config config/tiny-test.json --sequence-length 16` for a quick check. GPU access must be available to both processes. Python dependencies are in `tools/requirements-dev.txt`.

The runner exports native initial weights, every initial gradient, and the weights after one accumulated, clipped AdamW update. PyTorch loads those exact weights and token IDs. Both use the same MLA architecture, tied embedding, rotary convention, loss normalization, decay exclusions, optimizer constants, precision, and accumulation. FP16 matrix computation retains FP32 master weights, gradients, and moments in both engines, without loss scaling.

Each PyTorch variant must pass numerical checks before its timing is reported. FP16 checks allow rounding drift: tiny gradients can change sign and AdamW can turn that into a larger coordinate difference. Global gradient and update comparisons also check direction and relative size, so a skipped update cannot pass merely because the final weights remain close to their initial values. Limits and observed errors are saved in each parity report.

Two attention implementations are tested: explicit matrix/softmax attention and [PyTorch SDPA](https://docs.pytorch.org/docs/2.10/generated/torch.nn.functional.scaled_dot_product_attention.html). Optional `torch.compile` captures the complete model forward, cross-entropy loss, and generated backward; clipping and AdamW run outside the compiled graph. `--torch-fused` additionally gates and times fused AdamW alongside foreach AdamW. A failed compiled variant is recorded with its log and receives no speed result. The strict CPU correctness oracle remains separate from this benchmark model.

The current runner times gradient reset, forward, backward, global clipping, AdamW, GPU completion, and reading the scalar loss. Native per-microbatch token/target/coefficient staging is included; PyTorch uses preloaded device inputs. Initial loading, compilation, validation, and checkpoints are excluded. Gradient reset was excluded in the historical results below. Clipping uses reduction norms instead of allocating a squared copy of every gradient. Runs execute serially in separate processes; MPS CPU fallback is disabled. `--threads` controls PyTorch and Accelerate's CPU thread limit. `comparison.json` records median/p95, throughput, memory, versions, workload, exact commands, and binary/source/input checksums. The synthetic batch is repeated during accumulation and subsequent updates; this is an execution test, not a language-quality assessment.

Peak RSS is a process high-water mark, including libraries and startup allocations. It does not uniformly account for private GPU allocations across these runtimes. PyTorch's additional MPS driver allocation figure is measured after the run and is a different quantity. Neither is an isolated attention-memory measurement. The native performance process omits the separate CPU oracle and tensor export so those do not inflate its measurement.

## Measured results

Apple M2 Max, 64 GiB unified memory, macOS 26.7.1, native Release build, PyTorch 2.10.0 MPS, full 35,763,840-parameter model, length 128, three warmups and thirty measured updates per variant. All variants passed initial-loss, every-gradient, and accumulated-update gates. These are single serial trials, with compilation outside timing. All entries below are milliseconds per completed update; compiled variants include cross-entropy and its backward.

| Precision, microbatch, accumulation | Native median / p95 | Torch explicit eager | Torch SDPA eager | Torch explicit compiled | Torch SDPA compiled |
| --- | ---: | ---: | ---: | ---: | ---: |
| FP32, 2, 4 | 87.462 / 88.847 | 109.746 | 110.569 | 78.399 | 78.794 |
| FP16, 2, 4 | 80.385 / 82.130 | 324.173 | 602.084 | 76.679 | 76.652 |
| FP16, 8, 1 | 50.071 / 50.788 | 98.918 | 173.767 | 50.960 | 50.911 |

Our original FP32 shape is about 10% slower than compiled PyTorch SDPA and faster than eager execution. FP16 eager SDPA is particularly slow in this tested setup; compilation largely removes the difference. An SDPA API call alone does not guarantee a faster full training loop on this device.

At the larger FP16 microbatch, native throughput rises from **12,739 to 20,451 tokens/s**, about **60.5%**, at the same 1,024 tokens per update. Native and compiled PyTorch are about even there; the 1.7% timing difference is too small to establish a robust advantage from one trial. The native engine submits two graphs instead of five, with one host wait in both cases. This tuning works in both stacks.

FP32 global gradient relative L2 errors are below `2.4e-6` and update errors below `1.4e-4`. In FP16, gradient error reaches about `0.66%` and the first update differs by about `9.1%` in relative L2, with update cosine similarity above `0.9959`. These are precision-tolerant execution checks, not evidence that FP16 follows the same training trajectory. Loss after the repeated-batch updates remains finite; held-out quality is outside this benchmark.

See [FP32 results](benchmarks/stack-comparison/fp32-batch2/comparison.json), [FP16 results](benchmarks/stack-comparison/fp16-batch2/comparison.json), and [larger-microbatch results](benchmarks/stack-comparison/fp16-batch8/comparison.json). Each directory preserves parity reports and compiler logs. [Record details](benchmarks/stack-comparison/README.md) explain source snapshots, staging, and why tensor binaries are regenerated rather than stored in the documentation.

The final Release suite passes **7/7 targets: 61 C++ cases and 8 Python cases**, with actual Metal and loopback access. The new regression checks both PyTorch attention paths, every gradient, one accumulated update, global error metrics, and corrupted-input rejection. [Verification record](verification/stack-comparison/results.json), [full log](verification/stack-comparison/release-tests.log).

## Improve our baseline

**Try a larger microbatch first.** Four small matrix products generally use the GPU less efficiently than one larger product. Compare batch two with accumulation four against batch eight with accumulation one: both process 1,024 tokens per update at length 128. The larger activation workspace uses more memory. The synthetic IDs also differ across these batch shapes, so this experiment measures execution efficiency, not an identical training trajectory. Preserve the effective batch and supervised-token normalization in a real-data experiment.

**Bucket variable-length examples next.** Padding is work that buys no supervised signal. Group similar lengths, retain causal and padding masks, and count supervised tokens per second alongside processed tokens. Packing additionally needs attention boundaries between examples. Verify held-out loss and consumed-data replay before using either in a long run.

**Profile before adding another kernel.** The whole-model graph and bounded submission already remove many dispatches and host waits. Our optional tiled Flash path saves attention storage but was slower in measured full-model runs; `auto` keeps reference attention. More fusion only helps if its saved dispatch or memory cost exceeds its added computation. Use the [attention guide](attention_and_synchronization.md) to isolate these effects.

For optimizer, precision, or architecture changes, run the same data split and several seeds. Tune learning rates for each method, then compare elapsed time to a fixed held-out NLL, plus final NLL at a fixed time budget. Fast synthetic updates alone cannot establish a better training stack.

## Borrowing ideas from vLLM

[vLLM-Metal](https://github.com/vllm-project/vllm-metal) supplies Apple Silicon inference through MLX with vLLM's serving scheduler and paged cache management. It can accelerate generation of RL rollouts; it does not execute our backward or optimizer update. Its [model matrix](https://github.com/vllm-project/vllm-metal/blob/main/docs/supported_models.md) includes MLA families, but our custom architecture and checkpoint are not registered there.

Our MLA server now has separate request caches sharing immutable weights and compiled graphs. Its workers decode independently. The new Gemma server supplies continuous batching. For further MLA work, add continuous batching: each decode step serves active requests, admits new ones, and removes finished ones. Add paged allocation and shared-prefix reuse after measuring fragmentation and concurrency. Preserve our compressed latent/rotary cache semantics rather than replacing them with expanded per-head K/V.

A direct comparison of our custom MLA model with vLLM-Metal still needs a compatible model implementation, weight/config export, and tokenizer conversion first. Gate token IDs, logits, prefill, and cached decode against our engine before benchmarking concurrency 1/4/8, first-token latency, inter-token latency, p95 request latency, output tokens/s, and memory. The training runner makes no inference or vLLM performance claim. The [Gemma serving runner](mac_training_and_serving.md#serve-and-compare) supplies that separate common-model comparison.

Use the [throughput follow-up](throughput_followup.md) for later native-engine changes. Earlier PyTorch comparisons retain their original versions, timing scopes and numerical gates; they are not new measurements of the current engine.
