# Mac training and serving

This folder contains two engines. The original C++/MPSGraph engine trains the custom 35.8M MLA model from random initialization, including SFT, DPO and online GRPO. The new C++/MLX engine runs **Gemma 4 E2B text inference** with a pretrained 4-bit checkpoint. Gemma has its own architecture and 262,144-token vocabulary; it cannot use our MLA weights or uint16 tokenizer format.

The objective is a fast, verifiable Mac implementation. Performance depends on shape, precision, software version and concurrency. The measurements below do not establish superiority across models or training tasks.

## What changed

| Change | Intuition | Where to look |
| --- | --- | --- |
| First backward overwrites gradients | If the first batch replaces the gradient, clearing and adding zeros wastes memory traffic. Later batches still add their gradients. | `MpsExecution::zero_grad`, `encode_gradients`, graph mode 6 in `src/backends/mps_backend.mm` |
| Fused Metal AdamW | Read each gradient/weight/moment once, then clip and update together. Two reductions compute the global gradient norm without a tensor-sized squared temporary. | `src/backends/adamw.metal`, `MetalAdamW` |
| Inference FP16 weights stored once | Convert when loading, then reuse the smaller weights for every token. Training retains FP32 master weights. | `MpsExecution` constructor and `GraphBuilder::weights` |
| Independent MLA serving sessions | Requests share immutable weights and compiled graphs while each owns its cache, staging and output. One generation no longer locks the entire server. | `InferenceEngine::fork`, `Execution::fork_inference`, `Server::sessions_` |
| Packed Gemma linear layers | Keep weights at four bits in GPU memory; quantized matrix kernels consume them directly. Embedding lookup dequantizes only selected rows. | `GemmaEngine::Impl::linear` and `embedding` |
| Gemma KV reuse | Its final 20 layers reuse earlier keys/values, so they need no separate cache. Sliding layers retain a 512-token ring. | `source_layer`, `KV::cursor` in `src/infer/gemma.cpp` |
| Reduced-query prefill | Once all cache-producing layers have processed a prompt, only the last query is needed through cache-reusing layers to predict the next token. | `reduce_prefill_queries`; disable with `--full-prefill` |
| Continuous Gemma batching | Each decode call serves the current active requests together; finished or cancelled requests leave and new requests enter. Equal-length prompts can prefill together. | `tools/gemma_runtime.py`, `GemmaScheduler` |
| Packed batch KV reuse | Keep compatible batches' combined caches. Aligned rows append together; selected unequal rows write at their own positions and cursors. The automatic policy avoids measured small-batch regressions. | `BatchCache`, `append_cache`, `--all-batch-kv`, `--independent-kv` |
| Inline serving telemetry | Return current engine status with prefill/decode/release, removing the separate status request after each step. | `include_info` in `apps/gemma_main.cpp` and `GemmaProcess::call` |

AdamW keeps FP32 gradients, moments and master weights and uses checked, transactional output buffers. It falls back to MPSGraph on devices without the required Metal argument-buffer support. [Muon on Metal](muon_on_metal.md) now exposes momentum, Nesterov and iteration settings, with batched orthogonalization for repeated matrix shapes. Flash attention remains optional: it saves attention memory, but the reference path remains faster on the measured M2 Max training shapes. Bounded submission still groups backward passes and the update behind one completion fence per window.

The Gemma architecture and scheduling are our code; its matrix, quantization, normalization, RoPE and attention kernels come from [MLX](https://github.com/ml-explore/mlx). The small Python service handles the Hugging Face tokenizer, HTTP and scheduling; it does not run the model forward. This is a text, greedy-decoding implementation. Images, audio, tools, general sampling, LoRA and Gemma training are outside this path. The custom MLA server has independent worker sessions; continuous batching is implemented in the Gemma server.

## Set up Gemma and vLLM-Metal

The verified optional environment uses Python 3.12, Apple silicon, MLX 0.32.1, a pinned MLX-LM commit, Transformers 5.19.0 and vLLM/vLLM-Metal 0.30.0. The supplied MLX library was built for macOS 26.2; these runs used 26.7.1. The base MLA build does not depend on MLX or this Python environment.

Install `uv` and the existing CMake/Xcode prerequisites, then run from `CPP_Completed`:

```sh
python3 tools/setup_metal.py --with-vllm --download-gemma
```

Omit `--with-vllm` for our Gemma engine alone. `--skip-install` rebuilds against an existing environment. The script uses workspace caches, downloads the public [4-bit checkpoint](https://huggingface.co/mlx-community/gemma-4-E2B-it-4bit/tree/238767527555cb75a05732a84dff5d6ba0dd6809), and records its repository revision and weights checksum. The complete tested environment is pinned in `tools/requirements-metal.txt`. Model and environment files occupy several GiB and are ignored by Git.

To configure the optional target yourself:

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DMINI_MLA_ENABLE_GEMMA=ON \
  -DMLX_DIR="$PWD/.venv-serving/lib/python3.12/site-packages/mlx/share/cmake/MLX"
cmake --build build -j 8
```

Verify model numerics and the real HTTP interface:

```sh
.venv-serving/bin/python tests/test_gemma.py --binary build/bin/mini_mla_gemma \
  --model artifacts/gemma-4-E2B-it-4bit --output artifacts/new-gemma-parity.json
.venv-serving/bin/python tests/test_gemma_http.py --binary build/bin/mini_mla_gemma \
  --model artifacts/gemma-4-E2B-it-4bit --output artifacts/new-gemma-http.json
```

The C++ runner also accepts JSON lines on stdin (`prefill`, `prefill_batch`, `decode`, `release`, `info`). Each session retains its logical cache; compatible active requests share packed backing storage until membership changes. Batched decode accepts different current lengths, while `prefill_batch` requires equal prompt lengths. Context and session limits are explicit. Optional `logits` saves an FP32 `.npy` for inspection. `include_info: true` returns status in the same response. `MINI_MLA_GEMMA_TRACE` saves intermediate tensors to an existing directory and is for debugging, never timing.

## Serve and compare

```sh
.venv-serving/bin/python tools/gemma_server.py \
  --model artifacts/gemma-4-E2B-it-4bit --max-batch 8 --context-limit 4096
```

```sh
curl -N http://127.0.0.1:8081/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"mini-gemma-e2b","messages":[{"role":"user","content":"What is gradient descent?"}],"temperature":0,"max_tokens":64,"stream":true}'
```

The service supports `/health`, `/v1/models`, text completions, text chat completions, SSE, token usage, stop strings and disconnect cancellation. When idle it waits `--gather-ms` (default 2) for requests that arrive together, so a burst shares one prefill and stays on the aligned cache path; 0 admits immediately. The [throughput follow-up](throughput_followup.md#gemma-serving-layer) measures that trade and the incremental detokenization that keeps long responses cheap. Queueing, context and batch sizes are bounded. Use one server worker: the scheduler owns one model process. Increase `--max-batch` for throughput while measuring first-token and request latency.

This comparison runs **the actual vLLM-Metal HTTP server**, serially against ours, with the same checkpoint, token IDs, BF16 activations and packed 4-bit weights:

```sh
.venv-serving/bin/python tools/compare_serving.py \
  --model artifacts/gemma-4-E2B-it-4bit --output artifacts/new-serving-comparison \
  --concurrency 1 4 8 --prompt-length 128 --max-tokens 64 \
  --warmup 1 --repetitions 3 --full-prefill-baseline --vllm-compiled-mlp
```

Use `--vllm-first` for a repeat in reversed order. `--vllm-compiled-mlp` also runs with vLLM-Metal's optional compiled-MLP flag, with its decode pipeline enabled in both variants. The installed 0.30.0 wrapper policy targets Qwen MLP classes, not Gemma, so that flag does not add a Gemma optimization here; treat its Gemma result as another trial. It may help a model covered by that policy; inspect the installed implementation and measure your workload.

Use `--independent-kv-baseline` to isolate packed-cache reuse, `--native-only` for native ablations, and `--mixed-lengths` to vary input lengths within a burst. A preserved earlier implementation can be compared with `--binary` and `--native-source-root`. The [performance audit](performance_audit.md) records these follow-up measurements and their limits.

The default cache policy selects shapes from the measured M2 Max results. `--all-batch-kv-baseline` adds a forced-reuse comparison, useful when tuning another Mac. The server itself accepts `--all-batch-kv` and `--independent-kv`; its health response identifies the active policy.

`--vllm-batch-tokens` controls vLLM's prefill budget; it defaults to 512. The audit also measures 1,024. This matters for admission and first-token latency, so preserve it in comparisons rather than treating a single scheduler configuration as vLLM's maximum performance.

The runner records greedy outputs against MLX-LM, client-observed first-token latency, token intervals, p95 latency, aggregate output tokens/s, commands, versions and hashes. It disables prefix reuse and speculative decoding. EOS is ignored to force equally long outputs; these fixed-length completions are throughput inputs, not a quality evaluation. Input tokenization and startup are excluded from throughput; GPU work, scheduling, sampling, output detokenization and HTTP/SSE are included. Native caches grow with requests; vLLM reserves a paged pool, so memory numbers have different scopes.

Our custom MLA checkpoint still needs architecture registration, config/weight export and tokenizer conversion before vLLM can load it. Gemma provides a real common-model comparison now. [vLLM-Metal](https://github.com/vllm-project/vllm-metal) already supports Gemma through MLX, paged caches and its scheduler; its [supported model list](https://docs.vllm.ai/projects/vllm-metal/en/stable/supported_models/) describes compatibility. Reduced-query prefill is also an existing optimization in that stack; this work does not claim a novel upstream contribution.

## Training comparison

Use the strongest available PyTorch paths, including compiled forward/loss/backward and both foreach and fused AdamW:

```sh
python3 tools/compare_training.py --device mps --precision fp16 \
  --sequence-length 128 --microbatch 8 --grad-accum 1 \
  --warmup 3 --repetitions 30 --torch-compile --torch-fused \
  --output artifacts/new-training-comparison
```

Each path must pass the identical-weight gradient and accumulated-update gates. Timing now includes gradient reset: native lazy reset and PyTorch `set_to_none=True`. This changes the scope from historical reports. Both retain FP32 masters, gradients and moments; FP16 compute is unscaled. The final FP16 comparison observed up to 0.66% global gradient relative L2 error and 9.04% first-update relative error, with update cosine similarity at least 0.9959. This permits precision drift, rather than establishing identical training trajectories. The repeated synthetic batch measures update execution; compare held-out loss over several seeds to assess training quality per second. Larger microbatches often matter more than another custom kernel.

## Measured results

Apple M2 Max, 64 GiB unified memory, macOS 26.7.1. Serving uses Gemma 4 E2B, identical pinned 4-bit text weights, 128 input tokens, 64 generated tokens, one warmup and three measured bursts per concurrency. Below is the repeat with vLLM tested first. The compiled-MLP flag does not wrap Gemma in this installed version; differences between those columns are trial variability.

| Concurrent requests | Our output tokens/s | vLLM stock | vLLM with compiled-MLP flag | Our median / p95 first-token latency | Best vLLM median / p95 first-token latency |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 114.0 | 93.5 | 92.3 | 34.0 / 34.1 ms | 51.5 / 53.5 ms |
| 4 | 204.0 | 187.8 | 189.0 | 83.7 / 130.0 ms | 163.5 / 172.9 ms |
| 8 | 226.8 | 221.7 | 223.3 | 225.7 / 229.3 ms | 296.8 / 318.5 ms |

Our throughput lead over the best tested vLLM mode is approximately **22%, 8% and 1.5%** respectively. The first trial measured 118.9/213.1/229.5 tokens/s for ours and 94.4/188.2/220.9 for stock vLLM. The small difference at eight requests is sensitive to run variability. Our fast and full-prefill paths matched all eight 64-token MLX-LM reference outputs in the first trial; the fast path also matched in the repeat. vLLM matched one complete reference output and diverged on the others. A differing greedy sequence is recorded explicitly; its cause and broader quality impact have not been established.

Full-prefill mode measured 106.3/179.0/193.5 tokens/s in the first trial. Reduced-query prefill chiefly lowers first-token work. It remains configurable because equivalent BF16 computations can round differently.

The full custom MLA training benchmark uses length 128, microbatch eight, accumulation one, three warmups and thirty updates. **Native: 48.591 ms/update (21,074 tokens/s). PyTorch 2.10 explicit compiled + fused AdamW: 47.804 ms; compiled SDPA + fused AdamW: 47.878 ms.** Native is about 1.6% slower than the fastest tested PyTorch path, rather than superior. Eager explicit PyTorch took 91.834–95.486 ms. All eight variants passed loss, every-gradient and complete-update gates. A separate full-model FP32 check also passed every-gradient and clipped fused-AdamW update comparison against PyTorch. A separate PyTorch 2.13 run was much slower (best 122.562 ms); that version difference is not used to claim a general 2.5× training advantage.

The native optimizer ablation reduced full-model FP16 updates from 51.646 to 48.459 ms and FP32 updates from 87.095 to 83.387 ms. Those earlier paired timings exclude gradient reset. With cached FP16 inference weights, one native MLA decode token after a 256-token prompt improved from **3.542 to 3.184 ms**, while 128-token prefill changed from 4.860 to 5.029 ms. Both inference paths had the same maximum logit error (about 0.00364) against CPU FP32; this is a decode improvement, not a prefill win.

Records: [training](benchmarks/mac-optimized/training-torch210/comparison.json), [first serving trial](benchmarks/mac-optimized/serving-first/comparison.json), [serving repeat](benchmarks/mac-optimized/serving-repeat/comparison.json), [inference ablation](benchmarks/mac-optimized/inference-weight-ablation), and [verification](verification/mac-optimized/results.json). The verification includes **64 C++ cases, eight Python cases, 23 Gemma engine checks and nine Gemma HTTP checks**, actual Metal shader validation, six CPU/tools UBSan targets, and an 80-step FP16 SFT run from random initialization that learns `Say hi.` → `Hi!`.


## Why the serving lead shrinks with concurrency

On the repeat above, aggregate throughput leads the best vLLM mode by about 22%, 8% and 1.5% at concurrency one, four and eight. Our throughput still rises from 114 to 227 tokens/s; the relative advantage shrinks as vLLM catches up. Eight requests is close to a tie given the small three-burst sample, not evidence of a general advantage.

Each GPU batch has fixed scheduling/dispatch costs plus matrix, attention and memory work. Larger batches spread fixed costs across more tokens and use the GPU more fully. Both engines use MLX kernels for their heavy operations, so a smaller overhead advantage matters less as GPU work grows. This is an explanation consistent with the results; we have not profiled the precise GPU bottleneck. Client-observed token intervals also converge: roughly 8.3 versus 10.3 ms at one request, and 32.3 versus 31.9 ms at eight. Aggregate throughput and each user's token latency are different metrics. Our shorter prefill/first-token time still helps total throughput in this workload.

The follow-up [performance audit](performance_audit.md) profiles the process boundary and implements packed-cache reuse and inline telemetry. Remaining experiments identified in the code:

| Target | Why it could help | What to measure |
| --- | --- | --- |
| Preallocated or paged KV storage | Stable batches now avoid per-row repacking, including unequal positions. Global caches still concatenate growing histories, membership changes rebuild, and mixed lengths still require padding. | GPU memory traffic, allocations, mixed-length logits and cache-wrap correctness. |
| Length buckets and interleaved prefill | Mixed lengths add padding; an admitted long prompt currently finishes its prefill before decoding resumes. | Throughput plus queued/active request p95 latency, with realistic arrival rates. |
| Overlap token handling and next GPU step | The scheduler decodes full token history and emits events before submitting the next step. | CPU detokenization time, idle GPU time, stop/cancellation behavior and SSE token ordering. |

Measure one change at a time across concurrency and context lengths. Prefix reuse and speculative decoding are additional workload-dependent options; they need explicit common settings and quality checks in a comparison. These remaining entries are further work.

## Numerical limits and next experiments

BF16 results can change when batch shapes select different GEMM or attention kernels. Full-prefill mode is useful for close comparison with the official model. Reduced-query prefill is mathematically equivalent, but its smaller matrix shapes can round differently. Preserve both the numerical tests and greedy-output checks; exact token agreement on a finite test set is not a general quality guarantee. The sliding ring uses the same physical ordering as MLX-LM to avoid unnecessary BF16 reduction differences at wraparound.

For the next performance experiment, first profile realistic workloads. Useful extensions are packed/bucketed supervised data, Gemma prompt-length buckets, device-side stochastic sampling, paged/prefix caches, and LoRA with an independent quality gate. A larger serving scheduler can reduce overhead and fragmentation; a fast inference path alone does not implement Gemma backward or training. The [throughput follow-up](throughput_followup.md) records the later runtime and scheduler changes.
