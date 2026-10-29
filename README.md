# Custom MLA training and inference engine

A standalone training and inference engine for Apple-silicon Macs. Runtime code is C++20 and Objective-C++; CPU execution uses Accelerate, and GPU execution uses Metal/MPSGraph.

**Recorded M2 Max results: approximately 22% higher single-request Gemma throughput than vLLM-Metal, and up to 2.13× training throughput over this engine's previous version.**

## Performance

Measured on an Apple M2 Max with 64 GiB unified memory and macOS 26.7.1. The comparisons below name their baselines separately: external serving-stack results and improvements to the native engine.

### Compared with vLLM-Metal

An earlier same-checkpoint HTTP/SSE comparison against the actual **vLLM-Metal 0.30.0** server used Gemma 4 E2B, packed 4-bit weights, BF16 activations, 128 input tokens and 64 generated tokens. The table uses the better vLLM throughput from the two recorded trials at each concurrency.

| Concurrent requests | Native output tokens/s | vLLM-Metal output tokens/s | Native throughput lead |
| --- | ---: | ---: | ---: |
| 1 | 114.0 | 93.5 | **21.8%** |
| 4 | 204.0 | 189.0 | **7.9%** |
| 8 | 226.8 | 223.3 | **1.5%** |

At one request, median time to first token was **34.0 ms versus 51.5 ms: 34% lower**. Each concurrency had one warmup and three measured bursts, with prefix caching and speculative decoding disabled. The small eight-request lead is within plausible run variability. Greedy outputs differed between backends, so the timing result does not establish quality equivalence. This comparison predates the latest serving-layer improvements; [raw results, versions, commands and token outputs](docs/benchmarks/mac-optimized/serving-repeat/comparison.json) are retained.

### Improvements to this engine

The later optimization pass alternated previous and updated binaries in the same session. Training uses the full 35.8M MLA model, FP16 compute and FP32 master weights and optimizer state. Training times are steady-state update periods, including gradient reset, forward, backward and optimizer work; compilation is excluded.

| Workload | Previous | Updated | Throughput gain |
| --- | ---: | ---: | ---: |
| AdamW, 8 sequences × 128 tokens | 48.27 ms | 41.79 ms | **15.5%** |
| Muon, 8 sequences × 128 tokens | 88.54 ms | 73.64 ms | **20.2%** |
| AdamW, 32 sequences × 256 tokens | 426.2 ms | 273.4 ms | **55.9%** |
| AdamW, 2 × 128 tokens, 16 accumulated microbatches | 300.0 ms | 140.6 ms | **113% · 2.13×** |
| MLA cached decode, 1,024-token context | 4.16 ms/token | 3.58 ms/token | **16.1%** |

The gains come from pipelined updates, coalesced accumulation, flattened matrix products with an FP32 vocabulary projection, fewer RoPE operations, and composed Muon iterations. The Gemma serving-layer update separately improved eight-request HTTP throughput from **223.4 to 229.0 tokens/s**, and long-output scheduler throughput from **225.5 to 240.7 tokens/s**.

These training gains compare against the previous native engine. The strongest matched compiled/fused PyTorch comparison predates these changes; a new matched comparison is needed before claiming that the updated trainer beats it. Numerical gates cover loss, gradients, updates and checkpoint replay. See the [optimization report and per-workload records](docs/throughput_followup.md) for methods, precision differences and rejected experiments.

## Engine

The default model has **35,763,840 parameters**: 10 decoder blocks, width 512, 8 MLA heads, query rank 192, KV rank 128, RMSNorm, SwiGLU, adjacent-pair RoPE, and a tied 16,384-token embedding. The small configuration exists for fast, reproducible correctness checks.

Implemented paths include pretraining, assistant-masked SFT, DPO with a frozen reference, online GRPO with clipped policy ratios and KL regularization, AdamW, optional hybrid Muon/AdamW, complete training checkpoints, FP16 exports, compressed-cache generation, and a local streaming HTTP server. FP16 GPU computation retains FP32 master weights, gradients, optimizer state, and loss calculations.

The optional [Mac training and serving guide](docs/mac_training_and_serving.md) covers fused AdamW, independent MLA server sessions, our C++ Gemma 4 E2B text engine with continuous batching, and a real same-weight vLLM-Metal comparison. Gemma uses a separate pretrained checkpoint and MLX kernels; the base MLA build remains self-contained.

The follow-up [performance audit](docs/performance_audit.md) covers packed-batch KV reuse, fewer serving round trips and the faster Metal Muon multiplication order, with retained comparison paths and full-model numerical checks.

The supplied data is deliberately small. It demonstrates learning and reproducibility; training a useful language model still requires an appropriate corpus and evaluation set.

## Build and run

Requirements: an Apple-silicon Mac, Xcode Command Line Tools, CMake 3.28+, and Python 3.10+. Python's standard library is sufficient for tokenization, dataset preparation, checkpoint inspection, and the smoke test. PyTorch is used only by the optional independent test oracle.

```sh
git clone https://github.com/Astrowebdeving/cpp_inference_train_engine.git
cd cpp_inference_train_engine
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j 8
build/bin/mini_mla_info
ctest --test-dir build --output-on-failure
python3 tools/smoke.py --output artifacts/demo-cpu
```

The smoke test creates a byte tokenizer dataset, trains the tiny model for 80 steps, verifies resume and export, and checks that `Say hi.` produces `Hi!`. Use a fresh output directory for each run:

```sh
python3 tools/smoke.py --backend mps --precision fp16 --output artifacts/demo-gpu
build/bin/mini_mla_generate --model artifacts/demo-cpu/run/model-latest \
  --prompt 'Say hi.' --chat --temperature 0 --max-new-tokens 12 --json
```

Model weights and demo outputs are generated locally and are not included in the repository. Use a fresh output directory for each smoke run. Metal and loopback tests report a skip when those facilities are unavailable to the process; a skip does not verify that path.

For the full 35.8M model's short, already verified pilot, prepared data and a portable run configuration are included:

```sh
build/bin/mini_mla_train --config config/pilot.json
```

This runs twelve FP32 GPU updates with held-out validation. Its tiny corpus exercises the full architecture; the examples remain numerical demonstrations.

## Prepare your own data

Pretraining input is UTF-8 JSONL with a `text` field, or a text file with one document per line. The tokenizer is a deterministic byte-level BPE with eight reserved chat tokens. Its own native format is used by both preprocessing and the C++ runtime; arbitrary Hugging Face tokenizer files are not interchangeable with it.

```sh
python3 tools/train_tokenizer.py --input /absolute/path/corpus.jsonl \
  --vocab-size 16384 --output data/tokenizer.json
python3 tools/prepare_dataset.py --input /absolute/path/corpus.jsonl \
  --tokenizer data/tokenizer.json --output data/pretrain \
  --validation-fraction 0.01 --sequence-length 256
```

The tokenizer trainer uses a bounded sample, controlled by `--sample-bytes`. Increase it for a diverse corpus. A small corpus may produce fewer merges than the vocabulary capacity; those unused slots are excluded from inference sampling. GRPO requires a complete vocabulary because it samples the policy's full action distribution. The supplied 264-token byte tokenizer satisfies that requirement.

Dataset preparation assigns whole documents to splits by content hash, appends EOS, writes checksummed uint16 shards, and records source and tokenizer hashes. Duplicate copies of a document receive the same split. Use a separate validation source when you need a curated benchmark. Native training rejects identical train/validation files; DPO and GRPO also reject identical records across splits.

SFT, DPO, and GRPO use these JSONL records:

```json
{"messages":[{"role":"user","content":"Say hi."},{"role":"assistant","content":"Hi!"}]}
{"prompt":"Say hi.","chosen":"Hi!","rejected":"No."}
{"prompt":"Say a.","answer":"a"}
```

Run `prepare_dataset.py --mode sft|dpo|grpo` with the corresponding file. SFT preparation emits token IDs and supervision masks; only assistant content and assistant EOS contribute to the loss. Long samples are rejected by default. `--truncate-prompt` explicitly permits removing prompt tokens while preserving the assistant response.

## Train and resume

```sh
build/bin/mini_mla_train --config config/mla36m-v1.json \
  --tokenizer data/tokenizer.json --train 'data/pretrain/train-*.tok' \
  --validation 'data/pretrain/val-*.tok' --output runs/pretrain \
  --backend mps --precision fp16 --sequence-length 256 \
  --microbatch 2 --grad-accum 16 --steps 1000 --warmup-steps 20 \
  --learning-rate 0.0003 --validate-every 100 --checkpoint-every 100
build/bin/mini_mla_train --resume runs/pretrain/checkpoint-latest
```

Training expands quoted file globs itself. Gradients from accumulated batches are normalized by the total supervised token count. On Metal, accumulated microbatches are merged into device passes of up to `--coalesce-tokens` tokens (default 4,096; 0 keeps one pass per microbatch), which is the same gradient computed far more efficiently, and pretraining and SFT encode the next update while the current one executes (`--pipeline off` disables this; it never changes results). Validation restarts from a fixed data state and logs loss, perplexity, entropy, and greedy samples. Metrics are JSONL; `rollouts.jsonl` records generated validation samples and RL rollouts.

Checkpoints include FP32 weights, optimizer moments, schedule options, consumed data position, dataset fingerprints, and RNG state. Resume requires matching data, backend, precision, and schedule. CPU replay is bit-for-bit verified. MPSGraph restores the same state and data order but subsequent computation can drift slightly; the measured full-model comparison is in the performance report. Changing the total steps changes the cosine schedule; start a new run with `--initialize` for that case. SIGINT/SIGTERM finishes the current update boundary and saves state. Existing outputs are never overwritten; replay requires an existing checkpoint to exactly match the computed state. Move later outputs aside before replaying a GPU run from an older checkpoint.

Initialize SFT from a training checkpoint to retain full-precision weights:

```sh
build/bin/mini_mla_train --config config/mla36m-v1.json \
  --tokenizer data/tokenizer.json --initialize runs/pretrain/checkpoint-latest \
  --train 'data/chat/train-*.sft.jsonl' --output runs/sft --mode sft \
  --backend mps --precision fp16 --sequence-length 256 \
  --microbatch 2 --grad-accum 8 --steps 500 --warmup-steps 20 --learning-rate 0.0001
```

For a small DPO or GRPO run with the supplied tokenizer and configuration:

```sh
build/bin/mini_mla_train --config config/tiny-test.json \
  --tokenizer examples/tokenizer.json --initialize artifacts/demo-cpu/run/checkpoint-latest \
  --train examples/preferences.jsonl --output runs/dpo --mode dpo \
  --backend cpu --sequence-length 48 --microbatch 1 --grad-accum 1 \
  --steps 16 --warmup-steps 1 --learning-rate 0.001 --dpo-beta 0.1

build/bin/mini_mla_train --config config/tiny-test.json \
  --tokenizer examples/tokenizer.json --train examples/rewards.jsonl \
  --output runs/grpo --mode grpo --backend cpu --sequence-length 48 \
  --microbatch 1 --grad-accum 1 --steps 80 --warmup-steps 2 \
  --learning-rate 0.002 --group-size 16 --completion-tokens 16 \
  --gradient-epochs 2 --reward contains --kl-coefficient 0.05
```

DPO automatically freezes a copy of the initial policy, or accepts `--reference MODEL_DIR`. It uses completion log-probability sums, including EOS. GRPO samples at temperature 1 without top-k/top-p filtering, saves the old-policy log probabilities, standardizes rewards within each group, and applies the clipped objective with the sampled k3 KL estimator. Constant-reward groups skip updates. `exact`, `contains`, and `json` are deterministic built-in rewards; extend `reward_for` to add a task-specific evaluator.

AdamW is the default. `--backend mps --optimizer muon` routes hidden matrices to FP32 Muon on Metal and retains AdamW for the tied embedding and norm scales. The longer name `hybrid_muon_adamw` remains supported. New runs batch same-shaped matrices and compose the Newton–Schulz iterations in the small Gram space; `--muon-layout individual` and `--muon-ns-order gram` or `chain` retain comparison paths. Old checkpoints keep their saved arithmetic. Momentum, Nesterov and iteration settings are exposed. See the [Muon guide and preset](docs/muon_on_metal.md) for commands, checkpoint behavior, measured cost and learning checks. Compare wall-clock time to validation quality before choosing it for a larger run.

## Generate, export, and serve

```sh
build/bin/mini_mla_generate --model runs/sft/model-latest --backend mps --precision fp16 \
  --prompt 'Write a short explanation of caching.' --chat --max-new-tokens 128 \
  --temperature 0.8 --top-p 0.95 --top-k 0 --seed 42
build/bin/mini_mla_inspect --model runs/sft/checkpoint-latest \
  --export artifacts/sft-fp32 --dtype f32
build/bin/mini_mla_serve --model artifacts/demo-cpu/run/model-latest \
  --host 127.0.0.1 --port 8080 --max-new-tokens 16
```

Generation supports repetition penalties, `--stop TEXT`, comma-separated `--stop-tokens`, raw prompts, `--chat`, and `--chat-json` message arrays. Stop strings are held back across token boundaries; output streams contain complete UTF-8 characters. Prompt plus requested completion must fit the configured context.

```sh
curl -N http://127.0.0.1:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"Say hi."}],"temperature":0,"max_tokens":12,"stream":true}'
curl http://127.0.0.1:8080/metrics
```

The MLA server implements `/healthz`, `/metrics`, `/v1/models`, `/v1/completions`, and `/v1/chat/completions`, including SSE and `[DONE]`. It has bounded workers and queue depth, request/body/context limits, disconnect cancellation, and timing counters. Each worker owns an inference session with private cache and staging; immutable model weights and GPU programs are shared. The server binds to loopback by default and has no authentication or TLS. Continuous batching is available in the separate Gemma server described in the [Mac guide](docs/mac_training_and_serving.md).

## Verify and optimize

```sh
python3 -m pip install -r tools/requirements-dev.txt
ctest --test-dir build --output-on-failure
cmake -S . -B build-ubsan -DCMAKE_BUILD_TYPE=Debug \
  -DMINI_MLA_ENABLE_MPS=OFF -DMINI_MLA_ENABLE_UBSAN=ON
cmake --build build-ubsan -j 8
UBSAN_OPTIONS=halt_on_error=1 ctest --test-dir build-ubsan --output-on-failure

build/bin/mini_mla_inspect --model artifacts/demo-cpu/run/model-latest \
  --tokens 1,3,5 --targets 3,5,7 --mask 0,1,1 --debug-output artifacts/oracle --debug-update
python3 tools/reference_model.py --compare artifacts/oracle
python3 tools/plot_metrics.py runs/pretrain/metrics.jsonl --output artifacts/training.png
build/bin/mini_mla_bench --kind train --config config/mla36m-v1.json \
  --backend mps --precision fp16 --sequence-length 128 --microbatch 2 \
  --warmup 3 --repetitions 20 --output artifacts/train-benchmark.json
```

The reference model checks logits, every parameter gradient, and one clipped AdamW update independently in PyTorch. Tests also exercise finite differences, causal masking, tokenizer parity, compressed-cache/full-forward parity, corruption rejection, replay, overfitting, preference signs, RL reward improvement, and HTTP concurrency.

For matched training throughput against eager and compiled PyTorch, run `python3 tools/compare_training.py --device mps --precision fp32 --torch-compile --output artifacts/stack-fp32`. It gates timing on matching gradients and an AdamW update, tests explicit attention and SDPA, and records versions, commands, and checksums. See the [stack comparison guide](docs/stack_comparisons.md) for results, memory limits, microbatch tuning, and later vLLM-style serving work.

See [architecture](docs/architecture.md), [formats and numerical conventions](docs/file_formats.md), and [performance and verification](docs/performance.md). Benchmark kinds are `matmul`, `norm`, `rope`, `attention`, `train`, `prefill`, and `decode`; each reports correctness error alongside timing. Custom Metal Flash attention is optional; `auto` currently retains the faster measured MPSGraph attention path.

## Attention and submission options

New training runs default to `--attention auto --submission-window 4`. The submission window batches GPU work before reading losses on the CPU. Explicit `--attention flash` enables tiled Metal attention with a complete backward pass; `--attention reference` retains the whole-model MPSGraph path. The Flash implementation uses linear attention storage but is slower in the measured workloads on this Mac. See the [usage and explanation guide](docs/attention_and_synchronization.md) and [measured comparisons](docs/performance.md#attention-and-submission-measurements).

The latest recorded full verification passed 71 C++ cases, ten Python tools cases, 51 Gemma engine checks, nine HTTP checks and seven scheduler checks. These are recorded runs on an M2 Max; they do not establish compatibility or maximum performance on every Mac. See the [verification record](docs/verification/throughput-followup/results.json) and [throughput follow-up](docs/throughput_followup.md) for the measured changes, numerical gates and limitations. Benchmark records retain their original commands and machine paths as historical provenance.

## License

The engine is [MIT licensed](LICENSE). Downloaded model weights and external dependencies retain their upstream licenses and terms; they are not included in the source distribution.
