# Throughput follow-up

This follows the [performance audit](performance_audit.md). It measures where time actually goes in each path, changes what the measurements support, and records what was tried and rejected. Every number is from one Apple M2 Max (64 GiB, macOS 26.7.1) with the original and final binaries run alternately in the same session. Other Macs, model shapes and software versions will differ.

## Results

Full 35.8M model, FP16 compute unless noted, three warmup updates, timing includes gradient reset, forward, backward and the optimizer. "Original" is the implementation as the audit left it.

| Training workload | Original | Final | Throughput |
| --- | ---: | ---: | ---: |
| AdamW, 8 × 128 (the audit's shape) | 48.27 ms | 41.79 ms | +15.5% |
| Muon, 8 × 128 | 88.54 ms | 73.64 ms | +20.2% |
| AdamW, FP32, 8 × 128 | 50.29 ms | 45.29 ms | +11.0% |
| AdamW, 64 × 128 | 307.8 ms | 264.3 ms | +16.5% |
| AdamW, 32 × 256 | 426.2 ms | 273.4 ms | +55.9% |
| AdamW, 8 × 512 | 213.7 ms | 159.1 ms | +34.3% |
| AdamW, 1 × 1536 | 118.3 ms | 85.2 ms | +38.8% |
| AdamW, sequence 256, microbatch 2 × accumulation 16 | 426.1 ms | 289.3 ms | +47.3% |
| AdamW, sequence 128, microbatch 2 × accumulation 16 | 300.0 ms | 140.6 ms | +113% |
| Muon, sequence 256, microbatch 2 × accumulation 16 | 471.5 ms | 327.5 ms | +44.0% |

| Inference workload | Original | Final | Speed |
| --- | ---: | ---: | ---: |
| MLA cached decode, FP16, 256-token context | 3.29 ms | 2.90 ms | +13.5% |
| MLA cached decode, FP16, 1,024-token context | 4.16 ms | 3.58 ms | +16.1% |
| MLA prefill, FP16, 512 tokens | 8.31 ms | 7.35 ms | +13.0% |
| Gemma HTTP, 1 / 4 / 8 concurrent, 64 tokens | 109.9 / 205.2 / 223.4 tok/s | 111.1 / 210.5 / 229.0 tok/s | +1.1 / +2.6 / +2.5% |
| Gemma scheduler, 8 streams × 1,024 tokens | 225.5 tok/s | 240.7 tok/s | +6.7% |

The gains are largest where the original was weakest: longer sequences, larger passes, and small accumulated microbatches. At the audit's own benchmark shape they are 15–20%. The Gemma engine itself is unchanged; its gains come from the serving layer around it. Records: [training](benchmarks/throughput-followup/training-reference-shape.json), [batch size](benchmarks/throughput-followup/training-batch-scaling.json), [sequence length](benchmarks/throughput-followup/training-sequence-length.json), [accumulation](benchmarks/throughput-followup/training-accumulation.json), [inference](benchmarks/throughput-followup/inference.json), [serving](benchmarks/throughput-followup/serving.json).

## Where the time goes

Measuring first changed the plan more than once, so the breakdown is worth keeping.

**Training, 8 × 128, FP16.** Forward plus loss takes about 14.7 ms, backward about 29 ms, the fused AdamW kernel 3.6 ms. Muon's update alone took 44.6 ms, as much as forward and backward together. Each layer costs about 1.15 ms forward and 2.6 ms backward, and its matrix products run near 5 TFLOP/s while the vocabulary product reaches 11. At this size the device is limited by MPS's efficiency on mid-sized products, not by arithmetic capacity.

**FP16 products are not faster than FP32 on this GPU.** Both peak near 11 TFLOP/s and FP32 is quicker on small shapes. FP16 training wins only about 12% in the backward pass. This ruled out running Muon's iterations in half precision before any code was written.

**MLA cached decode is 83% host encoding.** One token spends about 2.7 ms inside MPSGraph's encode call and 0.4 ms waiting for the GPU. Decode speed follows the number of graph operations, not GPU arithmetic.

**Gemma decode is 96–98% MLX kernel evaluation.** Batch one builds its graph in 0.34 ms and evaluates it in about 8 ms, which is roughly the GPU's memory bandwidth over 1.2 GB of packed weights. The process round trip is 0.04 ms. MLX 0.32.1 already ships batch-aware quantized kernels, so the engine was left alone.

## What changed

### Pipelined updates

`Execution::begin_step` encodes and commits update N+1 and only then waits for update N. `finish_step` lands the last one. The host's graph encoding (about 14 ms for AdamW, 23 ms for Muon) now overlaps device execution instead of alternating with it.

A probe suggested this could remove 30% of every update. It removes about 7%, because MPSGraph already flushes work to the device during its own encode. The result is still free: the device executes the same work in the same order, so results are bit-identical.

Two command batches can be alive at once, so everything the host writes for a batch (inputs, the AdamW address table, the optimizer scalars) and everything it reads back (losses, the gradient norm) lives in one of two banks that alternate. Every synchronous entry point first lands the outstanding update. A failed pipelined update is reported one call late and leaves the execution unusable; `step` keeps its transactional behaviour and is still what DPO and GRPO use.

The trainer pipelines pretraining and SFT. An update that is validated, checkpointed, or ends the run lands first, so checkpoints and data positions are unchanged. Other updates are reported one update late; `step_ms` is then the period between consecutive updates. Disable with `--pipeline off`. It is not part of a run's saved options because it cannot change results.

### Coalesced accumulation

Accumulating gradients over several microbatches equals one pass over their concatenation, because the coefficients already carry the update-wide normalization. The device is much more efficient per token on larger passes, so consecutive jobs of equal sequence length are merged up to `--coalesce-tokens` (default 4,096; 0 keeps one pass per microbatch). Sequences longer than 256 tokens shrink the allowance in proportion, since attention memory grows with length squared.

This is the largest practical change for the documented `--microbatch 2 --grad-accum 16` style of run, and it also merges DPO pairs and GRPO rollouts. Summation order differs from separate passes, so the setting is saved with the run and old checkpoints resume with 0. Raise the budget if memory allows: 8,192 measured another 5% at sequence 256.

### Flattened products and an FP32 vocabulary projection

A `[batch, time, in]` activation times a shared `[in, out]` weight is one matrix product. Left three-dimensional, MPSGraph runs `batch` separate products, and the weight gradient becomes `batch` products plus a reduction over them. Multi-sequence graphs now flatten to two dimensions. Single sequences are left alone: there is nothing to merge and the extra reshapes cost decode 3–10%.

Flattening exposed a hazard that needs stating precisely. MPS runs the FP16 product `[rows, 16384] × [16384, 512]`, the hidden-state gradient through the output projection, about ten times slower once `rows` passes roughly 1,100 (2.0 ms at 1,024 rows, 28 ms at 2,048). The original three-dimensional form only met this when a single sequence was that long, which is why 1 × 1536 was slow. Flattening alone would have met it at any batch above 1,024 tokens and made larger batches worse. The same product in FP32 has no such step, so the tied projection is computed in the token table's stored precision: FP32 for training, where the table is the FP32 master, and FP16 for inference, where it is stored FP16 and multiplies one row. Together the two changes are faster at every shape tested.

The FP32 projection does not make FP16 training less accurate. Against the FP32 gradient of the same update, the FP16 gradient's relative error is 0.53% before and after; the final norm's own gradient improves from 4.0% to 0.15% and the FP16 loss error halves. The two FP32 gradients agree to `6e-7`, so the restructuring itself is numerically neutral. `tools/benchmark_model.py`, the PyTorch reference used by `compare_training.py`, now computes its projection the same way. Against it the first FP16 update differs by 8.4% in relative L2 with cosine 0.9965; the original pair measured 9.0% and 0.9959. Most of that figure is sign flips of near-zero gradient elements under AdamW's first step, in both.

### RoPE in six operations

Rotation is now `x * C + swap(x) * S`, with pair cosines `C`, signed pair sines `S` and `swap` exchanging the members of each pair. This equals the textbook form term for term, so the forward pass is bit-identical, in six operations instead of eleven and with no slicing or concatenation. The tables are built once per graph rather than once per call. This is where most of the decode gain comes from, and it also took 4.4% off a training update.

### Prefill projects one position

Prefill selected the last prompt position after projecting every position onto the vocabulary. It now selects first.

### Composed Newton–Schulz order

Each iteration multiplies `X` by a polynomial `P_k` of the Gram matrix `A_k = X_k X_kᵀ`, and `A_{k+1} = P_k A_k P_k`. All of that lives in the narrow `[m, m]` space. The `composed` order forms `Q = P_{s-1} … P_0` there and touches `X` twice: once for `A_0` and once for `Q X`. That is `(4s − 3) m³ + 2 m²n` multiply-adds against `s (2 m²n + m³)` for `gram`, fewer exactly when `n > 1.5 m`. It applies to every Muon matrix in this model except the square `w_o`, which keeps `gram`.

`composed` is the default for new Metal runs; `gram` and `chain` remain, and saved runs keep their recorded order. On the full model's first update, against `gram`: initial weights and gradients bitwise equal, largest per-parameter update relative L2 difference `2.5e-4`, smallest update cosine `0.99999997`, largest weight difference `1.0e-7`. These pass the audit's gates (`1e-3`, `0.99999`, `2e-6`). The difference is about sixteen times the `1.6e-5` between `gram` and `chain`, because errors compound in the Gram space; it remains far tighter than the half-precision iterations common in other Muon implementations. An 80-step FP16 run reduced loss from 5.6082 to 0.1314 and learned `Say hi.` → `Hi!`. [Gate records](benchmarks/throughput-followup/muon-gates).

### Gemma serving layer

The scheduler decoded each request's entire token history on every token. With eight streams of 1,024 tokens that was 8.8% of wall time. The Gemma decoder is a single-character replacement, byte fallback and fusing, which concatenates one string per token except that a run of byte-fallback tokens decodes as a unit and skipped special tokens vanish first. Decoding therefore restarts exactly after any ordinary token. `incremental_decode_plan` recognizes that decoder shape and returns `None` for any other, in which case whole histories are decoded as before. Host overhead for that workload fell to 2.2%.

The same change closes a failure. Previously a byte run that had been streamed and was then invalidated by a later byte raised inside the scheduler thread and stopped it for every request. Text after the last settled point is no longer streamed early, and stop strings are held back relative to that point.

When idle, the scheduler waited 0.5 ms for requests arriving together, but requests over HTTP land a few milliseconds apart, so every burst was split across two or three prefills. A split costs an extra pass and leaves rows at unequal positions, which keeps the batch on the scatter path for the whole generation. The window is now 2 ms, restarts on each arrival, and is capped at four windows. This trades first-token latency for the earliest request in a burst against everyone else's: at eight concurrent requests the minimum rose from 61 to 204 ms while the median fell from 231 to 214 ms and p95 from 244 to 220 ms. A lone request waits 2 ms longer. `--gather-ms 0` admits immediately.

The server now wakes a streaming response from the scheduler thread instead of handing each token to a worker thread and probing for disconnects, and non-streaming responses wake on completion instead of polling every 20 ms.

## Reproducibility

FP16 Metal training gives identical bytes in separate processes, with the defaults above, for both optimizers, on the tiny model and the full one. A pipelined run and a serial run give identical weights and optimizer state.

This was nearly lost. Gathering embedding rows before the FP16 cast looks like a harmless saving, but it turns the embedding gradient into an FP32 scatter-add, and MPS accumulates rows for repeated token ids in an order that changes between runs. FP32 training has always varied between processes for that reason, in the original as well. The table is therefore still cast before the gather, and `test_metal_fp16_training_is_reproducible_and_pipeline_invariant` now guards the property.

## Tried and rejected

| Idea | Finding |
| --- | --- |
| Muon iterations in FP16 | FP16 products are not faster than FP32 on this GPU. |
| One product for the gate and up projections | +0.3%: those products are arithmetic-bound, not launch-bound. |
| MPSGraph optimization level 0 | Identical timing and results. |
| MPSGraph's normalization operation for RMSNorm | −1.6% on training only, no gain on decode, and it changes forward rounding. |
| Passing norm scales without the FP16 round trip | Numerically identical already; MPSGraph folds the casts. |
| Row-chunked FP16 vocabulary projection | 512-row chunks scale linearly but 1,024-row chunks do not, and the trigger is not understood. FP32 has no step at any size tested. |
| Gathering embeddings before the cast | Breaks FP16 reproducibility; see above. |
| Resolving Gemma weights once instead of by name | No measurable change in graph construction time. |
| Threaded CPU decode | CPU decode is bound by memory bandwidth over 143 MB of FP32 weights; threads would give about 1.5×, still behind Metal. |

## Reproduce

```sh
build/bin/mini_mla_bench --kind train --config config/mla36m-v1.json \
  --backend mps --precision fp16 --attention reference \
  --sequence-length 128 --microbatch 8 --grad-accum 1 \
  --warmup 3 --repetitions 30 --time-zero-grad --skip-cpu-check
```

Add `--pipeline off`, `--coalesce-tokens 0` or `--muon-ns-order gram` to isolate each change, and `--optimizer muon` for Muon. For a pipelined run `median_ms` is the steady period between updates; `mean_ms` starts from an idle device and charges every update, including the wait for the last. `tools/compare_training.py` keeps PyTorch's execution structure by default, one synchronized update and one pass per microbatch; pass `--native-pipeline on --native-coalesce-tokens 4096` to time the engine as the trainer runs it.

The [saved scripts](benchmarks/throughput-followup/scripts) show how each table was measured, including the matrix-layout, vocabulary-projection and cross-process micro-benchmarks. They carry the recorded workspace paths.

## Remaining work

| Target | Why |
| --- | --- |
| Batched MLA decode | GRPO rollouts decode one token of one sequence at a time. A rollout-heavy update spends far longer there than in its gradient step. |
| A decode path outside MPSGraph | Decode is bound by MPSGraph's per-operation encode cost. A small set of hand-written Metal kernels would not be. |
| Default backend for `generate` and `serve` | They default to CPU, which decodes the full model in about 4.7 ms per token against 2.9 ms on Metal. |
| Mixed-length Gemma prefill | Only equal-length prompts prefill together; others are admitted one group per scheduler pass. |
| Reproducible FP32 training | Needs an embedding gradient that does not depend on scatter order. |

## Verification

Release passes 7/7 targets with 71 C++ cases and 10 Python cases; the UBSan build passes its six enabled targets. Gemma passes 51 engine checks, 9 HTTP checks and 7 new scheduler checks that need no GPU. The new GPU tests require bitwise equality between pipelined and serial updates across both precisions, both optimizers, both attention paths and multi-window accumulation, and agreement between coalesced and separate passes within FP32 rounding. Two of the new tests were mutation-checked: collapsing the two banks to one, and gathering embeddings before the cast, each make the relevant test fail. With PyTorch installed, the independent oracle and the full-model FP16 gradient and update gates also pass. [Verification record](verification/throughput-followup/results.json).
