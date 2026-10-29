# Muon on Metal

Use `--backend mps --optimizer muon`. The short name selects the existing `hybrid_muon_adamw` optimizer: Muon updates hidden projection/FFN matrices, while AdamW updates the tied embedding and RMSNorm scales. The update runs on the GPU through MPSGraph's Metal kernels. It does not copy gradients to a CPU optimizer, and it does not require PyTorch or MLX.

Run from `CPP_Completed`, after preparing your own tokenizer and data as described in the README:

```sh
build/bin/mini_mla_train --config config/muon-metal.json \
  --tokenizer data/tokenizer.json --train 'data/pretrain/train-*.tok' \
  --validation 'data/pretrain/val-*.tok' --output runs/pretrain-muon
build/bin/mini_mla_train --resume runs/pretrain-muon/checkpoint-latest
```

Or add these options to an existing training command:

```sh
--backend mps --optimizer muon --muon-layout batched --muon-ns-order composed \
--muon-momentum 0.95 --muon-nesterov true --muon-ns-steps 5
```

The preset uses the 35.8M MLA architecture, FP16 forward/backward compute, FP32 optimizer arithmetic, reference attention selected by `auto`, and submission window four. It starts from random weights unless `--initialize` is supplied. The learning rate is a starting setting, not a tuned recommendation. These options apply to MLA pretraining, SFT, DPO and GRPO; the separate Gemma text engine does not implement training.

## What Muon does

Think of a weight matrix as learning several directions at once. A gradient can be dominated by a few large directions. Muon first smooths gradients with momentum, then approximately balances those matrix directions using Newton–Schulz iterations. This is different from AdamW's element-by-element scaling. It is useful for hidden matrices; embeddings and normalization scales keep AdamW, following the [Muon parameter-routing guidance](https://github.com/KellerJordan/Muon).

Our checkpointed `moonlight-fp32-v1` convention uses an EMA momentum, optional Nesterov lookahead, per-matrix Frobenius normalization, coefficients `(3.4445, -4.7750, 2.0315)`, and update scaling `0.2 * sqrt(max(rows, cols))`. The scale follows [Moonlight's example](https://github.com/MoonshotAI/Moonlight/blob/master/examples/toy_train.py). Iterations stay in FP32 and use a norm floor rather than an added epsilon. Tall matrices transpose internally to keep the smaller Gram matrix.

For a wide matrix `X`, `A = X Xᵀ` is a smaller square matrix. The new `gram` order computes `B = -4.775 A + 2.0315 A A`, then `X = 3.4445 X + B X`. The older `chain` order computes `A X` and then `A (A X)`. They represent the same polynomial, but the Gram order replaces one expensive rectangular multiplication with a smaller square multiplication, following the [upstream Muon iteration](https://github.com/KellerJordan/Muon/blob/master/muon.py). Floating-point rounding differs, so the order is checkpointed and both paths remain available.

The `composed` order goes one step further. Every iteration's polynomial `P_k` is a function of the Gram matrix alone, and the next Gram matrix is `P_k A_k P_k`, so the iterations can be multiplied together in the small `[m, m]` space as `Q = P_{s-1} … P_0` and applied to `X` once. That costs `(4s − 3) m³ + 2 m²n` multiply-adds against `s (2 m²n + m³)` for `gram`, which is fewer exactly when the matrix is wider than 1.5 times its narrow side. Squarer matrices use `gram` automatically. Rounding errors compound in the Gram space, so its first-update difference from `gram` is larger than the one between `gram` and `chain`; the [throughput follow-up](throughput_followup.md#composed-newtonschulz-order) records the measured difference, the gates it passes and the 11% it saves.

| Setting | Meaning |
| --- | --- |
| `--muon-momentum 0.95` | Smooth gradients over steps. Valid range is zero up to, but excluding, one. |
| `--muon-nesterov true` | Combine the current gradient with the updated momentum. `false` uses momentum alone. |
| `--muon-ns-steps 5` | Orthogonalization iterations, from 1 to 20. More iterations add matrix work and need their own quality comparison. |
| `--muon-layout batched` | Group same-shaped hidden matrices, up to 32 at a time, into batched GPU matrix operations. Each matrix retains its own momentum and normalization. Default for new runs. |
| `--muon-layout individual` | Previous per-matrix GPU path, retained for comparisons and old checkpoints. |
| `--muon-ns-order composed` | Compose the iterations in the Gram space and apply them to the matrix once. Default for new Metal runs. |
| `--muon-ns-order gram` | Smaller Gram-matrix multiplication order, one iteration at a time. The previous default. |
| `--muon-ns-order chain` | Previous arithmetic order, retained for old checkpoints and numerical comparisons. CPU uses this order. |
| `--grad-clip 1 --weight-decay 0.1` | Global norm clipping and decoupled decay. RMSNorm scales have no decay. |

All optimizer settings, layout and multiplication order are saved. Resume requires the saved settings. Old runs without `muon_layout` retain the individual path; those without `muon_ns_order` retain the chain order and their original options JSON, and runs saved with `gram` keep it. Use a fresh run with `--initialize` when changing settings. These execution choices only affect Metal; CPU retains its reference arithmetic.

## Measured cost

Apple M2 Max, 64 GiB, full 35.8M model, FP16 compute, sequence 128, microbatch eight, one accumulation, five NS iterations, window four. Timings include gradient reset, forward, backward and optimizer; compilation and CPU checks are outside the timer. Each trial has three warmup updates and 30 measured updates; the second trial reverses layout order. These earlier layout measurements both use `chain` arithmetic. The newer Gram comparison is recorded in the [performance audit](performance_audit.md).

| Path | Median step, first / repeat | Training tokens/s, first / repeat |
| --- | ---: | ---: |
| Individual Muon | 113.72 / 113.61 ms | 9,005 / 9,013 |
| Batched Muon | 96.22 / 96.01 ms | 10,643 / 10,665 |
| Fused AdamW, separate trial | 46.68 ms | 21,936 |

Batched Muon saves about 15% of step time, or provides about 18% more tokens/s, over individual Muon here. Both Muon layouts still use two graph submissions and one host wait per update. The packing changes operations inside the optimizer graph. Muon remains more expensive per step than AdamW; compare held-out quality per second over several seeds and tune learning rates separately before choosing an optimizer. The repeated synthetic batch above does not establish a quality advantage. [Commands, hashes and raw results](benchmarks/muon-metal/comparison.json) are preserved.

Reproduce a layout trial, then replace `batched` with `individual`:

```sh
build/bin/mini_mla_bench --kind train --config config/mla36m-v1.json \
  --backend mps --precision fp16 --attention reference --optimizer muon \
  --muon-layout batched --muon-ns-order chain \
  --sequence-length 128 --microbatch 8 --grad-accum 1 \
  --submission-window 4 --warmup 3 --repetitions 30 --time-zero-grad
```

## Verification and further learning

The Metal tests compare weights and both kinds of moment state against the individual path in FP32/FP16, and against CPU in FP32. They include tall/wide/square matrices, clipping on/off, Nesterov on/off, nonzero initial momentum, zero-gradient updates, signed accumulated losses, and save/reload followed by further updates. Replay is bitwise exact for these small tested cases; it is not a promise for every graph or hardware version. CLI tests check saved settings, invalid options and CPU replay. Old options continue to round-trip without changing their resume identity. The final suite passed 67 C++ cases and nine Python cases; six CPU/tools UBSan targets also passed. [Verification records](verification/muon-metal/results.json) include the logs and source hashes.

An actual 80-step FP16 Metal Muon SFT run from random initialization reduced loss from 5.6082 to 0.1313 with the Gram order and to 0.1314 with the composed order, and learned `Say hi.` → `Hi!` in both. [Current verification](verification/performance-audit/results.json) also checks every first-update gradient and weight on the full model. The tiny SFT run verifies learning and export, not general language quality. Reproduce it in a new output folder:

```sh
python3 tools/smoke.py --backend mps --precision fp16 --optimizer muon \
  --output artifacts/new-muon-demo
```

Keep the CPU and individual GPU paths as numerical references when changing the optimizer. Further experiments include separate Muon/AdamW learning rates and alternative orthogonalization precision; each requires checkpoint versioning and a held-out quality gate. The [throughput follow-up](throughput_followup.md#composed-newtonschulz-order) records the current composed-order default and its numerical comparisons.
