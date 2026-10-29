# Architecture

The engine has a CPU implementation with explicit backward functions and a GPU implementation compiled from the same architecture into MPSGraph. They share the parameter registry, objectives, tokenizer, data loader, checkpoints, sampling, and serving code.

## Source map

| Path | Responsibility |
| --- | --- |
| `include/mini_mla`, `src/core` | Results, checked metadata, owning buffers, borrowed views, JSON, mapping, hashes, half conversion |
| `src/model/config.cpp` | Architecture validation and exact parameter accounting |
| `src/model/ops.cpp` | Accelerate GEMM and CPU norm, RoPE, attention, SwiGLU, and loss derivatives |
| `src/model/model.cpp` | Stable parameter registry, forward tape, manual backward, tied head |
| `src/backends/mps_backend.mm` | Graph stages, autodiff, resident optimizer, accumulation windows, cached decode |
| `src/backends/mps_runtime.hpp`, `flash_attention.metal` | Command/resource ownership, tiled attention forward/backward, dispatch |
| `src/train/data.cpp` | Checked mmap shards, SFT masks, deterministic loader, bounded prefetch |
| `src/train/optimizer.cpp` | AdamW, role-based hybrid Muon, clipping, cosine schedule |
| `src/train/trainer.cpp` | Pretraining/SFT/DPO/GRPO orchestration, evaluation, replay state |
| `src/train/checkpoint.cpp` | Atomic named-tensor checkpoints and model exports |
| `src/infer` | Native BPE, chat formatting, sampling, UTF-8 streaming, compressed cache |
| `src/serve/server.cpp` | HTTP parsing, bounded queue, private worker sessions, SSE, cancellation |
| `apps` | Command-line entry points and benchmarks |
| `tools` | Standard-library preparation and inspection; optional PyTorch/NumPy/Matplotlib tools |
| `tests` | Numerical, state, corruption, learning, and server checks |

## Model equations

For each token, the attention block first RMS-normalizes the residual stream. It computes `q_latent = RMSNorm(x W_dq)` and projects it to per-head non-positional and rotary query coordinates with `W_uq`. A separate `W_dkv` produces a KV latent and one shared rotary key. The KV latent is RMS-normalized before `W_uk` and `W_uv` expand the non-positional keys and values. Adjacent pairs of query and shared-key rotary coordinates rotate by token position.

Scores are the sum of the non-positional and rotary dot products divided by `sqrt(head_dim_nope + head_dim_rope)`. Causal softmax excludes future positions. The attended per-head values concatenate before `W_o`; the result adds to the residual. A second RMSNorm feeds `SiLU(x W_gate) * (x W_up)`, followed by `W_down` and another residual addition. The final RMSNorm uses the token embedding transpose as the language-model head.

The registry contains the embedding, final norm, and 13 named parameters per block. Norm scales start at one; projections use seeded Gaussian initialization with residual projections scaled by `1/sqrt(2 * layers)`. Stable roles distinguish hidden matrices, the tied embedding, and norm scales. Optimizer routing never depends on rank alone.

## Storage and lifetimes

`Buffer` is move-only RAII storage. `Tensor` owns storage and checked dtype/device/shape metadata. `TensorView` and `MutableTensorView` borrow it; they do not extend the owner's lifetime. Typed host access requires a contiguous CPU layout with matching dtype. Matrix math uses contiguous row-major FP32 storage; Accelerate receives explicit transpose flags.

`Model::forward` and `Execution::forward` return borrowed output that remains valid until the next operation on that model/execution. Copy it before another forward, update, or decode if it must survive. An execution owns mutable activations and cache state and cannot be used concurrently. Each HTTP worker gets a private inference execution. Forks share immutable weights and compiled GPU programs; CPU full-forward calls still lock the model's shared tape.

Manual changes through `Model::parameters()` must call `mark_weights_changed()`. CPU projections and caches track this revision. Recreate GPU execution after host-side changes; MPS execution rejects a mismatched host revision. Native optimizer updates perform this invalidation automatically. Always prefill again after an update before decoding.

## GPU execution

Objective-C++ is confined to the backend. ARC owns graph/device objects; C++ owners retain the Metal buffers and command resources. Graph keys include operation mode, batch size, and time capacity. Each execution retains up to 12 whole-model forward/backward/prefill/decode programs. Inference forks share up to 24 immutable program prototypes while retaining private staging and outputs. A separate LRU retains up to 24 Flash stage programs; one block's stages bind each layer's parameters. Update programs are retained separately. Compilation is a cold-start cost.

GPU weights, accumulated gradients, and optimizer moments stay in shared Metal buffers. FP16 training casts FP32 master weights for most matrix operations, retaining FP32 normalization/loss and updates. The tied vocabulary projection uses the master table's FP32 precision to avoid the measured large-shape FP16 slow path. Inference stores immutable FP16 weights once. Reference attention uses whole-model autodiff and materializes quadratic score/probability tensors. The optional Flash path uses custom Metal tiled forward/backward, FP32 attention output and reductions, and linear attention storage. MPSGraph differentiates the surrounding stages; backward recomputes projection/FFN intermediates. It adds dispatch and computation overhead. `auto` retains the faster measured reference path on this Mac. See [attention and synchronization](attention_and_synchronization.md) for equations, limits, and selection.

`Execution::step` borrows batches and coefficient spans until it returns. Call `zero_grad` first. The trainer retains all batches and coefficients across each step, preserving the objective's existing normalization. Up to `submission_window` microbatches have separate host staging and scalar loss buffers. Graphs and Metal kernels are encoded into ordered command batches; gradient buffers alternate on the GPU. The optimizer is encoded with the final window. Host loss reads and staging reuse follow completion, and weights/moments swap only after successful checks. With a window greater than one, there is one CPU wait per window; window one preserves serial backward/update waits.

Command batches retain encoded input/output buffers and executables, including programs evicted from the LRU. Failure drains outstanding GPU work before resources can be reused. Public execution calls return completed results, allowing checkpoint/debug export and host gradient zeroing at explicit boundaries. DPO/GRPO probability evaluation, sampling, validation, and exports retain their required synchronization. Quantization, distributed execution, and continuous batching remain extensions.

## Compressed inference

Each layer caches the normalized KV latent and rotated shared positional key. Non-positional query scores absorb `W_uk` into the query. Weighted latent context projects through a combined `W_uv W_o`, avoiding an expanded value cache. The CPU reference can disable absorption to reconstruct keys and values for comparison.

At the default 2,048-token context, the logical cache contains `10 * 2048 * (128 + 16)` values: 11.25 MiB in FP32 or 5.625 MiB in FP16. Expanded per-head keys and values would occupy 80 MiB in FP32 for this architecture. Combined output projections consume additional memory. The GPU uses double-buffered cache updates, so logical cache bytes are smaller than total physical allocation.

GPU decode uses a power-of-two capacity bucket around the active context, capped at the configured maximum. Its causal mask excludes unused capacity; growing a bucket preserves previous cache entries. Prefill replaces the active cache. Both precisions are checked against full recomputation across bucket boundaries and after prompt replacement.

## Data and update semantics

Pretraining uses contiguous packed spans of `T + 1` tokens with stride `T`; the final incomplete span is dropped. An affine permutation with a coprime multiplier visits each span once per epoch without allocating a shuffle array. SFT stores variable-length examples padded to the requested training length. A two-entry producer queue prefetches batches. Checkpoints save the last consumed loader state, independent of producer progress.

SFT and preference examples are currently indexed in host memory; token shards remain mapped. DPO batches a chosen/rejected pair in one forward and normalizes by pair count. GRPO captures old-policy log probabilities before any group update, freezes reference weights, reuses the group for the requested gradient epochs, and skips groups without reward variation. Its reward, policy, and KL calculations only use completion positions.

All tensor shapes, vocabulary IDs, masks, and finite numerical outputs are validated at public operation boundaries. Checkpoint load also verifies tensor names, roles, layouts, file integrity, configuration, and tokenizer identity. Resume is intended for the same backend, precision, build, and data. CPU replay is bit-for-bit verified. Full-model MPS replay restores data/RNG state exactly and has a measured tolerance comparison; subsequent GPU arithmetic is not guaranteed bit-for-bit. Numerical equivalence across different operating systems or devices requires a new comparison.
