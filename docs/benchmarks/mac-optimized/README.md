# Mac optimization records

Recorded on an Apple M2 Max with 64 GiB unified memory and macOS 26.7.1, on 2026-10-07. Read [the guide](../../mac_training_and_serving.md) for results and reproduction commands.

- `training-torch210`: final native code versus PyTorch 2.10, eight eager/compiled, explicit/SDPA, foreach/fused variants. Gradient reset is timed. Every variant passed its gradient and update gates.
- `training-torch213`: earlier fused optimizer trial against PyTorch 2.13. It predates the tile-map refinement and is retained to make the large version difference visible. It is not the strongest training baseline.
- `serving-first`: our fast and full-prefill Gemma servers versus stock vLLM-Metal. The C++ file was subsequently formatted; the binary/source hashes identify the measured build.
- `serving-repeat`: vLLM tested first, both stock and opt-in compiled-MLP modes, followed by our fast server. This record uses the final formatted C++ runner. Both serving records use the same pinned checkpoint and pretokenized inputs, with prefix caching and speculative decoding disabled.
- `optimizer-ablation`: paired native timings before and after lazy gradient reset/fused AdamW. These earlier timings exclude gradient reset and precede the tile-map refinement; do not mix their scope with the current training runner.
- `inference-weight-ablation`: paired native MLA prefill/decode measurements using the saved pre-optimization executable and final cached-weight executable. `commands.json` includes both executable hashes.

Performance and numerical checks run separately, with serial GPU processes. Timings exclude compilation/loading. Serving includes HTTP/SSE, GPU completion, scheduling and output detokenization. EOS is ignored to force 64 output tokens; token agreement is recorded separately. These synthetic workloads do not measure language quality or training time to a validation target.

Tensor manifests are preserved; the roughly 429 MB training tensor binaries, compiler caches, Python environment and 3.55 GB pretrained checkpoint remain outside the documentation. Regenerate tensor data with `compare_training.py`. `gemma-source.json` records the public model revision and checksum; `setup_metal.py --download-gemma` downloads that revision. Process RSS, MLX active/peak allocations and vLLM's reserved KV pool have different accounting scopes and should not be compared as interchangeable memory measurements.

[Verification logs](../../verification/mac-optimized) include Release with actual Metal shader validation and loopback access, CPU UBSan, Gemma engine/HTTP checks and a tiny FP16 SFT learning run.
