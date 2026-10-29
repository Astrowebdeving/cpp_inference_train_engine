# Optimized Apple-silicon engine

This release brings the standalone C++/Metal MLA trainer and optional Gemma text server together with the recorded performance comparisons.

- **External serving comparison:** approximately 22% higher single-request Gemma throughput and 34% lower median first-token latency than vLLM-Metal 0.30.0 in the recorded M2 Max workload. At eight requests the throughput lead was only 1.5%, within plausible run variability.
- **Native training improvements:** 15.5% more AdamW throughput and 20.2% more Muon throughput at 8 × 128, 55.9% more at 32 × 256, and up to 2.13× with accumulated microbatches, against the previous native engine.
- **Native inference improvements:** cached MLA decode at a 1,024-token context improved from 4.16 to 3.58 ms/token. The updated Gemma serving layer reached 229.0 tokens/s at eight concurrent HTTP requests and 240.7 tokens/s for eight long-output scheduler streams.
- **Implementation:** overlapped CPU encoding and GPU updates, larger coalesced device passes, flattened products, an FP32 training vocabulary projection, six-operation RoPE, composed Muon iterations, incremental detokenization and event-driven streaming.
- **Verification:** recorded full checks cover 71 C++ cases, ten Python tools cases, 51 Gemma engine checks, nine HTTP checks and seven scheduler checks, with numerical gates, actual learning runs and checkpoint replay.

All measurements are from an M2 Max with 64 GiB unified memory. The vLLM comparison predates the last serving-layer update; the native improvement results compare the previous and updated binaries in one session. Workloads, precision, timing scopes and software versions matter. The training gains are not a new matched PyTorch comparison, and differing greedy outputs in the external serving test do not establish quality equivalence.

See the [README performance tables](../README.md#performance), [serving comparison record](benchmarks/mac-optimized/serving-repeat/comparison.json), [native optimization report](throughput_followup.md), and [verification record](verification/throughput-followup/results.json). The [light release review](release_review.md) records what was checked again on 2026-10-29 without rerunning heavy workloads.
