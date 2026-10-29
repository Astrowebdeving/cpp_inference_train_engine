# Standalone source review

Reviewed on 2026-10-29 with a deliberately limited check budget. This checkout contains the throughput follow-up recorded on 2026-10-09, including pipelined Metal updates, coalesced accumulation, composed Muon iterations, and the incremental Gemma scheduler.

The 17 source hashes and three runtime binary hashes in [the latest verification record](verification/throughput-followup/results.json) match the local checkout. The five CPU test targets passed again: 56 C++ cases in about 0.2 seconds. The seven host-only Gemma scheduler checks also passed using the local tokenizer, and the reviewed scheduler/server Python files passed lint. No model training, GPU validation or performance benchmarks were rerun for this review.

The recorded full verification contains 71 C++ cases, ten Python tools cases, 51 Gemma engine checks, nine HTTP checks, seven scheduler checks, CPU UBSan checks and actual learning/replay runs. These remain historical results on the recorded M2 Max, rather than fresh cross-device validation. The throughput improvements compare the updated engine with its own previous implementation; they do not establish a new win over the strongest PyTorch baseline or every vLLM configuration.

The standalone README now describes private server sessions correctly, uses portable checkout paths and reproducible smoke-output locations, and links only to documentation inside this project. The separate tutorial is not required or included. Source distributions omit build products, virtual environments, downloaded weights, training runs and local caches. Historical benchmark records retain their original paths and hashes as provenance.
