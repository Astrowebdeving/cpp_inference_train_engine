# Throughput follow-up records

See the [guide](../../throughput_followup.md) for results and limits.

Every comparison alternates the original and final binaries in one session; `runs_ms` keeps each run. The original binaries were copied before the main build tree was rebuilt. Timings on this Mac drift by a few percent over an hour, so numbers from different sessions are not compared.

| File | Contents |
| --- | --- |
| `training-reference-shape.json` | AdamW and Muon at 8 × 128, including the serial and `gram` variants of the final binary. |
| `training-batch-scaling.json` | 16, 32 and 64 sequences of 128 tokens. |
| `training-sequence-length.json` | 256 to 2,040 tokens per sequence. |
| `training-accumulation.json` | Microbatch 2 × accumulation 16, with and without coalescing. |
| `inference.json` | MLA cached decode and prefill, with the logit error against the CPU reference. |
| `serving.json` | Gemma HTTP bursts and long generations; the engine binary is the same in both runs. |
| `muon-gates` | Full-model first-update comparisons between Newton–Schulz orders. |
| `micro` | MPSGraph micro-benchmark output: matrix layout and the vocabulary projection. |
| `scripts` | What produced each record. They carry the recorded workspace paths. |

In `micro/vocabulary-head-variants.txt` the first row is the flattened FP16 projection, the form that meets MPS's slow path above about 1,100 rows. It is not the original engine's three-dimensional form, which met that path only when one sequence was that long.

`scripts/det.sh` and `scripts/detbench.sh` run the same training three times in separate processes and compare checkpoint hashes. `scripts/detmicro.mm` hashes the output of single operations across processes; every operation it covers is reproducible in isolation, which is how the embedding scatter was isolated to the training graph.

`sources/original` holds every source, test, tool and configuration file this work modified, as it was beforehand, with the earlier `README.md` and `performance_audit.md`. The project is not under version control, so this is the reference for reviewing or reverting a change. `docs/muon_on_metal.md`, `docs/mac_training_and_serving.md` and `docs/attention_and_synchronization.md` each gained a short passage and are not copied. The original binaries used for the comparisons were built from these sources and are not kept.
