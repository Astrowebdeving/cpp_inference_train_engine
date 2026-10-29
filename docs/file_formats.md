# Formats and numerical conventions

All formats are versioned and specific to this engine. Binary data uses little-endian IEEE FP32/FP16 or uint16, as appropriate. Unsupported architectures and malformed files fail explicitly.

## Tokenizer

The JSON format is `mini-mla-tokenizer`, version 1, with `pretokenizer: bytes-v1`, `vocab_size`, sequential `[left, right, new_id]` merges, and `specials`. Base IDs 0–255 represent individual bytes. Merge IDs begin at 256, and each merge may only reference earlier IDs. Ranked merges are applied with stable left-position tie breaking; there is no regular-expression pretokenizer or Unicode normalization.

The final eight IDs are fixed, in this order: `<|pad|>`, `<|bos|>`, `<|eos|>`, `<|user|>`, `<|assistant|>`, `<|system|>`, `<|tool|>`, `<|sep|>`. Literal spellings in ordinary text encode as bytes unless special-token parsing is explicitly enabled. UTF-8 bytes round-trip unchanged, including embedded NUL.

The tokenizer hash is SHA-256 of the exact JSON file bytes, including whitespace. Reformatting the file changes its identity. The Python exporter writes sorted keys, compact separators, ASCII-escaped JSON strings, and a trailing newline. Both preparation and the C++ runtime hash the same bytes. Unassigned merge capacity has empty token bytes and cannot be sampled by ordinary generation.

Chat encoding is BOS, followed by each message's role token, newline, content, EOS, and newline. A generation prompt ends with the assistant role token and newline. SFT supervises assistant content and its EOS; its role marker and surrounding newlines are masked out.

## Token shards

The `.tok` header is exactly 160 bytes. The payload immediately follows with `token_count` little-endian uint16 IDs.

| Offset | Bytes | Field |
| ---: | ---: | --- |
| 0 | 8 | ASCII `MLATOK01` |
| 8 | 4 | Version, 1 |
| 12 | 4 | Header size, 160 |
| 16 | 4 | Dtype code, 1 for uint16 |
| 20 | 4 | Vocabulary size |
| 24 | 8 | Token count |
| 32 | 32 | Tokenizer file SHA-256 |
| 64 | 32 | Payload SHA-256 |
| 96 | 64 | Zero-reserved bytes |

Load validates header fields, exact file size, hashes, reserved bytes, and every token ID. Sidecar JSON and the dataset manifest record split, counts, source hashes, and preprocessing policy. Raw JSONL for DPO/GRPO contains text, rather than pre-tokenized completion IDs; training encodes it with the checkpoint's tokenizer contract.

## Checkpoint directory

`manifest.json` declares the format (`mini-mla-checkpoint` or `mini-mla-model`), version, architecture hash, tokenizer hash, parameter count, step, file hashes, and named tensor entries. Each entry includes name, role, shape, dtype, filename, byte offset, byte count, and CRC32. Offsets must be contiguous and nonoverlapping, and each required tensor must appear exactly once.

Training checkpoints contain `weights.bin`, `optimizer.bin`, `config.json`, `tokenizer.json`, `trainer_state.json`, `manifest.json`, and `COMPLETE`. Weights and moments are FP32. Model exports omit training state and moments and store FP16 or FP32 weights. The contents of `COMPLETE` are the SHA-256 of the manifest. Full-file SHA-256 and per-tensor CRC reject truncation and corruption before use.

Saving writes a sibling temporary directory, flushes and synchronizes files, writes `COMPLETE` last, synchronizes the directory, renames it into place, and synchronizes its parent. `checkpoint-latest` and `model-latest` are atomically replaced relative symlinks. An existing non-symlink at either link path is rejected.

Trainer state contains immutable options, consumed loader state, workflow RNG, token count, dataset and validation fingerprints, reference weight hash, compiler/device information, and best validation cost. Pretraining/SFT/DPO use loss as the validation cost; GRPO uses `1 - reward`. Large RNG and loader counters serialize as decimal strings to preserve all 64 bits through JSON.

Resuming an older checkpoint in a directory with later checkpoints verifies the replayed FP32 weights, moments, and trainer state before reusing any existing output. A conflicting checkpoint or export fails without replacement. Use `--initialize` and a fresh output directory to branch with new hyperparameters or datasets.

## Numerical rules

Norm epsilon is inside the RMSNorm square root. Rotary coordinates use adjacent pairs, not a split-half convention. Attention scales by the combined query/key dimension and uses a strict causal mask. Softmax and cross entropy subtract or account for the row maximum; CPU accumulation uses wider scalars where useful.

Next-token targets shift by one. Loss coefficients are zero for padding and masked prompt positions. Ordinary supervised updates divide by the total supervised positions across all accumulated batches. DPO uses signed per-completion coefficients from its logistic objective and sums completion log probabilities. GRPO uses a sequence mean of completion-token means, clipped ratios, and the k3 sampled KL estimator.

AdamW applies global norm clipping before moment updates, bias correction by completed step, and decoupled weight decay. Norm scales do not decay. Hybrid Muon uses hidden-matrix momentum, Nesterov mixing, five Newton–Schulz iterations by default, coefficients `3.4445, -4.775, 2.0315`, and update scaling `0.2 * sqrt(max(rows, cols))`; the embedding and norm scales still use AdamW. Its serialized variant is `moonlight-fp32-v1`.

FP16 exports round weights once to IEEE binary16. FP16 GPU execution additionally introduces arithmetic drift; compare both logits and held-out loss before accepting a precision change. The performance report contains measured drift and test tolerances. A model export is suitable for inference or initialization, while a training checkpoint is required for exact optimizer/data/schedule resume.
