import argparse
import array
import hashlib
import os
import struct
from pathlib import Path

from common import Tokenizer, canonical, input_paths, records, sha256_file


class ShardWriter:
    def __init__(self, output, split, tokenizer, limit):
        self.output, self.split, self.tokenizer, self.limit = output, split, tokenizer, limit
        self.index = self.count = self.total = 0
        self.stream = None
        self.paths = []

    def append(self, ids):
        while ids:
            if self.stream is None:
                self.path = self.output / f"{self.split}-{self.index:05d}.tok"
                self.temp = self.path.with_suffix(".tok.tmp")
                self.stream = self.temp.open("w+b")
                self.stream.write(bytes(160))
                self.digest = hashlib.sha256()
                self.count = 0
            n = min(len(ids), self.limit - self.count)
            values = array.array("H", ids[:n])
            if os.sys.byteorder != "little":
                values.byteswap()
            payload = values.tobytes()
            self.stream.write(payload)
            self.digest.update(payload)
            self.count += n
            self.total += n
            ids = ids[n:]
            if self.count == self.limit:
                self.finish()

    def finish(self):
        if self.stream is None:
            return
        header = struct.pack(
            "<8sIIIIQ32s32s64s",
            b"MLATOK01",
            1,
            160,
            1,
            self.tokenizer.vocab_size,
            self.count,
            bytes.fromhex(self.tokenizer.hash),
            self.digest.digest(),
            bytes(64),
        )
        self.stream.seek(0)
        self.stream.write(header)
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.stream.close()
        self.temp.replace(self.path)
        self.paths.append(str(self.path))
        self.path.with_suffix(".tok.json").write_text(
            canonical(
                {
                    "format": "mini-mla-token-shard",
                    "version": 1,
                    "split": self.split,
                    "token_count": self.count,
                    "tokenizer_sha256": self.tokenizer.hash,
                    "payload_sha256": self.digest.hexdigest(),
                }
            )
        )
        self.index += 1
        self.stream = None


def main():
    parser = argparse.ArgumentParser(
        description="Stream text or chat JSONL into checked training files."
    )
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=["pretrain", "sft", "dpo", "grpo"], default="pretrain")
    parser.add_argument("--validation-fraction", type=float, default=0.01)
    parser.add_argument("--shard-tokens", type=int, default=10_000_000)
    parser.add_argument("--sequence-length", type=int, default=2048)
    parser.add_argument("--truncate-prompt", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.validation_fraction < 1 or args.shard_tokens < 2 or args.sequence_length < 1:
        parser.error("Invalid split, shard size, or context length")
    tokenizer = Tokenizer.load(args.tokenizer)
    paths = input_paths(args.input)
    args.output.mkdir(parents=True, exist_ok=True)
    if (
        list(args.output.glob("train-*"))
        or list(args.output.glob("val-*"))
        or (args.output / "manifest.json").exists()
    ):
        raise ValueError("Output already contains a dataset; choose a new directory")
    writers = {
        s: ShardWriter(args.output, s, tokenizer, args.shard_tokens) for s in ("train", "val")
    }
    streams, files = {}, {}
    counts, supervised = {"train": 0, "val": 0}, {"train": 0, "val": 0}
    documents = byte_count = truncated = 0
    try:
        for row in records(paths):
            identity = row["text"] if args.mode == "pretrain" else canonical(row)
            if not isinstance(identity, str):
                raise ValueError("Expected text field")
            raw = identity.encode("utf-8", errors="strict")
            split = (
                "val"
                if int.from_bytes(hashlib.sha256(raw).digest()[:8], "big") / 2**64
                < args.validation_fraction
                else "train"
            )
            documents += 1
            byte_count += len(raw)
            counts[split] += 1
            if args.mode == "pretrain":
                writers[split].append(tokenizer.encode(identity) + [tokenizer.specials["<|eos|>"]])
                continue
            if split not in streams:
                suffix = "sft.jsonl" if args.mode == "sft" else f"{args.mode}.jsonl"
                files[split] = args.output / f"{split}-00000.{suffix}"
                streams[split] = (
                    files[split]
                    .with_suffix(files[split].suffix + ".tmp")
                    .open("w", encoding="utf-8")
                )
            if args.mode == "sft":
                ids, mask = tokenizer.chat_sample(row["messages"])
                excess = len(ids) - (args.sequence_length + 1)
                if excess > 0:
                    first_response = mask.index(1)
                    if not args.truncate_prompt or excess >= first_response - 1:
                        raise ValueError(
                            "SFT sample exceeds context; use an explicit prompt truncation policy or a longer context"
                        )
                    ids = ids[:1] + ids[1 + excess :]
                    mask = mask[:1] + mask[1 + excess :]
                    truncated += 1
                streams[split].write(canonical({"tokens": ids, "mask": mask}))
                supervised[split] += sum(mask[1:])
            else:
                fields = (
                    ("prompt", "chosen", "rejected") if args.mode == "dpo" else ("prompt", "answer")
                )
                if not all(isinstance(row.get(key), str) for key in fields):
                    raise ValueError(f"{args.mode} requires string fields: {fields}")
                streams[split].write(canonical({key: row[key] for key in fields}))
        for writer in writers.values():
            writer.finish()
        for split, stream in streams.items():
            stream.flush()
            os.fsync(stream.fileno())
            stream.close()
            temporary = files[split].with_suffix(files[split].suffix + ".tmp")
            temporary.replace(files[split])
            files[split].with_name(files[split].name + ".manifest.json").write_text(
                canonical(
                    {
                        "format": f"mini-mla-{args.mode}-v1",
                        "tokenizer_sha256": tokenizer.hash,
                        "payload_sha256": sha256_file(files[split]),
                        "samples": counts[split],
                        "supervised_tokens": supervised[split],
                    }
                )
            )
    finally:
        for stream in streams.values():
            if not stream.closed:
                stream.close()
        for writer in writers.values():
            if writer.stream is not None:
                writer.stream.close()
    manifest = {
        "format": "mini-mla-dataset-v1",
        "mode": args.mode,
        "tokenizer_sha256": tokenizer.hash,
        "split_rule": "sha256-document-first64-v1",
        "validation_fraction": args.validation_fraction,
        "documents": documents,
        "bytes": byte_count,
        "documents_by_split": counts,
        "tokens_by_split": {s: w.total for s, w in writers.items()},
        "truncated_samples": truncated,
        "sources": [{"path": str(p), "sha256": sha256_file(p)} for p in paths],
    }
    (args.output / "manifest.json").write_text(canonical(manifest))
    print(canonical(manifest).strip())


if __name__ == "__main__":
    main()
