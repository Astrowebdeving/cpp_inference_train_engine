import argparse
import heapq
from collections import defaultdict
from pathlib import Path

from common import SPECIALS, Tokenizer, canonical, input_paths, records


def train(documents, vocab_size, min_frequency=2, max_token_bytes=256):
    nodes, vocabulary = [], {i: bytes([i]) for i in range(256)}
    occurrences = defaultdict(set)
    for text in documents:
        data = text.encode("utf-8", errors="strict")
        start = len(nodes)
        for i, token in enumerate(data):
            nodes.append(
                [token, start + i - 1 if i else -1, start + i + 1 if i + 1 < len(data) else -1]
            )
        for i in range(start, len(nodes) - 1):
            occurrences[nodes[i][0], nodes[i + 1][0]].add(i)
    heap = [(-len(positions), pair) for pair, positions in occurrences.items()]
    heapq.heapify(heap)
    merges = []
    for token in range(256, vocab_size - 8):
        while heap:
            negative_count, pair = heapq.heappop(heap)
            count = len(occurrences[pair])
            if (
                -negative_count == count
                and len(vocabulary[pair[0]]) + len(vocabulary[pair[1]]) <= max_token_bytes
            ):
                break
        else:
            break
        if count < min_frequency:
            break
        touched = set()

        def remove(left):
            if left >= 0 and nodes[left][2] >= 0:
                right = nodes[left][2]
                key = nodes[left][0], nodes[right][0]
                occurrences[key].discard(left)
                touched.add(key)

        def add(left):
            if left >= 0 and nodes[left][2] >= 0:
                right = nodes[left][2]
                key = nodes[left][0], nodes[right][0]
                occurrences[key].add(left)
                touched.add(key)

        for left in sorted(occurrences[pair]):
            right = nodes[left][2]
            if right < 0 or (nodes[left][0], nodes[right][0]) != pair:
                continue
            before, after = nodes[left][1], nodes[right][2]
            remove(before)
            remove(left)
            remove(right)
            nodes[left][0], nodes[left][2], nodes[right][2] = token, after, -2
            if after >= 0:
                nodes[after][1] = left
            add(before)
            add(left)
        vocabulary[token] = vocabulary[pair[0]] + vocabulary[pair[1]]
        merges.append([pair[0], pair[1], token])
        for key in touched:
            heapq.heappush(heap, (-len(occurrences[key]), key))
    return {
        "format": "mini-mla-tokenizer",
        "version": 1,
        "pretokenizer": "bytes-v1",
        "vocab_size": vocab_size,
        "merges": merges,
        "specials": {name: vocab_size - 8 + i for i, name in enumerate(SPECIALS)},
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train a deterministic native-compatible byte BPE."
    )
    parser.add_argument("--input", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=16384)
    parser.add_argument("--sample-bytes", type=int, default=2 * 1024 * 1024)
    parser.add_argument("--min-frequency", type=int, default=2)
    args = parser.parse_args()
    if not 264 <= args.vocab_size <= 65535 or args.sample_bytes < 1 or args.min_frequency < 1:
        parser.error("Invalid vocabulary size, sample budget, or minimum frequency")
    documents, size = [], 0
    for record in records(input_paths(args.input)):
        text = record.get("text")
        if not isinstance(text, str):
            raise ValueError("Tokenizer training requires a text field")
        remaining = args.sample_bytes - size
        raw = text.encode("utf-8")[:remaining]
        sample = raw.decode("utf-8", errors="ignore")
        documents.append(sample)
        size += len(sample.encode("utf-8"))
        if len(text.encode("utf-8")) >= remaining:
            break
    if not size:
        raise ValueError("Empty training sample")
    value = train(documents, args.vocab_size, args.min_frequency)
    Tokenizer(value)
    destination = args.output if args.output.suffix == ".json" else args.output / "tokenizer.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(canonical(value))
    print(
        canonical(
            {
                "output": str(destination),
                "sample_bytes": size,
                "merges": len(value["merges"]),
                "active_tokens": len(value["merges"]) + 264,
                "vocab_capacity": args.vocab_size,
            }
        ).strip()
    )


if __name__ == "__main__":
    main()
