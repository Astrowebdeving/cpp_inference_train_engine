import glob
import hashlib
import heapq
import json
from pathlib import Path

SPECIALS = (
    "<|pad|>",
    "<|bos|>",
    "<|eos|>",
    "<|user|>",
    "<|assistant|>",
    "<|system|>",
    "<|tool|>",
    "<|sep|>",
)


def input_paths(patterns):
    paths = sorted({Path(p) for pattern in patterns for p in (glob.glob(pattern) or [pattern])})
    for p in paths:
        if not p.is_file():
            raise ValueError(f"Input file does not exist: {p}")
    return paths


def records(paths):
    for path in paths:
        with path.open(encoding="utf-8", errors="strict") as stream:
            for number, line in enumerate(stream, 1):
                if path.suffix == ".jsonl":
                    if not line.strip():
                        continue
                    try:
                        row = json.loads(line)
                    except json.JSONDecodeError as error:
                        raise ValueError(f"{path}:{number}: {error}") from error
                    if not isinstance(row, dict):
                        raise ValueError(f"{path}:{number}: expected an object")
                    yield row
                else:
                    yield {"text": line}


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True) + "\n"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


class Tokenizer:
    def __init__(self, value, digest=None):
        if (
            value["format"] != "mini-mla-tokenizer"
            or value["version"] != 1
            or value["pretokenizer"] != "bytes-v1"
        ):
            raise ValueError("Unsupported tokenizer")
        self.value = value
        self.vocab_size = value["vocab_size"]
        if not 264 <= self.vocab_size <= 65535:
            raise ValueError("Vocabulary must be in [264,65535]")
        self.specials = value["specials"]
        if self.specials != {name: self.vocab_size - 8 + i for i, name in enumerate(SPECIALS)}:
            raise ValueError("Invalid special token mapping")
        self.vocabulary = {i: bytes([i]) for i in range(256)}
        self.merges = {}
        for expected, (left, right, token) in enumerate(value["merges"], 256):
            if (
                token != expected
                or token >= self.vocab_size - 8
                or left not in self.vocabulary
                or right not in self.vocabulary
                or (left, right) in self.merges
            ):
                raise ValueError("Invalid BPE merge")
            self.merges[left, right] = token
            self.vocabulary[token] = self.vocabulary[left] + self.vocabulary[right]
        self.vocabulary.update({i: s.encode() for s, i in self.specials.items()})
        self.hash = digest or hashlib.sha256(canonical(value).encode()).hexdigest()

    @classmethod
    def load(cls, path):
        raw = Path(path).read_bytes()
        return cls(json.loads(raw), hashlib.sha256(raw).hexdigest())

    def encode(self, text):
        raw = text.encode("utf-8", errors="strict")
        if not self.merges or len(raw) < 2:
            return list(raw)
        nodes = [[token, i - 1, i + 1, 0, True] for i, token in enumerate(raw)]
        nodes[-1][2] = -1
        heap = []

        def push(left):
            if left < 0 or not nodes[left][4] or nodes[left][2] < 0:
                return
            right = nodes[left][2]
            rank = self.merges.get((nodes[left][0], nodes[right][0]))
            if rank is not None:
                heapq.heappush(heap, (rank, left, right, nodes[left][3], nodes[right][3]))

        for i in range(len(nodes)):
            push(i)
        while heap:
            rank, left, right, lv, rv = heapq.heappop(heap)
            a, b = nodes[left], nodes[right]
            if not a[4] or not b[4] or a[2] != right or a[3] != lv or b[3] != rv:
                continue
            a[0], a[2], a[3], b[4] = rank, b[2], a[3] + 1, False
            if b[2] >= 0:
                nodes[b[2]][1] = left
            push(a[1])
            push(left)
        return [node[0] for node in nodes if node[4]]

    def encode_reference(self, text):
        ids = list(text.encode("utf-8"))
        while len(ids) > 1:
            pairs = [
                (self.merges.get((a, b), 65536), i) for i, (a, b) in enumerate(zip(ids, ids[1:]))
            ]
            rank, position = min(pairs)
            if rank == 65536:
                break
            ids[position : position + 2] = [rank]
        return ids

    def decode_bytes(self, ids, skip_special=False):
        specials = set(self.specials.values()) if skip_special else set()
        return b"".join(self.vocabulary[i] for i in ids if i not in specials)

    def chat_sample(self, messages):
        ids, mask = [self.specials["<|bos|>"]], [0]
        for row in messages:
            role, content = row["role"], row["content"]
            if role not in ("system", "user", "assistant", "tool") or not isinstance(content, str):
                raise ValueError("Invalid chat role or content")
            ids.extend([self.specials[f"<|{role}|>"], 10])
            mask.extend([0, 0])
            tokens = self.encode(content)
            ids.extend(tokens + [self.specials["<|eos|>"], 10])
            mask.extend([int(role == "assistant")] * (len(tokens) + 1) + [0])
        if not any(mask[1:]):
            raise ValueError("Chat sample has no assistant response")
        return ids, mask
