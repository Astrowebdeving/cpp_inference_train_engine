import argparse
import json
import queue
import random
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from gemma_runtime import GemmaScheduler, Request, incremental_decode_plan


class FakeNative:
    """Stands in for the engine process: records commands and answers with fixed tokens."""

    def __init__(self):
        self.commands = []
        self.info = {}

    def call(self, command, **fields):
        self.commands.append((command, fields))
        if command == "prefill_batch":
            return {"tokens": [1000] * len(fields["sessions"]), "elapsed_ms": 0.0}
        if command == "decode":
            return {"tokens": [1001] * len(fields["sessions"]), "elapsed_ms": 0.0}
        return {"ok": True}

    def close(self):
        pass


def scheduler_for(tokenizer, plan="detect", gather_ms=2.0, run=False):
    """A scheduler without an engine process, for exercising host-side logic only."""
    s = object.__new__(GemmaScheduler)
    s.tokenizer = tokenizer
    s.decode_plan = incremental_decode_plan(tokenizer) if plan == "detect" else plan
    s.gather_seconds = gather_ms / 1000
    s.context_limit = 1 << 20
    s.max_batch = 8
    s.pending = queue.Queue(maxsize=128)
    s.stopping = threading.Event()
    s.native = FakeNative()
    s.eos = {1, 106}
    s.counters = {
        "prefill_batches": 0,
        "decode_batches": 0,
        "decode_tokens": 0,
        "max_decode_batch": 0,
        "completed_requests": 0,
    }
    if run:
        s.thread = threading.Thread(target=s._run, daemon=True)
        s.thread.start()
    return s


def stream(scheduler, tokens, stop=(), ignore_eos=True, max_tokens=None):
    """Feeds tokens through the scheduler's token handler and returns what a client sees."""
    r = Request([2], max_tokens or len(tokens), True, ignore_eos, tuple(stop))
    r.events = queue.Queue()
    texts = []
    for token in tokens:
        running = scheduler._token(r, token)
        texts.append(r.text)
        if not running:
            break
    emitted, result = "", None
    while not r.events.empty():
        event = r.events.get()
        if "done" in event:
            result = event["done"]
        else:
            emitted += event["text"]
    return texts, emitted, result


def reference(tokenizer, eos, tokens, stop=(), ignore_eos=True, max_tokens=None):
    """The whole-history algorithm this replaces: final text and reason for each prefix."""
    limit = max_tokens or len(tokens)
    for n in range(1, len(tokens) + 1):
        text = tokenizer.decode(tokens[:n], skip_special_tokens=True)
        reason = "stop" if tokens[n - 1] in eos and not ignore_eos else None
        for s in stop:
            index = text.find(s)
            if index >= 0:
                text, reason = text[:index], "stop"
        if n >= limit and reason is None:
            reason = "length"
        if reason:
            return n, text, reason
    return len(tokens), text, None


def verify(model, output):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
    checks = []
    plan = incremental_decode_plan(tokenizer)
    assert plan is not None, "The Gemma decoder should allow incremental decoding"
    byte, skipped = plan
    assert len(byte) == 256 and len(skipped) > 0
    for token in list(skipped)[:50]:
        assert tokenizer.decode([token], skip_special_tokens=True) == ""

    class Stripping:
        class backend_tokenizer:
            class decoder:
                @staticmethod
                def __getstate__():
                    return json.dumps({"type": "Metaspace", "replacement": "▁"}).encode()

    class Fusing:
        class backend_tokenizer:
            class decoder:
                @staticmethod
                def __getstate__():
                    wide = {"type": "Replace", "pattern": {"String": "ab"}, "content": " "}
                    return json.dumps({"type": "Sequence", "decoders": [wide]}).encode()

    assert incremental_decode_plan(Stripping) is None
    assert incremental_decode_plan(Fusing) is None
    assert incremental_decode_plan(object()) is None
    checks.append("decode plan accepts the Gemma decoder and rejects position-dependent ones")

    rng = random.Random(7)
    vocabulary = len(tokenizer)
    byte_ids = sorted(byte)
    by_value = {int(tokenizer.convert_ids_to_tokens(i)[3:5], 16): i for i in byte_ids}
    specials = sorted(skipped)[:12]

    def as_bytes(text):
        return [by_value[b] for b in text.encode()]

    samples = [
        "Gradient descent updates weights against the gradient of the loss.",
        "  Leading spaces,   doubled   spaces and a trailing one ",
        "Ünïcödé, 日本語のテキスト, 한국어, العربية, עברית and emoji 🙂🚀👩‍👩‍👧‍👦 mixed.",
        "Punctuation ! ? , . : ; n't 're 's ( x ) [ y ] { z } -- ...",
        "def f(x):\n\treturn x ** 2  # tabs and\r\nnewlines\n\n\n",
    ]
    sequences = [tokenizer.encode(text, add_special_tokens=False) for text in samples]
    sequences += [[rng.randrange(vocabulary) for _ in range(120)] for _ in range(60)]
    for _ in range(80):
        # Byte runs that are valid, cut short, or spoiled, between ordinary and skipped tokens.
        sequence = []
        for _ in range(rng.randrange(4, 14)):
            kind = rng.random()
            if kind < 0.3:
                sequence += as_bytes(rng.choice(["é", "日", "🙂", "ß", "€", "a"]))
            elif kind < 0.5:
                sequence += as_bytes(rng.choice(["é", "日", "🙂"]))[: rng.randrange(1, 3)]
            elif kind < 0.65:
                sequence.append(rng.choice(byte_ids))
            elif kind < 0.8:
                sequence.append(rng.choice(specials))
            else:
                sequence += tokenizer.encode(
                    rng.choice([" word", "x", " 2", ".", "\n"]), add_special_tokens=False
                )
        sequences.append(sequence)
    incremental = scheduler_for(tokenizer)
    prefixes = 0
    for sequence in sequences:
        texts, emitted, result = stream(incremental, sequence)
        for n, text in enumerate(texts, 1):
            assert text == tokenizer.decode(sequence[:n], skip_special_tokens=True), (
                sequence[:n],
                text,
            )
            prefixes += 1
        assert result is not None and result["finish_reason"] == "length"
        assert emitted == result["text"] == texts[-1]
    checks.append(f"incremental text equals whole-history decoding at {prefixes} prefixes")

    # A valid character that a later byte invalidates. Streaming it early is what the
    # whole-history handler did, and the rewrite then stopped the scheduler for every request.
    rewrite = tokenizer.encode("ok", add_special_tokens=False) + as_bytes("é") + [by_value[0x80]]
    whole = scheduler_for(tokenizer, plan=None)
    try:
        stream(whole, rewrite + tokenizer.encode(" end", add_special_tokens=False))
        raised = False
    except RuntimeError:
        raised = True
    assert raised, "Expected the whole-history handler to reject rewritten streamed text"
    texts, emitted, result = stream(
        incremental, rewrite + tokenizer.encode(" end", add_special_tokens=False)
    )
    assert emitted == result["text"] == texts[-1]
    assert "�" in result["text"] and result["text"].endswith(" end")
    checks.append("rewritten byte runs are withheld instead of stopping the scheduler")

    # Emitted text is always final: every streamed prefix must survive in the full text.
    for sequence in sequences[-80:]:
        r = Request([2], len(sequence), True, True, ())
        r.events = queue.Queue()
        sent = ""
        for n, token in enumerate(sequence, 1):
            incremental._token(r, token)
            while not r.events.empty():
                event = r.events.get()
                if "done" not in event:
                    sent += event["text"]
            if n < len(sequence):
                assert tokenizer.decode(sequence, skip_special_tokens=True).startswith(sent)
    checks.append("streamed text is never revised")

    stops = 0
    for sequence in sequences:
        full = tokenizer.decode(sequence, skip_special_tokens=True)
        if len(full) < 8:
            continue
        for _ in range(4):
            a = rng.randrange(1, len(full) - 3)
            stop_set = [full[a : a + rng.randrange(1, 6)]]
            if rng.random() < 0.4:
                b = rng.randrange(0, len(full) - 1)
                stop_set.append(full[b : b + rng.randrange(1, 9)])
            if any(not s for s in stop_set):
                continue
            ignore_eos = rng.random() < 0.5
            limit = rng.randrange(1, len(sequence) + 1) if rng.random() < 0.3 else None
            n, text, reason = reference(
                tokenizer, incremental.eos, sequence, stop_set, ignore_eos, limit
            )
            texts, emitted, result = stream(incremental, sequence, stop_set, ignore_eos, limit)
            assert result is not None, (sequence, stop_set)
            assert (len(texts), result["text"], result["finish_reason"]) == (n, text, reason), (
                sequence,
                stop_set,
                result,
                (n, text, reason),
            )
            assert emitted == text
            assert all(s not in emitted for s in stop_set)
            stops += 1
    checks.append(f"stop strings, EOS and length limits match the whole-history handler ({stops})")

    # The point of the change: work per token must not grow with the response.
    ordinary = tokenizer.encode(" ".join(samples[:1] * 200), add_special_tokens=False)[:2000]
    windows = []
    original = tokenizer.decode

    def counting(ids, **kwargs):
        windows.append(len(ids))
        return original(ids, **kwargs)

    tokenizer.decode = counting
    try:
        stream(incremental, ordinary)
    finally:
        del tokenizer.decode
    assert len(windows) == len(ordinary) and max(windows) <= 2, max(windows)
    checks.append("decode window stays at one token for ordinary text over 2,000 tokens")

    # Requests arriving a few milliseconds apart share one prefill when the gather window
    # covers the gap, and are split when it is disabled.
    for gather_ms, expect_single in [(40.0, True), (0.0, False)]:
        s = scheduler_for(tokenizer, gather_ms=gather_ms, run=True)
        try:
            requests = []
            for _ in range(4):
                requests.append(s.submit([2, 3, 4], 2, ignore_eos=True))
                time.sleep(0.004)
            for r in requests:
                r.future.result(timeout=10)
        finally:
            s.stopping.set()
            s.thread.join(timeout=5)
        sizes = [len(f["sessions"]) for c, f in s.native.commands if c == "prefill_batch"]
        assert sum(sizes) == 4
        assert (sizes == [4]) == expect_single, (gather_ms, sizes)
    isolated = scheduler_for(tokenizer, gather_ms=2.0, run=True)
    try:
        start = time.perf_counter()
        isolated.submit([2, 3, 4], 1, ignore_eos=True).future.result(timeout=10)
        waited = time.perf_counter() - start
    finally:
        isolated.stopping.set()
        isolated.thread.join(timeout=5)
    assert waited < 0.05, waited
    checks.append("gather window joins near-simultaneous requests without stalling a lone one")

    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"status": "passed", "checks": checks}, indent=2) + "\n")
    print(f"Gemma scheduler: {len(checks)} checks passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.model.resolve(), args.output.resolve())
