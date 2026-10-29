import json
import os
import queue
import select
import subprocess
import threading
import time
from concurrent.futures import Future
from dataclasses import dataclass, field
from pathlib import Path


def incremental_decode_plan(tokenizer):
    """Classify tokens so text can be decoded incrementally, or return None.

    A decoder built only from single-character replacement, byte fallback and fusing
    concatenates one string per token, with two exceptions: a run of byte-fallback tokens is
    decoded as a single UTF-8 unit, and skipped special tokens vanish before runs are formed.
    Decoding is therefore exact when it restarts just after any ordinary token. Returns the
    byte-fallback and skipped token ids for such a tokenizer. Any other decoder (leading-space
    stripping, cleanup passes, word pieces) gets None, and callers decode whole histories.
    """
    try:
        backend = tokenizer.backend_tokenizer
        decoder = json.loads(backend.decoder.__getstate__())
        steps = decoder["decoders"] if decoder.get("type") == "Sequence" else [decoder]
        for step in steps:
            if step.get("type") == "Replace":
                if len(step.get("pattern", {}).get("String", "")) != 1:
                    return None
            elif step.get("type") not in {"ByteFallback", "Fuse"}:
                return None
        if getattr(tokenizer, "clean_up_tokenization_spaces", False):
            return None
        skipped = {i for i, token in backend.get_added_tokens_decoder().items() if token.special}
        hexadecimal = set("0123456789abcdefABCDEF")
        byte = {
            i
            for token, i in backend.get_vocab(with_added_tokens=True).items()
            if len(token) == 6
            and token.startswith("<0x")
            and token.endswith(">")
            and set(token[3:5]) <= hexadecimal
        }
        return frozenset(byte), frozenset(skipped)
    except Exception:
        return None


class GemmaProcess:
    def __init__(
        self,
        binary,
        model,
        context_limit=4096,
        sessions=8,
        timeout=120,
        full_prefill=False,
        batch_kv=True,
        all_batch_kv=False,
    ):
        if all_batch_kv and not batch_kv:
            raise ValueError("Conflicting Gemma batch-cache options")
        self.timeout = timeout
        self.buffer = bytearray()
        command = [
            str(binary),
            "--model",
            str(model),
            "--context-limit",
            str(context_limit),
            "--sessions",
            str(sessions),
        ]
        if full_prefill:
            command.append("--full-prefill")
        if not batch_kv:
            command.append("--independent-kv")
        if all_batch_kv:
            command.append("--all-batch-kv")
        self.process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            bufsize=0,
        )
        try:
            self.info = self._read()
            if "error" in self.info:
                raise RuntimeError(self.info["error"])
        except BaseException:
            self.close()
            raise

    def _read(self):
        deadline = time.monotonic() + self.timeout
        while True:
            newline = self.buffer.find(b"\n")
            if newline >= 0:
                line = bytes(self.buffer[:newline])
                del self.buffer[: newline + 1]
                return json.loads(line)
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Native Gemma command timed out")
            ready, _, _ = select.select([self.process.stdout], [], [], min(remaining, 0.5))
            if ready:
                block = os.read(self.process.stdout.fileno(), 65536)
                if not block:
                    raise RuntimeError(f"Native Gemma exited with code {self.process.poll()}")
                self.buffer.extend(block)
                if len(self.buffer) > 4 * 1024 * 1024:
                    raise RuntimeError("Native Gemma response exceeds limit")

    def call(self, command, **fields):
        payload = json.dumps({"command": command, **fields}, separators=(",", ":")).encode() + b"\n"
        remaining = memoryview(payload)
        while remaining:
            written = os.write(self.process.stdin.fileno(), remaining)
            remaining = remaining[written:]
        result = self._read()
        if "error" in result:
            raise ValueError(result["error"])
        if "info" in result:
            self.info = result.pop("info")
        elif command == "info":
            self.info = result
        elif fields.get("include_info"):
            self.info = self.call("info")
        return result

    def close(self):
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=5)
        for stream in [self.process.stdin, self.process.stdout]:
            if stream:
                stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@dataclass
class Request:
    prompt: list[int]
    max_tokens: int
    streaming: bool = False
    ignore_eos: bool = False
    stop: tuple[str, ...] = ()
    created: float = field(default_factory=time.perf_counter)
    future: Future = field(default_factory=Future)
    events: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=512))
    cancelled: threading.Event = field(default_factory=threading.Event)
    tokens: list[int] = field(default_factory=list)
    timestamps: list[float] = field(default_factory=list)
    text: str = ""
    emitted: str = ""
    settled: str = ""  # decoded text of tokens[:settled_tokens]; later tokens cannot change it
    settled_tokens: int = 0
    prefill_ms: float = 0
    session: int = -1
    notify: object = None  # optional callable, invoked from the scheduler thread per event


class GemmaScheduler:
    def __init__(
        self,
        binary,
        model,
        tokenizer,
        context_limit=4096,
        max_batch=8,
        max_queue=128,
        full_prefill=False,
        batch_kv=True,
        all_batch_kv=False,
        gather_ms=2.0,
    ):
        if not 1 <= max_batch <= 64 or not 1 <= max_queue <= 4096:
            raise ValueError("Invalid scheduler capacity")
        if not 0 <= gather_ms <= 100:
            raise ValueError("Invalid admission gather window")
        self.tokenizer = tokenizer
        self.decode_plan = incremental_decode_plan(tokenizer)
        # When idle, wait this long for requests that arrive together, restarting the wait on
        # each arrival up to four windows in total. Requests admitted in one prefill share
        # positions, which keeps the batch on the aligned cache path for the whole generation;
        # a burst split across prefills pays an extra pass and decodes on the scatter path.
        self.gather_seconds = gather_ms / 1000
        self.context_limit = context_limit
        self.max_batch = max_batch
        self.pending = queue.Queue(maxsize=max_queue)
        self.stopping = threading.Event()
        self.native = GemmaProcess(
            binary,
            model,
            context_limit,
            max_batch,
            full_prefill=full_prefill,
            batch_kv=batch_kv,
            all_batch_kv=all_batch_kv,
        )
        config = json.loads((Path(model) / "config.json").read_text())
        eos = config.get("eos_token_id", [1, 106, 50])
        self.eos = set(eos if isinstance(eos, list) else [eos])
        self.counters = {
            "prefill_batches": 0,
            "decode_batches": 0,
            "decode_tokens": 0,
            "max_decode_batch": 0,
            "completed_requests": 0,
        }
        self.thread = threading.Thread(target=self._run, name="gemma-metal", daemon=True)
        self.thread.start()

    def submit(self, prompt, max_tokens, ignore_eos=False, stop=(), streaming=False):
        if self.stopping.is_set():
            raise RuntimeError("Gemma scheduler is stopping")
        if not prompt or max_tokens < 1 or len(prompt) + max_tokens > self.context_limit:
            raise ValueError("Prompt plus requested tokens exceeds context limit")
        if any(type(t) is not int or not 0 <= t < 262144 for t in prompt):
            raise ValueError("Invalid Gemma token ID")
        if len(stop) > 4 or any(not isinstance(s, str) or not s or len(s) > 256 for s in stop):
            raise ValueError("At most four nonempty stop strings of length <= 256 are allowed")
        request = Request(list(prompt), max_tokens, streaming, ignore_eos, tuple(stop))
        self.pending.put_nowait(request)
        return request

    def _event(self, r, value):
        if not r.streaming:
            return
        try:
            r.events.put_nowait(value)
        except queue.Full:
            r.cancelled.set()
            return
        if r.notify is not None:
            r.notify()

    def _finish(self, r, reason):
        if r.future.done():
            return
        if len(r.text) > len(r.emitted):
            self._event(r, {"text": r.text[len(r.emitted) :]})
            r.emitted = r.text
        now = time.perf_counter()
        result = {
            "text": r.text,
            "tokens": r.tokens,
            "finish_reason": reason,
            "prompt_tokens": len(r.prompt),
            "completion_tokens": len(r.tokens),
            "prefill_ms": r.prefill_ms,
            "latency_ms": (now - r.created) * 1000,
            "ttft_ms": (r.timestamps[0] - r.created) * 1000 if r.timestamps else None,
            "token_timestamps": r.timestamps,
        }
        r.future.set_result(result)
        self._event(r, {"done": result})
        self.counters["completed_requests"] += 1
        if r.session >= 0:
            self.native.call("release", session=r.session, include_info=True)

    def _token(self, r, token):
        r.tokens.append(token)
        r.timestamps.append(time.perf_counter())
        searched = len(r.settled)
        if self.decode_plan is None:
            text = self.tokenizer.decode(r.tokens, skip_special_tokens=True)
            searched, final = 0, None
        else:
            # Decode only the tokens after the last settled point. The result equals decoding
            # the whole history, at a cost that no longer grows with the response length.
            byte, skipped = self.decode_plan
            text = r.settled + self.tokenizer.decode(
                r.tokens[r.settled_tokens :], skip_special_tokens=True
            )
            # An ordinary token closes any byte run before it. A skipped token closes nothing,
            # so it settles only when nothing is open. Text after the settled point can still
            # be rewritten by a later byte, and is never streamed early.
            if token not in byte and (
                token not in skipped or r.settled_tokens == len(r.tokens) - 1
            ):
                r.settled, r.settled_tokens = text, len(r.tokens)
            final = len(r.settled)
        reason = None
        if token in self.eos and not r.ignore_eos:
            reason = "stop"
        for stop in r.stop:
            index = text.find(stop, max(0, searched - len(stop) + 1))
            if index >= 0:
                text = text[:index]
                reason = "stop"
        if len(r.tokens) >= r.max_tokens and reason is None:
            reason = "length"
        r.text = text
        hold = max(map(len, r.stop), default=0)
        if reason:
            stable = text
        elif final is None:
            stable = text[: max(0, len(text) - hold)].rstrip("\ufffd")
        else:
            # A stop string can only newly match where text is new or rewritten, at or after the
            # settled point, so it starts no earlier than `hold` characters before that point.
            stable = text[: max(len(r.emitted), final - hold)]
        if not stable.startswith(r.emitted):
            raise RuntimeError("Tokenizer changed text already emitted by streaming")
        self._event(r, {"text": stable[len(r.emitted) :], "token_ids": [token]})
        r.emitted = stable
        if reason:
            self._finish(r, reason)
            return False
        return True

    def _fail(self, r, error):
        if not r.future.done():
            r.future.set_exception(error)
            self._event(r, {"error": str(error)})
        if r.session >= 0:
            try:
                self.native.call("release", session=r.session, include_info=True)
            except Exception:
                pass

    def _run(self):
        active = []
        waiting = []
        incoming = []
        try:
            while not self.stopping.is_set():
                if not active and not waiting:
                    try:
                        waiting.append(self.pending.get(timeout=0.05))
                    except queue.Empty:
                        continue
                    deadline = time.perf_counter() + 4 * self.gather_seconds
                    while len(waiting) < self.max_batch and time.perf_counter() < deadline:
                        try:
                            waiting.append(self.pending.get(timeout=self.gather_seconds))
                        except queue.Empty:
                            break
                while len(waiting) < self.max_batch:
                    try:
                        waiting.append(self.pending.get_nowait())
                    except queue.Empty:
                        break
                live = []
                for r in active:
                    if r.cancelled.is_set():
                        self._finish(r, "cancelled")
                    else:
                        live.append(r)
                active = live
                live = []
                for r in waiting:
                    if r.cancelled.is_set():
                        self._finish(r, "cancelled")
                    else:
                        live.append(r)
                waiting = live
                free = self.max_batch - len(active)
                if free and waiting:
                    length = len(waiting[0].prompt)
                    incoming = [r for r in waiting if len(r.prompt) == length][:free]
                    incoming_ids = {id(r) for r in incoming}
                    waiting = [r for r in waiting if id(r) not in incoming_ids]
                    occupied = {r.session for r in active}
                    slots = [i for i in range(self.max_batch) if i not in occupied]
                    for r, slot in zip(incoming, slots):
                        r.session = slot
                    try:
                        output = self.native.call(
                            "prefill_batch",
                            sessions=[r.session for r in incoming],
                            prompts=[r.prompt for r in incoming],
                            include_info=True,
                        )
                        self.counters["prefill_batches"] += 1
                        for r, token in zip(incoming, output["tokens"], strict=True):
                            r.prefill_ms = output["elapsed_ms"]
                            if r.cancelled.is_set():
                                self._finish(r, "cancelled")
                            elif self._token(r, token):
                                active.append(r)
                    except ValueError as error:
                        for r in incoming:
                            self._fail(r, error)
                    incoming = []
                if active:
                    output = self.native.call(
                        "decode",
                        sessions=[r.session for r in active],
                        tokens=[r.tokens[-1] for r in active],
                        include_info=True,
                    )
                    self.counters["decode_batches"] += 1
                    self.counters["decode_tokens"] += len(active)
                    self.counters["max_decode_batch"] = max(
                        self.counters["max_decode_batch"], len(active)
                    )
                    live = []
                    for r, token in zip(active, output["tokens"], strict=True):
                        if r.cancelled.is_set():
                            self._finish(r, "cancelled")
                        elif self._token(r, token):
                            live.append(r)
                    active = live
        except BaseException as error:
            self.stopping.set()
            for r in active + waiting + incoming:
                self._fail(r, error)
        finally:
            for r in active + waiting:
                self._fail(r, RuntimeError("Gemma scheduler stopped"))
            while True:
                try:
                    self._fail(self.pending.get_nowait(), RuntimeError("Gemma scheduler stopped"))
                except queue.Empty:
                    break
            self.native.close()

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=125)
        if self.thread.is_alive():
            self.native.close()
            self.thread.join(timeout=5)
