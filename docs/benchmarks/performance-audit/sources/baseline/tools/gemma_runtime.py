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


class GemmaProcess:
    def __init__(
        self, binary, model, context_limit=4096, sessions=8, timeout=120, full_prefill=False
    ):
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
    prefill_ms: float = 0
    session: int = -1


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
    ):
        if not 1 <= max_batch <= 64 or not 1 <= max_queue <= 4096:
            raise ValueError("Invalid scheduler capacity")
        self.tokenizer = tokenizer
        self.context_limit = context_limit
        self.max_batch = max_batch
        self.pending = queue.Queue(maxsize=max_queue)
        self.stopping = threading.Event()
        self.native = GemmaProcess(
            binary, model, context_limit, max_batch, full_prefill=full_prefill
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
            self.native.call("release", session=r.session)

    def _token(self, r, token):
        r.tokens.append(token)
        r.timestamps.append(time.perf_counter())
        text = self.tokenizer.decode(r.tokens, skip_special_tokens=True)
        reason = None
        if token in self.eos and not r.ignore_eos:
            reason = "stop"
        for stop in r.stop:
            index = text.find(stop)
            if index >= 0:
                text = text[:index]
                reason = "stop"
        if len(r.tokens) >= r.max_tokens and reason is None:
            reason = "length"
        r.text = text
        hold = max(map(len, r.stop), default=0)
        stable = text if reason else text[: max(0, len(text) - hold)].rstrip("\ufffd")
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
                self.native.call("release", session=r.session)
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
                        time.sleep(0.0005)
                    except queue.Empty:
                        continue
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
                self.native.info = self.native.call("info")
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
