"""Scheduler-only timing (no HTTP): where does a request's wall time go?"""
import json, sys, time, statistics
tools, binary = sys.argv[1], sys.argv[2]
sys.path.insert(0, tools)
import gemma_runtime as runtime
from transformers import AutoTokenizer
root = "/Users/tu15/alex_cpp/CPP_Completed"
model = f"{root}/artifacts/gemma-4-E2B-it-4bit"
tokenizer = AutoTokenizer.from_pretrained(model, local_files_only=True)
calls = []
original = runtime.GemmaProcess.call
def timed(self, command, **fields):
    t = time.perf_counter(); result = original(self, command, **fields)
    calls.append((command, (time.perf_counter() - t) * 1000, result.get("elapsed_ms", 0.0))); return result
runtime.GemmaProcess.call = timed
inputs = json.load(open(sys.argv[3]))
scheduler = runtime.GemmaScheduler(binary, model, tokenizer, context_limit=4096, max_batch=8)
try:
    for count, max_tokens in [(1, 64), (8, 64), (1, 1024), (8, 1024)]:
        prompts = inputs["prompts"][:count]
        for r in [scheduler.submit(p, 16, ignore_eos=True, streaming=False) for p in prompts]:
            r.future.result(timeout=120)
        calls.clear(); start = time.perf_counter()
        requests = [scheduler.submit(p, max_tokens, ignore_eos=True, streaming=False) for p in prompts]
        for r in requests: r.future.result(timeout=600)
        wall = (time.perf_counter() - start) * 1000
        decode = [c for c in calls if c[0] == "decode"]; prefill = [c for c in calls if c[0].startswith("prefill")]
        native = sum(c[2] for c in decode) + sum(c[2] for c in prefill)
        print(f"{count} x {max_tokens:4d} tokens: wall {wall:8.1f} ms  native engine {native:8.1f} ms  outside engine {wall - native:7.1f} ms ({100 * (wall - native) / wall:4.1f}%)  "
              f"{count * max_tokens * 1000 / wall:6.1f} tok/s", flush=True)
finally:
    scheduler.close()
