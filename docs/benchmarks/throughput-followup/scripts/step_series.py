import json, sys, statistics
sys.path.insert(0, "/Users/tu15/alex_cpp/CPP_Completed/tools")
from gemma_runtime import GemmaProcess
root = "/Users/tu15/alex_cpp/CPP_Completed"
binary, inputs = sys.argv[1], json.load(open(sys.argv[2]))
extra = sys.argv[3:]
with GemmaProcess(binary, f"{root}/artifacts/gemma-4-E2B-it-4bit", context_limit=4096, sessions=8) as engine:
    for batch in (1, 8):
        handles = list(range(batch))
        for trial in range(3):
            out = engine.call("prefill_batch", sessions=handles, prompts=inputs["prompts"][:batch])
            tokens, series = out["tokens"], []
            for step in range(63):
                r = engine.call("decode", sessions=handles, tokens=tokens)
                tokens = r["tokens"]; series.append(r["elapsed_ms"])
            for h in handles:
                engine.call("release", session=h)
            if trial == 0:
                continue
            print(f"batch {batch} trial {trial}: prefill {out['elapsed_ms']:6.1f} ms | decode mean {statistics.mean(series):6.2f} median {statistics.median(series):6.2f} "
                  f"min {min(series):6.2f} max {max(series):6.2f} | first 6: {[round(x, 1) for x in series[:6]]} | steps over 1.15x median: {sum(x > 1.15 * statistics.median(series) for x in series)}")
            if trial == 2:
                print("   every 4th step:", [round(x, 1) for x in series[::4]])
