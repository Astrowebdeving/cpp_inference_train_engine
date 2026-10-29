import json, statistics, subprocess
ROOT = "/Users/tu15/alex_cpp/CPP_Completed"
S = "/private/tmp/claude-504/-Users-tu15-alex-cpp/31e524c9-a66a-4a3a-8be5-706d442a7ae5/scratchpad"
bins = {"original": f"{S}/bin_base/mini_mla_bench", "final": f"{ROOT}/build/bin/mini_mla_bench"}
cases = [("decode", "fp16", 256), ("decode", "fp32", 256), ("decode", "fp16", 1024), ("prefill", "fp16", 128), ("prefill", "fp32", 128), ("prefill", "fp16", 512), ("prefill", "fp16", 1024)]
out = {}
for kind, precision, seq in cases:
    vals = {k: [] for k in bins}; err = {}
    for r in range(4):
        for label in (list(bins) if r % 2 == 0 else list(bins)[::-1]):
            cmd = [bins[label], "--kind", kind, "--config", f"{ROOT}/config/mla36m-v1.json", "--sequence-length", str(seq), "--warmup", "5", "--repetitions", "60", "--backend", "mps", "--precision", precision]
            d = json.loads(subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT, check=True).stdout.strip().splitlines()[-1])
            vals[label].append(d["median_ms"]); err[label] = d["correctness_error"]
    o, n = statistics.median(vals["original"]), statistics.median(vals["final"])
    name = f"{kind} {precision} ctx {seq}"
    out[name] = {"original_ms": round(o, 3), "final_ms": round(n, 3), "error_original": err["original"], "error_final": err["final"]}
    print(f"{name:26s} original {o:7.3f} ms   final {n:7.3f} ms   {100 * (o / n - 1):+5.1f}% speed   max logit error vs CPU: {err['original']:.2e} -> {err['final']:.2e}", flush=True)
json.dump(out, open(f"{S}/final-inference.json", "w"), indent=2)
