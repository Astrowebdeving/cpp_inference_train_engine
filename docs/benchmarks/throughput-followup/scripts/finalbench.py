"""Interleaved original-vs-final training benchmark. Prints a JSON summary per workload."""
import json, statistics, subprocess, sys
ROOT = "/Users/tu15/alex_cpp/CPP_Completed"
S = "/private/tmp/claude-504/-Users-tu15-alex-cpp/31e524c9-a66a-4a3a-8be5-706d442a7ae5/scratchpad"
ORIG, FINAL = f"{S}/bin_base/mini_mla_bench", f"{ROOT}/build/bin/mini_mla_bench"
def run(binary, precision, seq, mb, accum, reps, extra):
    cmd = [binary, "--kind", "train", "--config", f"{ROOT}/config/mla36m-v1.json", "--backend", "mps", "--attention", "reference",
           "--precision", precision, "--sequence-length", str(seq), "--microbatch", str(mb), "--grad-accum", str(accum),
           "--submission-window", "4", "--warmup", "3", "--repetitions", str(reps), "--time-zero-grad", "--skip-cpu-check"] + extra
    out = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if out.returncode:
        raise SystemExit(f"FAILED {cmd}\n{out.stdout[-300:]}{out.stderr[-300:]}")
    return json.loads(out.stdout.strip().splitlines()[-1])
def workload(name, precision, seq, mb, accum, variants, rounds=3, reps=30):
    tokens = seq * mb * accum
    samples = {label: [] for label, _, _ in variants}
    for r in range(rounds):
        for label, binary, extra in (variants if r % 2 == 0 else variants[::-1]):
            d = run(binary, precision, seq, mb, accum, reps, extra)
            samples[label].append(d.get("mean_ms", d["median_ms"]) if label.endswith("*") else d["median_ms"])
    base = statistics.median(samples[variants[0][0]])
    print(f"\n{name}  ({tokens} tokens/update)")
    row = {}
    for label, _, _ in variants:
        m = statistics.median(samples[label])
        row[label] = {"ms": round(m, 2), "tokens_per_second": round(tokens * 1000 / m), "runs_ms": [round(x, 2) for x in samples[label]]}
        print(f"  {label:34s} {m:8.2f} ms  {tokens * 1000 / m:8.0f} tok/s  {100 * (base / m - 1):+6.1f}% throughput   runs {[round(x, 1) for x in samples[label]]}")
    return {name: row}
results = {}
which = sys.argv[1] if len(sys.argv) > 1 else "all"
if which in ("all", "ref"):
    results |= workload("AdamW FP16, 8 x 128", "fp16", 128, 8, 1, [("original", ORIG, ["--optimizer", "adamw"]), ("final, serial", FINAL, ["--optimizer", "adamw", "--pipeline", "off"]), ("final (default, pipelined)", FINAL, ["--optimizer", "adamw"])])
    results |= workload("Muon FP16, 8 x 128", "fp16", 128, 8, 1, [("original (gram)", ORIG, ["--optimizer", "muon", "--muon-ns-order", "gram"]), ("final, gram order", FINAL, ["--optimizer", "muon", "--muon-ns-order", "gram"]), ("final (default, composed)", FINAL, ["--optimizer", "muon"])])
    results |= workload("AdamW FP32, 8 x 128", "fp32", 128, 8, 1, [("original", ORIG, ["--optimizer", "adamw"]), ("final (default)", FINAL, ["--optimizer", "adamw"])], rounds=2)
if which in ("all", "scale"):
    for mb in (16, 32, 64):
        results |= workload(f"AdamW FP16, {mb} x 128", "fp16", 128, mb, 1, [("original", ORIG, ["--optimizer", "adamw"]), ("final (default)", FINAL, ["--optimizer", "adamw"])], rounds=2, reps=12)
if which in ("all", "accum"):
    results |= workload("AdamW FP16, seq 256, microbatch 2 x accum 16", "fp16", 256, 2, 16, [("original", ORIG, ["--optimizer", "adamw"]), ("final, no merge", FINAL, ["--optimizer", "adamw", "--coalesce-tokens", "0"]), ("final (default, 4096/pass)", FINAL, ["--optimizer", "adamw"]), ("final, 8192/pass", FINAL, ["--optimizer", "adamw", "--coalesce-tokens", "8192"])], rounds=2, reps=8)
    results |= workload("AdamW FP16, seq 128, microbatch 2 x accum 16", "fp16", 128, 2, 16, [("original", ORIG, ["--optimizer", "adamw"]), ("final (default, 4096/pass)", FINAL, ["--optimizer", "adamw"])], rounds=2, reps=10)
    results |= workload("Muon FP16, seq 256, microbatch 2 x accum 16", "fp16", 256, 2, 16, [("original (gram)", ORIG, ["--optimizer", "muon", "--muon-ns-order", "gram"]), ("final (default)", FINAL, ["--optimizer", "muon"])], rounds=2, reps=8)
json.dump(results, open(f"{S}/final-train-{which}.json", "w"), indent=2)
if which == "seq":
    for mb, seq, reps in ((32, 256, 8), (8, 512, 10), (4, 1024, 10), (1, 1024, 16), (1, 1536, 12), (1, 2040, 10)):
        results |= workload(f"AdamW FP16, {mb} x {seq}", "fp16", seq, mb, 1, [("original", ORIG, ["--optimizer", "adamw"]), ("final (default)", FINAL, ["--optimizer", "adamw"])], rounds=2, reps=reps)
    json.dump(results, open(f"{S}/final-train-seq.json", "w"), indent=2)
