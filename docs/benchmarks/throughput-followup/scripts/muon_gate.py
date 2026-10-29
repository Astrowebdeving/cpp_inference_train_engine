"""Full-model first-update gate between two Muon NS orders (same thresholds as the audit)."""
import json, subprocess, sys
from pathlib import Path
import numpy as np
root = Path("/Users/tu15/alex_cpp/CPP_Completed")
out, binary, a_name, b_name = Path(sys.argv[1]), sys.argv[2], sys.argv[3], sys.argv[4]
precision = sys.argv[5] if len(sys.argv) > 5 else "fp16"
base = [binary, "--kind", "train", "--config", str(root / "config/mla36m-v1.json"), "--backend", "mps",
        "--precision", precision, "--attention", "reference", "--sequence-length", "128", "--microbatch", "8",
        "--grad-accum", "1", "--submission-window", "4", "--time-zero-grad", "--optimizer", "muon",
        "--muon-layout", "batched", "--warmup", "0", "--repetitions", "1"]
def run(order):
    path = out / f"{order}-{precision}"
    subprocess.run(base + ["--muon-ns-order", order, "--comparison-data", str(path)], check=True,
                   stdout=subprocess.DEVNULL, cwd=root)
    meta = json.loads((path / "comparison.json").read_text())
    values = np.memmap(path / "tensors.bin", dtype=np.float32, mode="r")
    return {t["name"]: values[t["offset"] // 4:(t["offset"] + t["nbytes"]) // 4] for t in meta["tensors"]}
a, b = run(a_name), run(b_name)
for name in a:
    if not name.endswith(".final"):
        np.testing.assert_array_equal(a[name], b[name], err_msg=name)
rows = []
for name in a:
    if not name.endswith(".weight"):
        continue
    prefix = name[:-7]
    da = a[prefix + ".final"].astype(np.float64) - a[name]
    db = b[prefix + ".final"].astype(np.float64) - b[name]
    na, nb = np.linalg.norm(da), np.linalg.norm(db)
    rows.append((prefix, float(np.linalg.norm(db - da) / max(na, 1e-20)),
                 float(np.dot(da, db) / (na * nb)) if na and nb else 1.0, float(np.max(np.abs(db - da)))))
worst = max(rows, key=lambda r: r[1])
print(f"{a_name} vs {b_name} ({precision}): {len(rows)} parameters; initial weights and gradients bitwise equal")
print(f"  max update relative L2  {max(r[1] for r in rows):.3e}   (gate 1e-3)   worst: {worst[0]}")
print(f"  min update cosine       {min(r[2] for r in rows):.12f}   (gate 0.99999)")
print(f"  max abs weight diff     {max(r[3] for r in rows):.3e}   (gate 2e-6)")
ok = max(r[1] for r in rows) <= 1e-3 and min(r[2] for r in rows) >= 0.99999 and max(r[3] for r in rows) <= 2e-6
print("  GATE", "PASSED" if ok else "FAILED")
(out / f"gate-{a_name}-vs-{b_name}-{precision}.json").write_text(json.dumps({
    "status": "passed" if ok else "failed", "compared": [a_name, b_name], "precision": precision, "parameters": len(rows),
    "initial_weights_and_gradients": "bitwise equal",
    "max_update_relative_l2": max(r[1] for r in rows), "min_update_cosine": min(r[2] for r in rows),
    "max_absolute_weight_difference": max(r[3] for r in rows), "worst_parameter": worst[0],
    "gates": {"update_relative_l2": 1e-3, "update_cosine": 0.99999, "absolute_weight_difference": 2e-6}}, indent=2) + "\n")
sys.exit(0 if ok else 1)
