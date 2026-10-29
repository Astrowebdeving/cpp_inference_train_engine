import hashlib
import json
import shutil
import subprocess
from pathlib import Path
import sys
sys.path.insert(0, "/Users/tu15/alex_cpp/CPP_Completed/tools")
from inspect_checkpoint import load_weights
import numpy as np

root = Path("/Users/tu15/alex_cpp/CPP_Completed")
work = Path("/private/tmp/mla-attention-work")
report = {}
for name, step in [("flash-sft", 40), ("flash-dpo", 4), ("flash-grpo", 4)]:
    run = work / name / ("run" if name == "flash-sft" else "")
    latest = (run / "checkpoint-latest").resolve()
    manifest, expected = load_weights(latest)
    original_optimizer = np.fromfile(latest / "optimizer.bin", dtype="<f4").copy()
    expected_state = json.loads((latest / "trainer_state.json").read_text())
    reference = run / "reference/weights.bin"
    ref_hash = hashlib.sha256(reference.read_bytes()).hexdigest() if reference.exists() else None
    shutil.move(str(latest), str(run / "original-final-checkpoint"))
    exported = (run / "model-latest").resolve()
    shutil.move(str(exported), str(run / "original-final-model"))
    command = [str(root / "build/bin/mini_mla_train"), "--resume", str(run / f"checkpoint-{step:08d}")]
    with (work / f"{name}-replay.log").open("w") as log:
        subprocess.run(command, cwd=root, stdout=log, stderr=subprocess.STDOUT, check=True)
    _, actual = load_weights(run / "checkpoint-latest")
    max_weight = 0
    for key in expected:
        np.testing.assert_allclose(actual[key], expected[key], atol=2e-6, rtol=1e-5)
        max_weight = max(max_weight, float(np.max(np.abs(actual[key] - expected[key]))))
    actual_optimizer = np.fromfile(run / "checkpoint-latest/optimizer.bin", dtype="<f4")
    np.testing.assert_allclose(actual_optimizer, original_optimizer, atol=2e-6, rtol=1e-5)
    actual_state = json.loads((run / "checkpoint-latest/trainer_state.json").read_text())
    for key in ["options", "workflow_rng", "data_fingerprint", "validation_fingerprint", "tokens_seen"]:
        assert actual_state[key] == expected_state[key], key
    if "data_state" in expected_state:
        assert actual_state["data_state"] == expected_state["data_state"]
    assert not ref_hash or hashlib.sha256(reference.read_bytes()).hexdigest() == ref_hash
    report[name] = {"passed": True, "resume_step": step, "max_weight_error": max_weight,
                    "max_optimizer_error": float(np.max(np.abs(actual_optimizer-original_optimizer))),
                    "state_matches": True, "reference_unchanged": True if ref_hash else None,
                    "command": command, "atol": 2e-6, "rtol": 1e-5}
(work / "flash-replay-report.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(report, indent=2))
