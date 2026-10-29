import argparse
import gc
import hashlib
import json
import math
import os
import platform
import resource
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_input(directory):
    import numpy as np

    metadata = json.loads((directory / "comparison.json").read_text())
    if metadata["format"] != "mini-mla-training-comparison-v1":
        raise ValueError("Unsupported training comparison format")
    binary = directory / "tensors.bin"
    if sha256(binary) != metadata["tensors_sha256"]:
        raise ValueError("Comparison tensor checksum failed")
    entries = {e["name"]: e for e in metadata["tensors"]}
    if len(entries) != len(metadata["tensors"]):
        raise ValueError("Duplicate comparison tensor names")
    offset = 0
    for e in metadata["tensors"]:
        if e["offset"] != offset or e["nbytes"] != math.prod(e["shape"]) * 4:
            raise ValueError("Invalid comparison tensor extent")
        offset += e["nbytes"]
    if offset != binary.stat().st_size:
        raise ValueError("Unindexed comparison tensor bytes")

    def tensor(name):
        e = entries[name]
        return np.memmap(binary, mode="r", dtype="<f4", offset=e["offset"], shape=tuple(e["shape"]))

    return metadata, entries, tensor


class VectorError:
    def __init__(self):
        self.expected_square = 0.0
        self.actual_square = 0.0
        self.error_square = 0.0
        self.dot = 0.0

    def add(self, actual, expected, initial=None):
        import numpy as np

        actual = np.asarray(actual, dtype=np.float64)
        expected = np.asarray(expected, dtype=np.float64)
        if initial is not None:
            initial = np.asarray(initial, dtype=np.float64)
            actual = actual - initial
            expected = expected - initial
        self.expected_square += float(np.square(expected).sum())
        self.actual_square += float(np.square(actual).sum())
        self.error_square += float(np.square(actual - expected).sum())
        self.dot += float((actual * expected).sum())

    def check(self, relative_limit, cosine_limit):
        relative = math.sqrt(self.error_square / max(self.expected_square, 1e-60))
        cosine = self.dot / max(math.sqrt(self.expected_square * self.actual_square), 1e-60)
        if not math.isfinite(relative) or relative > relative_limit or cosine < cosine_limit:
            raise AssertionError(f"Vector mismatch: relative L2 error {relative}, cosine {cosine}")
        return {
            "relative_l2_error": relative,
            "cosine_similarity": cosine,
            "relative_l2_limit": relative_limit,
            "cosine_limit": cosine_limit,
        }


def worker(args):
    import numpy as np
    import torch
    from benchmark_model import BenchmarkModel

    if args.device == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("PyTorch MPS is unavailable to this process")
    torch.set_num_threads(args.threads)
    metadata, entries, tensor = load_input(args.input)
    workload = metadata["workload"]
    weights = {n[:-7]: tensor(n) for n in entries if n.endswith(".weight")}
    model = BenchmarkModel(
        metadata["config"], weights, workload["precision"], args.torch_attention
    ).to(args.device)
    del weights
    gc.collect()
    parameters = list(model.parameters())
    decay, no_decay = [], []
    for name, entry in entries.items():
        if name.endswith(".weight"):
            (no_decay if entry["role"] == "norm_scale" else decay).append(model.master(name[:-7]))
    o = metadata["optimizer"]
    optimizer = torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": o["weight_decay"]},
            {"params": no_decay, "weight_decay": 0},
        ],
        lr=metadata["learning_rate"],
        betas=(o["beta1"], o["beta2"]),
        eps=o["epsilon"],
        foreach=args.torch_optimizer == "foreach",
        fused=args.torch_optimizer == "fused",
    )
    ids = torch.tensor(metadata["input"], device=args.device).reshape(
        workload["batch"], workload["sequence_length"]
    )
    targets = torch.tensor(metadata["targets"], device=args.device)

    def sync():
        if args.device == "mps":
            torch.mps.synchronize()

    def loss():
        logits = model(ids).reshape(-1, metadata["config"]["vocab_size"])
        return torch.nn.functional.cross_entropy(logits, targets, reduction="mean")

    if args.compiled:
        loss = torch.compile(loss, fullgraph=True, dynamic=False)

    started = time.perf_counter()
    initial = loss()
    initial.backward()
    sync()
    initial_loss = initial.item()
    del initial
    initial_ms = (time.perf_counter() - started) * 1000
    errors = {}
    vector_errors = {}
    if args.verify:
        atol, rtol = (1e-4, 0.002) if workload["precision"] == "fp32" else (0.001, 0.02)
        if abs(initial_loss - metadata["initial_loss"]) > (
            1e-4 if workload["precision"] == "fp32" else 0.005
        ):
            raise AssertionError("Initial training loss mismatch")
        gradient_error = VectorError()
        for name in entries:
            if name.endswith(".gradient"):
                actual = model.master(name[:-9]).grad.detach().cpu().numpy()
                expected = tensor(name)
                np.testing.assert_allclose(actual, expected, atol=atol, rtol=rtol, err_msg=name)
                errors[name] = float(np.max(np.abs(actual - expected)))
                gradient_error.add(actual, expected)
        vector_errors["gradient"] = gradient_error.check(
            0.005 if workload["precision"] == "fp32" else 0.02, 0.99
        )
    samples = []
    last_loss = initial_loss
    for i in range(-args.warmup, args.repetitions):
        sync()
        started = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        total = torch.zeros((), device=args.device)
        for _ in range(workload["grad_accum"]):
            micro_loss = loss() / workload["grad_accum"]
            micro_loss.backward()
            total = total + micro_loss.detach()
            del micro_loss
        with torch.no_grad():
            norm = torch.linalg.vector_norm(
                torch.stack([torch.linalg.vector_norm(p.grad) for p in parameters])
            )
            scale = (
                (o["clip_norm"] / norm.clamp_min(1e-30)).clamp(max=1)
                if o["clip_norm"] > 0
                else torch.ones_like(norm)
            )
            torch._foreach_mul_([p.grad for p in parameters], scale)
        optimizer.step()
        sync()
        last_loss = total.item()
        elapsed = (time.perf_counter() - started) * 1000
        if not math.isfinite(last_loss):
            raise AssertionError("Nonfinite training loss")
        if i >= 0:
            samples.append(elapsed)
    if args.verify:
        update_error = VectorError()
        for name in entries:
            if name.endswith(".final"):
                actual = model.master(name[:-6]).detach().cpu().numpy()
                expected = tensor(name)
                np.testing.assert_allclose(
                    actual,
                    expected,
                    atol=1e-4 if workload["precision"] == "fp32" else 0.002,
                    rtol=0.003 if workload["precision"] == "fp32" else 0.02,
                    err_msg=name,
                )
                errors[name] = float(np.max(np.abs(actual - expected)))
                update_error.add(actual, expected, tensor(name[:-6] + ".weight"))
        vector_errors["update"] = update_error.check(
            0.02 if workload["precision"] == "fp32" else 0.15,
            0.99 if workload["precision"] == "fp32" else 0.985,
        )
    with torch.no_grad():
        finite = (
            torch.stack(
                [torch.isfinite(p).all() & torch.isfinite(p.grad).all() for p in parameters]
            )
            .all()
            .item()
        )
        if not finite:
            raise AssertionError("Nonfinite final weights or gradients")
    samples.sort()
    median = samples[len(samples) // 2]
    result = {
        "framework": "pytorch",
        "torch_version": torch.__version__,
        "device": args.device,
        "precision": workload["precision"],
        "attention": args.torch_attention,
        "execution": "compile-loss" if args.compiled else "eager",
        "optimizer_foreach": args.torch_optimizer == "foreach",
        "optimizer": args.torch_optimizer,
        "gradient_reset": "set_to_none=True",
        "master_gradient_moment_dtype": "fp32",
        "grad_scaling": False,
        "threads": args.threads,
        "batch": workload["batch"],
        "sequence_length": workload["sequence_length"],
        "grad_accum": workload["grad_accum"],
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "median_ms": median,
        "p95_ms": samples[math.ceil(len(samples) * 0.95) - 1],
        "tokens_per_second": workload["batch"]
        * workload["sequence_length"]
        * workload["grad_accum"]
        * 1000
        / median,
        "initial_loss": initial_loss,
        "last_loss": last_loss,
        "initial_forward_backward_ms": initial_ms,
        "peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "verified": args.verify,
        "max_absolute_errors": errors,
        "vector_errors": vector_errors,
        "cpu_fallback": os.environ.get("PYTORCH_ENABLE_MPS_FALLBACK"),
    }
    if args.device == "mps":
        result["mps_tensor_bytes_after_run"] = torch.mps.current_allocated_memory()
        result["mps_driver_bytes_after_run"] = torch.mps.driver_allocated_memory()
    args.output.write_text(json.dumps(result, indent=2) + "\n")


def compare(args):
    if args.output.exists():
        raise ValueError("Output already exists; choose a new directory")
    if not 0 <= args.warmup <= 100 or not 1 <= args.repetitions <= 1000 or args.threads < 1:
        raise ValueError("Invalid benchmark limits")
    args.output.mkdir(parents=True)
    env = os.environ.copy()
    env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    env["PYTORCH_MPS_FAST_MATH"] = "0"
    env["VECLIB_MAXIMUM_THREADS"] = str(args.threads)
    env["TORCHINDUCTOR_CACHE_DIR"] = str(args.output / "compiler-cache")
    commands = []

    def run(name, command):
        commands.append({"name": name, "command": list(map(str, command))})
        (args.output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        with (args.output / (name + ".log")).open("w") as log:
            subprocess.run(
                list(map(str, command)),
                cwd=ROOT,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
                timeout=600,
            )

    common = [
        args.build / "bin/mini_mla_bench",
        "--kind",
        "train",
        "--config",
        args.config,
        "--backend",
        args.device,
        "--precision",
        args.precision,
        "--attention",
        "reference",
        "--sequence-length",
        args.sequence_length,
        "--microbatch",
        args.microbatch,
        "--grad-accum",
        args.grad_accum,
        "--submission-window",
        args.submission_window,
        "--skip-cpu-check",
        "--time-zero-grad",
    ]
    run(
        "native-parity",
        common
        + [
            "--warmup",
            0,
            "--repetitions",
            1,
            "--comparison-data",
            args.output / "input",
            "--output",
            args.output / "native-parity.json",
        ],
    )
    variants = []
    failed = []
    options = [
        (a, c, o)
        for a in ["reference", "sdpa"]
        for c in ([False, True] if args.torch_compile else [False])
        for o in (["foreach", "fused"] if args.torch_fused else ["foreach"])
    ]
    for attention, compiled, optimizer_kind in options:
        name = "torch-" + attention + ("-compiled" if compiled else "-eager")
        if optimizer_kind != "foreach":
            name += "-" + optimizer_kind
        flags = ["--compiled"] if compiled else []
        base = [
            sys.executable,
            Path(__file__).resolve(),
            "--worker",
            "--input",
            args.output / "input",
            "--device",
            args.device,
            "--torch-attention",
            attention,
            "--threads",
            args.threads,
            "--torch-optimizer",
            optimizer_kind,
            *flags,
        ]
        try:
            run(
                name + "-parity",
                base
                + [
                    "--verify",
                    "--warmup",
                    0,
                    "--repetitions",
                    1,
                    "--output",
                    args.output / (name + "-parity.json"),
                ],
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if not compiled and optimizer_kind == "foreach":
                raise
            failed.append(
                {
                    "variant": name,
                    "status": "failed correctness or compilation gate",
                    "log": name + "-parity.log",
                }
            )
            continue
        variants.append((name, base))
    run(
        "native-performance",
        common
        + [
            "--warmup",
            args.warmup,
            "--repetitions",
            args.repetitions,
            "--output",
            args.output / "native-performance.json",
        ],
    )
    reports = []
    native = json.loads((args.output / "native-performance.json").read_text())
    for name, base in variants:
        try:
            run(
                name,
                base
                + [
                    "--warmup",
                    args.warmup,
                    "--repetitions",
                    args.repetitions,
                    "--output",
                    args.output / (name + ".json"),
                ],
            )
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired):
            if "--compiled" not in base and "fused" not in base:
                raise
            failed.append(
                {"variant": name, "status": "failed performance run", "log": name + ".log"}
            )
            continue
        report = json.loads((args.output / (name + ".json")).read_text())
        report["native_throughput_ratio"] = report["median_ms"] / native["median_ms"]
        report["parity_report"] = name + "-parity.json"
        reports.append(report)
    inputs = json.loads((args.output / "input/comparison.json").read_text())
    sources = [
        ROOT / "apps/bench_main.cpp",
        ROOT / "src/backends/mps_backend.mm",
        ROOT / "src/backends/adamw.metal",
        ROOT / "tools/benchmark_model.py",
        Path(__file__).resolve(),
        ROOT / "tools/reference_model.py",
    ]
    result = {
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "macOS": platform.mac_ver()[0],
            "machine": platform.machine(),
            "python": sys.version,
            "blas_threads": args.threads,
            "native_binary_sha256": sha256(args.build / "bin/mini_mla_bench"),
        },
        "config": inputs["config"],
        "parameter_count": sum(
            math.prod(e["shape"]) for e in inputs["tensors"] if e["name"].endswith(".weight")
        ),
        "native": native,
        "pytorch": reports,
        "failed_variants": failed,
        "commands": commands,
        "sources_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in sources},
        "input_sha256": inputs["tensors_sha256"],
        "timing_scope": "gradient reset (PyTorch set_to_none=True), forward, backward, clipping, AdamW, completion fence, scalar loss read; native per-microbatch host staging included; initial loading, compilation and validation excluded",
        "memory_scope": "separate performance processes; RSS includes libraries, model loading and compiled caches; native performance excludes the CPU oracle and tensor dump",
        "limits": [
            "Synthetic repeated pretraining batch; this does not measure data loading, checkpoint I/O, RL rollouts or time to validation quality.",
            "Both variants retain FP32 master weights, gradients and moments; FP16 computation is unscaled. FP16 numerical gates permit rounding drift, including some coordinate sign changes, and check global gradient/update direction and magnitude.",
            "No cross-hardware or vLLM training comparison is implied. Compiled variants capture the model and cross-entropy loss with generated backward, while clipping and the selected foreach/fused AdamW run outside the compiled graph.",
        ],
    }
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        f"Native: {native['median_ms']:.3f} ms/update, {native['tokens_per_second']:.0f} tokens/s"
    )
    for r in reports:
        print(
            f"PyTorch {r['attention']} {r['execution']} {r['optimizer']}: {r['median_ms']:.3f} ms/update; native throughput ratio {r['native_throughput_ratio']:.2f}x"
        )
    for failure in failed:
        print(f"{failure['variant']}: {failure['status']}; see {failure['log']}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare the native engine and equivalent PyTorch training with numerical gates."
    )
    parser.add_argument("--config", type=Path, default=ROOT / "config/mla36m-v1.json")
    parser.add_argument("--build", type=Path, default=ROOT / "build")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "mps"], default="mps")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--sequence-length", type=int, default=128)
    parser.add_argument("--microbatch", type=int, default=2)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--submission-window", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repetitions", type=int, default=30)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--torch-fused", action="store_true", help="Also gate and time fused AdamW")
    parser.add_argument(
        "--torch-optimizer", choices=["foreach", "fused"], default="foreach", help=argparse.SUPPRESS
    )
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--input", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--torch-attention", choices=["reference", "sdpa"], help=argparse.SUPPRESS)
    parser.add_argument("--compiled", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--verify", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    args.output = args.output.resolve()
    args.config = args.config.resolve()
    args.build = args.build.resolve()
    if args.worker:
        worker(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
