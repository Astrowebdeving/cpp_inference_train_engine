import argparse
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_NAME = "mini-gemma-e2b"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def percentile(values, fraction):
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * fraction) - 1)]


def free_port():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def request(base, prompt, max_tokens, barrier=None):
    data = json.dumps(
        {
            "model": MODEL_NAME,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
            "return_token_ids": True,
        }
    ).encode()
    req = urllib.request.Request(
        base + "/v1/completions", data, {"Content-Type": "application/json"}
    )
    if barrier:
        barrier.wait(timeout=30)
    start = time.perf_counter()
    tokens, timestamps, text = [], [], ""
    usage = None
    first = None
    done = False
    with urllib.request.urlopen(req, timeout=120) as response:
        for line in response:
            if not line.startswith(b"data: "):
                continue
            message = line[6:].strip()
            if message == b"[DONE]":
                done = True
                break
            event = json.loads(message)
            if "error" in event:
                raise RuntimeError(event["error"])
            now = time.perf_counter()
            for choice in event.get("choices", []):
                ids = choice.get("token_ids") or []
                if ids and first is None:
                    first = now
                for token in ids:
                    tokens.append(token)
                    timestamps.append(now - start)
                text += choice.get("text") or ""
            if event.get("usage"):
                usage = event["usage"]
    elapsed = time.perf_counter() - start
    if (
        not done
        or not usage
        or usage["completion_tokens"] != max_tokens
        or len(tokens) != max_tokens
    ):
        raise AssertionError(f"Incomplete SSE response: {done=}, {usage=}, {len(tokens)=}")
    return {
        "latency_ms": elapsed * 1000,
        "ttft_ms": (first - start) * 1000,
        "tokens": tokens,
        "text": text,
        "usage": usage,
        "token_arrival_seconds": timestamps,
    }


def burst(base, prompts, count, max_tokens):
    barrier = threading.Barrier(count + 1)
    with ThreadPoolExecutor(max_workers=count) as pool:
        futures = [
            pool.submit(request, base, prompts[i], max_tokens, barrier) for i in range(count)
        ]
        start = time.perf_counter()
        barrier.wait(timeout=30)
        results = [future.result(timeout=180) for future in futures]
    elapsed = time.perf_counter() - start
    return {
        "wall_seconds": elapsed,
        "output_tokens_per_second": count * max_tokens / elapsed,
        "requests": results,
    }


def stop(process):
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def wait_ready(process, base, path, timeout):
    start = time.perf_counter()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"Server exited with code {process.returncode}; inspect its log")
        try:
            with urllib.request.urlopen(base + path, timeout=1) as response:
                if response.status == 200:
                    return time.perf_counter() - start
        except (OSError, urllib.error.URLError):
            pass
        time.sleep(0.25)
    raise TimeoutError("Server did not become ready")


def reference(args):
    import mlx.core as mx
    from mlx_lm import load

    inputs = json.loads(args.input.read_text())
    model, _ = load(args.model)
    outputs = []
    for prompt in inputs["prompts"]:
        cache = model.make_cache()
        logits = model(mx.array([prompt]), cache=cache)[:, -1, :]
        tokens = []
        for i in range(inputs["max_tokens"]):
            token = mx.argmax(logits, axis=-1)
            mx.eval(token)
            tokens.append(token.item())
            if i + 1 < inputs["max_tokens"]:
                logits = model(mx.reshape(token, (1, 1)), cache=cache)[:, -1, :]
        outputs.append(tokens)
    args.output.write_text(json.dumps({"backend": "mlx-lm", "tokens": outputs}, indent=2) + "\n")


def run(args):
    from transformers import AutoTokenizer

    args.output = args.output.resolve()
    args.model = args.model.resolve()
    args.binary = args.binary.resolve()
    args.native_source_root = args.native_source_root.resolve()
    source_files = [
        Path(__file__).resolve(),
        args.native_source_root / "tools/gemma_server.py",
        args.native_source_root / "tools/gemma_runtime.py",
        args.native_source_root / "src/infer/gemma.cpp",
        args.native_source_root / "apps/gemma_main.cpp",
    ]
    for path in [args.binary, *source_files]:
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.independent_kv_baseline:
        help_text = subprocess.check_output([str(args.binary), "--help"], text=True)
        if "--independent-kv" not in help_text:
            raise ValueError("The independent KV comparison requires a supporting native binary")
    args.output.mkdir(parents=True, exist_ok=False)
    tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
    prompts = []
    questions = [
        "Explain gradient descent.",
        "Explain attention.",
        "What does a loss function do?",
        "Why do we use a validation set?",
        "Explain overfitting.",
        "Explain matrix multiplication.",
        "What is a token?",
        "Explain a learning rate.",
    ]
    for i, question in enumerate(questions[: max(args.concurrency)]):
        length = args.prompt_length - (16 * (i % 4) if args.mixed_lengths else 0)
        content = (
            question
            + " Context: a neural network learns from examples by updating weights."
            * args.prompt_length
        )
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": content}],
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=False,
            return_dict=False,
        )
        prompts.append(ids[: length - 8] + ids[-8:])
    inputs = {
        "prompts": prompts,
        "max_tokens": args.max_tokens,
        "prompt_length": args.prompt_length,
        "prompt_lengths": [len(prompt) for prompt in prompts],
        "force_length_ignore_eos": True,
    }
    input_file = args.output / "inputs.json"
    input_file.write_text(json.dumps(inputs, indent=2) + "\n")
    env = os.environ.copy()
    env.update(
        {
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
            "VLLM_LOGGING_LEVEL": "WARNING",
            "PYTORCH_ENABLE_MPS_FALLBACK": "0",
            "VECLIB_MAXIMUM_THREADS": "1",
            "VLLM_MLX_DEVICE": "gpu",
            "VLLM_METAL_DECODE_PIPELINE": "1",
        }
    )
    env.pop("MINI_MLA_GEMMA_TRACE", None)
    commands = []
    ref_command = [
        sys.executable,
        __file__,
        "--reference",
        "--model",
        str(args.model),
        "--input",
        str(input_file),
        "--output",
        str(args.output / "mlx-reference.json"),
    ]
    commands.append(ref_command)
    with (args.output / "mlx-reference.log").open("w") as log:
        subprocess.run(
            ref_command, env=env, check=True, timeout=600, stdout=log, stderr=subprocess.STDOUT
        )
    expected = json.loads((args.output / "mlx-reference.json").read_text())["tokens"]
    results = []
    variants = ["native"] if args.native_only else ["native", "vllm-metal"]
    if args.independent_kv_baseline:
        variants.insert(1, "native-independent-kv")
    if args.vllm_compiled_mlp:
        variants.append("vllm-metal-compiled-mlp")
    if args.full_prefill_baseline:
        variants.insert(1, "native-full-prefill")
    if args.vllm_first:
        variants.sort(key=lambda variant: not variant.startswith("vllm"))
    for variant in variants:
        port = free_port()
        base = f"http://127.0.0.1:{port}"
        if variant.startswith("native"):
            command = [
                sys.executable,
                str(args.native_source_root / "tools/gemma_server.py"),
                "--model",
                str(args.model),
                "--binary",
                str(args.binary),
                "--context-limit",
                str(args.context_limit),
                "--max-batch",
                str(max(args.concurrency)),
                "--port",
                str(port),
            ]
            if variant == "native-full-prefill":
                command.append("--full-prefill")
            if variant == "native-independent-kv":
                command.append("--independent-kv")
            health_path = "/health"
        else:
            command = [
                sys.executable,
                "-m",
                "vllm.entrypoints.openai.api_server",
                "--model",
                str(args.model),
                "--served-model-name",
                MODEL_NAME,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--dtype",
                "bfloat16",
                "--max-model-len",
                str(args.context_limit),
                "--max-num-seqs",
                str(max(args.concurrency)),
                "--max-num-batched-tokens",
                str(args.vllm_batch_tokens),
                "--gpu-memory-utilization",
                "0.2",
                "--kv-cache-memory-bytes",
                str(args.vllm_kv_bytes),
                "--no-enable-prefix-caching",
                "--limit-mm-per-prompt",
                '{"image":0,"audio":0,"video":0}',
                "--enforce-eager",
                "--no-enable-log-requests",
            ]
            health_path = "/health"
        commands.append(command)
        (args.output / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        variant_env = env.copy()
        variant_env["VLLM_METAL_COMPILED_MLP"] = "1" if variant.endswith("compiled-mlp") else "0"
        with (args.output / (variant + ".log")).open("w") as log:
            process = subprocess.Popen(
                command, env=variant_env, stdout=log, stderr=subprocess.STDOUT
            )
            try:
                startup = wait_ready(process, base, health_path, args.startup_timeout)
                report = {
                    "variant": variant,
                    "startup_seconds": startup,
                    "command": command,
                    "environment_overrides": {
                        key: variant_env[key]
                        for key in [
                            "VLLM_MLX_DEVICE",
                            "VLLM_METAL_DECODE_PIPELINE",
                            "VLLM_METAL_COMPILED_MLP",
                        ]
                    },
                    "workloads": [],
                    "parity": [],
                }
                for i, prompt in enumerate(prompts):
                    actual = request(base, prompt, args.max_tokens)
                    exact = sum(a == b for a, b in zip(actual["tokens"], expected[i], strict=True))
                    prefix = 0
                    for a, b in zip(actual["tokens"], expected[i], strict=True):
                        if a != b:
                            break
                        prefix += 1
                    report["parity"].append(
                        {
                            "prompt": i,
                            "exact_positions": exact,
                            "total_positions": args.max_tokens,
                            "matching_prefix": prefix,
                            "tokens": actual["tokens"],
                        }
                    )
                report["exact_greedy_reference_match"] = all(
                    r["exact_positions"] == args.max_tokens for r in report["parity"]
                )
                for count in args.concurrency:
                    for _ in range(args.warmup):
                        burst(base, prompts, count, args.max_tokens)
                    samples = [
                        burst(base, prompts, count, args.max_tokens)
                        for _ in range(args.repetitions)
                    ]
                    requests = [r for sample in samples for r in sample["requests"]]
                    intervals = [
                        1000 * (b - a)
                        for r in requests
                        for a, b in zip(r["token_arrival_seconds"], r["token_arrival_seconds"][1:])
                    ]
                    report["workloads"].append(
                        {
                            "concurrency": count,
                            "samples": samples,
                            "median_output_tokens_per_second": percentile(
                                [r["output_tokens_per_second"] for r in samples], 0.5
                            ),
                            "median_ttft_ms": percentile([r["ttft_ms"] for r in requests], 0.5),
                            "p95_ttft_ms": percentile([r["ttft_ms"] for r in requests], 0.95),
                            "median_request_ms": percentile(
                                [r["latency_ms"] for r in requests], 0.5
                            ),
                            "p95_request_ms": percentile([r["latency_ms"] for r in requests], 0.95),
                            "median_observed_token_interval_ms": percentile(intervals, 0.5),
                        }
                    )
                if variant.startswith("native"):
                    with urllib.request.urlopen(base + health_path, timeout=10) as response:
                        report["native_status"] = json.load(response)
                (args.output / (variant + ".json")).write_text(json.dumps(report, indent=2) + "\n")
                results.append(report)
                print(
                    variant,
                    [
                        (w["concurrency"], round(w["median_output_tokens_per_second"], 2))
                        for w in report["workloads"]
                    ],
                    "exact reference",
                    report["exact_greedy_reference_match"],
                    flush=True,
                )
            finally:
                stop(process)
    versions = {}
    for package in [
        "mlx",
        "mlx-lm",
        "vllm",
        "vllm-metal",
        "torch",
        "transformers",
        "fastapi",
        "uvicorn",
    ]:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    result = {
        "recorded_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "environment": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "versions": versions,
        },
        "model": str(args.model),
        "model_weights_sha256": sha256(args.model / "model.safetensors"),
        "config_sha256": sha256(args.model / "config.json"),
        "input_sha256": sha256(input_file),
        "native_binary_sha256": sha256(args.binary),
        "commands": commands,
        "source_sha256": {
            str(p.relative_to(ROOT) if p.is_relative_to(ROOT) else p): sha256(p)
            for p in source_files
        },
        "prompt_length": args.prompt_length,
        "prompt_lengths": inputs["prompt_lengths"],
        "mixed_lengths": args.mixed_lengths,
        "max_tokens": args.max_tokens,
        "warmup": args.warmup,
        "repetitions": args.repetitions,
        "vllm_reserved_kv_bytes": args.vllm_kv_bytes,
        "vllm_max_batched_tokens": args.vllm_batch_tokens,
        "results": results,
        "method": "Serial server processes; identical pretokenized prompts, BF16 activations and packed 4-bit weights; temperature 0, EOS ignored to force equal output lengths; prefix caching and speculative decoding disabled; HTTP/SSE, scheduler, GPU completion, sampling and output detokenization timed. Startup and input tokenization excluded from throughput.",
        "limits": [
            "Token arrivals are observed at the HTTP client; coalesced chunks can assign one timestamp to several tokens.",
            "A changed greedy sequence can diverge after one rounded logit decision; exact positions and matching prefixes are reported separately from numerical parity.",
            "Native memory grows with live private caches; vLLM reserves a fixed paged pool. These are different allocation policies.",
            "Only this checkpoint, device and workload were measured; no training or other-model performance claim is implied.",
            "vLLM uses the recorded batch-token and KV-pool settings; these are tested configurations, not a search over all scheduler settings.",
        ],
    }
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare our C++ Gemma HTTP server with vLLM-Metal"
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--binary", type=Path, default=ROOT / "build/bin/mini_mla_gemma")
    parser.add_argument("--native-source-root", type=Path, default=ROOT)
    parser.add_argument("--native-only", action="store_true")
    parser.add_argument("--independent-kv-baseline", action="store_true")
    parser.add_argument("--mixed-lengths", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8])
    parser.add_argument("--prompt-length", type=int, default=128)
    parser.add_argument("--max-tokens", type=int, default=64)
    parser.add_argument("--context-limit", type=int, default=2048)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--vllm-kv-bytes", type=int, default=512 * 1024 * 1024)
    parser.add_argument("--vllm-batch-tokens", type=int, default=512)
    parser.add_argument("--startup-timeout", type=int, default=300)
    parser.add_argument("--full-prefill-baseline", action="store_true")
    parser.add_argument(
        "--vllm-compiled-mlp", action="store_true", help="Also test vLLM's opt-in compiled MLP"
    )
    parser.add_argument(
        "--vllm-first", action="store_true", help="Reverse server order for a repeat"
    )
    parser.add_argument("--reference", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--input", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.reference:
        reference(args)
    else:
        if (
            not all(1 <= c <= 8 for c in args.concurrency)
            or args.prompt_length < 16
            or (args.mixed_lengths and args.prompt_length < 64)
            or (args.native_only and args.vllm_compiled_mlp)
            or args.max_tokens < 2
            or args.prompt_length + args.max_tokens > args.context_limit
            or args.repetitions < 1
            or args.warmup < 0
            or args.vllm_batch_tokens < max(args.concurrency)
        ):
            parser.error("Invalid comparison workload")
        run(args)
