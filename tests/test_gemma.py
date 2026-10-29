import argparse
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from gemma_runtime import GemmaProcess, GemmaScheduler


def verify(binary, model_dir, output):
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load
    from transformers import AutoTokenizer

    model, _ = load(model_dir)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, local_files_only=True)
    report = {
        "cases": [],
        "cache_cases": [],
        "observed_max_absolute_logit_error": 0.0,
        "observed_max_relative_l2_error": 0.0,
        "checks": 0,
    }
    with tempfile.TemporaryDirectory(prefix="mini-gemma-parity-") as temporary:
        path = Path(temporary) / "logits.npy"
        with GemmaProcess(
            binary, model_dir, context_limit=2048, sessions=4, full_prefill=True, all_batch_kv=True
        ) as native:
            report["native"] = native.info

            def compare(actual, expected, label, max_absolute=0.75, max_relative=0.012):
                expected = expected.astype(mx.float32)
                mx.eval(expected)
                reference = np.asarray(expected)
                result = np.load(path)
                difference = result.astype(np.float64) - reference.astype(np.float64)
                absolute = float(np.max(np.abs(difference)))
                relative = float(np.linalg.norm(difference) / max(np.linalg.norm(reference), 1e-12))
                if relative > max_relative or absolute > max_absolute:
                    raise AssertionError(
                        f"{label}: max absolute error {absolute}, relative L2 {relative}"
                    )
                report["observed_max_absolute_logit_error"] = max(
                    report["observed_max_absolute_logit_error"], absolute
                )
                report["observed_max_relative_l2_error"] = max(
                    report["observed_max_relative_l2_error"], relative
                )
                top = np.argmax(reference, axis=-1)
                ranks = [
                    int(np.count_nonzero(row > row[token]))
                    for row, token in zip(reference, actual["tokens"], strict=True)
                ]
                if any(rank > 2 for rank in ranks):
                    raise AssertionError(
                        f"{label}: native greedy token outside reference top three: {ranks}"
                    )
                report["cases"].append(
                    {
                        "case": label,
                        "max_absolute_error": absolute,
                        "relative_l2_error": relative,
                        "native_tokens": actual["tokens"],
                        "reference_tokens": top.tolist(),
                        "reference_ranks": ranks,
                    }
                )
                report["checks"] += 1

            prompts = [
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": text}],
                    tokenize=True,
                    add_generation_prompt=True,
                    enable_thinking=False,
                    return_dict=False,
                )
                for text in ["What is 2 + 2?", "Explain gradient descent in one sentence."]
            ]
            caches = [model.make_cache() for _ in prompts]
            for session, prompt in enumerate(prompts):
                actual = native.call("prefill", session=session, tokens=prompt, logits=str(path))
                expected = model(mx.array([prompt]), cache=caches[session])[:, -1, :]
                compare(actual, expected, f"prefill-{session}")
            for step in range(6):
                tokens = [100000 + step, 200000 + step]
                actual = native.call("decode", sessions=[0, 1], tokens=tokens, logits=str(path))
                expected = mx.concatenate(
                    [
                        model(mx.array([[token]]), cache=cache)[:, -1, :]
                        for token, cache in zip(tokens, caches, strict=True)
                    ],
                    axis=0,
                )
                compare(actual, expected, f"mixed-length-decode-{step}")
            for session in [0, 1]:
                native.call("release", session=session)
            prompt = [2] + [500 + i % 100 for i in range(519)]
            cache = model.make_cache()
            for offset in range(0, len(prompt), 512):
                expected = model(mx.array([prompt[offset : offset + 512]]), cache=cache)[:, -1, :]
                mx.eval(expected)
            actual = native.call("prefill", session=0, tokens=prompt, logits=str(path))
            compare(actual, expected, "sliding-boundary-prefill")
            for step in range(4):
                token = 200000 + step
                actual = native.call("decode", sessions=[0], tokens=[token], logits=str(path))
                expected = model(mx.array([[token]]), cache=cache)[:, -1, :]
                compare(actual, expected, f"sliding-boundary-decode-{step}")
            native.call("release", session=0)
            prompt = prompts[0]
            actual = native.call(
                "prefill_batch", sessions=[0, 1, 2], prompts=[prompt] * 3, logits=str(path)
            )
            expected = model(mx.array([prompt] * 3), cache=model.make_cache())[:, -1, :]
            compare(actual, expected, "batched-prefill")
            with GemmaProcess(
                binary,
                model_dir,
                context_limit=2048,
                sessions=4,
                full_prefill=True,
                batch_kv=False,
            ) as independent:
                reference_path = Path(temporary) / "independent.npy"
                prompt = [2] + [500 + i % 97 for i in range(511)]
                handles = [0, 1, 2]
                for engine, target in ((native, path), (independent, reference_path)):
                    engine.call(
                        "prefill_batch", sessions=handles, prompts=[prompt] * 3, logits=str(target)
                    )

                def check_cache(handles, steps, label):
                    before = native.call("info")["batch_cache_reuses"]
                    for step in range(steps):
                        tokens = [100000 + step + handle * 500 for handle in handles]
                        actual = native.call(
                            "decode",
                            sessions=handles,
                            tokens=tokens,
                            logits=str(path),
                            include_info=True,
                        )
                        baseline = independent.call(
                            "decode", sessions=handles, tokens=tokens, logits=str(reference_path)
                        )
                        np.testing.assert_array_equal(
                            np.load(path), np.load(reference_path), err_msg=f"{label}-{step}"
                        )
                        if actual["tokens"] != baseline["tokens"]:
                            raise AssertionError(f"{label}: KV reuse changed greedy tokens")
                        report["checks"] += 1
                    reused = native.info["batch_cache_reuses"] - before
                    if reused < steps - 1:
                        raise AssertionError(f"{label}: expected packed reuse, got {reused} steps")
                    report["cache_cases"].append(
                        {
                            "case": label,
                            "steps": steps,
                            "bitwise_equal_logits": True,
                            "reused_batches": reused,
                        }
                    )

                check_cache(handles, 6, "packed-sliding-wrap")
                handles = [2, 0, 1]
                check_cache(handles, 2, "packed-reorder")
                for engine in (native, independent):
                    engine.call("release", session=1)
                check_cache([2, 0], 2, "packed-shrink")
                for engine in (native, independent):
                    engine.call("prefill", session=0, tokens=[2, 300, 301])
                    try:
                        engine.call("prefill", session=0, tokens=[262144])
                    except ValueError:
                        pass
                    else:
                        raise AssertionError("Invalid prefill token was accepted")
                check_cache([2, 0], 6, "packed-mixed-replacement")
                for engine in (native, independent):
                    engine.call("prefill", session=0, tokens=prompt + [501, 502])
                check_cache([2, 0], 4, "packed-different-ring-cursors")
                for engine in (native, independent):
                    engine.call("prefill", session=0, tokens=prompt[:-2])
                check_cache([2, 0], 6, "packed-mixed-enter-ring")
                if native.info["batch_cache_reuses"] < 8:
                    raise AssertionError("Packed KV reuse was not exercised")
                report["checks"] += 1
            for command, fields in [
                ("decode", {"sessions": [0, 0], "tokens": [1, 1]}),
                ("decode", {"sessions": [99], "tokens": [1]}),
                ("decode", {"sessions": [0], "tokens": [262144]}),
                ("prefill", {"session": 3, "tokens": [2] * 2049}),
            ]:
                try:
                    native.call(command, **fields)
                except ValueError:
                    report["checks"] += 1
                else:
                    raise AssertionError(f"Invalid {command} was accepted")
            for session in range(4):
                native.call("release", session=session)
            info = native.call("info")
            if info["kv_bytes"] != 0 or info["active_sessions"] != 0:
                raise AssertionError("Released sessions retain KV cache")
            report["checks"] += 1
        with GemmaProcess(binary, model_dir, context_limit=2048, sessions=4) as native:
            for session, prompt in enumerate(prompts):
                actual = native.call("prefill", session=session, tokens=prompt, logits=str(path))
                expected = model(mx.array([prompt]), cache=model.make_cache())[:, -1, :]
                compare(actual, expected, f"reduced-query-prefill-{session}", 2.0, 0.02)
                if actual["tokens"] != mx.argmax(expected, axis=-1).tolist():
                    raise AssertionError("Reduced-query prefill changed the tested greedy token")
            for step in range(3):
                native.call(
                    "decode",
                    sessions=[0, 1],
                    tokens=[100000 + step, 200000 + step],
                    include_info=True,
                )
            if native.info["batch_kv_policy"] != "auto" or native.info["batch_cache_reuses"] != 0:
                raise AssertionError("Auto policy reused unequal-position small caches")
            report["checks"] += 1
        scheduler = GemmaScheduler(binary, model_dir, tokenizer, context_limit=2048, max_batch=4)
        try:
            requests = [scheduler.submit(prompt, 12, ignore_eos=True) for prompt in prompts * 2]
            results = [request.future.result(timeout=60) for request in requests]
            if (
                any(len(r["tokens"]) != 12 for r in results)
                or scheduler.counters["max_decode_batch"] < 2
            ):
                raise AssertionError("Continuous batching did not process concurrent requests")
            report["scheduler"] = scheduler.counters
            report["checks"] += 1
            cancelled = scheduler.submit(prompts[0], 128)
            cancelled.cancelled.set()
            result = cancelled.future.result(timeout=30)
            if result["finish_reason"] != "cancelled":
                raise AssertionError("Cancellation failed")
            report["checks"] += 1
        finally:
            scheduler.close()
    report["status"] = "passed"
    report["limits"] = {
        "full_prefill_max_absolute_logit_error": 0.75,
        "full_prefill_relative_l2_error": 0.012,
        "reduced_prefill_max_absolute_logit_error": 2.0,
        "reduced_prefill_relative_l2_error": 0.02,
        "reduced_prefill_exact_tested_greedy_match": True,
        "native_greedy_reference_rank": 2,
        "precision": "BF16 activations, identical affine group-64 4-bit weights",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Gemma: {report['checks']} checks passed; max relative logit error {report['observed_max_relative_l2_error']:.6g}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.binary.resolve(), args.model.resolve(), args.output.resolve())
