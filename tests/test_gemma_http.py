import argparse
import json
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from compare_serving import free_port, stop, wait_ready
from compare_serving import request as stream_request


def verify(binary, model, output):
    checks = []
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    command = [
        sys.executable,
        str(Path(__file__).resolve().parents[1] / "tools/gemma_server.py"),
        "--model",
        str(model),
        "--binary",
        str(binary),
        "--port",
        str(port),
        "--context-limit",
        "2048",
        "--max-batch",
        "4",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryFile(mode="w+") as log:
        process = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
        try:
            wait_ready(process, base, "/health", 120)

            def call(path, payload=None):
                data = json.dumps(payload).encode() if payload is not None else None
                req = urllib.request.Request(
                    base + path, data, {"Content-Type": "application/json"}
                )
                try:
                    with urllib.request.urlopen(req, timeout=30) as response:
                        return response.status, json.load(response)
                except urllib.error.HTTPError as error:
                    return error.code, json.load(error)

            code, models = call("/v1/models")
            assert code == 200 and models["data"][0]["id"] == "mini-gemma-e2b"
            checks.append("model discovery")
            code, result = call(
                "/v1/chat/completions",
                {
                    "messages": [{"role": "user", "content": "What is 2 + 2?"}],
                    "max_tokens": 16,
                    "temperature": 0,
                    "return_token_ids": True,
                },
            )
            assert code == 200 and "4" in result["choices"][0]["message"]["content"]
            assert result["usage"]["completion_tokens"] == len(result["choices"][0]["token_ids"])
            checks.append("chat generation and token usage")
            prompt = [2, 22190, 107, 2364, 107]
            streaming = stream_request(base, prompt, 16)
            code, result = call(
                "/v1/completions",
                {
                    "prompt": prompt,
                    "max_tokens": 16,
                    "temperature": 0,
                    "ignore_eos": True,
                    "return_token_ids": True,
                },
            )
            assert code == 200 and result["choices"][0]["token_ids"] == streaming["tokens"]
            checks.append("stream/nonstream token equality and SSE completion")
            code, stopped = call(
                "/v1/chat/completions",
                {
                    "messages": [{"role": "user", "content": "What is 2 + 2?"}],
                    "max_tokens": 16,
                    "stop": "4",
                },
            )
            assert code == 200 and "4" not in stopped["choices"][0]["message"]["content"]
            assert stopped["choices"][0]["finish_reason"] == "stop"
            checks.append("stop strings withheld")
            for payload, expected in [
                ({"prompt": prompt, "temperature": 1}, 400),
                ({"prompt": [262144]}, 400),
                ({"prompt": prompt, "model": "unknown"}, 404),
                ({"prompt": [2] * 2048, "max_tokens": 1}, 400),
            ]:
                code, _ = call("/v1/completions", payload)
                assert code == expected, (payload, code)
                checks.append(f"invalid request rejected ({expected})")
            req = urllib.request.Request(
                base + "/v1/completions",
                json.dumps(
                    {
                        "prompt": prompt,
                        "max_tokens": 1000,
                        "stream": True,
                        "ignore_eos": True,
                        "return_token_ids": True,
                    }
                ).encode(),
                {"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as response:
                for line in response:
                    if b'"token_ids": [' in line:
                        break
                else:
                    raise AssertionError("Disconnected stream produced no tokens")
            deadline = time.monotonic() + 10
            while time.monotonic() < deadline:
                _, health = call("/health")
                if health["engine"]["active_sessions"] == 0:
                    break
                time.sleep(0.02)
            else:
                raise AssertionError("Disconnected request retained its session")
            checks.append("disconnect cancellation releases cache")
            output.write_text(
                json.dumps({"status": "passed", "checks": checks, "command": command}, indent=2)
                + "\n"
            )
            print(f"Gemma HTTP: {len(checks)} checks passed")
        except BaseException:
            log.seek(0)
            sys.stderr.write(log.read())
            raise
        finally:
            stop(process)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    verify(args.binary.resolve(), args.model.resolve(), args.output.resolve())
