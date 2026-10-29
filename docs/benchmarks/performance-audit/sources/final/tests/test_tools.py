import concurrent.futures
import hashlib
import http.client
import importlib.util
import json
import random
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BIN = (
    Path(sys.argv.pop(1)).resolve()
    if len(sys.argv) > 1 and not sys.argv[1].startswith("-")
    else ROOT / "build/bin"
)
sys.path.insert(0, str(ROOT / "tools"))
from common import Tokenizer, canonical
from inspect_checkpoint import inspect
from train_tokenizer import train


def run(command, expect=0):
    result = subprocess.run(
        list(map(str, command)), cwd=ROOT, capture_output=True, text=True, timeout=120
    )
    if result.returncode != expect:
        raise AssertionError(
            f"{command}\nexit={result.returncode}\n{result.stderr}\n{result.stdout[-2000:]}"
        )
    return result.stdout


class Tools(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="mla-tools-")
        cls.path = Path(cls.directory.name)
        cls.tokenizer_path = ROOT / "examples/tokenizer.json"
        cls.tokenizer = Tokenizer.load(cls.tokenizer_path)
        cls.sft = cls.path / "sft"
        run(
            [
                sys.executable,
                ROOT / "tools/prepare_dataset.py",
                "--input",
                ROOT / "examples/chat.jsonl",
                "--tokenizer",
                cls.tokenizer_path,
                "--mode",
                "sft",
                "--output",
                cls.sft,
                "--validation-fraction",
                0,
                "--sequence-length",
                48,
            ]
        )
        cls.training = [
            BIN / "mini_mla_train",
            "--config",
            ROOT / "config/tiny-test.json",
            "--tokenizer",
            cls.tokenizer_path,
            "--train",
            cls.sft / "train-*.sft.jsonl",
            "--backend",
            "cpu",
            "--mode",
            "sft",
            "--sequence-length",
            48,
            "--microbatch",
            2,
            "--grad-accum",
            2,
            "--warmup-steps",
            2,
            "--steps",
            80,
            "--learning-rate",
            0.006,
            "--checkpoint-every",
            40,
            "--log-every",
            20,
        ]
        cls.run_path = cls.path / "sft-run"
        run(cls.training + ["--output", cls.run_path])
        cls.model = cls.run_path / "model-latest"

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_tokenizers_match_heap_and_slow_reference(self):
        corpus = ["aba abc banana café 世界\n"] * 20 + ["ababa\x00✨ repeated words words"] * 10
        value = train(corpus, 320)
        tokenizer = Tokenizer(value)
        path = self.path / "bpe.json"
        path.write_text(canonical(value))
        rng = random.Random(91)
        texts = ["", "ababa", "<|eos|>", "café 世界\n\x00🚀"]
        texts += [
            "".join(rng.choices("abc café✨世界\n", k=rng.randrange(1, 200))) for _ in range(40)
        ]
        for text in texts:
            self.assertEqual(tokenizer.encode(text), tokenizer.encode_reference(text))
            self.assertEqual(tokenizer.decode_bytes(tokenizer.encode(text)), text.encode())
            input_path = self.path / "input.txt"
            input_path.write_text(text)
            result = json.loads(
                run(
                    [
                        BIN / "mini_mla_tokenize",
                        "--tokenizer",
                        path,
                        "--input",
                        input_path,
                        "--json",
                    ]
                )
            )
            self.assertEqual(result["tokens"], tokenizer.encode(text))
            self.assertEqual(result["decoded"], text)

    def test_dataset_splits_are_reproducible_and_content_based(self):
        source = self.path / "documents.jsonl"
        documents = [f"document {i} contains useful text and repeated words.\n" for i in range(80)]
        source.write_text("".join(canonical({"text": text}) for text in documents + documents[:20]))
        roots = [self.path / "split-a", self.path / "split-b"]
        for path in roots:
            run(
                [
                    sys.executable,
                    ROOT / "tools/prepare_dataset.py",
                    "--input",
                    source,
                    "--tokenizer",
                    self.tokenizer_path,
                    "--output",
                    path,
                    "--validation-fraction",
                    0.2,
                    "--shard-tokens",
                    100000,
                ]
            )
        for split in ("train", "val"):
            self.assertEqual(
                (roots[0] / f"{split}-00000.tok").read_bytes(),
                (roots[1] / f"{split}-00000.tok").read_bytes(),
            )
        for text in documents[:20]:
            split = (
                "val"
                if int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") / 2**64 < 0.2
                else "train"
            )
            needle = bytes().join(int(i).to_bytes(2, "little") for i in self.tokenizer.encode(text))
            self.assertIn(needle, (roots[0] / f"{split}-00000.tok").read_bytes()[160:])
            self.assertNotIn(
                needle,
                (roots[0] / f"{'train' if split == 'val' else 'val'}-00000.tok").read_bytes()[160:],
            )

    def test_sft_overfits_and_resume_matches_uninterrupted(self):
        manifest, _ = inspect(self.run_path / "checkpoint-latest")
        self.assertEqual(manifest["step"], 80)
        metrics = [
            json.loads(line) for line in (self.run_path / "metrics.jsonl").read_text().splitlines()
        ]
        self.assertLess(metrics[-1]["train_loss"], 0.08)
        generation = json.loads(
            run(
                [
                    BIN / "mini_mla_generate",
                    "--model",
                    self.model,
                    "--prompt",
                    "Say hi.",
                    "--chat",
                    "--temperature",
                    0,
                    "--max-new-tokens",
                    12,
                    "--json",
                ]
            )
        )
        self.assertEqual(generation["text"], "Hi!")
        base = [
            BIN / "mini_mla_generate",
            "--model",
            self.model,
            "--prompt",
            "Say hi.",
            "--chat",
            "--temperature",
            0,
            "--max-new-tokens",
            12,
            "--json",
        ]
        stopped = json.loads(run(base + ["--stop", "i!"]))
        self.assertEqual(stopped["text"], "H")
        self.assertEqual(stopped["finish_reason"], "stop")
        stopped = json.loads(run(base + ["--stop-tokens", ord("i")]))
        self.assertEqual(stopped["text"], "H")
        before = (self.run_path / "checkpoint-latest/weights.bin").read_bytes()
        moments = (self.run_path / "checkpoint-latest/optimizer.bin").read_bytes()
        shutil.rmtree(self.run_path / "checkpoint-00000080")
        shutil.rmtree(self.run_path / "model-80")
        run([BIN / "mini_mla_train", "--resume", self.run_path / "checkpoint-00000040"])
        self.assertEqual((self.run_path / "checkpoint-latest/weights.bin").read_bytes(), before)
        self.assertEqual((self.run_path / "checkpoint-latest/optimizer.bin").read_bytes(), moments)
        repeated = [
            json.loads(line) for line in (self.run_path / "metrics.jsonl").read_text().splitlines()
        ]
        self.assertEqual(metrics[-1]["train_loss"], repeated[-1]["train_loss"])
        run(
            [
                BIN / "mini_mla_train",
                "--resume",
                self.run_path / "checkpoint-latest",
                "--sequence-length",
                47,
            ],
            expect=1,
        )

    def test_muon_cli_settings_and_resume(self):
        output = self.path / "muon-run"
        command = [
            BIN / "mini_mla_train",
            "--config",
            ROOT / "config/tiny-test.json",
            "--tokenizer",
            self.tokenizer_path,
            "--train",
            self.sft / "train-*.sft.jsonl",
            "--output",
            output,
            "--backend",
            "cpu",
            "--mode",
            "sft",
            "--sequence-length",
            48,
            "--microbatch",
            2,
            "--grad-accum",
            1,
            "--steps",
            12,
            "--warmup-steps",
            2,
            "--learning-rate",
            0.006,
            "--checkpoint-every",
            6,
            "--log-every",
            6,
            "--optimizer",
            "muon",
            "--muon-momentum",
            0.8,
            "--muon-nesterov",
            "false",
            "--muon-ns-steps",
            3,
            "--muon-layout",
            "batched",
            "--muon-ns-order",
            "gram",
        ]
        run(command)
        options = json.loads((output / "options.json").read_text())
        self.assertEqual(options["muon_layout"], "batched")
        self.assertEqual(options["muon_ns_order"], "gram")
        self.assertEqual(options["optimizer"]["kind"], "hybrid_muon_adamw")
        self.assertAlmostEqual(options["optimizer"]["momentum"], 0.8)
        self.assertEqual(options["optimizer"]["nesterov"], False)
        self.assertEqual(options["optimizer"]["ns_steps"], 3)
        before = (output / "checkpoint-latest/weights.bin").read_bytes()
        moments = (output / "checkpoint-latest/optimizer.bin").read_bytes()
        shutil.rmtree(output / "checkpoint-00000012")
        shutil.rmtree(output / "model-12")
        run([BIN / "mini_mla_train", "--resume", output / "checkpoint-00000006"])
        self.assertEqual((output / "checkpoint-latest/weights.bin").read_bytes(), before)
        self.assertEqual((output / "checkpoint-latest/optimizer.bin").read_bytes(), moments)
        benchmark = [
            BIN / "mini_mla_bench",
            "--kind",
            "train",
            "--backend",
            "cpu",
            "--config",
            ROOT / "config/tiny-test.json",
            "--sequence-length",
            3,
            "--warmup",
            0,
            "--repetitions",
            1,
        ]
        for option, value in (
            ("muon-momentum", "1"),
            ("muon-ns-steps", "0"),
            ("muon-nesterov", "yes"),
            ("muon-layout", "invalid"),
            ("muon-ns-order", "invalid"),
        ):
            run(benchmark + ["--optimizer", "muon", f"--{option}", value], expect=1)
        run(benchmark + ["--muon-ns-steps", 3], expect=1)

    def test_native_debug_state_matches_independent_pytorch(self):
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("numpy") is None:
            self.skipTest("Install tools/requirements-dev.txt for the independent oracle")
        debug = self.path / "oracle"
        run(
            [
                BIN / "mini_mla_inspect",
                "--model",
                self.model,
                "--tokens",
                "1,3,5",
                "--targets",
                "3,5,7",
                "--mask",
                "0,1,1",
                "--debug-output",
                debug,
                "--debug-update",
            ]
        )
        result = json.loads(
            run([sys.executable, ROOT / "tools/reference_model.py", "--compare", debug])
        )
        self.assertTrue(result["passed"])

    def test_training_comparison_checks_gradients_and_updates_before_timing(self):
        if importlib.util.find_spec("torch") is None or importlib.util.find_spec("numpy") is None:
            self.skipTest("Install tools/requirements-dev.txt for training comparisons")
        destination = self.path / "training-comparison"
        run(
            [
                sys.executable,
                ROOT / "tools/compare_training.py",
                "--config",
                ROOT / "config/tiny-test.json",
                "--build",
                BIN.parent,
                "--device",
                "cpu",
                "--sequence-length",
                8,
                "--microbatch",
                2,
                "--grad-accum",
                2,
                "--warmup",
                0,
                "--repetitions",
                2,
                "--output",
                destination,
            ]
        )
        report = json.loads((destination / "comparison.json").read_text())
        self.assertEqual(report["parameter_count"], 26704)
        self.assertIsNone(report["native"]["correctness_error"])
        self.assertEqual({r["attention"] for r in report["pytorch"]}, {"reference", "sdpa"})
        from compare_training import load_input

        _, entries, _ = load_input(destination / "input")
        checked = {n for n in entries if n.endswith((".gradient", ".final"))}
        for result in report["pytorch"]:
            parity = json.loads((destination / result["parity_report"]).read_text())
            self.assertTrue(parity["verified"])
            self.assertEqual(set(parity["max_absolute_errors"]), checked)
            self.assertLess(parity["vector_errors"]["gradient"]["relative_l2_error"], 0.001)
            self.assertLess(parity["vector_errors"]["update"]["relative_l2_error"], 0.001)
            self.assertGreater(parity["vector_errors"]["update"]["cosine_similarity"], 0.99)
            self.assertGreater(result["median_ms"], 0)
            self.assertEqual(result["repetitions"], 2)
            self.assertEqual(result["master_gradient_moment_dtype"], "fp32")
        binary = destination / "input/tensors.bin"
        with binary.open("r+b") as stream:
            value = stream.read(1)[0]
            stream.seek(0)
            stream.write(bytes([value ^ 1]))
        with self.assertRaisesRegex(ValueError, "checksum"):
            load_input(destination / "input")

    def test_preference_training_increases_margin(self):
        destination = self.path / "dpo"
        run(
            [
                BIN / "mini_mla_train",
                "--config",
                ROOT / "config/tiny-test.json",
                "--tokenizer",
                self.tokenizer_path,
                "--initialize",
                self.model,
                "--train",
                ROOT / "examples/preferences.jsonl",
                "--output",
                destination,
                "--mode",
                "dpo",
                "--backend",
                "cpu",
                "--sequence-length",
                48,
                "--microbatch",
                1,
                "--grad-accum",
                1,
                "--steps",
                16,
                "--warmup-steps",
                1,
                "--learning-rate",
                0.001,
                "--log-every",
                4,
                "--checkpoint-every",
                8,
            ]
        )
        metrics = [
            json.loads(line) for line in (destination / "metrics.jsonl").read_text().splitlines()
        ]
        self.assertAlmostEqual(metrics[0]["train_loss"], 0.69314718056, places=5)
        self.assertLess(metrics[-1]["train_loss"], metrics[0]["train_loss"] * 0.8)
        reference = (destination / "reference/weights.bin").read_bytes()
        weights = (destination / "checkpoint-latest/weights.bin").read_bytes()
        moments = (destination / "checkpoint-latest/optimizer.bin").read_bytes()
        shutil.rmtree(destination / "checkpoint-00000016")
        shutil.rmtree(destination / "model-16")
        run([BIN / "mini_mla_train", "--resume", destination / "checkpoint-00000008"])
        self.assertEqual((destination / "reference/weights.bin").read_bytes(), reference)
        self.assertEqual((destination / "checkpoint-latest/weights.bin").read_bytes(), weights)
        self.assertEqual((destination / "checkpoint-latest/optimizer.bin").read_bytes(), moments)

    def test_grpo_improves_fixed_seed_rewards_and_resumes_exactly(self):
        destination = self.path / "grpo"
        validation = self.path / "reward-validation.jsonl"
        validation.write_text(canonical({"prompt": "Say a!", "answer": "a"}))
        run(
            [
                BIN / "mini_mla_train",
                "--config",
                ROOT / "config/tiny-test.json",
                "--tokenizer",
                self.tokenizer_path,
                "--train",
                ROOT / "examples/rewards.jsonl",
                "--validation",
                validation,
                "--output",
                destination,
                "--mode",
                "grpo",
                "--backend",
                "cpu",
                "--sequence-length",
                48,
                "--microbatch",
                1,
                "--grad-accum",
                1,
                "--steps",
                80,
                "--warmup-steps",
                2,
                "--learning-rate",
                0.002,
                "--group-size",
                16,
                "--completion-tokens",
                16,
                "--gradient-epochs",
                2,
                "--kl-coefficient",
                0.05,
                "--reward",
                "contains",
                "--log-every",
                20,
                "--validate-every",
                20,
                "--validation-batches",
                1,
                "--checkpoint-every",
                40,
            ]
        )
        metrics = [
            json.loads(line) for line in (destination / "metrics.jsonl").read_text().splitlines()
        ]
        evaluated = [row for row in metrics if "val_reward" in row]
        self.assertGreater(evaluated[-1]["val_reward"], evaluated[0]["val_reward"])
        self.assertGreater(evaluated[-1]["val_kl"], 0)
        self.assertLess(evaluated[-1]["val_kl"], 0.5)
        weights = (destination / "checkpoint-latest/weights.bin").read_bytes()
        moments = (destination / "checkpoint-latest/optimizer.bin").read_bytes()
        reference = (destination / "reference/weights.bin").read_bytes()
        shutil.rmtree(destination / "checkpoint-00000080")
        shutil.rmtree(destination / "model-80")
        run([BIN / "mini_mla_train", "--resume", destination / "checkpoint-00000040"])
        self.assertEqual((destination / "checkpoint-latest/weights.bin").read_bytes(), weights)
        self.assertEqual((destination / "checkpoint-latest/optimizer.bin").read_bytes(), moments)
        self.assertEqual((destination / "reference/weights.bin").read_bytes(), reference)
        replayed = [
            json.loads(line) for line in (destination / "metrics.jsonl").read_text().splitlines()
        ]
        self.assertEqual(metrics[-1]["val_reward"], replayed[-1]["val_reward"])
        validation.write_text(canonical({"prompt": "Say b!", "answer": "b"}))
        output = subprocess.run(
            [str(BIN / "mini_mla_train"), "--resume", str(destination / "checkpoint-latest")],
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(output.returncode, 1)
        self.assertIn("Validation data changed", output.stderr)


class Server(unittest.TestCase):
    def test_http_streaming_validation_and_concurrency(self):
        with tempfile.TemporaryDirectory(prefix="mla-server-") as directory:
            path = Path(directory)
            run(
                [
                    sys.executable,
                    ROOT / "tools/prepare_dataset.py",
                    "--input",
                    ROOT / "examples/chat.jsonl",
                    "--tokenizer",
                    ROOT / "examples/tokenizer.json",
                    "--mode",
                    "sft",
                    "--output",
                    path / "data",
                    "--validation-fraction",
                    0,
                    "--sequence-length",
                    48,
                ]
            )
            run(
                [
                    BIN / "mini_mla_train",
                    "--config",
                    ROOT / "config/tiny-test.json",
                    "--tokenizer",
                    ROOT / "examples/tokenizer.json",
                    "--train",
                    path / "data/train-00000.sft.jsonl",
                    "--output",
                    path / "run",
                    "--mode",
                    "sft",
                    "--backend",
                    "cpu",
                    "--sequence-length",
                    48,
                    "--steps",
                    1,
                    "--warmup-steps",
                    0,
                    "--microbatch",
                    1,
                    "--grad-accum",
                    1,
                ]
            )
            with socket.socket() as reservation:
                try:
                    reservation.bind(("127.0.0.1", 0))
                except PermissionError:
                    self.skipTest("Loopback access is restricted in this test process")
                port = reservation.getsockname()[1]
            process = subprocess.Popen(
                [
                    str(BIN / "mini_mla_serve"),
                    "--model",
                    str(path / "run/model-latest"),
                    "--port",
                    str(port),
                    "--max-new-tokens",
                    "16",
                ],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            try:
                ready = False
                for _ in range(100):
                    if process.poll() is not None:
                        raise AssertionError(process.stderr.read().decode())
                    try:
                        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=1)
                        connection.request("GET", "/healthz")
                        response = connection.getresponse()
                        ready = response.status == 200
                        response.read()
                        connection.close()
                        if ready:
                            break
                    except OSError:
                        time.sleep(0.02)
                self.assertTrue(ready)

                def request(body, endpoint="/v1/completions"):
                    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
                    connection.request(
                        "POST", endpoint, json.dumps(body), {"Content-Type": "application/json"}
                    )
                    response = connection.getresponse()
                    status, payload = response.status, response.read().decode()
                    connection.close()
                    return status, payload

                body = {"prompt": "Hi", "max_tokens": 8, "temperature": 0, "seed": 42}
                status, payload = request(body)
                self.assertEqual(status, 200)
                completion = json.loads(payload)["choices"][0]["text"]
                status, stream = request(dict(body, stream=True))
                self.assertEqual(status, 200)
                self.assertIn("data: [DONE]", stream)
                events = [
                    json.loads(line[6:])
                    for line in stream.splitlines()
                    if line.startswith("data: ") and line != "data: [DONE]"
                ]
                self.assertEqual(
                    "".join(event["choices"][0]["text"] for event in events), completion
                )
                status, payload = request(
                    {"messages": [{"role": "user", "content": "Hi"}], "max_tokens": 2},
                    "/v1/chat/completions",
                )
                self.assertEqual(status, 200)
                self.assertEqual(json.loads(payload)["choices"][0]["message"]["role"], "assistant")
                for invalid in (
                    {"prompt": "Hi", "max_tokens": 1000},
                    {"prompt": "Hi", "temperature": -1},
                    {"prompt": "Hi", "top_p": 0},
                    {"prompt": "Hi", "n": 2},
                ):
                    self.assertEqual(request(invalid)[0], 400)
                with concurrent.futures.ThreadPoolExecutor(max_workers=6) as executor:
                    responses = list(executor.map(lambda _: request(body), range(6)))
                self.assertTrue(all(status == 200 for status, _ in responses))
                ids = [json.loads(payload)["id"] for _, payload in responses]
                self.assertEqual(len(ids), len(set(ids)))
                with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
                    client.sendall(
                        b"POST /v1/completions HTTP/1.1\r\nContent-Length: 0\r\nContent-Length: 0\r\n\r\n"
                    )
                    self.assertIn(b"400 Bad Request", client.recv(4096))
                with socket.create_connection(("127.0.0.1", port), timeout=5) as client:
                    payload = b'{"prompt":"Hi","stream":true,"max_tokens":16}'
                    header = f"POST /v1/completions HTTP/1.1\r\nContent-Length: {len(payload)}\r\n\r\n".encode()
                    client.sendall(header + payload)
                    self.assertIn(b"200 OK", client.recv(4096))
                connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
                connection.request("GET", "/metrics")
                response = connection.getresponse()
                metrics = response.read().decode()
                self.assertIn("mini_mla_completed_total", metrics)
                self.assertIn("mini_mla_first_token_seconds_sum", metrics)
                connection.close()
            finally:
                process.send_signal(signal.SIGTERM)
                try:
                    process.communicate(timeout=15)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.communicate()


if __name__ == "__main__":
    unittest.main(verbosity=2)
