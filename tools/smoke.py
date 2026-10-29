import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(
        description="Train, resume, export, and generate with a small working model."
    )
    parser.add_argument("--backend", choices=["cpu", "mps"], default="cpu")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--optimizer", choices=["adamw", "muon"], default="adamw")
    parser.add_argument("--attention", choices=["auto", "reference", "flash"], default="auto")
    parser.add_argument("--submission-window", type=int, choices=range(1, 17), default=4)
    parser.add_argument("--build", type=Path, default=root / "build")
    parser.add_argument("--output", type=Path, default=root / "artifacts/demo")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        parser.error(f"Output already exists: {output}; choose another --output")
    binaries = args.build.resolve() / "bin"
    if not (binaries / "mini_mla_train").is_file():
        parser.error("Build the CMake project before running this script")

    def run(command, capture=False):
        return subprocess.run(
            list(map(str, command)),
            cwd=root,
            check=True,
            text=True,
            stdout=subprocess.PIPE if capture else None,
        ).stdout

    run(
        [
            sys.executable,
            root / "tools/prepare_dataset.py",
            "--input",
            root / "examples/chat.jsonl",
            "--tokenizer",
            root / "examples/tokenizer.json",
            "--output",
            output / "data",
            "--mode",
            "sft",
            "--validation-fraction",
            0,
            "--sequence-length",
            48,
        ]
    )
    run(
        [
            binaries / "mini_mla_train",
            "--config",
            root / "config/tiny-test.json",
            "--tokenizer",
            root / "examples/tokenizer.json",
            "--train",
            output / "data/train-00000.sft.jsonl",
            "--output",
            output / "run",
            "--mode",
            "sft",
            "--backend",
            args.backend,
            "--precision",
            args.precision,
            "--optimizer",
            args.optimizer,
            "--attention",
            args.attention,
            "--submission-window",
            args.submission_window,
            "--sequence-length",
            48,
            "--microbatch",
            2,
            "--grad-accum",
            2,
            "--steps",
            80,
            "--warmup-steps",
            2,
            "--learning-rate",
            0.006,
            "--log-every",
            20,
            "--checkpoint-every",
            40,
        ]
    )
    run([binaries / "mini_mla_train", "--resume", output / "run/checkpoint-latest"])
    raw = run(
        [
            binaries / "mini_mla_generate",
            "--model",
            output / "run/model-latest",
            "--chat",
            "--prompt",
            "Say hi.",
            "--max-new-tokens",
            12,
            "--temperature",
            0,
            "--backend",
            args.backend,
            "--precision",
            args.precision,
            "--attention",
            args.attention,
            "--json",
        ],
        capture=True,
    )
    generation = json.loads(raw)
    if generation["text"] != "Hi!":
        raise AssertionError(f"Demo did not learn its response: {generation['text']!r}")
    print(f"Model: {output / 'run/model-latest'}\nSay hi. -> {generation['text']}")


if __name__ == "__main__":
    main()
