import argparse
from pathlib import Path

import numpy as np
from inspect_checkpoint import load_weights


def main():
    parser = argparse.ArgumentParser(
        description="Export native weights or a gradient debug bundle into an NPZ archive."
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if (args.input / "debug.json").exists():
        import json

        info = json.loads((args.input / "debug.json").read_bytes())
        tensors = {
            entry["name"]: np.memmap(
                args.input / "tensors.bin",
                dtype="<f4",
                mode="r",
                offset=entry["offset"],
                shape=tuple(entry["shape"]),
            ).copy()
            for entry in info["tensors"]
        }
    else:
        _, tensors = load_weights(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.output, **tensors)
    print(f"{len(tensors)} tensors -> {args.output}")


if __name__ == "__main__":
    main()
