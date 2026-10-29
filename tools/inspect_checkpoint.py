import argparse
import hashlib
import json
import math
import mmap
import zlib
from pathlib import Path

from common import Tokenizer, sha256_file


def cpp_json(value):
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        if not math.isfinite(value):
            raise ValueError("Nonfinite JSON number")
        return format(value, ".17g")
    if isinstance(value, str):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if isinstance(value, list):
        return "[" + ",".join(map(cpp_json, value)) + "]"
    return "{" + ",".join(cpp_json(k) + ":" + cpp_json(value[k]) for k in sorted(value)) + "}"


def inspect(path):
    path = Path(path)
    raw = (path / "manifest.json").read_bytes()
    if (path / "COMPLETE").read_text() != hashlib.sha256(raw).hexdigest() + "\n":
        raise ValueError("Incomplete checkpoint or damaged manifest")
    manifest = json.loads(raw)
    if (
        manifest["format"] not in ("mini-mla-checkpoint", "mini-mla-model")
        or manifest["version"] != 1
    ):
        raise ValueError("Unsupported format")
    config = json.loads((path / "config.json").read_bytes())
    if hashlib.sha256(cpp_json(config).encode()).hexdigest() != manifest["model_config_sha256"]:
        raise ValueError("Model configuration checksum failed")
    tokenizer = Tokenizer.load(path / "tokenizer.json")
    if (
        tokenizer.hash != manifest["tokenizer_sha256"]
        or tokenizer.vocab_size != config["vocab_size"]
    ):
        raise ValueError("Tokenizer checksum or vocabulary failed")
    allowed = {"weights.bin", "optimizer.bin", "trainer_state.json"}
    for name, digest in manifest["sha256"].items():
        if name not in allowed or sha256_file(path / name) != digest:
            raise ValueError(f"File checksum failed: {name}")
    ends, names, parameter_count = {}, set(), 0
    for entry in manifest["tensors"]:
        name, file = entry["name"], entry["file"]
        dimensions, dtype = entry["shape"], entry["dtype"]
        if (
            name in names
            or file not in {"weights.bin", "optimizer.bin"}
            or file not in manifest["sha256"]
        ):
            raise ValueError(f"Duplicate tensor or invalid file: {name}")
        if (
            len(dimensions) != 2
            or not all(isinstance(d, int) and d > 0 for d in dimensions)
            or dtype not in ("f16", "f32")
        ):
            raise ValueError(f"Invalid shape or dtype: {name}")
        if entry["offset"] != ends.get(file, 0) or entry["nbytes"] != math.prod(dimensions) * (
            2 if dtype == "f16" else 4
        ):
            raise ValueError(f"Invalid tensor extent: {name}")
        with (
            (path / file).open("rb") as stream,
            mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data,
        ):
            start, end = entry["offset"], entry["offset"] + entry["nbytes"]
            if end > len(data) or zlib.crc32(data[start:end]) != entry["crc32"]:
                raise ValueError(f"Tensor checksum failed: {name}")
        ends[file] = end
        names.add(name)
        if file == "weights.bin":
            parameter_count += math.prod(dimensions)
    if parameter_count != manifest["parameter_count"]:
        raise ValueError("Parameter count mismatch")
    for file, end in ends.items():
        if (path / file).stat().st_size != end:
            raise ValueError("Trailing unindexed tensor data")
    return manifest, config


def load_weights(path):
    import numpy as np

    path = Path(path)
    manifest, config = inspect(path)
    weights = {}
    for entry in manifest["tensors"]:
        if entry["file"] != "weights.bin":
            continue
        value = np.memmap(
            path / "weights.bin",
            dtype="<f2" if entry["dtype"] == "f16" else "<f4",
            mode="r",
            offset=entry["offset"],
            shape=tuple(entry["shape"]),
        )
        weights[entry["name"]] = np.asarray(value, dtype=np.float32).copy()
        if not np.isfinite(weights[entry["name"]]).all():
            raise ValueError(f"Nonfinite weights: {entry['name']}")
    return config, weights


def main():
    parser = argparse.ArgumentParser(
        description="Verify and inspect a native model or resumable checkpoint."
    )
    parser.add_argument("path", type=Path)
    parser.add_argument("--tensors", action="store_true")
    args = parser.parse_args()
    manifest, config = inspect(args.path)
    output = {
        key: manifest[key]
        for key in ("format", "version", "step", "dtype", "parameter_count", "tokenizer_sha256")
    }
    output["config"] = config
    if args.tensors:
        output["tensors"] = manifest["tensors"]
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
