import argparse
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO = "mlx-community/gemma-4-E2B-it-4bit"
REVISION = "238767527555cb75a05732a84dff5d6ba0dd6809"


def run(command, env):
    print(" ".join(map(str, command)), flush=True)
    subprocess.run(list(map(str, command)), env=env, check=True)


def main():
    parser = argparse.ArgumentParser(description="Set up the optional Gemma C++ / Metal path")
    parser.add_argument("--python", default="python3.12")
    parser.add_argument("--environment", type=Path, default=ROOT / ".venv-serving")
    parser.add_argument("--build", type=Path, default=ROOT / "build")
    parser.add_argument("--with-vllm", action="store_true")
    parser.add_argument("--download-gemma", action="store_true")
    parser.add_argument("--skip-install", action="store_true")
    args = parser.parse_args()
    if sys.platform != "darwin" or platform.machine() != "arm64":
        parser.error("This setup requires Apple silicon and macOS")
    uv = shutil.which("uv")
    if not uv:
        parser.error("Install uv, then rerun this command")
    env = os.environ.copy()
    cache = ROOT / "artifacts/setup-cache"
    env["UV_CACHE_DIR"] = str(cache / "uv")
    env["HF_HOME"] = str(cache / "huggingface")
    python = args.environment.resolve() / "bin/python"
    if not python.exists():
        run([uv, "venv", args.environment.resolve(), "--python", args.python], env)
    version = subprocess.check_output(
        [str(python), "-c", "import sys; print('.'.join(map(str, sys.version_info[:2])))"],
        text=True,
        env=env,
    ).strip()
    if version != "3.12":
        parser.error("The pinned vLLM wheels and verified setup use Python 3.12")
    if not args.skip_install:
        requirements = "requirements-metal.txt" if args.with_vllm else "requirements-gemma.txt"
        run([uv, "pip", "install", "--python", python, "-r", ROOT / "tools" / requirements], env)
    if args.download_gemma:
        script = """
import hashlib,json,sys
from pathlib import Path
from huggingface_hub import snapshot_download
repo,revision,directory=sys.argv[1:]
p=Path(snapshot_download(repo,revision=revision,local_dir=directory,allow_patterns=[
 'config.json','generation_config.json','tokenizer.json','tokenizer_config.json',
 'chat_template.jinja','model.safetensors','model.safetensors.index.json',
 'README.md','processor_config.json']))
h=hashlib.sha256()
with (p/'model.safetensors').open('rb') as f:
 while block:=f.read(1024*1024): h.update(block)
(p/'mini_mla_source.json').write_text(json.dumps({
 'repository':repo,'revision':revision,'model_weights_sha256':h.hexdigest()
},indent=2)+'\\n')
print(p)
"""
        run([python, "-c", script, REPO, REVISION, ROOT / "artifacts/gemma-4-E2B-it-4bit"], env)
    location = subprocess.check_output(
        [str(python), "-c", "import mlx; print(next(iter(mlx.__path__)))"],
        env=env,
        text=True,
    ).strip()
    run(
        [
            "cmake",
            "-S",
            ROOT,
            "-B",
            args.build.resolve(),
            "-DCMAKE_BUILD_TYPE=Release",
            "-DMINI_MLA_ENABLE_GEMMA=ON",
            f"-DMLX_DIR={location}/share/cmake/MLX",
        ],
        env,
    )
    run(["cmake", "--build", args.build.resolve(), "-j", "8"], env)
    print(f"Ready: {args.build.resolve() / 'bin/mini_mla_gemma'}")


if __name__ == "__main__":
    main()
