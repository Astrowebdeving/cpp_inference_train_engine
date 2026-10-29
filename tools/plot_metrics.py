import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(
        description="Plot loss, throughput, and post-training metrics from a run."
    )
    parser.add_argument("run", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = args.run / "metrics.jsonl" if args.run.is_dir() else args.run
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if not rows:
        raise ValueError("No metrics to plot")
    figure, axes = plt.subplots(2, 2, figsize=(10, 6), constrained_layout=True)
    for axis, keys, label in zip(
        axes.flat,
        [
            ("train_loss", "val_loss"),
            ("tokens_per_second",),
            ("grad_norm",),
            ("reward", "val_reward", "kl"),
        ],
        ["Loss", "Tokens / second", "Gradient norm", "Reward / KL"],
    ):
        for key in keys:
            points = [(row["step"], row[key]) for row in rows if key in row]
            if points:
                axis.plot(*zip(*points), label=key, linewidth=1.4)
        axis.set_xlabel("Optimizer step")
        axis.set_ylabel(label)
        axis.grid(alpha=0.2)
        if axis.lines:
            axis.legend(frameon=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    print(args.output)


if __name__ == "__main__":
    main()
