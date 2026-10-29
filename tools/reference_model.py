import argparse
import json
from pathlib import Path

import numpy as np
import torch
from inspect_checkpoint import load_weights


class ReferenceModel(torch.nn.Module):
    def __init__(self, config, weights):
        super().__init__()
        self.config = config
        self.weights = torch.nn.ParameterDict(
            {
                k.replace(".", "__"): torch.nn.Parameter(torch.from_numpy(v.copy()))
                for k, v in weights.items()
            }
        )

    def weight(self, name):
        return self.weights[name.replace(".", "__")]

    def norm(self, x, weight):
        return (
            x
            * torch.rsqrt(x.square().mean(dim=-1, keepdim=True) + self.config["norm_epsilon"])
            * weight
        )

    def rope(self, x):
        width = x.shape[-1]
        frequency = self.config["rope_theta"] ** (-torch.arange(0, width, 2, dtype=x.dtype) / width)
        angle = torch.arange(x.shape[1], dtype=x.dtype)[:, None] * frequency[None, :]
        cosine, sine = angle.cos()[None, :, None, :], angle.sin()[None, :, None, :]
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack(
            (even * cosine - odd * sine, even * sine + odd * cosine), dim=-1
        ).flatten(-2)

    def forward(self, ids):
        c = self.config
        batch, time = ids.shape
        heads, nope, rope, dv = (
            c["n_heads"],
            c["head_dim_nope"],
            c["head_dim_rope"],
            c["head_dim_value"],
        )
        embedding = self.weight("token_embedding")
        x = embedding[ids]
        causal = torch.ones(time, time, dtype=torch.bool).triu(1)
        for layer in range(c["n_layers"]):

            def w(name):
                return self.weight(f"layers.{layer}.{name}")

            h = self.norm(x, w("attn.norm"))
            cq = self.norm(h @ w("attn.w_dq"), w("attn.q_norm"))
            q = (cq @ w("attn.w_uq")).view(batch, time, heads, nope + rope)
            q = torch.cat((q[..., :nope], self.rope(q[..., nope:])), dim=-1)
            packed = h @ w("attn.w_dkv")
            latent = self.norm(packed[..., : c["kv_lora_rank"]], w("attn.kv_norm"))
            kr = self.rope(packed[..., c["kv_lora_rank"] :].view(batch, time, 1, rope))
            kn = (latent @ w("attn.w_uk")).view(batch, time, heads, nope)
            k = torch.cat((kn, kr.expand(batch, time, heads, rope)), dim=-1)
            v = (latent @ w("attn.w_uv")).view(batch, time, heads, dv)
            scores = (
                q.transpose(1, 2) @ k.transpose(1, 2).transpose(-1, -2) / ((nope + rope) ** 0.5)
            )
            probabilities = scores.masked_fill(causal, float("-inf")).softmax(dim=-1)
            output = (
                (probabilities @ v.transpose(1, 2)).transpose(1, 2).reshape(batch, time, heads * dv)
            )
            x = x + output @ w("attn.w_o")
            h = self.norm(x, w("ffn.norm"))
            gate = h @ w("ffn.w_gate")
            x = x + (torch.nn.functional.silu(gate) * (h @ w("ffn.w_up"))) @ w("ffn.w_down")
        return self.norm(x, self.weight("final_norm")) @ embedding.t()


def debug_tensors(path):
    path = Path(path)
    info = json.loads((path / "debug.json").read_bytes())
    values = {
        entry["name"]: np.memmap(
            path / "tensors.bin",
            dtype="<f4",
            mode="r",
            offset=entry["offset"],
            shape=tuple(entry["shape"]),
        ).copy()
        for entry in info["tensors"]
    }
    return info, values


def evaluate(config, weights, ids, targets, mask):
    torch.set_num_threads(1)
    model = ReferenceModel(config, weights)
    logits = model(torch.tensor([ids], dtype=torch.long)).reshape(len(ids), config["vocab_size"])
    coefficients = torch.tensor(mask, dtype=torch.float32)
    coefficients = coefficients / coefficients.sum()
    nll = torch.nn.functional.cross_entropy(logits, torch.tensor(targets), reduction="none")
    loss = (nll * coefficients).sum()
    loss.backward()
    gradients = {name: model.weight(name).grad.detach().numpy().copy() for name in weights}
    return model, logits.detach().numpy(), float(loss.detach()), gradients


def compare(path, atol=0.0001, rtol=0.002):
    info, tensors = debug_tensors(path)
    weights = {
        entry["name"][:-7]: tensors[entry["name"]]
        for entry in info["tensors"]
        if entry["name"].endswith(".weight")
    }
    ids = list(map(int, info["input"].split(",")))
    targets = list(map(int, info["targets"].split(",")))
    mask = list(map(int, info["mask"].split(","))) if info["mask"] else [1] * len(ids)
    model, logits, loss, gradients = evaluate(info["config"], weights, ids, targets, mask)
    errors = {
        "logits": float(np.max(np.abs(logits - tensors["logits"]))),
        "loss": abs(loss - info["loss"]),
    }
    np.testing.assert_allclose(logits, tensors["logits"], atol=atol, rtol=rtol)
    if abs(loss - info["loss"]) > atol:
        raise AssertionError(f"Loss mismatch: {loss} versus {info['loss']}")
    for name, gradient in gradients.items():
        expected = tensors[name + ".gradient"]
        errors[name] = float(np.max(np.abs(gradient - expected)))
        np.testing.assert_allclose(gradient, expected, atol=atol, rtol=rtol, err_msg=name)
    if any(name.endswith(".updated") for name in tensors):
        o = info["optimizer"]
        if o["kind"] != "adamw":
            raise ValueError("Update comparison currently expects AdamW")
        decay, no_decay = [], []
        for name in weights:
            (no_decay if ("norm" in name) else decay).append(model.weight(name))
        optimizer = torch.optim.AdamW(
            [
                {"params": decay, "weight_decay": o["weight_decay"]},
                {"params": no_decay, "weight_decay": 0},
            ],
            lr=info["learning_rate"],
            betas=(o["beta1"], o["beta2"]),
            eps=o["epsilon"],
            foreach=False,
        )
        if o["clip_norm"] > 0:
            norm = sum(p.grad.double().square().sum() for p in model.parameters()).sqrt()
            scale = min(1.0, o["clip_norm"] / max(float(norm), 1e-30))
            for p in model.parameters():
                p.grad.mul_(scale)
        optimizer.step()
        for name in weights:
            actual = model.weight(name).detach().numpy()
            expected = tensors[name + ".updated"]
            errors[name + ".updated"] = float(np.max(np.abs(actual - expected)))
            np.testing.assert_allclose(
                actual, expected, atol=atol, rtol=rtol, err_msg=name + ".updated"
            )
    return {"reference_loss": loss, "max_absolute_errors": errors, "passed": True}


def main():
    parser = argparse.ArgumentParser(
        description="Independent PyTorch oracle for native MLA forward, backward, and AdamW."
    )
    parser.add_argument("--model", type=Path)
    parser.add_argument("--tokens", default="1,2,3")
    parser.add_argument("--targets", default="2,3,4")
    parser.add_argument("--mask", default="")
    parser.add_argument("--compare", type=Path)
    parser.add_argument("--atol", type=float, default=0.0001)
    parser.add_argument("--rtol", type=float, default=0.002)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.compare:
        report = compare(args.compare, args.atol, args.rtol)
    else:
        if args.model is None:
            parser.error("--model or --compare is required")
        config, weights = load_weights(args.model)
        ids, targets = (
            list(map(int, args.tokens.split(","))),
            list(map(int, args.targets.split(","))),
        )
        mask = list(map(int, args.mask.split(","))) if args.mask else [1] * len(ids)
        _, logits, loss, gradients = evaluate(config, weights, ids, targets, mask)
        report = {
            "loss": loss,
            "logits": logits.tolist(),
            "gradient_norms": {k: float(np.linalg.norm(v)) for k, v in gradients.items()},
        }
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
