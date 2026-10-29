import torch
from reference_model import ReferenceModel


class BenchmarkModel(ReferenceModel):
    def __init__(self, config, weights, precision, attention):
        super().__init__(config, weights)
        self.compute_dtype = torch.float16 if precision == "fp16" else torch.float32
        self.attention = attention
        width = config["head_dim_rope"]
        frequency = config["rope_theta"] ** (-torch.arange(0, width, 2).float() / width)
        angles = torch.arange(config["max_sequence_length"]).float()[:, None] * frequency
        self.register_buffer("cosine", angles.cos().to(self.compute_dtype)[None, :, None, :])
        self.register_buffer("sine", angles.sin().to(self.compute_dtype)[None, :, None, :])
        length = config["max_sequence_length"]
        self.register_buffer("causal", torch.ones(length, length, dtype=torch.bool).triu(1))

    def weight(self, name):
        return super().weight(name).to(self.compute_dtype)

    def master(self, name):
        return self.weights[name.replace(".", "__")]

    def norm(self, x, weight):
        f = x.float()
        inverse = torch.rsqrt(
            f.square().sum(-1, keepdim=True) / f.shape[-1] + self.config["norm_epsilon"]
        )
        return (f * inverse * weight.float()).to(self.compute_dtype)

    def rope(self, x):
        cosine, sine = self.cosine[:, : x.shape[1]], self.sine[:, : x.shape[1]]
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
        x = torch.nn.functional.embedding(ids, embedding)
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
            q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
            if self.attention == "sdpa":
                output = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, is_causal=True, dropout_p=0.0
                )
            else:
                scores = (q.float() @ k.float().transpose(-1, -2)) * ((nope + rope) ** -0.5)
                probabilities = (
                    scores.masked_fill(self.causal[:time, :time], -1e30)
                    .softmax(-1)
                    .to(self.compute_dtype)
                )
                output = probabilities @ v
            output = output.transpose(1, 2).reshape(batch, time, heads * dv)
            x = x + output @ w("attn.w_o")
            h = self.norm(x, w("ffn.norm"))
            gate = h @ w("ffn.w_gate")
            x = x + ((gate * gate.sigmoid()) * (h @ w("ffn.w_up"))) @ w("ffn.w_down")
        return (self.norm(x, self.weight("final_norm")) @ embedding.t()).float()
