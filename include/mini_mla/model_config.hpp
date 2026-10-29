#pragma once
#include "mini_mla/json.hpp"

namespace mini_mla {
struct ModelConfig {
    int vocab_size = 264, max_sequence_length = 64, n_layers = 2, d_model = 32, n_heads = 2;
    int head_dim_nope = 12, head_dim_rope = 4, head_dim_value = 16;
    int q_lora_rank = 16, kv_lora_rank = 8, d_ff = 64;
    float norm_epsilon = 1e-5F, rope_theta = 10000.0F;
    [[nodiscard]] int qk_dim() const { return head_dim_nope + head_dim_rope; }
    [[nodiscard]] int kv_width() const { return kv_lora_rank + head_dim_rope; }
    [[nodiscard]] int attn_width() const { return n_heads * head_dim_value; }
    [[nodiscard]] std::size_t parameter_count() const;
    [[nodiscard]] Status validate() const;
    [[nodiscard]] Json json() const;
    [[nodiscard]] std::string hash() const;
    [[nodiscard]] static Result<ModelConfig> from_json(const Json&);
    [[nodiscard]] static Result<ModelConfig> load(const std::filesystem::path&);
};
} // namespace mini_mla
