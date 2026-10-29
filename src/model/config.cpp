#include "mini_mla/io.hpp"
#include "mini_mla/model_config.hpp"

#include <cmath>

namespace mini_mla {
std::size_t ModelConfig::parameter_count() const {
    const auto d = static_cast<std::size_t>(d_model);
    const auto h = static_cast<std::size_t>(n_heads);
    const auto q = static_cast<std::size_t>(q_lora_rank);
    const auto kv = static_cast<std::size_t>(kv_lora_rank);
    return static_cast<std::size_t>(vocab_size) * d + d +
           static_cast<std::size_t>(n_layers) *
               (d * q + q * h * static_cast<std::size_t>(qk_dim()) +
                d * static_cast<std::size_t>(kv_width()) +
                kv * h * static_cast<std::size_t>(head_dim_nope + head_dim_value) +
                h * static_cast<std::size_t>(head_dim_value) * d +
                3 * d * static_cast<std::size_t>(d_ff) + 2 * d + q + kv);
}
Status ModelConfig::validate() const {
    for (auto [name, value] : {std::pair{"vocab_size", vocab_size},
                               {"max_sequence_length", max_sequence_length},
                               {"n_layers", n_layers},
                               {"d_model", d_model},
                               {"n_heads", n_heads},
                               {"head_dim_nope", head_dim_nope},
                               {"head_dim_rope", head_dim_rope},
                               {"head_dim_value", head_dim_value},
                               {"q_lora_rank", q_lora_rank},
                               {"kv_lora_rank", kv_lora_rank},
                               {"d_ff", d_ff}})
        if (value < 1 || value > 65535)
            return Status::failure(ErrorCode::invalid_argument,
                                   std::string(name) + " must be in [1, 65535]");
    if (vocab_size > 65535 || n_layers > 128 || n_heads > 256 || d_model % n_heads ||
        head_dim_rope % 2)
        return Status::failure(
            ErrorCode::invalid_argument,
            "Invalid vocabulary/layer/head count, d_model divisibility, or odd RoPE dimension");
    if (!std::isfinite(norm_epsilon) || norm_epsilon <= 0 || !std::isfinite(rope_theta) ||
        rope_theta <= 1)
        return Status::failure(ErrorCode::invalid_argument,
                               "Invalid normalization epsilon or RoPE theta");
    if (parameter_count() > 1000000000ULL)
        return Status::failure(ErrorCode::invalid_argument,
                               "Configuration exceeds one billion parameters");
    return ok();
}
Json ModelConfig::json() const {
    return Json::Object{{"model_type", "mla_dense_decoder"},
                        {"version", 1},
                        {"vocab_size", vocab_size},
                        {"max_sequence_length", max_sequence_length},
                        {"n_layers", n_layers},
                        {"d_model", d_model},
                        {"n_heads", n_heads},
                        {"head_dim_nope", head_dim_nope},
                        {"head_dim_rope", head_dim_rope},
                        {"head_dim_value", head_dim_value},
                        {"q_lora_rank", q_lora_rank},
                        {"kv_lora_rank", kv_lora_rank},
                        {"d_ff", d_ff},
                        {"norm_epsilon", static_cast<double>(norm_epsilon)},
                        {"rope_theta", static_cast<double>(rope_theta)},
                        {"norm", "rmsnorm"},
                        {"activation", "swiglu"},
                        {"mla_latent_norms", true},
                        {"bias", false},
                        {"tie_word_embeddings", true}};
}
std::string ModelConfig::hash() const {
    return hash_text(json().dump());
}
Result<ModelConfig> ModelConfig::from_json(const Json& j) {
    try {
        ModelConfig c;
        c.vocab_size = j.at("vocab_size").integer();
        c.max_sequence_length = j.at("max_sequence_length").integer();
        c.n_layers = j.at("n_layers").integer();
        c.d_model = j.at("d_model").integer();
        c.n_heads = j.at("n_heads").integer();
        c.head_dim_nope = j.at("head_dim_nope").integer();
        c.head_dim_rope = j.at("head_dim_rope").integer();
        c.head_dim_value = j.at("head_dim_value").integer();
        c.q_lora_rank = j.at("q_lora_rank").integer();
        c.kv_lora_rank = j.at("kv_lora_rank").integer();
        c.d_ff = j.at("d_ff").integer();
        c.norm_epsilon = static_cast<float>(j.at("norm_epsilon").number());
        c.rope_theta = static_cast<float>(j.at("rope_theta").number());
        if (j.str_or("model_type", "mla_dense_decoder") != "mla_dense_decoder" ||
            j.num_or("version", 1) != 1 || j.str_or("norm", "rmsnorm") != "rmsnorm" ||
            j.str_or("activation", "swiglu") != "swiglu" || j.bool_or("bias", false) ||
            !j.bool_or("tie_word_embeddings", true) || !j.bool_or("mla_latent_norms", true))
            return Result<ModelConfig>::failure(ErrorCode::unsupported,
                                                "Only v1 tied-embedding, bias-free MLA with latent "
                                                "RMSNorm and SwiGLU is supported");
        auto valid = c.validate();
        if (!valid)
            return Result<ModelConfig>::failure(valid.error());
        if (j.contains("expected_parameter_count") &&
            j.at("expected_parameter_count").size_integer() != c.parameter_count())
            return Result<ModelConfig>::failure(
                ErrorCode::invalid_argument,
                "Expected parameter count disagrees with architecture");
        return Result<ModelConfig>::success(c);
    } catch (const std::exception& e) {
        return Result<ModelConfig>::failure(ErrorCode::invalid_argument, e.what());
    }
}
Result<ModelConfig> ModelConfig::load(const std::filesystem::path& p) {
    auto j = read_json(p);
    if (!j)
        return Result<ModelConfig>::failure(j.error());
    return from_json(j.value());
}
} // namespace mini_mla
