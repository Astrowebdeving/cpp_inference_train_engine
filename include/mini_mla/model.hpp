#pragma once
#include "mini_mla/model_config.hpp"
#include "mini_mla/ops.hpp"

#include <memory>

namespace mini_mla {
enum class ParameterRole { hidden_matrix, token_embedding, norm_scale };
[[nodiscard]] std::string_view role_name(ParameterRole) noexcept;
struct Parameter {
    std::string name;
    ParameterRole role;
    Matrix value, gradient;
};
enum LayerParameter : std::size_t {
    attn_norm,
    w_dq,
    q_norm,
    w_uq,
    w_dkv,
    kv_norm,
    w_uk,
    w_uv,
    w_o,
    ffn_norm,
    w_gate,
    w_up,
    w_down,
    layer_parameter_count
};
struct LayerTape {
    Matrix h, raw_cq, cq, q, kv_pack, raw_ckv, ckv, k_rope, k_nope, k, v;
    Matrix probabilities, attention_output, residual, h2, gate, up, gated, output;
};
class Model {
  public:
    [[nodiscard]] static Result<std::unique_ptr<Model>> create(ModelConfig, std::uint64_t seed = 42,
                                                               bool trainable = true,
                                                               bool initialize_weights = true);
    [[nodiscard]] const ModelConfig& config() const noexcept { return config_; }
    [[nodiscard]] std::vector<Parameter>& parameters() noexcept { return parameters_; }
    [[nodiscard]] const std::vector<Parameter>& parameters() const noexcept { return parameters_; }
    [[nodiscard]] Parameter& layer(int l, LayerParameter p) {
        return parameters_.at(2 + static_cast<std::size_t>(l) * layer_parameter_count + p);
    }
    [[nodiscard]] const Parameter& layer(int l, LayerParameter p) const {
        return parameters_.at(2 + static_cast<std::size_t>(l) * layer_parameter_count + p);
    }
    [[nodiscard]] const std::vector<LayerTape>& tape() const noexcept { return tape_; }
    [[nodiscard]] Result<const Matrix*> forward(std::span<const std::uint16_t> tokens, int batch,
                                                int sequence);
    [[nodiscard]] Status backward(const Matrix& logits_gradient);
    void zero_grad();
    [[nodiscard]] bool gradients_finite() const;
    [[nodiscard]] std::uint64_t revision() const noexcept { return revision_; }
    void mark_weights_changed() noexcept {
        ++revision_;
        forward_ready_ = false;
    }

  private:
    explicit Model(ModelConfig c) : config_(c) {}
    struct BackwardWorkspace {
        Matrix dy, residual_grad, gated_grad, gate_grad, up_grad, h2_grad, tmp, tmp2;
        Matrix o_grad, q_grad, k_grad, v_grad, scratch, k_nope_grad, rope_grad;
        Matrix ckv_grad, kv_raw_grad, kv_pack_grad, h_grad, cq_grad, raw_cq_grad, dx;
    } work_;
    ModelConfig config_;
    std::vector<Parameter> parameters_;
    std::vector<LayerTape> tape_;
    Matrix embedding_output_, final_hidden_, logits_;
    std::vector<std::uint16_t> tokens_;
    int batch_ = 0, sequence_ = 0;
    bool forward_ready_ = false;
    std::uint64_t revision_ = 0;
};
} // namespace mini_mla
