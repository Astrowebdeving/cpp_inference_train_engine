#include "mini_mla/model.hpp"

#include "mini_mla/rng.hpp"

#include <cmath>
#include <new>

namespace mini_mla {
std::string_view role_name(ParameterRole role) noexcept {
    switch (role) {
    case ParameterRole::hidden_matrix:
        return "hidden_matrix";
    case ParameterRole::token_embedding:
        return "token_embedding";
    case ParameterRole::norm_scale:
        return "norm_scale";
    }
    return "invalid";
}
Result<std::unique_ptr<Model>> Model::create(ModelConfig c, std::uint64_t seed, bool trainable,
                                             bool initialize_weights) {
    auto valid = c.validate();
    if (!valid)
        return Result<std::unique_ptr<Model>>::failure(valid.error());
    try {
        auto m = std::unique_ptr<Model>(new Model(c));
        m->parameters_.reserve(2 + static_cast<std::size_t>(c.n_layers) * layer_parameter_count);
        Rng rng(seed);
        auto add = [&](std::string name, ParameterRole role, int r, int cols,
                       bool residual = false) {
            Parameter p{std::move(name), role, Matrix(r, cols), Matrix()};
            if (trainable)
                p.gradient.resize(r, cols);
            float scale = residual ? 0.02F / std::sqrt(static_cast<float>(2 * c.n_layers)) : 0.02F;
            if (initialize_weights)
                for (auto& x : p.value.values)
                    x = role == ParameterRole::norm_scale ? 1.0F : rng.normal() * scale;
            m->parameters_.push_back(std::move(p));
        };
        add("token_embedding", ParameterRole::token_embedding, c.vocab_size, c.d_model);
        add("final_norm", ParameterRole::norm_scale, 1, c.d_model);
        for (int l = 0; l < c.n_layers; ++l) {
            auto prefix = "layers." + std::to_string(l) + ".";
            auto norm = [&](std::string name, int width) {
                add(prefix + name, ParameterRole::norm_scale, 1, width);
            };
            auto matrix = [&](std::string name, int r, int cols, bool residual = false) {
                add(prefix + name, ParameterRole::hidden_matrix, r, cols, residual);
            };
            norm("attn.norm", c.d_model);
            matrix("attn.w_dq", c.d_model, c.q_lora_rank);
            norm("attn.q_norm", c.q_lora_rank);
            matrix("attn.w_uq", c.q_lora_rank, c.n_heads * c.qk_dim());
            matrix("attn.w_dkv", c.d_model, c.kv_width());
            norm("attn.kv_norm", c.kv_lora_rank);
            matrix("attn.w_uk", c.kv_lora_rank, c.n_heads * c.head_dim_nope);
            matrix("attn.w_uv", c.kv_lora_rank, c.attn_width());
            matrix("attn.w_o", c.attn_width(), c.d_model, true);
            norm("ffn.norm", c.d_model);
            matrix("ffn.w_gate", c.d_model, c.d_ff);
            matrix("ffn.w_up", c.d_model, c.d_ff);
            matrix("ffn.w_down", c.d_ff, c.d_model, true);
        }
        std::size_t count = 0;
        for (const auto& p : m->parameters_)
            count += p.value.size();
        if (count != c.parameter_count())
            detail::fatal("Parameter registry disagrees with model configuration");
        m->tape_.resize(static_cast<std::size_t>(c.n_layers));
        return Result<std::unique_ptr<Model>>::success(std::move(m));
    } catch (const std::bad_alloc&) {
        return Result<std::unique_ptr<Model>>::failure(ErrorCode::out_of_memory,
                                                       "Model allocation failed");
    }
}
Result<const Matrix*> Model::forward(std::span<const std::uint16_t> tokens, int batch,
                                     int sequence) {
    forward_ready_ = false;
    const auto& c = config_;
    if (batch < 1 || sequence < 1 || sequence > c.max_sequence_length ||
        batch > INT_MAX / sequence ||
        tokens.size() != static_cast<std::size_t>(batch) * static_cast<std::size_t>(sequence))
        return Result<const Matrix*>::failure(ErrorCode::invalid_argument,
                                              "Input batch or sequence shape is invalid");
    for (auto id : tokens)
        if (id >= c.vocab_size)
            return Result<const Matrix*>::failure(ErrorCode::invalid_argument,
                                                  "Input token exceeds vocabulary");
    try {
        batch_ = batch;
        sequence_ = sequence;
        tokens_.assign(tokens.begin(), tokens.end());
        int n = batch * sequence;
        embedding_output_.resize(n, c.d_model);
        const auto& embedding = parameters_[0].value;
        for (int i = 0; i < n; ++i)
            for (int j = 0; j < c.d_model; ++j)
                embedding_output_(i, j) = embedding(tokens[static_cast<std::size_t>(i)], j);
        for (int l = 0; l < c.n_layers; ++l) {
            auto& t = tape_[static_cast<std::size_t>(l)];
            const auto& x =
                l == 0 ? embedding_output_ : tape_[static_cast<std::size_t>(l - 1)].output;
            auto w = [&](LayerParameter p) -> const Matrix& { return layer(l, p).value; };
            ops::rmsnorm(x, w(attn_norm), t.h, c.norm_epsilon);
            ops::matmul(t.h, w(w_dq), t.raw_cq);
            ops::rmsnorm(t.raw_cq, w(q_norm), t.cq, c.norm_epsilon);
            ops::matmul(t.cq, w(w_uq), t.q);
            ops::rope(t.q, batch, sequence, c.n_heads, c.qk_dim(), c.head_dim_nope, c.head_dim_rope,
                      c.rope_theta);
            ops::matmul(t.h, w(w_dkv), t.kv_pack);
            t.raw_ckv.resize(n, c.kv_lora_rank);
            t.k_rope.resize(n, c.head_dim_rope);
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < c.kv_lora_rank; ++j)
                    t.raw_ckv(i, j) = t.kv_pack(i, j);
                for (int j = 0; j < c.head_dim_rope; ++j)
                    t.k_rope(i, j) = t.kv_pack(i, c.kv_lora_rank + j);
            }
            ops::rmsnorm(t.raw_ckv, w(kv_norm), t.ckv, c.norm_epsilon);
            ops::rope(t.k_rope, batch, sequence, 1, c.head_dim_rope, 0, c.head_dim_rope,
                      c.rope_theta);
            ops::matmul(t.ckv, w(w_uk), t.k_nope);
            ops::matmul(t.ckv, w(w_uv), t.v);
            t.k.resize(n, c.n_heads * c.qk_dim());
            for (int i = 0; i < n; ++i)
                for (int h = 0; h < c.n_heads; ++h) {
                    for (int j = 0; j < c.head_dim_nope; ++j)
                        t.k(i, h * c.qk_dim() + j) = t.k_nope(i, h * c.head_dim_nope + j);
                    for (int j = 0; j < c.head_dim_rope; ++j)
                        t.k(i, h * c.qk_dim() + c.head_dim_nope + j) = t.k_rope(i, j);
                }
            ops::attention(t.q, t.k, t.v, t.probabilities, t.attention_output, batch, sequence,
                           c.n_heads, c.qk_dim(), c.head_dim_value);
            ops::matmul(t.attention_output, w(w_o), t.residual);
            ops::add_inplace(t.residual, x);
            ops::rmsnorm(t.residual, w(ffn_norm), t.h2, c.norm_epsilon);
            ops::matmul(t.h2, w(w_gate), t.gate);
            ops::matmul(t.h2, w(w_up), t.up);
            ops::swiglu(t.gate, t.up, t.gated);
            ops::matmul(t.gated, w(w_down), t.output);
            ops::add_inplace(t.output, t.residual);
        }
        ops::rmsnorm(tape_.back().output, parameters_[1].value, final_hidden_, c.norm_epsilon);
        ops::matmul(final_hidden_, embedding, logits_, false, true);
        if (!ops::finite(logits_.values))
            return Result<const Matrix*>::failure(ErrorCode::numerical_error,
                                                  "Nonfinite model output");
        forward_ready_ = true;
        return Result<const Matrix*>::success(&logits_);
    } catch (const std::bad_alloc&) {
        return Result<const Matrix*>::failure(
            ErrorCode::out_of_memory,
            "Forward workspace allocation failed; reduce sequence length or batch");
    }
}
void Model::zero_grad() {
    for (auto& p : parameters_) {
        p.gradient.resize(p.value.rows, p.value.cols);
        p.gradient.zero();
    }
}
bool Model::gradients_finite() const {
    for (const auto& p : parameters_)
        if (!ops::finite(p.gradient.values))
            return false;
    return true;
}
Status Model::backward(const Matrix& g) {
    if (!forward_ready_ || g.rows != logits_.rows || g.cols != logits_.cols ||
        !ops::finite(g.values))
        return Status::failure(
            ErrorCode::invalid_argument,
            "Backward requires the latest forward pass and a finite matching gradient");
    for (const auto& p : parameters_)
        if (p.gradient.size() != p.value.size())
            return Status::failure(ErrorCode::invalid_argument,
                                   "Call zero_grad before backward on an inference-only model");
    const auto& c = config_;
    auto& s = work_;
    int n = batch_ * sequence_;
    ops::matmul(g, parameters_[0].value, s.tmp);
    ops::matmul(g, final_hidden_, parameters_[0].gradient, true, false, true);
    ops::rmsnorm_backward(tape_.back().output, parameters_[1].value, s.tmp, s.dy,
                          parameters_[1].gradient, c.norm_epsilon);
    for (int l = c.n_layers - 1; l >= 0; --l) {
        const auto& t = tape_[static_cast<std::size_t>(l)];
        const auto& x = l == 0 ? embedding_output_ : tape_[static_cast<std::size_t>(l - 1)].output;
        auto w = [&](LayerParameter p) -> const Matrix& { return layer(l, p).value; };
        auto dw = [&](LayerParameter p) -> Matrix& { return layer(l, p).gradient; };
        ops::linear_backward(t.gated, w(w_down), s.dy, s.gated_grad, dw(w_down));
        ops::swiglu_backward(t.gate, t.up, s.gated_grad, s.gate_grad, s.up_grad);
        ops::linear_backward(t.h2, w(w_gate), s.gate_grad, s.h2_grad, dw(w_gate));
        ops::linear_backward(t.h2, w(w_up), s.up_grad, s.tmp, dw(w_up));
        ops::add_inplace(s.h2_grad, s.tmp);
        ops::rmsnorm_backward(t.residual, w(ffn_norm), s.h2_grad, s.residual_grad, dw(ffn_norm),
                              c.norm_epsilon);
        ops::add_inplace(s.residual_grad, s.dy);
        ops::linear_backward(t.attention_output, w(w_o), s.residual_grad, s.o_grad, dw(w_o));
        ops::attention_backward(t.q, t.k, t.v, t.probabilities, s.o_grad, s.q_grad, s.k_grad,
                                s.v_grad, s.scratch, batch_, sequence_, c.n_heads, c.qk_dim(),
                                c.head_dim_value);
        ops::rope(s.q_grad, batch_, sequence_, c.n_heads, c.qk_dim(), c.head_dim_nope,
                  c.head_dim_rope, c.rope_theta, 0, true);
        s.k_nope_grad.resize(n, c.n_heads * c.head_dim_nope);
        s.rope_grad.resize(n, c.head_dim_rope);
        s.rope_grad.zero();
        for (int i = 0; i < n; ++i)
            for (int h = 0; h < c.n_heads; ++h) {
                for (int j = 0; j < c.head_dim_nope; ++j)
                    s.k_nope_grad(i, h * c.head_dim_nope + j) = s.k_grad(i, h * c.qk_dim() + j);
                for (int j = 0; j < c.head_dim_rope; ++j)
                    s.rope_grad(i, j) += s.k_grad(i, h * c.qk_dim() + c.head_dim_nope + j);
            }
        ops::rope(s.rope_grad, batch_, sequence_, 1, c.head_dim_rope, 0, c.head_dim_rope,
                  c.rope_theta, 0, true);
        ops::linear_backward(t.ckv, w(w_uk), s.k_nope_grad, s.ckv_grad, dw(w_uk));
        ops::linear_backward(t.ckv, w(w_uv), s.v_grad, s.tmp, dw(w_uv));
        ops::add_inplace(s.ckv_grad, s.tmp);
        ops::rmsnorm_backward(t.raw_ckv, w(kv_norm), s.ckv_grad, s.kv_raw_grad, dw(kv_norm),
                              c.norm_epsilon);
        s.kv_pack_grad.resize(n, c.kv_width());
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < c.kv_lora_rank; ++j)
                s.kv_pack_grad(i, j) = s.kv_raw_grad(i, j);
            for (int j = 0; j < c.head_dim_rope; ++j)
                s.kv_pack_grad(i, c.kv_lora_rank + j) = s.rope_grad(i, j);
        }
        ops::linear_backward(t.h, w(w_dkv), s.kv_pack_grad, s.h_grad, dw(w_dkv));
        ops::linear_backward(t.cq, w(w_uq), s.q_grad, s.cq_grad, dw(w_uq));
        ops::rmsnorm_backward(t.raw_cq, w(q_norm), s.cq_grad, s.raw_cq_grad, dw(q_norm),
                              c.norm_epsilon);
        ops::linear_backward(t.h, w(w_dq), s.raw_cq_grad, s.tmp, dw(w_dq));
        ops::add_inplace(s.h_grad, s.tmp);
        ops::rmsnorm_backward(x, w(attn_norm), s.h_grad, s.dx, dw(attn_norm), c.norm_epsilon);
        ops::add_inplace(s.dx, s.residual_grad);
        s.dy = s.dx;
    }
    for (int i = 0; i < n; ++i)
        for (int j = 0; j < c.d_model; ++j)
            parameters_[0].gradient(tokens_[static_cast<std::size_t>(i)], j) += s.dy(i, j);
    forward_ready_ = false;
    if (!gradients_finite())
        return Status::failure(ErrorCode::numerical_error, "Nonfinite model gradient");
    return ok();
}
} // namespace mini_mla
