#include "mini_mla/optimizer.hpp"

#include <cmath>
#include <numbers>

namespace mini_mla {
Status OptimizerConfig::validate() const {
    if (kind != "adamw" && kind != "hybrid_muon_adamw")
        return Status::failure(ErrorCode::unsupported,
                               "Optimizer must be adamw or hybrid_muon_adamw");
    if (!std::isfinite(beta1) || beta1 < 0 || beta1 >= 1 || !std::isfinite(beta2) || beta2 < 0 ||
        beta2 >= 1 || !std::isfinite(epsilon) || epsilon <= 0 || !std::isfinite(weight_decay) ||
        weight_decay < 0 || !std::isfinite(clip_norm) || clip_norm < 0 ||
        !std::isfinite(momentum) || momentum < 0 || momentum >= 1 || ns_steps < 1 || ns_steps > 20)
        return Status::failure(ErrorCode::invalid_argument, "Invalid optimizer hyperparameters");
    return ok();
}
Json OptimizerConfig::json() const {
    return Json::Object{{"kind", kind},
                        {"beta1", static_cast<double>(beta1)},
                        {"beta2", static_cast<double>(beta2)},
                        {"epsilon", static_cast<double>(epsilon)},
                        {"weight_decay", static_cast<double>(weight_decay)},
                        {"clip_norm", static_cast<double>(clip_norm)},
                        {"momentum", static_cast<double>(momentum)},
                        {"nesterov", nesterov},
                        {"ns_steps", ns_steps},
                        {"muon_variant", "moonlight-fp32-v1"}};
}
Result<OptimizerConfig> OptimizerConfig::from_json(const Json& j) {
    try {
        OptimizerConfig c;
        c.kind = j.str_or("kind", "adamw");
        c.beta1 = static_cast<float>(j.num_or("beta1", c.beta1));
        c.beta2 = static_cast<float>(j.num_or("beta2", c.beta2));
        c.epsilon = static_cast<float>(j.num_or("epsilon", c.epsilon));
        c.weight_decay = static_cast<float>(j.num_or("weight_decay", c.weight_decay));
        c.clip_norm = static_cast<float>(j.num_or("clip_norm", c.clip_norm));
        c.momentum = static_cast<float>(j.num_or("momentum", c.momentum));
        c.nesterov = j.bool_or("nesterov", c.nesterov);
        if (j.contains("ns_steps"))
            c.ns_steps = j.at("ns_steps").integer();
        if (j.str_or("muon_variant", "moonlight-fp32-v1") != "moonlight-fp32-v1")
            return Result<OptimizerConfig>::failure(ErrorCode::unsupported, "Unknown Muon variant");
        auto valid = c.validate();
        if (!valid)
            return Result<OptimizerConfig>::failure(valid.error());
        return Result<OptimizerConfig>::success(c);
    } catch (const std::exception& e) {
        return Result<OptimizerConfig>::failure(ErrorCode::invalid_argument, e.what());
    }
}
Status Optimizer::initialize(const Model& m) {
    auto valid = config_.validate();
    if (!valid)
        return valid;
    states_.resize(m.parameters().size());
    steps_ = 0;
    for (std::size_t i = 0; i < states_.size(); ++i) {
        const auto& p = m.parameters()[i];
        auto& s = states_[i];
        s.first.resize(p.value.rows, p.value.cols);
        s.first.zero();
        if (config_.kind == "hybrid_muon_adamw" && p.role == ParameterRole::hidden_matrix)
            s.second = Matrix();
        else {
            s.second.resize(p.value.rows, p.value.cols);
            s.second.zero();
        }
    }
    return ok();
}
void Optimizer::muon_update(Parameter& p, OptimizerState& s, float lr, float gs) {
    bool transpose = p.value.rows > p.value.cols;
    x_.resize(transpose ? p.value.cols : p.value.rows, transpose ? p.value.rows : p.value.cols);
    double sq = 0;
    for (int i = 0; i < p.value.rows; ++i)
        for (int j = 0; j < p.value.cols; ++j) {
            float g = p.gradient(i, j) * gs;
            s.first(i, j) = config_.momentum * s.first(i, j) + (1 - config_.momentum) * g;
            float v = config_.nesterov
                          ? (1 - config_.momentum) * g + config_.momentum * s.first(i, j)
                          : s.first(i, j);
            x_(transpose ? j : i, transpose ? i : j) = v;
            sq += static_cast<double>(v) * v;
        }
    float inv =
        static_cast<float>(1.0 / std::max(std::sqrt(sq), static_cast<double>(config_.epsilon)));
    for (auto& v : x_.values)
        v *= inv;
    for (int k = 0; k < config_.ns_steps; ++k) {
        ops::matmul(x_, x_, a_, false, true);
        ops::matmul(a_, x_, ax_);
        ops::matmul(a_, ax_, aax_);
        for (std::size_t i = 0; i < x_.size(); ++i)
            x_.values[i] =
                3.4445F * x_.values[i] - 4.7750F * ax_.values[i] + 2.0315F * aax_.values[i];
    }
    float scale = 0.2F * std::sqrt(static_cast<float>(std::max(p.value.rows, p.value.cols)));
    for (int i = 0; i < p.value.rows; ++i)
        for (int j = 0; j < p.value.cols; ++j)
            p.value(i, j) = p.value(i, j) * (1 - lr * config_.weight_decay) -
                            lr * scale * x_(transpose ? j : i, transpose ? i : j);
}
Result<double> Optimizer::step(Model& m, float lr) {
    if (!std::isfinite(lr) || lr < 0 || states_.size() != m.parameters().size())
        return Result<double>::failure(ErrorCode::invalid_argument,
                                       "Invalid learning rate or uninitialized optimizer");
    double sq = 0;
    for (std::size_t i = 0; i < states_.size(); ++i) {
        const auto& p = m.parameters()[i];
        const auto& s = states_[i];
        bool muon = config_.kind == "hybrid_muon_adamw" && p.role == ParameterRole::hidden_matrix;
        auto matches = [&](const Matrix& x) {
            return x.rows == p.value.rows && x.cols == p.value.cols && x.size() == p.value.size();
        };
        if (!matches(p.gradient) || !matches(s.first) ||
            (muon ? s.second.size() != 0 : !matches(s.second)))
            return Result<double>::failure(ErrorCode::invalid_argument,
                                           "Optimizer state or gradient shape mismatch");
        for (auto g : p.gradient.values) {
            if (!std::isfinite(g))
                return Result<double>::failure(ErrorCode::numerical_error,
                                               "Nonfinite gradient before optimizer update");
            sq += static_cast<double>(g) * g;
        }
    }
    double norm = std::sqrt(sq);
    float gs = config_.clip_norm > 0
                   ? static_cast<float>(std::min(1.0, static_cast<double>(config_.clip_norm) /
                                                          std::max(norm, 1e-30)))
                   : 1.0F;
    m.mark_weights_changed();
    ++steps_;
    float bc1 = static_cast<float>(
        1.0 - std::pow(static_cast<double>(config_.beta1), static_cast<double>(steps_)));
    float bc2 = static_cast<float>(
        1.0 - std::pow(static_cast<double>(config_.beta2), static_cast<double>(steps_)));
    for (std::size_t k = 0; k < states_.size(); ++k) {
        auto& p = m.parameters()[k];
        auto& s = states_[k];
        if (config_.kind == "hybrid_muon_adamw" && p.role == ParameterRole::hidden_matrix)
            muon_update(p, s, lr, gs);
        else {
            float decay = p.role == ParameterRole::norm_scale ? 0 : config_.weight_decay;
            for (std::size_t i = 0; i < p.value.size(); ++i) {
                float g = p.gradient.values[i] * gs;
                s.first.values[i] = config_.beta1 * s.first.values[i] + (1 - config_.beta1) * g;
                s.second.values[i] =
                    config_.beta2 * s.second.values[i] + (1 - config_.beta2) * g * g;
                p.value.values[i] = p.value.values[i] * (1 - lr * decay) -
                                    lr * (s.first.values[i] / bc1) /
                                        (std::sqrt(s.second.values[i] / bc2) + config_.epsilon);
            }
        }
        if (!ops::finite(p.value.values))
            return Result<double>::failure(ErrorCode::numerical_error,
                                           "Nonfinite optimizer output: " + p.name);
    }
    return Result<double>::success(norm);
}
float cosine_learning_rate(std::uint64_t done, std::uint64_t total, std::uint64_t warmup,
                           float peak, float minimum) {
    if (warmup && done < warmup)
        return peak * static_cast<float>(done + 1) / static_cast<float>(warmup);
    if (total <= warmup)
        return peak;
    double t = std::clamp(static_cast<double>(done - std::min(done, warmup)) /
                              static_cast<double>(total - warmup),
                          0.0, 1.0);
    return peak *
           static_cast<float>(minimum + (1 - minimum) * 0.5 * (1 + std::cos(std::numbers::pi * t)));
}
} // namespace mini_mla
