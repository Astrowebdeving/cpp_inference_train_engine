#pragma once
#include "mini_mla/model.hpp"

namespace mini_mla {
struct OptimizerConfig {
    std::string kind = "adamw";
    float beta1 = 0.9F, beta2 = 0.95F, epsilon = 1e-8F, weight_decay = 0.1F, clip_norm = 1.0F;
    float momentum = 0.95F;
    bool nesterov = true;
    int ns_steps = 5;
    [[nodiscard]] Status validate() const;
    [[nodiscard]] Json json() const;
    [[nodiscard]] static Result<OptimizerConfig> from_json(const Json&);
};
struct OptimizerState {
    Matrix first, second;
};
class Optimizer {
  public:
    explicit Optimizer(OptimizerConfig config = {}) : config_(std::move(config)) {}
    [[nodiscard]] Status initialize(const Model&);
    [[nodiscard]] Result<double> step(Model&, float learning_rate);
    [[nodiscard]] const OptimizerConfig& config() const { return config_; }
    [[nodiscard]] std::uint64_t steps() const { return steps_; }
    [[nodiscard]] std::vector<OptimizerState>& states() { return states_; }
    [[nodiscard]] const std::vector<OptimizerState>& states() const { return states_; }
    void set_steps(std::uint64_t step) { steps_ = step; }

  private:
    void muon_update(Parameter&, OptimizerState&, float learning_rate, float gradient_scale);
    OptimizerConfig config_;
    std::uint64_t steps_ = 0;
    std::vector<OptimizerState> states_;
    Matrix x_, a_, ax_, aax_;
};
[[nodiscard]] float cosine_learning_rate(std::uint64_t completed_steps, std::uint64_t total_steps,
                                         std::uint64_t warmup_steps, float peak,
                                         float minimum_fraction);
} // namespace mini_mla
