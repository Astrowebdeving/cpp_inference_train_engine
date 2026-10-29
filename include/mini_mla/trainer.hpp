#pragma once
#include "mini_mla/inference.hpp"

namespace mini_mla {
struct TrainOptions {
    std::filesystem::path model_config = "config/mla36m-v1.json", tokenizer,
                          output = "runs/mla36m-v1", resume, initialize, reference;
    std::vector<std::filesystem::path> train, validation;
    std::string mode = "pretrain", backend = "mps", precision = "fp32", reward = "exact";
    std::string attention = "auto";
    int submission_window = 4;
    std::string muon_layout = "batched";
    std::string muon_ns_order = "composed";
    int coalesce_tokens = 4096;
    // Encode the next update while the device runs the current one. This reorders host work
    // only, so it is not part of a run's saved arithmetic and may differ between resumes.
    bool pipeline = true;
    int sequence = 256, microbatch = 2, accumulation = 16, log_every = 10, checkpoint_every = 100,
        validate_every = 100, validation_batches = 4;
    std::uint64_t steps = 1000, warmup = 20, seed = 42;
    float learning_rate = 3e-4F, minimum_lr_fraction = 0.1F, dpo_beta = 0.1F;
    int group_size = 4, completion_tokens = 32, gradient_epochs = 1;
    float clip_epsilon = 0.2F, kl_coefficient = 0.02F;
    OptimizerConfig optimizer;
    [[nodiscard]] Status validate() const;
    [[nodiscard]] Json json() const;
    [[nodiscard]] static Result<TrainOptions> from_json(const Json&);
};
struct DpoObjective {
    double loss = 0, margin = 0;
    float chosen_nll_coefficient = 0, rejected_nll_coefficient = 0;
};
[[nodiscard]] DpoObjective dpo_objective(double chosen, double rejected, double reference_chosen,
                                         double reference_rejected, float beta);
struct GrpoObjective {
    double loss = 0, kl = 0;
    float nll_coefficient = 0;
};
[[nodiscard]] Result<GrpoObjective> grpo_objective(float policy_log_prob, float old_log_prob,
                                                   float reference_log_prob, float advantage,
                                                   float clip_epsilon, float kl_coefficient);
[[nodiscard]] Result<std::vector<float>> group_advantages(std::span<const float> rewards,
                                                          float epsilon = 1e-6F);
[[nodiscard]] Result<Batch> completion_batch(std::span<const std::uint16_t> prompt,
                                             std::span<const std::uint16_t> completion,
                                             int sequence, std::uint16_t pad);
[[nodiscard]] Status train_model(TrainOptions, std::function<bool()> stop_requested = {});
} // namespace mini_mla
