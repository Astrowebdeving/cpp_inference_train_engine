#pragma once
#include "mini_mla/data.hpp"
#include "mini_mla/optimizer.hpp"

namespace mini_mla {
struct ExecutionOptions {
    std::string attention = "auto";
    int submission_window = 4;
    std::string muon_layout = "batched";
    std::string muon_ns_order = "gram";
    [[nodiscard]] Status validate() const;
};
struct GradientBatch {
    const Batch* batch = nullptr;
    std::span<const float> coefficients;
};
struct StepResult {
    double loss = 0, gradient_norm = 0;
};
struct ExecutionStats {
    std::uint64_t graph_submissions = 0, host_waits = 0, metal_kernel_submissions = 0;
};
class Execution {
  public:
    virtual ~Execution() = default;
    [[nodiscard]] virtual std::string device_name() const = 0;
    [[nodiscard]] virtual std::string attention_name() const { return "reference"; }
    [[nodiscard]] virtual ExecutionStats stats() const { return {}; }
    [[nodiscard]] virtual Result<std::unique_ptr<Execution>> fork_inference() {
        return Result<std::unique_ptr<Execution>>::failure(
            ErrorCode::unsupported, "Execution cannot fork inference sessions");
    }
    [[nodiscard]] virtual Result<const Matrix*> forward(std::span<const std::uint16_t>, int batch,
                                                        int sequence) = 0;
    [[nodiscard]] virtual Result<std::vector<float>> log_probabilities(const Batch&) = 0;
    [[nodiscard]] virtual Result<double> accumulate(const Batch&,
                                                    std::span<const float> coefficients) = 0;
    [[nodiscard]] virtual Status zero_grad() = 0;
    [[nodiscard]] virtual Result<double> update(float learning_rate) = 0;
    [[nodiscard]] virtual Result<StepResult> step(std::span<const GradientBatch>,
                                                  float learning_rate);
    [[nodiscard]] virtual Status synchronize_to_host() = 0;
    [[nodiscard]] virtual Result<std::span<const float>>
        prefill(std::span<const std::uint16_t>) = 0;
    [[nodiscard]] virtual Result<std::span<const float>> decode(std::uint16_t) = 0;
};
[[nodiscard]] Result<std::unique_ptr<Execution>> make_execution(Model&, Optimizer*,
                                                                std::string_view backend,
                                                                std::string_view precision = "fp32",
                                                                const ExecutionOptions& = {});
[[nodiscard]] Result<std::unique_ptr<Execution>>
make_mps_execution(Model&, Optimizer*, std::string_view precision, const ExecutionOptions& = {});
[[nodiscard]] Result<Json> metal_device_info();
} // namespace mini_mla
