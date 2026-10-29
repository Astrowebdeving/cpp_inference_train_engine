#pragma once
#include "mini_mla/checkpoint.hpp"
#include "mini_mla/execution.hpp"

#include <functional>

namespace mini_mla {
class CpuDecoder {
  public:
    explicit CpuDecoder(Model& model);
    [[nodiscard]] Result<std::span<const float>> prefill(std::span<const std::uint16_t>);
    [[nodiscard]] Result<std::span<const float>> decode(std::uint16_t);
    void set_absorbed(bool enabled) { absorbed_ = enabled; }
    [[nodiscard]] std::size_t cache_bytes() const;

  private:
    Model& model_;
    std::vector<Matrix> latent_, rope_, output_projection_;
    LayerTape work_;
    Matrix hidden_, query_latent_, latent_context_, probabilities_, attention_result_,
        reconstructed_key_, reconstructed_value_, logits_;
    int position_ = 0;
    bool absorbed_ = true;
    std::uint64_t cache_revision_ = 0, projection_revision_ = UINT64_MAX;
    void prepare_projections();
};
struct GenerationConfig {
    int max_new_tokens = 128, top_k = 0;
    float temperature = 0.8F, top_p = 0.95F, repetition_penalty = 1;
    std::uint64_t seed = 42;
    std::vector<std::string> stop;
    std::vector<std::vector<std::uint16_t>> stop_tokens;
    [[nodiscard]] Status validate() const;
};
struct Generation {
    std::string text, finish_reason;
    std::vector<std::uint16_t> tokens;
    std::size_t prompt_tokens = 0;
    double prefill_ms = 0, decode_ms = 0, time_to_first_token_ms = 0;
};
[[nodiscard]] Result<std::uint16_t> sample_token(std::span<const float>,
                                                 std::span<const std::uint16_t> history,
                                                 const GenerationConfig&, Rng&,
                                                 std::span<const std::uint16_t> banned = {});
class InferenceEngine {
  public:
    [[nodiscard]] static Result<std::unique_ptr<InferenceEngine>>
    load(const std::filesystem::path&, std::string_view backend = "cpu",
         std::string_view precision = "fp32", const ExecutionOptions& = {});
    [[nodiscard]] Result<std::unique_ptr<InferenceEngine>> fork();
    [[nodiscard]] Result<Generation> generate(std::string_view prompt, const GenerationConfig&,
                                              std::function<bool(std::string_view)> callback = {});
    [[nodiscard]] Result<Generation>
    generate_tokens(std::span<const std::uint16_t> prompt, const GenerationConfig&,
                    std::function<bool(std::string_view)> callback = {});
    [[nodiscard]] const Tokenizer& tokenizer() const { return *tokenizer_; }
    [[nodiscard]] const ModelConfig& config() const { return model_->config(); }
    [[nodiscard]] std::string device_name() const { return execution_->device_name(); }

  private:
    std::shared_ptr<Model> model_;
    std::shared_ptr<const Tokenizer> tokenizer_;
    std::shared_ptr<const std::vector<std::uint16_t>> banned_tokens_;
    std::unique_ptr<Execution> execution_;
};
} // namespace mini_mla
