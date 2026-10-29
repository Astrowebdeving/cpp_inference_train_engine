#include "mini_mla/inference.hpp"

#include <Accelerate/Accelerate.h>
#include <chrono>
#include <cmath>
#include <mutex>
#include <numeric>

namespace mini_mla {
CpuDecoder::CpuDecoder(Model& model) : model_(model) {
    const auto& c = model.config();
    latent_.resize(static_cast<std::size_t>(c.n_layers));
    rope_.resize(latent_.size());
    output_projection_.resize(latent_.size());
    for (std::size_t l = 0; l < latent_.size(); ++l) {
        latent_[l].resize(c.max_sequence_length, c.kv_lora_rank);
        rope_[l].resize(c.max_sequence_length, c.head_dim_rope);
    }
    query_latent_.resize(c.n_heads, c.kv_lora_rank);
    latent_context_.resize(1, c.n_heads * c.kv_lora_rank);
    probabilities_.resize(c.n_heads, c.max_sequence_length);
}
std::size_t CpuDecoder::cache_bytes() const {
    const auto& c = model_.config();
    return static_cast<std::size_t>(c.n_layers) * static_cast<std::size_t>(c.max_sequence_length) *
           static_cast<std::size_t>(c.kv_width()) * sizeof(float);
}
void CpuDecoder::prepare_projections() {
    if (projection_revision_ == model_.revision())
        return;
    const auto& c = model_.config();
    for (int l = 0; l < c.n_layers; ++l) {
        auto& out = output_projection_[static_cast<std::size_t>(l)];
        out.resize(c.n_heads * c.kv_lora_rank, c.d_model);
        const auto& uv = model_.layer(l, w_uv).value;
        const auto& wo = model_.layer(l, w_o).value;
        for (int h = 0; h < c.n_heads; ++h)
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, c.kv_lora_rank, c.d_model,
                        c.head_dim_value, 1, uv.data() + h * c.head_dim_value, c.attn_width(),
                        wo.data() + static_cast<std::size_t>(h * c.head_dim_value * c.d_model),
                        c.d_model, 0,
                        out.data() + static_cast<std::size_t>(h * c.kv_lora_rank * c.d_model),
                        c.d_model);
    }
    projection_revision_ = model_.revision();
}
Result<std::span<const float>> CpuDecoder::prefill(std::span<const std::uint16_t> ids) {
    position_ = 0;
    if (ids.empty() || ids.size() > static_cast<std::size_t>(model_.config().max_sequence_length))
        return Result<std::span<const float>>::failure(ErrorCode::invalid_argument,
                                                       "Prompt exceeds context or is empty");
    auto forward = model_.forward(ids, 1, static_cast<int>(ids.size()));
    if (!forward)
        return Result<std::span<const float>>::failure(forward.error());
    prepare_projections();
    for (std::size_t l = 0; l < latent_.size(); ++l) {
        std::copy(model_.tape()[l].ckv.values.begin(), model_.tape()[l].ckv.values.end(),
                  latent_[l].values.begin());
        std::copy(model_.tape()[l].k_rope.values.begin(), model_.tape()[l].k_rope.values.end(),
                  rope_[l].values.begin());
    }
    position_ = static_cast<int>(ids.size());
    cache_revision_ = model_.revision();
    logits_.resize(1, model_.config().vocab_size);
    std::copy_n(forward.value()->data() + (ids.size() - 1) * static_cast<std::size_t>(logits_.cols),
                logits_.cols, logits_.data());
    return Result<std::span<const float>>::success(logits_.values);
}
Result<std::span<const float>> CpuDecoder::decode(std::uint16_t token) {
    const auto& c = model_.config();
    if (position_ < 1 || position_ >= c.max_sequence_length || token >= c.vocab_size ||
        cache_revision_ != model_.revision())
        return Result<std::span<const float>>::failure(
            ErrorCode::invalid_argument,
            "Decode requires current-weight prefill, available context, and a valid token");
    hidden_.resize(1, c.d_model);
    for (int j = 0; j < c.d_model; ++j)
        hidden_(0, j) = model_.parameters()[0].value(token, j);
    auto& t = work_;
    int length = position_ + 1;
    for (int l = 0; l < c.n_layers; ++l) {
        auto w = [&](LayerParameter p) -> const Matrix& { return model_.layer(l, p).value; };
        auto& cache = latent_[static_cast<std::size_t>(l)];
        auto& kr = rope_[static_cast<std::size_t>(l)];
        ops::rmsnorm(hidden_, w(attn_norm), t.h, c.norm_epsilon);
        ops::matmul(t.h, w(w_dq), t.raw_cq);
        ops::rmsnorm(t.raw_cq, w(q_norm), t.cq, c.norm_epsilon);
        ops::matmul(t.cq, w(w_uq), t.q);
        ops::rope(t.q, 1, 1, c.n_heads, c.qk_dim(), c.head_dim_nope, c.head_dim_rope, c.rope_theta,
                  position_);
        ops::matmul(t.h, w(w_dkv), t.kv_pack);
        t.raw_ckv.resize(1, c.kv_lora_rank);
        t.k_rope.resize(1, c.head_dim_rope);
        for (int j = 0; j < c.kv_lora_rank; ++j)
            t.raw_ckv(0, j) = t.kv_pack(0, j);
        for (int j = 0; j < c.head_dim_rope; ++j)
            t.k_rope(0, j) = t.kv_pack(0, c.kv_lora_rank + j);
        ops::rmsnorm(t.raw_ckv, w(kv_norm), t.ckv, c.norm_epsilon);
        ops::rope(t.k_rope, 1, 1, 1, c.head_dim_rope, 0, c.head_dim_rope, c.rope_theta, position_);
        for (int j = 0; j < c.kv_lora_rank; ++j)
            cache(position_, j) = t.ckv(0, j);
        for (int j = 0; j < c.head_dim_rope; ++j)
            kr(position_, j) = t.k_rope(0, j);
        if (!absorbed_) {
            reconstructed_key_.resize(length, c.n_heads * c.head_dim_nope);
            reconstructed_value_.resize(length, c.attn_width());
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, length,
                        c.n_heads * c.head_dim_nope, c.kv_lora_rank, 1, cache.data(),
                        c.kv_lora_rank, w(w_uk).data(), c.n_heads * c.head_dim_nope, 0,
                        reconstructed_key_.data(), reconstructed_key_.cols);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, length, c.attn_width(),
                        c.kv_lora_rank, 1, cache.data(), c.kv_lora_rank, w(w_uv).data(),
                        c.attn_width(), 0, reconstructed_value_.data(), reconstructed_value_.cols);
            t.attention_output.resize(1, c.attn_width());
        }
        for (int h = 0; h < c.n_heads; ++h) {
            float* scores =
                probabilities_.data() + static_cast<std::size_t>(h * c.max_sequence_length);
            const float* q = t.q.data() + h * c.qk_dim();
            if (absorbed_) {
                float* latent_q = query_latent_.data() + h * c.kv_lora_rank;
                cblas_sgemv(CblasRowMajor, CblasNoTrans, c.kv_lora_rank, c.head_dim_nope, 1,
                            w(w_uk).data() + h * c.head_dim_nope, c.n_heads * c.head_dim_nope, q, 1,
                            0, latent_q, 1);
                cblas_sgemv(CblasRowMajor, CblasNoTrans, length, c.kv_lora_rank, 1, cache.data(),
                            c.kv_lora_rank, latent_q, 1, 0, scores, 1);
            } else
                cblas_sgemv(CblasRowMajor, CblasNoTrans, length, c.head_dim_nope, 1,
                            reconstructed_key_.data() + h * c.head_dim_nope,
                            reconstructed_key_.cols, q, 1, 0, scores, 1);
            cblas_sgemv(CblasRowMajor, CblasNoTrans, length, c.head_dim_rope, 1, kr.data(),
                        c.head_dim_rope, q + c.head_dim_nope, 1, 1, scores, 1);
            float scale = 1.0F / std::sqrt(static_cast<float>(c.qk_dim())), max = scores[0] * scale;
            for (int i = 0; i < length; ++i) {
                scores[i] *= scale;
                max = std::max(max, scores[i]);
            }
            float sum = 0;
            for (int i = 0; i < length; ++i) {
                scores[i] = std::exp(scores[i] - max);
                sum += scores[i];
            }
            for (int i = 0; i < length; ++i)
                scores[i] /= sum;
            if (absorbed_)
                cblas_sgemv(CblasRowMajor, CblasTrans, length, c.kv_lora_rank, 1, cache.data(),
                            c.kv_lora_rank, scores, 1, 0,
                            latent_context_.data() + h * c.kv_lora_rank, 1);
            else
                cblas_sgemv(CblasRowMajor, CblasTrans, length, c.head_dim_value, 1,
                            reconstructed_value_.data() + h * c.head_dim_value, c.attn_width(),
                            scores, 1, 0, t.attention_output.data() + h * c.head_dim_value, 1);
        }
        if (absorbed_)
            ops::matmul(latent_context_, output_projection_[static_cast<std::size_t>(l)],
                        t.residual);
        else
            ops::matmul(t.attention_output, w(w_o), t.residual);
        ops::add_inplace(t.residual, hidden_);
        ops::rmsnorm(t.residual, w(ffn_norm), t.h2, c.norm_epsilon);
        ops::matmul(t.h2, w(w_gate), t.gate);
        ops::matmul(t.h2, w(w_up), t.up);
        ops::swiglu(t.gate, t.up, t.gated);
        ops::matmul(t.gated, w(w_down), hidden_);
        ops::add_inplace(hidden_, t.residual);
    }
    ops::rmsnorm(hidden_, model_.parameters()[1].value, t.h, c.norm_epsilon);
    ops::matmul(t.h, model_.parameters()[0].value, logits_, false, true);
    ++position_;
    if (!ops::finite(logits_.values))
        return Result<std::span<const float>>::failure(ErrorCode::numerical_error,
                                                       "Nonfinite decode output");
    return Result<std::span<const float>>::success(logits_.values);
}
namespace {
class CpuExecution final : public Execution {
  public:
    CpuExecution(Model& m, Optimizer* o,
                 std::shared_ptr<std::mutex> guard = std::make_shared<std::mutex>())
        : model_(m), model_guard_(std::move(guard)), optimizer_(o), decoder_(m) {}
    std::string device_name() const override { return "CPU / Accelerate"; }
    Result<std::unique_ptr<Execution>> fork_inference() override {
        if (optimizer_)
            return Result<std::unique_ptr<Execution>>::failure(ErrorCode::invalid_argument,
                                                               "Training execution cannot fork");
        return Result<std::unique_ptr<Execution>>::success(
            std::make_unique<CpuExecution>(model_, nullptr, model_guard_));
    }
    Result<const Matrix*> forward(std::span<const std::uint16_t> ids, int b, int t) override {
        std::lock_guard lock(*model_guard_);
        auto result = model_.forward(ids, b, t);
        if (!result || optimizer_)
            return result;
        output_ = *result.value();
        return Result<const Matrix*>::success(&output_);
    }
    Result<std::vector<float>> log_probabilities(const Batch& b) override {
        auto out = forward(b.input, b.batch, b.sequence);
        if (!out)
            return Result<std::vector<float>>::failure(out.error());
        coefficients_.assign(b.target.size(), 0);
        std::vector<float> lp;
        auto loss = ops::weighted_nll(*out.value(), b.target, coefficients_, nullptr, &lp);
        if (!loss)
            return Result<std::vector<float>>::failure(loss.error());
        return Result<std::vector<float>>::success(std::move(lp));
    }
    Result<double> accumulate(const Batch& b, std::span<const float> coefficients) override {
        if (!optimizer_)
            return Result<double>::failure(ErrorCode::invalid_argument,
                                           "Inference execution has no optimizer");
        auto out = forward(b.input, b.batch, b.sequence);
        if (!out)
            return Result<double>::failure(out.error());
        auto loss = ops::weighted_nll(*out.value(), b.target, coefficients, &gradient_);
        if (!loss)
            return loss;
        auto backward = model_.backward(gradient_);
        if (!backward)
            return Result<double>::failure(backward.error());
        return loss;
    }
    Status zero_grad() override {
        model_.zero_grad();
        return ok();
    }
    Result<double> update(float lr) override {
        if (!optimizer_)
            return Result<double>::failure(ErrorCode::invalid_argument,
                                           "Inference execution has no optimizer");
        return optimizer_->step(model_, lr);
    }
    Status synchronize_to_host() override { return ok(); }
    Result<std::span<const float>> prefill(std::span<const std::uint16_t> p) override {
        std::lock_guard lock(*model_guard_);
        return decoder_.prefill(p);
    }
    Result<std::span<const float>> decode(std::uint16_t t) override { return decoder_.decode(t); }

  private:
    Model& model_;
    Matrix output_;
    std::shared_ptr<std::mutex> model_guard_;
    Optimizer* optimizer_;
    CpuDecoder decoder_;
    Matrix gradient_;
    std::vector<float> coefficients_;
};
double milliseconds(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
        .count();
}
std::size_t stream_prefix(std::string_view s, std::size_t limit) {
    std::size_t i = 0;
    while (i < limit) {
        auto c = static_cast<unsigned char>(s[i]);
        std::size_t n = c < 0xc0 ? 1 : c < 0xe0 ? 2 : c < 0xf0 ? 3 : c <= 0xf4 ? 4 : 1;
        if (i + n > limit)
            break;
        i += n;
    }
    return i;
}
} // namespace
Status ExecutionOptions::validate() const {
    if (attention != "auto" && attention != "reference" && attention != "flash")
        return Status::failure(ErrorCode::invalid_argument,
                               "Attention must be auto, reference, or flash");
    if (submission_window < 1 || submission_window > 16)
        return Status::failure(ErrorCode::invalid_argument, "Submission window must be in [1, 16]");
    if (muon_layout != "batched" && muon_layout != "individual")
        return Status::failure(ErrorCode::invalid_argument,
                               "Muon layout must be batched or individual");
    if (muon_ns_order != "composed" && muon_ns_order != "gram" && muon_ns_order != "chain")
        return Status::failure(ErrorCode::invalid_argument,
                               "Muon NS order must be composed, gram, or chain");
    if (coalesce_tokens < 0 || coalesce_tokens > 1048576)
        return Status::failure(ErrorCode::invalid_argument,
                               "Coalesced pass tokens must be in [0, 1048576]");
    return ok();
}
Result<StepResult> Execution::step(std::span<const GradientBatch> batches, float lr) {
    if (batches.empty() || !std::isfinite(lr) || lr < 0)
        return Result<StepResult>::failure(ErrorCode::invalid_argument, "Invalid training step");
    for (const auto& job : batches)
        if (!job.batch || job.coefficients.size() != job.batch->target.size() ||
            !ops::finite(job.coefficients))
            return Result<StepResult>::failure(ErrorCode::invalid_argument,
                                               "Invalid gradient batch");
    StepResult result;
    for (const auto& job : batches) {
        auto loss = accumulate(*job.batch, job.coefficients);
        if (!loss)
            return Result<StepResult>::failure(loss.error());
        result.loss += loss.value();
    }
    auto norm = update(lr);
    if (!norm)
        return Result<StepResult>::failure(norm.error());
    result.gradient_norm = norm.value();
    return Result<StepResult>::success(result);
}
Result<std::optional<StepResult>> Execution::begin_step(std::span<const GradientBatch> batches,
                                                        float lr) {
    auto current = step(batches, lr);
    if (!current)
        return Result<std::optional<StepResult>>::failure(current.error());
    return Result<std::optional<StepResult>>::success(
        std::exchange(deferred_step_, current.value()));
}
Result<std::optional<StepResult>> Execution::finish_step() {
    return Result<std::optional<StepResult>>::success(std::exchange(deferred_step_, std::nullopt));
}
Result<std::unique_ptr<Execution>> make_execution(Model& m, Optimizer* o, std::string_view backend,
                                                  std::string_view precision,
                                                  const ExecutionOptions& options) {
    auto valid = options.validate();
    if (!valid)
        return Result<std::unique_ptr<Execution>>::failure(valid.error());
    if (backend == "mps")
        return make_mps_execution(m, o, precision, options);
    if (backend != "cpu" || precision != "fp32")
        return Result<std::unique_ptr<Execution>>::failure(ErrorCode::unsupported,
                                                           "Select cpu/fp32 or mps/fp32|fp16");
    if (options.attention == "flash")
        return Result<std::unique_ptr<Execution>>::failure(ErrorCode::unsupported,
                                                           "Flash attention requires Metal");
    return Result<std::unique_ptr<Execution>>::success(std::make_unique<CpuExecution>(m, o));
}
Status GenerationConfig::validate() const {
    if (max_new_tokens < 0 || max_new_tokens > 65535 || top_k < 0 || !std::isfinite(temperature) ||
        temperature < 0 || !std::isfinite(top_p) || top_p <= 0 || top_p > 1 ||
        !std::isfinite(repetition_penalty) || repetition_penalty <= 0)
        return Status::failure(ErrorCode::invalid_argument, "Invalid generation settings");
    for (const auto& s : stop)
        if (s.empty() || s.size() > 4096)
            return Status::failure(ErrorCode::invalid_argument,
                                   "Stop strings must be nonempty and <=4096 bytes");
    for (const auto& s : stop_tokens)
        if (s.empty())
            return Status::failure(ErrorCode::invalid_argument,
                                   "Stop token sequences must be nonempty");
    return ok();
}
Result<std::uint16_t> sample_token(std::span<const float> logits,
                                   std::span<const std::uint16_t> history,
                                   const GenerationConfig& c, Rng& rng,
                                   std::span<const std::uint16_t> banned) {
    auto valid = c.validate();
    if (!valid)
        return Result<std::uint16_t>::failure(valid.error());
    if (logits.empty() || logits.size() > 65535 || !ops::finite(logits))
        return Result<std::uint16_t>::failure(ErrorCode::numerical_error, "Invalid sampler logits");
    if (c.temperature == 0) {
        std::vector<bool> excluded(banned.empty() ? 0 : logits.size());
        for (auto id : banned)
            if (id < excluded.size())
                excluded[id] = true;
        std::vector<bool> repeated(c.repetition_penalty == 1 ? 0 : logits.size());
        for (auto id : history)
            if (id < repeated.size())
                repeated[id] = true;
        std::size_t best = logits.size();
        float maximum = -std::numeric_limits<float>::infinity();
        for (std::size_t i = 0; i < logits.size(); ++i) {
            if (!excluded.empty() && excluded[i])
                continue;
            float score = logits[i];
            if (!repeated.empty() && repeated[i])
                score = score > 0 ? score / c.repetition_penalty : score * c.repetition_penalty;
            if (best == logits.size() || score > maximum) {
                best = i;
                maximum = score;
            }
        }
        if (best == logits.size())
            return Result<std::uint16_t>::failure(ErrorCode::invalid_argument,
                                                  "All tokens are banned");
        return Result<std::uint16_t>::success(static_cast<std::uint16_t>(best));
    }
    std::vector<float> scores(logits.begin(), logits.end());
    std::vector<bool> seen(c.repetition_penalty == 1 ? 0 : logits.size());
    for (auto id : history)
        if (id < seen.size() && !seen[id]) {
            seen[id] = true;
            scores[id] = scores[id] > 0 ? scores[id] / c.repetition_penalty
                                        : scores[id] * c.repetition_penalty;
        }
    for (auto id : banned)
        if (id < scores.size())
            scores[id] = -std::numeric_limits<float>::infinity();
    std::vector<std::uint16_t> order;
    order.reserve(scores.size());
    for (std::size_t i = 0; i < scores.size(); ++i)
        if (std::isfinite(scores[i]))
            order.push_back(static_cast<std::uint16_t>(i));
    if (order.empty())
        return Result<std::uint16_t>::failure(ErrorCode::invalid_argument, "All tokens are banned");
    auto compare = [&](auto a, auto b) {
        return scores[a] > scores[b] || (scores[a] == scores[b] && a < b);
    };
    if (c.top_k > 0 && static_cast<std::size_t>(c.top_k) < order.size()) {
        std::nth_element(order.begin(), order.begin() + c.top_k, order.end(), compare);
        order.resize(static_cast<std::size_t>(c.top_k));
    }
    std::sort(order.begin(), order.end(), compare);
    double sum = 0;
    std::vector<double> probability(order.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        probability[i] =
            std::exp(static_cast<double>(scores[order[i]] - scores[order[0]]) / c.temperature);
        sum += probability[i];
    }
    double kept = 0;
    std::size_t count = 0;
    do {
        kept += probability[count++];
    } while (count < order.size() && kept < static_cast<double>(c.top_p) * sum);
    double draw = static_cast<double>(rng.uniform()) * kept;
    for (std::size_t i = 0; i < count; ++i) {
        draw -= probability[i];
        if (draw <= 0)
            return Result<std::uint16_t>::success(order[i]);
    }
    return Result<std::uint16_t>::success(order[count - 1]);
}
Result<std::unique_ptr<InferenceEngine>> InferenceEngine::load(const std::filesystem::path& path,
                                                               std::string_view backend,
                                                               std::string_view precision,
                                                               const ExecutionOptions& options) {
    auto loaded = load_checkpoint(path);
    if (!loaded)
        return Result<std::unique_ptr<InferenceEngine>>::failure(loaded.error());
    auto e = std::make_unique<InferenceEngine>();
    e->model_ = std::move(loaded.value().model);
    e->tokenizer_ = std::make_shared<const Tokenizer>(std::move(loaded.value().tokenizer));
    auto execution = make_execution(*e->model_, nullptr, backend, precision, options);
    if (!execution)
        return Result<std::unique_ptr<InferenceEngine>>::failure(execution.error());
    e->execution_ = std::move(execution).value();
    auto banned = std::make_shared<std::vector<std::uint16_t>>();
    for (int i = 0; i < e->tokenizer_->vocab_size(); ++i)
        if ((e->tokenizer_->is_special(static_cast<std::uint16_t>(i)) &&
             i != e->tokenizer_->special("<|eos|>")) ||
            e->tokenizer_->token_bytes(static_cast<std::uint16_t>(i)).empty())
            banned->push_back(static_cast<std::uint16_t>(i));
    e->banned_tokens_ = std::move(banned);
    return Result<std::unique_ptr<InferenceEngine>>::success(std::move(e));
}
Result<std::unique_ptr<InferenceEngine>> InferenceEngine::fork() {
    auto execution = execution_->fork_inference();
    if (!execution)
        return Result<std::unique_ptr<InferenceEngine>>::failure(execution.error());
    auto session = std::make_unique<InferenceEngine>();
    session->model_ = model_;
    session->tokenizer_ = tokenizer_;
    session->banned_tokens_ = banned_tokens_;
    session->execution_ = std::move(execution).value();
    return Result<std::unique_ptr<InferenceEngine>>::success(std::move(session));
}
Result<Generation> InferenceEngine::generate(std::string_view prompt, const GenerationConfig& c,
                                             std::function<bool(std::string_view)> callback) {
    auto ids = tokenizer_->encode(prompt);
    if (!ids)
        return Result<Generation>::failure(ids.error());
    ids.value().insert(ids.value().begin(), tokenizer_->special("<|bos|>"));
    return generate_tokens(ids.value(), c, std::move(callback));
}
Result<Generation>
InferenceEngine::generate_tokens(std::span<const std::uint16_t> prompt, const GenerationConfig& c,
                                 std::function<bool(std::string_view)> callback) {
    auto valid = c.validate();
    if (!valid)
        return Result<Generation>::failure(valid.error());
    if (prompt.empty() || prompt.size() + static_cast<std::size_t>(c.max_new_tokens) >
                              static_cast<std::size_t>(config().max_sequence_length))
        return Result<Generation>::failure(ErrorCode::invalid_argument,
                                           "Prompt plus requested tokens exceeds context");
    auto started = std::chrono::steady_clock::now();
    auto out = execution_->prefill(prompt);
    if (!out)
        return Result<Generation>::failure(out.error());
    Generation result;
    result.prompt_tokens = prompt.size();
    result.prefill_ms = milliseconds(started);
    result.finish_reason = "length";
    std::vector<std::uint16_t> history(prompt.begin(), prompt.end());
    Rng rng(c.seed);
    std::size_t emitted = 0, hold = 0;
    for (const auto& stop : c.stop)
        hold = std::max(hold, stop.size());
    for (const auto& sequence : c.stop_tokens) {
        std::size_t bytes = 0;
        for (auto token : sequence) {
            if (token >= tokenizer_->vocab_size())
                return Result<Generation>::failure(ErrorCode::invalid_argument,
                                                   "Stop token exceeds vocabulary");
            bytes += tokenizer_->token_bytes(token).size();
        }
        hold = std::max(hold, bytes);
    }
    for (int i = 0; i < c.max_new_tokens; ++i) {
        if (callback && !callback({})) {
            result.finish_reason = "cancelled";
            break;
        }
        auto sampled = sample_token(out.value(), history, c, rng, *banned_tokens_);
        if (!sampled)
            return Result<Generation>::failure(sampled.error());
        if (i == 0)
            result.time_to_first_token_ms = milliseconds(started);
        auto token = sampled.value();
        result.tokens.push_back(token);
        history.push_back(token);
        if (token == tokenizer_->special("<|eos|>")) {
            result.finish_reason = "stop";
            break;
        }
        bool stopped = false;
        for (const auto& sequence : c.stop_tokens)
            if (result.tokens.size() >= sequence.size() &&
                std::equal(sequence.rbegin(), sequence.rend(), result.tokens.rbegin())) {
                stopped = true;
                break;
            }
        if (stopped) {
            auto text = tokenizer_->decode(result.tokens, true);
            if (!text)
                return Result<Generation>::failure(text.error());
            result.text = std::move(text).value();
            for (const auto& sequence : c.stop_tokens)
                if (result.tokens.size() >= sequence.size() &&
                    std::equal(sequence.rbegin(), sequence.rend(), result.tokens.rbegin())) {
                    auto tail = tokenizer_->decode(sequence, true);
                    if (tail && tail.value().size() <= result.text.size())
                        result.text.resize(result.text.size() - tail.value().size());
                    break;
                }
            result.finish_reason = "stop";
            break;
        }
        result.text += tokenizer_->token_bytes(token);
        for (const auto& stop : c.stop) {
            auto p = result.text.find(stop);
            if (p != std::string::npos) {
                result.text.resize(p);
                result.finish_reason = "stop";
                stopped = true;
                break;
            }
        }
        std::size_t safe =
            stream_prefix(result.text, stopped                     ? result.text.size()
                                       : result.text.size() > hold ? result.text.size() - hold
                                                                   : 0);
        if (callback && safe > emitted) {
            if (!callback(
                    clean_utf8(std::string_view(result.text).substr(emitted, safe - emitted)))) {
                result.finish_reason = "cancelled";
                break;
            }
            emitted = safe;
        }
        if (stopped)
            break;
        if (i + 1 < c.max_new_tokens) {
            auto step = std::chrono::steady_clock::now();
            out = execution_->decode(token);
            if (!out)
                return Result<Generation>::failure(out.error());
            result.decode_ms += milliseconds(step);
        }
    }
    if (callback && result.finish_reason != "cancelled" && result.text.size() > emitted &&
        !callback(clean_utf8(std::string_view(result.text).substr(emitted))))
        result.finish_reason = "cancelled";
    result.text = clean_utf8(result.text);
    return Result<Generation>::success(std::move(result));
}
} // namespace mini_mla
