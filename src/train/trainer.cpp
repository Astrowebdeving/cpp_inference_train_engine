#include "mini_mla/trainer.hpp"

#include "mini_mla/workflow.hpp"

#include <chrono>
#include <fstream>
#include <iomanip>
#include <numeric>
#include <set>
#include <sstream>

namespace mini_mla {
namespace {
Json path_array(const std::vector<std::filesystem::path>& paths) {
    Json::Array a;
    for (const auto& p : paths)
        a.emplace_back(std::filesystem::absolute(p).string());
    return a;
}
std::vector<std::filesystem::path> parse_paths(const Json& j) {
    std::vector<std::filesystem::path> p;
    for (const auto& x : j.array())
        p.emplace_back(x.string());
    return p;
}
std::string absolute_path(const std::filesystem::path& p) {
    return p.empty() ? "" : std::filesystem::absolute(p).string();
}
std::uint64_t state_integer(const Json& j, std::string_view key) {
    const auto& s = j.at(key).string();
    std::size_t used;
    auto n = std::stoull(s, &used);
    if (used != s.size() || s.starts_with('-'))
        throw std::invalid_argument("Invalid trainer integer");
    return n;
}
std::vector<Json> read_records(std::span<const std::filesystem::path> paths,
                               std::string& fingerprint) {
    std::vector<Json> records;
    std::string identity;
    for (const auto& path : paths) {
        auto file = take(MappedFile::open(path));
        identity += hex(sha256(file.bytes()));
        std::string_view text(reinterpret_cast<const char*>(file.bytes().data()),
                              file.bytes().size());
        for (std::size_t start = 0; start < text.size();) {
            auto end = text.find('\n', start);
            if (end == std::string_view::npos)
                end = text.size();
            auto line = text.substr(start, end - start);
            if (line.find_first_not_of(" \r\t") != std::string_view::npos)
                records.push_back(take(Json::parse(line)));
            start = end + 1;
        }
    }
    if (records.empty())
        throw std::invalid_argument("Post-training dataset is empty");
    fingerprint = hash_text(identity);
    return records;
}
void require_distinct_splits(const TrainOptions& o, const Tokenizer& tokenizer) {
    std::set<std::string> training;
    auto digest = [&](const std::filesystem::path& path) {
        if (o.mode == "pretrain")
            return take(TokenShard::open(path, tokenizer.vocab_size(), tokenizer.hash()))
                .payload_hash();
        auto file = take(MappedFile::open(path));
        return hex(sha256(file.bytes()));
    };
    for (const auto& path : o.train)
        if (!training.insert(digest(path)).second)
            throw std::invalid_argument("Duplicate training data file");
    for (const auto& path : o.validation)
        if (training.contains(digest(path)))
            throw std::invalid_argument("Training and validation content overlap");
}
void require_distinct_records(const std::vector<Json>& training,
                              const std::vector<Json>& validation) {
    std::set<std::string> hashes;
    for (const auto& row : training)
        hashes.insert(hash_text(row.dump()));
    for (const auto& row : validation)
        if (hashes.contains(hash_text(row.dump())))
            throw std::invalid_argument("Training and validation records overlap");
}
std::unique_ptr<Model> clone(const Model& m) {
    auto copy = take(Model::create(m.config(), 0, false, false));
    for (std::size_t i = 0; i < m.parameters().size(); ++i)
        copy->parameters()[i].value = m.parameters()[i].value;
    return copy;
}
Batch preference_batch(const Json& row, const Tokenizer& tokenizer, int sequence) {
    auto prompt = take(tokenizer.chat_prompt(
        Json::Array{Json::Object{{"role", "user"}, {"content", row.at("prompt").string()}}}));
    auto completion = [&](std::string_view key) {
        auto ids = take(tokenizer.encode(row.at(key).string()));
        ids.push_back(tokenizer.special("<|eos|>"));
        return take(completion_batch(prompt, ids, sequence, tokenizer.special("<|pad|>")));
    };
    auto chosen = completion("chosen"), rejected = completion("rejected");
    chosen.batch = 2;
    chosen.input.insert(chosen.input.end(), rejected.input.begin(), rejected.input.end());
    chosen.target.insert(chosen.target.end(), rejected.target.begin(), rejected.target.end());
    chosen.mask.insert(chosen.mask.end(), rejected.mask.begin(), rejected.mask.end());
    return chosen;
}
DpoObjective preference_objective(Execution& policy, Execution& reference, const Batch& batch,
                                  float beta) {
    auto p = take(policy.log_probabilities(batch)), r = take(reference.log_probabilities(batch));
    double chosen = 0, rejected = 0, ref_chosen = 0, ref_rejected = 0;
    for (std::size_t t = 0; t < batch.target.size(); ++t)
        if (batch.mask[t] != 0) {
            if (t < static_cast<std::size_t>(batch.sequence)) {
                chosen += p[t];
                ref_chosen += r[t];
            } else {
                rejected += p[t];
                ref_rejected += r[t];
            }
        }
    return dpo_objective(chosen, rejected, ref_chosen, ref_rejected, beta);
}
std::string trim(std::string s) {
    auto a = s.find_first_not_of(" \r\n\t");
    if (a == std::string::npos)
        return {};
    auto b = s.find_last_not_of(" \r\n\t");
    return s.substr(a, b - a + 1);
}
float reward_for(std::string_view completion, const Json& row, std::string_view kind) {
    auto answer = row.str_or("answer", "");
    if (kind == "exact")
        return trim(std::string(completion)) == answer ? 1 : 0;
    if (kind == "contains") {
        if (answer.empty())
            throw std::invalid_argument("Contains reward needs a nonempty answer");
        return completion.find(answer) != std::string_view::npos ? 1 : 0;
    }
    return Json::parse(completion).has_value() ? 1 : 0;
}
std::string greedy_sample(Execution& e, const Tokenizer& tokenizer,
                          std::span<const std::uint16_t> prompt, int limit) {
    auto logits = take(e.prefill(prompt));
    Rng rng(0);
    GenerationConfig gc;
    gc.temperature = 0;
    std::vector<std::uint16_t> history(prompt.begin(), prompt.end()), response, banned;
    for (int id = 0; id < tokenizer.vocab_size(); ++id)
        if ((tokenizer.is_special(static_cast<std::uint16_t>(id)) &&
             id != tokenizer.special("<|eos|>")) ||
            tokenizer.token_bytes(static_cast<std::uint16_t>(id)).empty())
            banned.push_back(static_cast<std::uint16_t>(id));
    for (int i = 0; i < limit; ++i) {
        auto token = take(sample_token(logits, history, gc, rng, banned));
        if (token == tokenizer.special("<|eos|>"))
            break;
        history.push_back(token);
        response.push_back(token);
        if (i + 1 < limit)
            logits = take(e.decode(token));
    }
    return clean_utf8(take(tokenizer.decode(response, true)));
}
double entropy(const Matrix& logits, std::span<const float> mask) {
    double total = 0;
    for (int i = 0; i < logits.rows; ++i) {
        if (mask[static_cast<std::size_t>(i)] == 0)
            continue;
        auto row = std::span{logits.data() + static_cast<std::size_t>(i * logits.cols),
                             static_cast<std::size_t>(logits.cols)};
        double max = *std::max_element(row.begin(), row.end()), z = 0, weighted = 0;
        for (float value : row) {
            double p = std::exp(value - max);
            z += p;
            weighted += p * (value - max);
        }
        total += std::log(z) - weighted / z;
    }
    return total;
}
struct Rollout {
    Batch batch;
    std::vector<float> old_log_probs;
    std::string text;
    float reward = 0;
};
Rollout rollout(Execution& e, const Tokenizer& tokenizer, std::span<const std::uint16_t> prompt,
                const TrainOptions& o, std::uint64_t seed) {
    auto logits = take(e.prefill(prompt));
    std::vector<std::uint16_t> response, history(prompt.begin(), prompt.end());
    Rng rng(seed);
    GenerationConfig gc;
    gc.temperature = 1;
    gc.top_p = 1;
    gc.max_new_tokens = o.completion_tokens;
    int length = std::min(o.completion_tokens, o.sequence + 1 - static_cast<int>(prompt.size()));
    if (length < 1)
        throw std::invalid_argument("GRPO prompt leaves no room for a response");
    for (int i = 0; i < length; ++i) {
        auto token = take(sample_token(logits, history, gc, rng));
        response.push_back(token);
        history.push_back(token);
        if (token == tokenizer.special("<|eos|>"))
            break;
        if (i + 1 < length)
            logits = take(e.decode(token));
    }
    Rollout r;
    r.batch = take(completion_batch(prompt, response, o.sequence, tokenizer.special("<|pad|>")));
    r.old_log_probs = take(e.log_probabilities(r.batch));
    r.text = clean_utf8(take(tokenizer.decode(response, true)));
    return r;
}
ExecutionOptions execution_options(const TrainOptions& o) {
    return {o.attention, o.submission_window, o.muon_layout, o.muon_ns_order, o.coalesce_tokens};
}
} // namespace
Status TrainOptions::validate() const {
    if (mode != "pretrain" && mode != "sft" && mode != "dpo" && mode != "grpo")
        return Status::failure(ErrorCode::unsupported, "Mode must be pretrain, sft, dpo, or grpo");
    if (train.empty() || tokenizer.empty() || output.empty() || sequence < 1 || microbatch < 1 ||
        microbatch > 4096 || accumulation < 1 || accumulation > 1024 || !steps ||
        steps > 1000000000 || warmup > steps || log_every < 1 || checkpoint_every < 1 ||
        validate_every < 1 || validation_batches < 1)
        return Status::failure(ErrorCode::invalid_argument,
                               "Missing training files/tokenizer or invalid training limits");
    if (!std::isfinite(learning_rate) || learning_rate <= 0 ||
        !std::isfinite(minimum_lr_fraction) || minimum_lr_fraction < 0 || minimum_lr_fraction > 1 ||
        !std::isfinite(dpo_beta) || dpo_beta <= 0 || group_size < 2 || group_size > 64 ||
        completion_tokens < 1 || gradient_epochs < 1 || gradient_epochs > 16 ||
        !std::isfinite(clip_epsilon) || clip_epsilon <= 0 || clip_epsilon >= 1 ||
        !std::isfinite(kl_coefficient) || kl_coefficient < 0)
        return Status::failure(ErrorCode::invalid_argument,
                               "Invalid schedule, DPO, or GRPO settings");
    if (reward != "exact" && reward != "contains" && reward != "json")
        return Status::failure(ErrorCode::unsupported, "Reward must be exact, contains, or json");
    for (const auto& p : train)
        for (const auto& v : validation)
            if (std::filesystem::weakly_canonical(p) == std::filesystem::weakly_canonical(v))
                return Status::failure(ErrorCode::invalid_argument,
                                       "Training and validation files overlap");
    auto execution_valid = execution_options(*this).validate();
    if (!execution_valid)
        return execution_valid;
    return optimizer.validate();
}
Json TrainOptions::json() const {
    Json result = Json::Object{{"model_config", absolute_path(model_config)},
                               {"tokenizer", absolute_path(tokenizer)},
                               {"output", absolute_path(output)},
                               {"initialize", absolute_path(initialize)},
                               {"reference", absolute_path(reference)},
                               {"train", path_array(train)},
                               {"validation", path_array(validation)},
                               {"mode", mode},
                               {"backend", backend},
                               {"precision", precision},
                               {"reward", reward},
                               {"sequence_length", sequence},
                               {"microbatch", microbatch},
                               {"grad_accum", accumulation},
                               {"steps", static_cast<std::size_t>(steps)},
                               {"warmup_steps", static_cast<std::size_t>(warmup)},
                               {"seed", std::to_string(seed)},
                               {"learning_rate", static_cast<double>(learning_rate)},
                               {"minimum_lr_fraction", static_cast<double>(minimum_lr_fraction)},
                               {"log_every", log_every},
                               {"checkpoint_every", checkpoint_every},
                               {"validate_every", validate_every},
                               {"validation_batches", validation_batches},
                               {"dpo_beta", static_cast<double>(dpo_beta)},
                               {"group_size", group_size},
                               {"completion_tokens", completion_tokens},
                               {"gradient_epochs", gradient_epochs},
                               {"clip_epsilon", static_cast<double>(clip_epsilon)},
                               {"kl_coefficient", static_cast<double>(kl_coefficient)},
                               {"optimizer", optimizer.json()}};
    if (attention != "reference")
        result["attention"] = attention;
    if (submission_window != 1)
        result["submission_window"] = submission_window;
    if (muon_layout != "individual")
        result["muon_layout"] = muon_layout;
    if (muon_ns_order != "chain")
        result["muon_ns_order"] = muon_ns_order;
    if (coalesce_tokens != 0)
        result["coalesce_tokens"] = coalesce_tokens;
    return result;
}
Result<TrainOptions> TrainOptions::from_json(const Json& j) {
    try {
        TrainOptions o;
        o.model_config = j.str_or("model_config", o.model_config.string());
        o.tokenizer = j.str_or("tokenizer", "");
        o.output = j.str_or("output", o.output.string());
        o.initialize = j.str_or("initialize", "");
        o.reference = j.str_or("reference", "");
        if (j.contains("train"))
            o.train = parse_paths(j.at("train"));
        if (j.contains("validation"))
            o.validation = parse_paths(j.at("validation"));
        o.mode = j.str_or("mode", o.mode);
        o.backend = j.str_or("backend", o.backend);
        o.precision = j.str_or("precision", o.precision);
        o.attention = j.str_or("attention", "reference");
        o.submission_window =
            j.contains("submission_window") ? j.at("submission_window").integer() : 1;
        o.muon_layout = j.str_or("muon_layout", "individual");
        o.muon_ns_order = j.str_or("muon_ns_order", "chain");
        o.coalesce_tokens = j.contains("coalesce_tokens") ? j.at("coalesce_tokens").integer() : 0;
        o.reward = j.str_or("reward", o.reward);
        auto integer = [&](std::string_view key, int fallback) {
            return j.contains(key) ? j.at(key).integer() : fallback;
        };
        o.sequence = integer("sequence_length", o.sequence);
        o.microbatch = integer("microbatch", o.microbatch);
        o.accumulation = integer("grad_accum", o.accumulation);
        o.log_every = integer("log_every", o.log_every);
        o.checkpoint_every = integer("checkpoint_every", o.checkpoint_every);
        o.validate_every = integer("validate_every", o.validate_every);
        o.validation_batches = integer("validation_batches", o.validation_batches);
        o.group_size = integer("group_size", o.group_size);
        o.completion_tokens = integer("completion_tokens", o.completion_tokens);
        o.gradient_epochs = integer("gradient_epochs", o.gradient_epochs);
        if (j.contains("steps"))
            o.steps = j.at("steps").size_integer();
        if (j.contains("warmup_steps"))
            o.warmup = j.at("warmup_steps").size_integer();
        if (j.contains("seed"))
            o.seed =
                j.at("seed").is_string() ? state_integer(j, "seed") : j.at("seed").size_integer();
        o.learning_rate = static_cast<float>(j.num_or("learning_rate", o.learning_rate));
        o.minimum_lr_fraction =
            static_cast<float>(j.num_or("minimum_lr_fraction", o.minimum_lr_fraction));
        o.dpo_beta = static_cast<float>(j.num_or("dpo_beta", o.dpo_beta));
        o.clip_epsilon = static_cast<float>(j.num_or("clip_epsilon", o.clip_epsilon));
        o.kl_coefficient = static_cast<float>(j.num_or("kl_coefficient", o.kl_coefficient));
        if (j.contains("optimizer"))
            o.optimizer = take(OptimizerConfig::from_json(j.at("optimizer")));
        return Result<TrainOptions>::success(std::move(o));
    } catch (const std::exception& e) {
        return Result<TrainOptions>::failure(ErrorCode::invalid_argument, e.what());
    }
}
DpoObjective dpo_objective(double chosen, double rejected, double rc, double rr, float beta) {
    double margin = (chosen - rejected) - (rc - rr), z = beta * margin;
    double loss = std::max(-z, 0.0) + std::log1p(std::exp(-std::abs(z)));
    float scale = beta * ops::sigmoid(static_cast<float>(-z));
    return DpoObjective{loss, margin, scale, -scale};
}
Result<GrpoObjective> grpo_objective(float lp, float old, float ref, float advantage, float clip,
                                     float klc) {
    if (!std::isfinite(lp) || !std::isfinite(old) || !std::isfinite(ref) ||
        !std::isfinite(advantage) || !std::isfinite(clip) || clip <= 0 || clip >= 1 ||
        !std::isfinite(klc) || klc < 0 || std::abs(lp - old) > 80 || std::abs(ref - lp) > 80)
        return Result<GrpoObjective>::failure(ErrorCode::numerical_error,
                                              "Invalid or unbounded GRPO token statistics");
    double ratio = std::exp(static_cast<double>(lp) - old),
           clipped = std::clamp(ratio, 1.0 - clip, 1.0 + clip),
           delta = static_cast<double>(ref) - lp, exp_delta = std::exp(delta);
    double kl = exp_delta - delta - 1,
           objective = -std::min(ratio * advantage, clipped * advantage) + klc * kl;
    bool active = advantage >= 0 ? ratio <= 1 + clip : ratio >= 1 - clip;
    double coefficient = (active ? ratio * advantage : 0) + klc * (exp_delta - 1);
    if (!std::isfinite(objective) || std::abs(coefficient) > std::numeric_limits<float>::max())
        return Result<GrpoObjective>::failure(ErrorCode::numerical_error,
                                              "Nonfinite GRPO objective");
    return Result<GrpoObjective>::success(
        GrpoObjective{objective, kl, static_cast<float>(coefficient)});
}
Result<std::vector<float>> group_advantages(std::span<const float> rewards, float epsilon) {
    if (rewards.size() < 2 || !ops::finite(rewards) || !std::isfinite(epsilon) || epsilon <= 0)
        return Result<std::vector<float>>::failure(ErrorCode::invalid_argument,
                                                   "Invalid reward group");
    double mean = std::accumulate(rewards.begin(), rewards.end(), 0.0) /
                  static_cast<double>(rewards.size()),
           sq = 0;
    for (auto r : rewards)
        sq += (r - mean) * (r - mean);
    double scale = std::sqrt(sq / static_cast<double>(rewards.size())) + epsilon;
    std::vector<float> a;
    for (auto r : rewards)
        a.push_back(static_cast<float>((r - mean) / scale));
    return Result<std::vector<float>>::success(std::move(a));
}
Result<Batch> completion_batch(std::span<const std::uint16_t> prompt,
                               std::span<const std::uint16_t> response, int sequence,
                               std::uint16_t pad) {
    if (prompt.empty() || response.empty() || sequence < 1 ||
        prompt.size() + response.size() > static_cast<std::size_t>(sequence) + 1)
        return Result<Batch>::failure(
            ErrorCode::invalid_argument,
            "Completion sample exceeds context or has an empty prompt/response");
    std::vector<std::uint16_t> ids(prompt.begin(), prompt.end());
    ids.insert(ids.end(), response.begin(), response.end());
    Batch b;
    b.batch = 1;
    b.sequence = sequence;
    b.input.resize(static_cast<std::size_t>(sequence), pad);
    b.target = b.input;
    b.mask.resize(b.input.size(), 0);
    for (std::size_t i = 0; i + 1 < ids.size(); ++i) {
        b.input[i] = ids[i];
        b.target[i] = ids[i + 1];
        b.mask[i] = i + 1 >= prompt.size() ? 1 : 0;
    }
    return Result<Batch>::success(std::move(b));
}
Status train_model(TrainOptions o, std::function<bool()> stop_requested) {
    try {
        check(o.validate());
        Tokenizer tokenizer = take(Tokenizer::load(o.tokenizer));
        require_distinct_splits(o, tokenizer);
        std::unique_ptr<Model> model;
        std::unique_ptr<Optimizer> optimizer;
        Json restored;
        std::uint64_t tokens_seen = 0;
        double best_validation = -1;
        Rng random(o.seed);
        if (!o.resume.empty()) {
            auto checkpoint = take(load_checkpoint(o.resume, true));
            model = std::move(checkpoint.model);
            optimizer = std::move(checkpoint.optimizer);
            restored = std::move(checkpoint.trainer_state);
            if (o.json().dump() != restored.at("options").dump())
                throw std::invalid_argument("Resume options changed; restore the original "
                                            "schedule, backend, data, and objective");
            if (tokenizer.hash() != checkpoint.tokenizer.hash())
                throw std::invalid_argument("Resume tokenizer changed");
            tokens_seen = state_integer(restored, "tokens_seen");
            random.restore(state_integer(restored, "workflow_rng"));
            best_validation = restored.at("best_validation").number();
        } else {
            if (std::filesystem::exists(o.output / "options.json"))
                throw std::invalid_argument(
                    "Run already exists; use --resume or a new output folder");
            if (!o.initialize.empty()) {
                auto initial = take(load_checkpoint(o.initialize));
                model = std::move(initial.model);
                if (tokenizer.hash() != initial.tokenizer.hash())
                    throw std::invalid_argument("Initial model tokenizer differs");
                model->zero_grad();
            } else
                model = take(Model::create(take(ModelConfig::load(o.model_config)), o.seed));
            optimizer = std::make_unique<Optimizer>(o.optimizer);
            check(optimizer->initialize(*model));
        }
        if (model->config().vocab_size != tokenizer.vocab_size() ||
            o.sequence > model->config().max_sequence_length)
            throw std::invalid_argument(
                "Tokenizer vocabulary or training context disagrees with model");
        std::filesystem::create_directories(o.output);
        std::unique_ptr<Execution> reference_execution;
        std::unique_ptr<Model> reference_model;
        std::string reference_hash;
        if (o.mode == "dpo" || o.mode == "grpo") {
            if (o.reference.empty())
                o.reference = o.output / "reference";
            if (std::filesystem::exists(o.reference)) {
                auto loaded = take(load_checkpoint(o.reference));
                reference_model = std::move(loaded.model);
                if (loaded.tokenizer.hash() != tokenizer.hash() ||
                    reference_model->config().hash() != model->config().hash())
                    throw std::invalid_argument("Frozen reference is incompatible");
            } else {
                if (!restored.is_null() ||
                    (!o.reference.empty() && o.reference != o.output / "reference"))
                    throw std::invalid_argument("Requested frozen reference does not exist");
                reference_model = clone(*model);
                check(export_model(o.reference, *reference_model, o.tokenizer, "f32"));
            }
            reference_hash = take(read_json(o.reference / "manifest.json"))
                                 .at("sha256")
                                 .at("weights.bin")
                                 .string();
            if (!restored.is_null() &&
                restored.at("reference_weights_sha256").string() != reference_hash)
                throw std::invalid_argument("Frozen reference weights changed since checkpoint");
            reference_execution = take(make_execution(*reference_model, nullptr, o.backend,
                                                      o.precision, execution_options(o)));
        }
        auto execution = take(make_execution(*model, optimizer.get(), o.backend, o.precision,
                                             execution_options(o)));
        std::fprintf(stderr, "%s; %zu parameters; mode=%s; optimizer=%s\n",
                     execution->device_name().c_str(), model->config().parameter_count(),
                     o.mode.c_str(), optimizer->config().kind.c_str());
        check(write_json(o.output / "options.json", o.json()));
        std::ofstream metrics(o.output / "metrics.jsonl", std::ios::app),
            rollouts(o.output / "rollouts.jsonl", std::ios::app);
        if (!metrics || !rollouts)
            throw std::runtime_error("Cannot open training logs");
        std::unique_ptr<PrefetchLoader> prefetch;
        std::unique_ptr<DataLoader> validation;
        Json validation_start;
        std::vector<Json> records, validation_records;
        std::string fingerprint, validation_fingerprint;
        if (o.mode == "pretrain" || o.mode == "sft") {
            auto loader = o.mode == "sft"
                              ? take(DataLoader::sft(o.train, o.sequence, tokenizer.vocab_size(),
                                                     tokenizer.hash(), o.seed))
                              : take(DataLoader::tokens(o.train, o.sequence, tokenizer.vocab_size(),
                                                        tokenizer.hash(), o.seed));
            fingerprint = loader->fingerprint();
            if (!restored.is_null())
                check(loader->restore(restored.at("data_state")));
            prefetch = std::make_unique<PrefetchLoader>(std::move(loader), o.microbatch);
            if (!o.validation.empty()) {
                validation =
                    o.mode == "sft"
                        ? take(DataLoader::sft(o.validation, o.sequence, tokenizer.vocab_size(),
                                               tokenizer.hash(), 0))
                        : take(DataLoader::tokens(o.validation, o.sequence, tokenizer.vocab_size(),
                                                  tokenizer.hash(), 0));
                if (validation->fingerprint() == fingerprint)
                    throw std::invalid_argument("Training and validation data are identical");
                validation_start = validation->state();
                validation_fingerprint = validation->fingerprint();
            }
        } else {
            records = read_records(o.train, fingerprint);
            if (!restored.is_null() && restored.at("data_fingerprint").string() != fingerprint)
                throw std::invalid_argument("Post-training dataset changed since checkpoint");
            if (!o.validation.empty()) {
                validation_records = read_records(o.validation, validation_fingerprint);
                require_distinct_records(records, validation_records);
            }
            if (o.mode == "grpo")
                for (int id = 0; id < tokenizer.vocab_size(); ++id)
                    if (tokenizer.token_bytes(static_cast<std::uint16_t>(id)).empty())
                        throw std::invalid_argument("GRPO needs a complete tokenizer vocabulary; "
                                                    "train more merges or use the byte tokenizer");
        }
        if (!restored.is_null() &&
            restored.at("validation_fingerprint").string() != validation_fingerprint)
            throw std::invalid_argument("Validation data changed since checkpoint");
        auto save_state = [&] {
            check(execution->synchronize_to_host());
            Json state = Json::Object{{"options", o.json()},
                                      {"tokens_seen", std::to_string(tokens_seen)},
                                      {"workflow_rng", std::to_string(random.state())},
                                      {"best_validation", best_validation},
                                      {"data_fingerprint", fingerprint},
                                      {"validation_fingerprint", validation_fingerprint},
                                      {"reference_weights_sha256", reference_hash},
                                      {"compiler", std::string(__VERSION__)},
                                      {"device", execution->device_name()},
                                      {"source_revision", "unversioned"}};
            if (prefetch)
                state["data_state"] = prefetch->consumed_state();
            std::ostringstream number;
            number << std::setw(8) << std::setfill('0') << optimizer->steps();
            auto target = o.output / ("checkpoint-" + number.str());
            if (!std::filesystem::exists(target))
                check(save_checkpoint(target, *model, *optimizer, state, o.tokenizer));
            else {
                auto existing = take(load_checkpoint(target, true));
                bool equal = existing.model->config().hash() == model->config().hash() &&
                             existing.tokenizer.hash() == tokenizer.hash() &&
                             existing.optimizer->steps() == optimizer->steps() &&
                             existing.optimizer->config().json().dump() ==
                                 optimizer->config().json().dump() &&
                             existing.trainer_state.dump() == state.dump();
                if (equal)
                    for (std::size_t i = 0; i < model->parameters().size(); ++i)
                        equal = equal &&
                                existing.model->parameters()[i].value.values ==
                                    model->parameters()[i].value.values &&
                                existing.optimizer->states()[i].first.values ==
                                    optimizer->states()[i].first.values &&
                                existing.optimizer->states()[i].second.values ==
                                    optimizer->states()[i].second.values;
                if (!equal)
                    throw std::runtime_error("Existing checkpoint differs from replayed state: " +
                                             target.string());
            }
            check(update_latest_link(o.output / "checkpoint-latest", target));
        };
        std::uint64_t attempts = 0, starting_step = optimizer->steps();
        // An update whose device result has not been reported yet.
        struct Submitted {
            std::uint64_t step = 0, tokens_seen = 0;
            float lr = 0;
            std::size_t raw_tokens = 0, supervised = 0;
            double loss = 0, kl = 0, reward_mean = 0;
            ExecutionStats before, after;
            std::chrono::steady_clock::time_point started;
        };
        // Pretraining and SFT need nothing from the device between updates, so the next update
        // is encoded while the current one executes and its metrics are reported one update
        // late. Any update that is validated, checkpointed, or ends the run lands first. DPO and
        // GRPO read policy probabilities between updates and keep the synchronous step.
        bool pipelined = o.pipeline && prefetch && o.backend == "mps" && o.submission_window > 1;
        auto lands_first = [&](std::uint64_t step) {
            return step == o.steps || step % static_cast<std::uint64_t>(o.checkpoint_every) == 0 ||
                   (!o.validation.empty() &&
                    step % static_cast<std::uint64_t>(o.validate_every) == 0);
        };
        std::optional<Submitted> outstanding;
        auto started = std::chrono::steady_clock::now();
        // `finished` closes the update's timing: its own completion when it landed before
        // anything else was submitted, or the start of the next update when it is reported late,
        // which makes step_ms the period between consecutive updates.
        auto report = [&](const Submitted& submitted, const StepResult& result,
                          std::chrono::steady_clock::time_point finished) {
            double loss = prefetch ? result.loss : submitted.loss;
            double norm = result.gradient_norm, kl = submitted.kl,
                   reward_mean = submitted.reward_mean;
            auto step = submitted.step;
            auto raw_tokens = submitted.raw_tokens;
            float lr = submitted.lr;
            double elapsed =
                std::chrono::duration<double, std::milli>(finished - submitted.started).count();
            Json metric =
                Json::Object{{"step", static_cast<std::size_t>(step)},
                             {"tokens_seen", static_cast<std::size_t>(submitted.tokens_seen)},
                             {"train_loss", loss},
                             {"lr", static_cast<double>(lr)},
                             {"grad_norm", norm},
                             {"supervised_tokens", submitted.supervised},
                             {"tokens_per_second",
                              static_cast<double>(raw_tokens) * 1000 / std::max(elapsed, 1e-9)},
                             {"step_ms", elapsed},
                             {"mode", o.mode},
                             {"kl", kl},
                             {"reward", reward_mean}};
            metric["gpu_host_waits"] = static_cast<std::size_t>(submitted.after.host_waits -
                                                                submitted.before.host_waits);
            metric["graph_submissions"] = static_cast<std::size_t>(
                submitted.after.graph_submissions - submitted.before.graph_submissions);
            metric["attention"] = execution->attention_name();
            bool evaluate =
                (!o.validation.empty()) &&
                (step % static_cast<std::uint64_t>(o.validate_every) == 0 || step == o.steps);
            if (evaluate && validation) {
                check(validation->restore(validation_start));
                double sum = 0, count = 0, ent = 0;
                std::vector<std::uint16_t> sample_prompt;
                for (int v = 0; v < o.validation_batches; ++v) {
                    auto b = take(validation->next(o.microbatch));
                    auto logits = take(execution->forward(b.input, b.batch, b.sequence));
                    sum += take(ops::weighted_nll(*logits, b.target, b.mask));
                    ent += entropy(*logits, b.mask);
                    count += static_cast<double>(b.supervised_tokens());
                    if (v == 0) {
                        auto first =
                            std::find(b.mask.begin(), b.mask.begin() + b.sequence, 1.0F);
                        int length = o.mode == "sft"
                                         ? static_cast<int>(first - b.mask.begin()) + 1
                                         : std::min(16, b.sequence);
                        sample_prompt.assign(b.input.begin(), b.input.begin() + length);
                    }
                }
                double val = sum / count;
                if (best_validation < 0 || val < best_validation)
                    best_validation = val;
                metric["val_loss"] = val;
                metric["entropy"] = ent / count;
                metric["perplexity"] = std::exp(std::min(val, 80.0));
                int limit = std::min(32, model->config().max_sequence_length -
                                             static_cast<int>(sample_prompt.size()));
                if (limit > 0)
                    rollouts << Json(Json::Object{
                                         {"kind", "validation_sample"},
                                         {"step", static_cast<std::size_t>(step)},
                                         {"prompt", clean_utf8(take(tokenizer.decode(
                                                        sample_prompt, true)))},
                                         {"completion", greedy_sample(*execution, tokenizer,
                                                                      sample_prompt, limit)},
                                         {"temperature", 0}})
                                    .dump()
                             << '\n';
            } else if (evaluate && o.mode == "dpo") {
                double total_loss = 0, total_margin = 0, correct = 0;
                auto count =
                    std::min(validation_records.size(),
                             static_cast<std::size_t>(o.validation_batches * o.microbatch));
                for (std::size_t i = 0; i < count; ++i) {
                    auto pair = preference_batch(validation_records[i], tokenizer, o.sequence);
                    auto objective = preference_objective(*execution, *reference_execution,
                                                          pair, o.dpo_beta);
                    total_loss += objective.loss;
                    total_margin += objective.margin;
                    correct += objective.margin > 0 ? 1 : objective.margin == 0 ? 0.5 : 0;
                }
                double val = total_loss / static_cast<double>(count);
                if (best_validation < 0 || val < best_validation)
                    best_validation = val;
                metric["val_loss"] = val;
                metric["preference_accuracy"] = correct / static_cast<double>(count);
                metric["preference_margin"] = total_margin / static_cast<double>(count);
            } else if (evaluate && o.mode == "grpo") {
                double rewards = 0, total_kl = 0;
                std::size_t completions = 0, count_tokens = 0;
                auto count =
                    std::min(validation_records.size(),
                             static_cast<std::size_t>(o.validation_batches * o.microbatch));
                for (std::size_t i = 0; i < count; ++i) {
                    const auto& row = validation_records[i];
                    auto prompt = take(tokenizer.chat_prompt(Json::Array{Json::Object{
                        {"role", "user"}, {"content", row.at("prompt").string()}}}));
                    for (int sample = 0; sample < o.group_size; ++sample) {
                        auto r =
                            rollout(*execution, tokenizer, prompt, o,
                                    100000 + static_cast<std::uint64_t>(
                                                 i * static_cast<std::size_t>(o.group_size) +
                                                 static_cast<std::size_t>(sample)));
                        rewards += reward_for(r.text, row, o.reward);
                        ++completions;
                        auto reference = take(reference_execution->log_probabilities(r.batch));
                        for (std::size_t t = 0; t < reference.size(); ++t)
                            if (r.batch.mask[t] != 0) {
                                auto objective = take(grpo_objective(
                                    r.old_log_probs[t], r.old_log_probs[t], reference[t], 0,
                                    o.clip_epsilon, o.kl_coefficient));
                                total_kl += objective.kl;
                                ++count_tokens;
                            }
                    }
                }
                double val_reward = rewards / static_cast<double>(completions);
                double val_cost = 1 - val_reward;
                if (best_validation < 0 || val_cost < best_validation)
                    best_validation = val_cost;
                metric["val_reward"] = val_reward;
                metric["val_kl"] = total_kl / static_cast<double>(count_tokens);
            }
            if (evaluate || step % static_cast<std::uint64_t>(o.log_every) == 0 || step == 1 ||
                step == o.steps) {
                metrics << metric.dump() << '\n';
                metrics.flush();
                rollouts.flush();
                if (!metrics || !rollouts)
                    throw std::runtime_error("Training log write failed");
                std::fprintf(stderr, "step=%llu loss=%.6f lr=%.6g grad=%.4f tok/s=%.1f\n",
                             static_cast<unsigned long long>(step), loss,
                             static_cast<double>(lr), norm,
                             static_cast<double>(raw_tokens) * 1000 / std::max(elapsed, 1e-9));
            }
        };
        while (optimizer->steps() < o.steps && !(stop_requested && stop_requested())) {
            started = std::chrono::steady_clock::now();
            check(execution->zero_grad());
            double loss = 0, kl = 0, reward_mean = 0;
            std::size_t raw_tokens = 0, supervised = 0;
            std::vector<Rollout> group;
            std::vector<float> advantages;
            std::vector<Batch> gradient_batches;
            std::vector<std::vector<float>> gradient_coefficients;
            int groups = o.microbatch * o.accumulation;
            if (prefetch) {
                gradient_batches.reserve(static_cast<std::size_t>(o.accumulation));
                for (int i = 0; i < o.accumulation; ++i) {
                    gradient_batches.push_back(take(prefetch->next()));
                    supervised += gradient_batches.back().supervised_tokens();
                    raw_tokens += gradient_batches.back().input.size();
                }
                if (!supervised)
                    throw std::runtime_error("No supervised tokens in update");
                for (const auto& batch : gradient_batches) {
                    auto coefficients = batch.mask;
                    for (auto& v : coefficients)
                        v /= static_cast<float>(supervised);
                    gradient_coefficients.push_back(std::move(coefficients));
                }
            } else if (o.mode == "dpo") {
                for (int i = 0; i < groups; ++i) {
                    const auto& row =
                        records[static_cast<std::size_t>(random.bounded(records.size()))];
                    auto pair = preference_batch(row, tokenizer, o.sequence);
                    auto objective =
                        preference_objective(*execution, *reference_execution, pair, o.dpo_beta);
                    loss += objective.loss / groups;
                    auto coefficients = pair.mask;
                    for (std::size_t t = 0; t < coefficients.size(); ++t)
                        coefficients[t] *= (t < static_cast<std::size_t>(pair.sequence)
                                                ? objective.chosen_nll_coefficient
                                                : objective.rejected_nll_coefficient) /
                                           static_cast<float>(groups);
                    supervised += pair.supervised_tokens();
                    raw_tokens += pair.input.size();
                    gradient_batches.push_back(std::move(pair));
                    gradient_coefficients.push_back(std::move(coefficients));
                }
            } else {
                if (++attempts > (o.steps - starting_step + 1) * 32)
                    throw std::runtime_error("GRPO rewards are constant across too many groups; "
                                             "provide a policy or reward with a measurable signal");
                for (int g = 0; g < groups; ++g) {
                    const auto& row =
                        records[static_cast<std::size_t>(random.bounded(records.size()))];
                    auto prompt = take(tokenizer.chat_prompt(Json::Array{
                        Json::Object{{"role", "user"}, {"content", row.at("prompt").string()}}}));
                    std::vector<float> rewards;
                    for (int i = 0; i < o.group_size; ++i) {
                        auto seed = random.next();
                        auto r = rollout(*execution, tokenizer, prompt, o, seed);
                        r.reward = reward_for(r.text, row, o.reward);
                        rollouts << Json(
                                        Json::Object{{"prompt", row.at("prompt")},
                                                     {"completion", r.text},
                                                     {"reward", static_cast<double>(r.reward)},
                                                     {"seed", std::to_string(seed)},
                                                     {"policy_step",
                                                      static_cast<std::size_t>(optimizer->steps())},
                                                     {"evaluator", "mini-mla-" + o.reward + "-v1"}})
                                        .dump()
                                 << '\n';
                        rewards.push_back(r.reward);
                        reward_mean += r.reward / static_cast<double>(groups * o.group_size);
                        supervised += r.batch.supervised_tokens();
                        raw_tokens += r.batch.input.size();
                        group.push_back(std::move(r));
                    }
                    auto a = take(group_advantages(rewards));
                    advantages.insert(advantages.end(), a.begin(), a.end());
                }
            }
            if (o.mode == "grpo" &&
                std::all_of(advantages.begin(), advantages.end(), [](float a) { return a == 0; }))
                continue;
            int epochs = o.mode == "grpo" ? o.gradient_epochs : 1;
            for (int epoch = 0; epoch < epochs && optimizer->steps() < o.steps; ++epoch) {
                if (o.mode == "grpo") {
                    check(execution->zero_grad());
                    loss = kl = 0;
                    gradient_coefficients.clear();
                    for (std::size_t i = 0; i < group.size(); ++i) {
                        auto current = take(execution->log_probabilities(group[i].batch)),
                             reference =
                                 take(reference_execution->log_probabilities(group[i].batch));
                        auto coefficients = group[i].batch.mask;
                        double denominator =
                            static_cast<double>(group.size() * group[i].batch.supervised_tokens());
                        for (std::size_t t = 0; t < coefficients.size(); ++t) {
                            if (coefficients[t] == 0)
                                continue;
                            auto objective = take(
                                grpo_objective(current[t], group[i].old_log_probs[t], reference[t],
                                               advantages[i], o.clip_epsilon, o.kl_coefficient));
                            loss += objective.loss / denominator;
                            kl += objective.kl / denominator;
                            coefficients[t] =
                                static_cast<float>(objective.nll_coefficient / denominator);
                        }
                        gradient_coefficients.push_back(std::move(coefficients));
                    }
                }
                float lr = cosine_learning_rate(optimizer->steps(), o.steps, o.warmup,
                                                o.learning_rate, o.minimum_lr_fraction);
                std::vector<GradientBatch> jobs;
                for (std::size_t i = 0; i < gradient_coefficients.size(); ++i)
                    jobs.push_back({o.mode == "grpo" ? &group[i].batch : &gradient_batches[i],
                                    gradient_coefficients[i]});
                Submitted current;
                current.lr = lr;
                current.raw_tokens = raw_tokens;
                current.supervised = supervised;
                current.loss = loss;
                current.kl = kl;
                current.reward_mean = reward_mean;
                current.started = started;
                current.before = execution->stats();
                if (pipelined) {
                    auto previous = take(execution->begin_step(jobs, lr));
                    tokens_seen += raw_tokens;
                    current.step = optimizer->steps();
                    current.tokens_seen = tokens_seen;
                    current.after = execution->stats();
                    if (previous.has_value() != outstanding.has_value())
                        throw std::runtime_error("Pipelined update results are out of order");
                    if (previous)
                        report(*outstanding, *previous, started);
                    outstanding = current;
                    if (lands_first(current.step)) {
                        auto landed = take(execution->finish_step());
                        if (!landed)
                            throw std::runtime_error("Pipelined update result is missing");
                        outstanding->after = execution->stats();
                        report(*outstanding, *landed, std::chrono::steady_clock::now());
                        outstanding.reset();
                    }
                } else {
                    auto result = take(execution->step(jobs, lr));
                    tokens_seen += raw_tokens;
                    current.step = optimizer->steps();
                    current.tokens_seen = tokens_seen;
                    current.after = execution->stats();
                    report(current, result, std::chrono::steady_clock::now());
                    started = std::chrono::steady_clock::now();
                }
            }
            if (optimizer->steps() % static_cast<std::uint64_t>(o.checkpoint_every) == 0 ||
                optimizer->steps() == o.steps)
                save_state();
        }
        if (outstanding) {
            auto landed = take(execution->finish_step());
            if (!landed)
                throw std::runtime_error("Pipelined update result is missing");
            outstanding->after = execution->stats();
            report(*outstanding, *landed, std::chrono::steady_clock::now());
            outstanding.reset();
        }
        save_state();
        check(execution->synchronize_to_host());
        auto export_path = o.output / ("model-" + std::to_string(optimizer->steps()));
        if (!std::filesystem::exists(export_path))
            check(export_model(export_path, *model, o.tokenizer, "f16"));
        else {
            auto existing = take(load_checkpoint(export_path));
            if (existing.model->config().hash() != model->config().hash() ||
                existing.tokenizer.hash() != tokenizer.hash())
                throw std::runtime_error("Existing model export is incompatible");
            for (std::size_t i = 0; i < model->parameters().size(); ++i) {
                const auto& expected = model->parameters()[i].value.values;
                const auto& actual = existing.model->parameters()[i].value.values;
                for (std::size_t j = 0; j < expected.size(); ++j)
                    if (from_half(to_half(expected[j])) != actual[j])
                        throw std::runtime_error(
                            "Existing model export differs from replayed weights");
            }
        }
        check(update_latest_link(o.output / "model-latest", export_path));
        return ok();
    } catch (const Failure& e) {
        return Status::failure(e.error());
    } catch (const std::exception& e) {
        return Status::failure(ErrorCode::invalid_argument, e.what());
    }
}
} // namespace mini_mla
