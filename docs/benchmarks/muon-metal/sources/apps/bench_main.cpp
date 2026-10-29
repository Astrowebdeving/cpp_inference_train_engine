#include "app_utils.hpp"

#include <chrono>
#include <fstream>
#include <numeric>
#include <sys/resource.h>

namespace {
struct Timing {
    double median = 0, p95 = 0;
};
Timing time_it(int warmup, int repetitions, const std::function<void()>& prepare,
               const std::function<void()>& operation) {
    std::vector<double> samples;
    for (int i = -warmup; i < repetitions; ++i) {
        prepare();
        auto start = std::chrono::steady_clock::now();
        operation();
        double elapsed =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start)
                .count();
        if (i >= 0)
            samples.push_back(elapsed);
    }
    std::sort(samples.begin(), samples.end());
    return {samples[samples.size() / 2],
            samples[std::min(samples.size() - 1, static_cast<std::size_t>(std::ceil(
                                                     static_cast<double>(samples.size()) * 0.95)) -
                                                     1)]};
}
double difference(std::span<const float> a, std::span<const float> b) {
    if (a.size() != b.size())
        throw std::runtime_error("Benchmark output shape mismatch");
    double d = 0;
    for (std::size_t i = 0; i < a.size(); ++i)
        d = std::max(d, std::abs(static_cast<double>(a[i]) - b[i]));
    return d;
}
class ComparisonData {
  public:
    explicit ComparisonData(const std::filesystem::path& path) : path_(path) {
        if (!std::filesystem::create_directory(path_))
            throw std::invalid_argument("Comparison directory already exists");
        data_.open(path_ / "tensors.bin", std::ios::binary);
        if (!data_)
            throw std::runtime_error("Cannot create comparison tensors");
    }
    void append(const mini_mla::Model& model, std::string_view suffix, bool gradient = false) {
        for (const auto& p : model.parameters()) {
            const auto& value = gradient ? p.gradient : p.value;
            auto bytes = value.size() * sizeof(float);
            data_.write(reinterpret_cast<const char*>(value.data()),
                        static_cast<std::streamsize>(bytes));
            if (!data_)
                throw std::runtime_error("Comparison tensor write failed");
            tensors_.emplace_back(
                mini_mla::Json::Object{{"name", p.name + "." + std::string(suffix)},
                                       {"role", std::string(mini_mla::role_name(p.role))},
                                       {"shape", mini_mla::Json::Array{value.rows, value.cols}},
                                       {"offset", offset_},
                                       {"nbytes", bytes}});
            offset_ += bytes;
        }
    }
    void finish(mini_mla::Json metadata) {
        data_.flush();
        if (!data_)
            throw std::runtime_error("Comparison tensor flush failed");
        data_.close();
        metadata["format"] = "mini-mla-training-comparison-v1";
        metadata["tensors"] = std::move(tensors_);
        auto mapped = mini_mla::take(mini_mla::MappedFile::open(path_ / "tensors.bin"));
        auto digest = mini_mla::sha256(mapped.bytes());
        metadata["tensors_sha256"] = mini_mla::hex(digest);
        mini_mla::check(mini_mla::write_json(path_ / "comparison.json", metadata));
    }

  private:
    std::filesystem::path path_;
    std::ofstream data_;
    mini_mla::Json::Array tensors_;
    std::size_t offset_ = 0;
};
} // namespace
int main(int argc, char** argv) {
    return mini_mla::app_main(argc, argv, [](const mini_mla::Args& args) {
        using namespace mini_mla;
        args.known({"kind",
                    "config",
                    "backend",
                    "precision",
                    "sequence-length",
                    "microbatch",
                    "warmup",
                    "repetitions",
                    "m",
                    "n",
                    "k",
                    "output",
                    "attention",
                    "grad-accum",
                    "submission-window",
                    "comparison-data",
                    "skip-cpu-check",
                    "time-zero-grad",
                    "optimizer",
                    "weight-decay",
                    "grad-clip",
                    "muon-momentum",
                    "muon-ns-steps",
                    "muon-nesterov",
                    "muon-layout",
                    "help"});
        if (args.has("help")) {
            std::puts("mini_mla_bench --kind matmul|norm|rope|attention|train|prefill|decode "
                      "--config MODEL.json\n  --backend cpu|mps --precision fp32|fp16 "
                      "--sequence-length 32 --warmup 3 --repetitions 20\n  "
                      "--attention auto|reference|flash --grad-accum 1 --submission-window 4\n  "
                      "--comparison-data NEW_DIR --skip-cpu-check (training comparisons)\n  "
                      "--optimizer adamw|muon --muon-layout batched|individual\n  "
                      "--muon-momentum 0.95 --muon-ns-steps 5 --muon-nesterov true");
            return 0;
        }
        auto c = take(ModelConfig::load(args.get("config", "config/tiny-test.json")));
        auto kind = args.get("kind", "decode"), backend = args.get("backend", "cpu"),
             precision = args.get("precision", "fp32");
        if (kind != "train" && (args.has("comparison-data") || args.has("skip-cpu-check")))
            throw std::invalid_argument("Comparison options require --kind train");
        int sequence = args.integer("sequence-length", 32), batch = args.integer("microbatch", 1),
            warmup = args.integer("warmup", 3), repetitions = args.integer("repetitions", 20);
        int accumulation = args.integer("grad-accum", 1);
        ExecutionOptions execution_options{args.get("attention", "auto"),
                                           args.integer("submission-window", 4),
                                           args.get("muon-layout", "batched")};
        check(execution_options.validate());
        auto optimizer_config = optimizer_options(args);
        if (kind != "train" &&
            (args.has("optimizer") || args.has("muon-momentum") || args.has("muon-ns-steps") ||
             args.has("muon-nesterov") || args.has("muon-layout") || args.has("weight-decay") ||
             args.has("grad-clip")))
            throw std::invalid_argument("Optimizer options require --kind train");
        if (sequence < 1 || sequence >= c.max_sequence_length || batch < 1 || batch > 1024 ||
            warmup < 0 || repetitions < 1 || repetitions > 10000 || accumulation < 1 ||
            accumulation > 1024)
            throw std::invalid_argument("Invalid benchmark limits");
        Json report = Json::Object{{"kind", kind},
                                   {"backend", backend},
                                   {"precision", precision},
                                   {"sequence_length", sequence},
                                   {"batch", batch},
                                   {"grad_accum", accumulation},
                                   {"attention_requested", execution_options.attention},
                                   {"submission_window", execution_options.submission_window},
                                   {"warmup", warmup},
                                   {"repetitions", repetitions},
                                   {"compiler", std::string(__VERSION__)}};
        Timing timing;
        double error = 0;
        Json comparison_metadata;
        std::unique_ptr<ComparisonData> comparison;
        if (kind == "matmul" || kind == "norm" || kind == "rope" || kind == "attention") {
            if (backend != "cpu" || precision != "fp32")
                throw std::invalid_argument("Operator benchmarks use the CPU reference; use "
                                            "train/prefill/decode for MPSGraph");
            Rng rng(42);
            int m = args.integer("m", sequence * batch), n = args.integer("n", c.d_ff),
                k = args.integer("k", c.d_model);
            if (m < 1 || n < 1 || k < 1 ||
                static_cast<std::size_t>(m) * static_cast<std::size_t>(n) *
                        static_cast<std::size_t>(k) >
                    1000000000)
                throw std::invalid_argument("Invalid or oversized matrix benchmark");
            Matrix a(m, k), b(k, n), output, expected;
            for (auto& x : a.values)
                x = rng.normal();
            for (auto& x : b.values)
                x = rng.normal();
            if (kind == "matmul") {
                ops::matmul_naive(a, b, expected);
                ops::matmul(a, b, output);
                error = difference(expected.values, output.values);
                auto baseline = time_it(
                    1, std::min(repetitions, 5), [] {}, [&] { ops::matmul_naive(a, b, expected); });
                timing = time_it(warmup, repetitions, [] {}, [&] { ops::matmul(a, b, output); });
                report["baseline_median_ms"] = baseline.median;
                report["speedup"] = baseline.median / std::max(timing.median, 1e-9);
                report["dimensions"] = Json::Array{m, n, k};
                report["gflops"] = 2.0 * m * n * k / (timing.median * 1e6);
            } else if (kind == "norm") {
                Matrix norm(1, k);
                std::fill(norm.values.begin(), norm.values.end(), 1);
                timing = time_it(
                    warmup, repetitions, [] {},
                    [&] { ops::rmsnorm(a, norm, output, c.norm_epsilon); });
                double sum = 0;
                for (int j = 0; j < k; ++j)
                    sum += a(0, j) * a(0, j);
                double inv = 1 / std::sqrt(sum / k + c.norm_epsilon);
                for (int j = 0; j < k; ++j)
                    error = std::max(error, std::abs(output(0, j) - a(0, j) * inv));
            } else if (kind == "rope") {
                Matrix input(batch * sequence, c.n_heads * c.qk_dim());
                for (auto& x : input.values)
                    x = rng.normal();
                Matrix rotated = input;
                timing = time_it(
                    warmup, repetitions, [&] { rotated = input; },
                    [&] {
                        ops::rope(rotated, batch, sequence, c.n_heads, c.qk_dim(), c.head_dim_nope,
                                  c.head_dim_rope, c.rope_theta);
                    });
                ops::rope(rotated, batch, sequence, c.n_heads, c.qk_dim(), c.head_dim_nope,
                          c.head_dim_rope, c.rope_theta, 0, true);
                error = difference(input.values, rotated.values);
            } else {
                Matrix q(batch * sequence, c.n_heads * c.qk_dim()),
                    key = q, value(batch * sequence, c.attn_width()), probabilities;
                for (auto& x : q.values)
                    x = rng.normal();
                for (auto& x : key.values)
                    x = rng.normal();
                for (auto& x : value.values)
                    x = rng.normal();
                timing = time_it(
                    warmup, repetitions, [] {},
                    [&] {
                        ops::attention(q, key, value, probabilities, output, batch, sequence,
                                       c.n_heads, c.qk_dim(), c.head_dim_value);
                    });
                for (int row = 0; row < probabilities.rows; ++row) {
                    double sum = 0;
                    for (int j = 0; j < probabilities.cols; ++j)
                        sum += probabilities(row, j);
                    error = std::max(error, std::abs(sum - 1));
                }
            }
        } else {
            if (kind != "train" && kind != "prefill" && kind != "decode")
                throw std::invalid_argument("Unknown benchmark kind");
            auto model = take(Model::create(c, 42, kind == "train"));
            Optimizer optimizer(optimizer_config);
            if (kind == "train") {
                check(optimizer.initialize(*model));
                report["optimizer"] = optimizer.config().json();
                report["muon_layout"] = backend == "mps" ? execution_options.muon_layout : "cpu";
            }
            if (args.has("comparison-data")) {
                comparison = std::make_unique<ComparisonData>(args.required("comparison-data"));
                comparison->append(*model, "weight");
                comparison_metadata = Json::Object{{"config", c.json()},
                                                   {"optimizer", optimizer.config().json()},
                                                   {"learning_rate", 0.0003},
                                                   {"workload", report}};
            }
            auto execution = take(make_execution(*model, kind == "train" ? &optimizer : nullptr,
                                                 backend, precision, execution_options));
            report["device"] = execution->device_name();
            std::vector<std::uint16_t> prompt(static_cast<std::size_t>(sequence));
            for (std::size_t i = 0; i < prompt.size(); ++i)
                prompt[i] = static_cast<std::uint16_t>(i % static_cast<std::size_t>(c.vocab_size));
            if (kind == "train") {
                Batch b;
                b.batch = batch;
                b.sequence = sequence;
                b.input.resize(static_cast<std::size_t>(batch * sequence));
                b.target = b.input;
                b.mask.resize(b.input.size(), 1);
                for (std::size_t i = 0; i < b.input.size(); ++i) {
                    b.input[i] =
                        static_cast<std::uint16_t>(i % static_cast<std::size_t>(c.vocab_size));
                    b.target[i] = static_cast<std::uint16_t>(
                        (i + 1) % static_cast<std::size_t>(c.vocab_size));
                }
                auto coefficients = take(b.coefficients());
                double cpu_loss = 0;
                if (!args.has("skip-cpu-check")) {
                    auto expected = take(model->forward(b.input, batch, sequence));
                    cpu_loss = take(ops::weighted_nll(*expected, b.target, coefficients));
                }
                check(execution->zero_grad());
                double device_loss = take(execution->accumulate(b, coefficients));
                error = std::abs(device_loss - cpu_loss);
                report["initial_loss"] = device_loss;
                report["cpu_correctness_checked"] = !args.has("skip-cpu-check");
                if (comparison) {
                    check(execution->synchronize_to_host());
                    comparison->append(*model, "gradient", true);
                    comparison_metadata["initial_loss"] = device_loss;
                    Json::Array ids, targets;
                    for (auto id : b.input)
                        ids.emplace_back(static_cast<int>(id));
                    for (auto id : b.target)
                        targets.emplace_back(static_cast<int>(id));
                    comparison_metadata["input"] = std::move(ids);
                    comparison_metadata["targets"] = std::move(targets);
                }
                for (auto& coefficient : coefficients)
                    coefficient /= static_cast<float>(accumulation);
                std::vector<GradientBatch> jobs(static_cast<std::size_t>(accumulation),
                                                GradientBatch{&b, coefficients});
                auto before = execution->stats();
                StepResult last;
                bool time_reset = args.has("time-zero-grad");
                timing = time_it(
                    warmup, repetitions,
                    [&] {
                        if (!time_reset)
                            check(execution->zero_grad());
                    },
                    [&] {
                        if (time_reset)
                            check(execution->zero_grad());
                        last = take(execution->step(jobs, 0.0003F));
                    });
                report["gradient_reset_timed"] = time_reset;
                auto count = static_cast<double>(warmup + repetitions);
                report["host_waits_per_update"] =
                    static_cast<double>(execution->stats().host_waits - before.host_waits) / count;
                report["metal_kernel_submissions_per_update"] =
                    static_cast<double>(execution->stats().metal_kernel_submissions -
                                        before.metal_kernel_submissions) /
                    count;
                report["graph_submissions_per_update"] =
                    static_cast<double>(execution->stats().graph_submissions -
                                        before.graph_submissions) /
                    count;
                report["tokens_per_second"] =
                    static_cast<double>(batch * sequence) * accumulation * 1000 / timing.median;
                report["last_loss"] = last.loss;
                report["last_gradient_norm"] = last.gradient_norm;
                if (comparison) {
                    check(execution->synchronize_to_host());
                    comparison->append(*model, "final");
                }
            } else {
                auto expected = *take(model->forward(prompt, 1, sequence));
                auto prefill = take(execution->prefill(prompt));
                error = difference(
                    prefill, std::span{expected.data() +
                                           static_cast<std::size_t>((sequence - 1) * c.vocab_size),
                                       static_cast<std::size_t>(c.vocab_size)});
                if (kind == "prefill")
                    timing = time_it(
                        warmup, repetitions, [] {}, [&] { take(execution->prefill(prompt)); });
                else {
                    auto step = take(execution->decode(1));
                    auto full_ids = prompt;
                    full_ids.push_back(1);
                    auto full = take(model->forward(full_ids, 1, sequence + 1));
                    error = std::max(
                        error,
                        difference(step, std::span{full->data() + static_cast<std::size_t>(
                                                                      sequence * c.vocab_size),
                                                   static_cast<std::size_t>(c.vocab_size)}));
                    timing = time_it(
                        warmup, repetitions, [&] { take(execution->prefill(prompt)); },
                        [&] { take(execution->decode(1)); });
                    if (backend == "cpu") {
                        CpuDecoder fallback(*model);
                        fallback.set_absorbed(false);
                        auto baseline = time_it(
                            warmup, repetitions, [&] { take(fallback.prefill(prompt)); },
                            [&] { take(fallback.decode(1)); });
                        report["baseline_median_ms"] = baseline.median;
                        report["speedup"] = baseline.median / timing.median;
                    }
                }
                report["tokens_per_second"] =
                    (kind == "prefill" ? sequence : 1) * 1000.0 / timing.median;
                report["cache_bytes"] = static_cast<std::size_t>(c.n_layers) *
                                        static_cast<std::size_t>(c.max_sequence_length) *
                                        static_cast<std::size_t>(c.kv_width()) *
                                        (precision == "fp16" ? 2 : 4);
            }
            report["attention_effective"] = execution->attention_name();
        }
        struct rusage usage{};
        ::getrusage(RUSAGE_SELF, &usage);
        report["median_ms"] = timing.median;
        report["p95_ms"] = timing.p95;
        report["correctness_error"] = args.has("skip-cpu-check") ? Json() : Json(error);
        report["peak_rss_bytes"] = static_cast<std::size_t>(usage.ru_maxrss);
        if (comparison) {
            comparison_metadata["native_result"] = report;
            comparison->finish(std::move(comparison_metadata));
        }
        if (args.has("output"))
            check(write_json(args.get("output"), report));
        std::puts(report.dump().c_str());
        return 0;
    });
}
