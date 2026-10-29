#include "inference_sessions.hpp"
#include "mini_mla/checkpoint.hpp"
#include "mini_mla/execution.hpp"
#include "mini_mla/testing.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <unistd.h>

using namespace mini_mla;
static ModelConfig config() {
    ModelConfig c;
    c.vocab_size = 32;
    c.n_layers = 2;
    c.d_model = 8;
    c.n_heads = 2;
    c.head_dim_nope = 2;
    c.head_dim_rope = 2;
    c.head_dim_value = 3;
    c.q_lora_rank = 4;
    c.kv_lora_rank = 3;
    c.d_ff = 12;
    c.max_sequence_length = 128;
    return c;
}
static Batch batch() {
    return Batch{2, 3, {1, 3, 5, 2, 4, 6}, {3, 5, 7, 4, 6, 8}, {0, 1, 1, 1, 1, 1}};
}
static void require_result(const auto& result) {
    if (!result)
        std::fprintf(stderr, "%s\n", result.error().message.c_str());
}
MINI_MLA_TEST(mps_inference_forks_share_graphs_with_independent_request_caches) {
    MINI_MLA_CHECK(check_inference_sessions(config(), "mps", "fp32"));
    MINI_MLA_CHECK(check_inference_sessions(config(), "mps", "fp16"));
}
MINI_MLA_TEST(mps_lazy_zero_and_empty_updates_match_cpu) {
    auto a = take(Model::create(config(), 7)), b = take(Model::create(config(), 7));
    Optimizer ao, bo;
    MINI_MLA_REQUIRE(ao.initialize(*a));
    MINI_MLA_REQUIRE(bo.initialize(*b));
    auto cpu = take(make_execution(*a, &ao, "cpu"));
    auto gpu = take(make_execution(*b, &bo, "mps"));
    auto data = batch();
    auto coefficients = take(data.coefficients());
    for (int step = 0; step < 3; ++step) {
        MINI_MLA_REQUIRE(cpu->zero_grad());
        MINI_MLA_REQUIRE(gpu->zero_grad());
        if (step == 1) {
            MINI_MLA_REQUIRE(cpu->accumulate(data, coefficients));
            MINI_MLA_REQUIRE(gpu->accumulate(data, coefficients));
        }
        MINI_MLA_REQUIRE(cpu->zero_grad());
        MINI_MLA_REQUIRE(gpu->zero_grad());
        MINI_MLA_REQUIRE(gpu->synchronize_to_host());
        for (const auto& p : b->parameters())
            for (float value : p.gradient.values)
                MINI_MLA_CHECK(value == 0);
        MINI_MLA_REQUIRE(cpu->update(0.001F));
        MINI_MLA_REQUIRE(gpu->update(0.001F));
        MINI_MLA_REQUIRE(gpu->synchronize_to_host());
        for (std::size_t i = 0; i < a->parameters().size(); ++i)
            for (std::size_t j = 0; j < a->parameters()[i].value.size(); ++j)
                MINI_MLA_CHECK(std::abs(a->parameters()[i].value.values[j] -
                                        b->parameters()[i].value.values[j]) < 1e-6F);
    }
    MINI_MLA_CHECK(!gpu->fork_inference());
}
MINI_MLA_TEST(mps_forward_gradients_and_adamw_match_cpu) {
    auto cpu_model = Model::create(config(), 5);
    auto gpu_model = Model::create(config(), 5);
    MINI_MLA_REQUIRE(cpu_model && gpu_model);
    OptimizerConfig oc;
    oc.clip_norm = 0.5F;
    Optimizer cpu_optimizer(oc), gpu_optimizer(oc);
    MINI_MLA_REQUIRE(cpu_optimizer.initialize(*cpu_model.value()));
    MINI_MLA_REQUIRE(gpu_optimizer.initialize(*gpu_model.value()));
    auto cpu = make_execution(*cpu_model.value(), &cpu_optimizer, "cpu");
    auto gpu = make_execution(*gpu_model.value(), &gpu_optimizer, "mps");
    require_result(gpu);
    MINI_MLA_REQUIRE(cpu && gpu);
    auto b = batch();
    auto expected = cpu.value()->forward(b.input, 2, 3);
    std::fprintf(stderr, "Checking GPU forward\n");
    auto actual = gpu.value()->forward(b.input, 2, 3);
    require_result(actual);
    MINI_MLA_REQUIRE(expected && actual);
    for (std::size_t i = 0; i < expected.value()->size(); ++i)
        MINI_MLA_CHECK(std::abs(expected.value()->values[i] - actual.value()->values[i]) < 1e-4F);
    for (int step = 0; step < 3; ++step) {
        MINI_MLA_REQUIRE(cpu.value()->zero_grad());
        MINI_MLA_REQUIRE(gpu.value()->zero_grad());
        auto coefficients = b.coefficients(0.5F);
        MINI_MLA_REQUIRE(coefficients);
        std::fprintf(stderr, "Checking GPU backward step %d\n", step);
        for (int acc = 0; acc < 2; ++acc) {
            auto loss_cpu = cpu.value()->accumulate(b, coefficients.value());
            auto loss_gpu = gpu.value()->accumulate(b, coefficients.value());
            require_result(loss_gpu);
            MINI_MLA_REQUIRE(loss_cpu && loss_gpu);
            MINI_MLA_CHECK(std::abs(loss_cpu.value() - loss_gpu.value()) < 1e-4);
        }
        MINI_MLA_REQUIRE(gpu.value()->synchronize_to_host());
        for (std::size_t p = 0; p < cpu_model.value()->parameters().size(); ++p) {
            const auto& cg = cpu_model.value()->parameters()[p].gradient;
            const auto& gg = gpu_model.value()->parameters()[p].gradient;
            for (std::size_t i = 0; i < cg.size(); ++i)
                MINI_MLA_CHECK_MSG(std::abs(cg.values[i] - gg.values[i]) <
                                       0.0005F + 0.0005F * std::abs(cg.values[i]),
                                   cpu_model.value()->parameters()[p].name);
        }
        auto norm_cpu = cpu.value()->update(0.001F);
        auto norm_gpu = gpu.value()->update(0.001F);
        require_result(norm_gpu);
        MINI_MLA_REQUIRE(norm_cpu && norm_gpu);
        MINI_MLA_CHECK(std::abs(norm_cpu.value() - norm_gpu.value()) < 0.001);
        MINI_MLA_REQUIRE(gpu.value()->synchronize_to_host());
        for (std::size_t p = 0; p < cpu_model.value()->parameters().size(); ++p)
            for (std::size_t i = 0; i < cpu_model.value()->parameters()[p].value.size(); ++i)
                MINI_MLA_CHECK(std::abs(cpu_model.value()->parameters()[p].value.values[i] -
                                        gpu_model.value()->parameters()[p].value.values[i]) <
                               2e-5F);
    }
}
MINI_MLA_TEST(mps_cached_decode_and_fp16_match_full_recomputation) {
    auto made = Model::create(config(), 9, false);
    MINI_MLA_REQUIRE(made);
    auto& m = *made.value();
    for (auto precision : {"fp32", "fp16"}) {
        auto gpu = make_execution(m, nullptr, "mps", precision);
        require_result(gpu);
        MINI_MLA_REQUIRE(gpu);
        std::vector<std::uint16_t> tokens{1, 5, 8};
        auto prefill = gpu.value()->prefill(tokens);
        require_result(prefill);
        MINI_MLA_REQUIRE(prefill);
        auto full = m.forward(tokens, 1, 3);
        MINI_MLA_REQUIRE(full);
        float tolerance = std::string_view(precision) == "fp32" ? 1e-4F : 0.015F;
        for (int j = 0; j < config().vocab_size; ++j)
            MINI_MLA_CHECK(std::abs(prefill.value()[static_cast<std::size_t>(j)] -
                                    (*full.value())(2, j)) < tolerance);
        for (int i = 0; i < 14; ++i) {
            auto id = static_cast<std::uint16_t>((i * 3 + 2) % 32);
            tokens.push_back(id);
            auto step = gpu.value()->decode(id);
            require_result(step);
            MINI_MLA_REQUIRE(step);
            full = m.forward(tokens, 1, static_cast<int>(tokens.size()));
            MINI_MLA_REQUIRE(full);
            for (int j = 0; j < config().vocab_size; ++j)
                MINI_MLA_CHECK(std::abs(step.value()[static_cast<std::size_t>(j)] -
                                        (*full.value())(static_cast<int>(tokens.size() - 1), j)) <
                               tolerance);
        }
        tokens = {2, 3};
        prefill = gpu.value()->prefill(tokens);
        MINI_MLA_REQUIRE(prefill);
        full = m.forward(tokens, 1, 2);
        MINI_MLA_REQUIRE(full);
        for (int j = 0; j < config().vocab_size; ++j)
            MINI_MLA_CHECK(std::abs(prefill.value()[static_cast<std::size_t>(j)] -
                                    (*full.value())(1, j)) < tolerance);
    }
}
MINI_MLA_TEST(mps_muon_matches_cpu_and_half_training_learns) {
    auto cpu_model = Model::create(config(), 17), gpu_model = Model::create(config(), 17);
    MINI_MLA_REQUIRE(cpu_model && gpu_model);
    OptimizerConfig oc;
    oc.kind = "hybrid_muon_adamw";
    Optimizer co(oc), go(oc);
    MINI_MLA_REQUIRE(co.initialize(*cpu_model.value()));
    MINI_MLA_REQUIRE(go.initialize(*gpu_model.value()));
    auto gpu = make_execution(*gpu_model.value(), &go, "mps");
    MINI_MLA_REQUIRE(gpu);
    auto b = batch();
    auto coefficients = b.coefficients();
    MINI_MLA_REQUIRE(coefficients);
    auto cpu = make_execution(*cpu_model.value(), &co, "cpu");
    MINI_MLA_REQUIRE(cpu);
    MINI_MLA_REQUIRE(cpu.value()->zero_grad());
    MINI_MLA_REQUIRE(gpu.value()->zero_grad());
    MINI_MLA_REQUIRE(cpu.value()->accumulate(b, coefficients.value()));
    MINI_MLA_REQUIRE(gpu.value()->accumulate(b, coefficients.value()));
    MINI_MLA_REQUIRE(cpu.value()->update(0.0003F));
    MINI_MLA_REQUIRE(gpu.value()->update(0.0003F));
    MINI_MLA_REQUIRE(gpu.value()->synchronize_to_host());
    for (std::size_t i = 0; i < cpu_model.value()->parameters().size(); ++i)
        for (std::size_t j = 0; j < cpu_model.value()->parameters()[i].value.size(); ++j)
            MINI_MLA_CHECK(std::abs(cpu_model.value()->parameters()[i].value.values[j] -
                                    gpu_model.value()->parameters()[i].value.values[j]) < 2e-5F);
    auto model = Model::create(config(), 3);
    MINI_MLA_REQUIRE(model);
    Optimizer optimizer;
    MINI_MLA_REQUIRE(optimizer.initialize(*model.value()));
    auto half = make_execution(*model.value(), &optimizer, "mps", "fp16", {"flash", 4});
    MINI_MLA_REQUIRE(half);
    double initial = 0, final = 0;
    for (int i = 0; i < 100; ++i) {
        MINI_MLA_REQUIRE(half.value()->zero_grad());
        auto loss = half.value()->accumulate(b, coefficients.value());
        require_result(loss);
        MINI_MLA_REQUIRE(loss);
        if (i == 0)
            initial = loss.value();
        final = loss.value();
        MINI_MLA_REQUIRE(half.value()->update(0.008F));
    }
    MINI_MLA_REQUIRE(half.value()->synchronize_to_host());
    MINI_MLA_CHECK_MSG(final < initial * 0.2,
                       "initial=" + std::to_string(initial) + " final=" + std::to_string(final));
}
MINI_MLA_TEST(mps_muon_zero_gradients_keep_orthogonalization_finite) {
    auto a = take(Model::create(config(), 23)), b = take(Model::create(config(), 23));
    OptimizerConfig oc;
    oc.kind = "hybrid_muon_adamw";
    oc.momentum = 0;
    oc.nesterov = false;
    oc.ns_steps = 1;
    Optimizer ao(oc), bo(oc);
    MINI_MLA_REQUIRE(ao.initialize(*a));
    MINI_MLA_REQUIRE(bo.initialize(*b));
    auto cpu = take(make_execution(*a, &ao, "cpu"));
    auto gpu = take(make_execution(*b, &bo, "mps"));
    for (int step = 0; step < 2; ++step) {
        MINI_MLA_REQUIRE(cpu->zero_grad());
        MINI_MLA_REQUIRE(gpu->zero_grad());
        auto expected = cpu->update(0.001F), actual = gpu->update(0.001F);
        MINI_MLA_REQUIRE(expected && actual);
        MINI_MLA_CHECK(expected.value() == 0 && actual.value() == 0);
        MINI_MLA_REQUIRE(gpu->synchronize_to_host());
        for (std::size_t i = 0; i < a->parameters().size(); ++i) {
            for (std::size_t j = 0; j < a->parameters()[i].value.size(); ++j) {
                float value = b->parameters()[i].value.values[j];
                MINI_MLA_CHECK(std::isfinite(value));
                MINI_MLA_CHECK(std::abs(value - a->parameters()[i].value.values[j]) < 1e-7F);
                MINI_MLA_CHECK(bo.states()[i].first.values[j] == 0);
            }
            for (float value : bo.states()[i].second.values)
                MINI_MLA_CHECK(value == 0);
        }
    }
}
MINI_MLA_TEST(mps_batched_muon_matches_individual_and_checkpoint_resume) {
    struct Temporary {
        std::filesystem::path path =
            std::filesystem::temp_directory_path() / ("mla-muon-" + std::to_string(getpid()));
        ~Temporary() {
            std::error_code error;
            std::filesystem::remove_all(path, error);
        }
    } temp;
    auto c = config();
    c.vocab_size = 264;
    c.n_layers = 3;
    auto data = batch();
    auto coefficients = take(data.coefficients(0.5F));
    auto negative = coefficients;
    for (auto& value : negative)
        value *= -0.3F;
    std::array<GradientBatch, 2> jobs{{{&data, coefficients}, {&data, negative}}};
    for (auto precision : {"fp32", "fp16"}) {
        for (bool nesterov : {true, false}) {
            OptimizerConfig oc;
            oc.kind = "hybrid_muon_adamw";
            oc.nesterov = nesterov;
            oc.ns_steps = nesterov ? 5 : 3;
            oc.momentum = nesterov ? 0.95F : 0.8F;
            oc.clip_norm = nesterov ? 0.2F : 0;
            auto cpu_model = take(Model::create(c, 43));
            auto individual_model = take(Model::create(c, 43));
            auto batched_model = take(Model::create(c, 43));
            Optimizer co(oc), io(oc), bo(oc);
            MINI_MLA_REQUIRE(co.initialize(*cpu_model));
            MINI_MLA_REQUIRE(io.initialize(*individual_model));
            MINI_MLA_REQUIRE(bo.initialize(*batched_model));
            Rng random(73);
            for (auto& state : co.states()) {
                for (auto& value : state.first.values)
                    value = random.normal() * 0.001F;
                for (auto& value : state.second.values)
                    value = std::abs(random.normal()) * 0.00001F;
            }
            io.states() = co.states();
            bo.states() = co.states();
            auto cpu = take(make_execution(*cpu_model, &co, "cpu"));
            auto individual = take(make_execution(*individual_model, &io, "mps", precision,
                                                  {"reference", 4, "individual", "chain"}));
            auto batched = take(
                make_execution(*batched_model, &bo, "mps", precision, {"reference", 4, "batched"}));
            LoadedCheckpoint resumed;
            std::unique_ptr<Execution> replay;
            for (int step = 0; step < 5; ++step) {
                for (auto execution : {cpu.get(), individual.get(), batched.get(), replay.get()}) {
                    if (!execution)
                        continue;
                    MINI_MLA_REQUIRE(execution->zero_grad());
                    if (step == 2) {
                        auto norm = execution->update(0.001F);
                        MINI_MLA_REQUIRE(norm);
                        MINI_MLA_CHECK(norm.value() == 0);
                    } else {
                        auto result = execution->step(jobs, 0.001F);
                        require_result(result);
                        MINI_MLA_REQUIRE(result);
                        MINI_MLA_CHECK(std::isfinite(result.value().loss));
                    }
                    MINI_MLA_REQUIRE(execution->synchronize_to_host());
                }
                for (std::size_t i = 0; i < cpu_model->parameters().size(); ++i) {
                    auto check_values = [&](std::span<const float> a, std::span<const float> b,
                                            float tolerance) {
                        MINI_MLA_REQUIRE(a.size() == b.size());
                        for (std::size_t j = 0; j < a.size(); ++j)
                            MINI_MLA_CHECK_MSG(std::isfinite(a[j]) && std::isfinite(b[j]) &&
                                                   std::abs(a[j] - b[j]) < tolerance,
                                               cpu_model->parameters()[i].name);
                    };
                    check_values(individual_model->parameters()[i].value.values,
                                 batched_model->parameters()[i].value.values, 2e-5F);
                    check_values(io.states()[i].first.values, bo.states()[i].first.values, 2e-5F);
                    check_values(io.states()[i].second.values, bo.states()[i].second.values, 2e-5F);
                    if (std::string_view(precision) == "fp32") {
                        check_values(cpu_model->parameters()[i].value.values,
                                     batched_model->parameters()[i].value.values, 5e-5F);
                        check_values(co.states()[i].first.values, bo.states()[i].first.values,
                                     5e-5F);
                        check_values(co.states()[i].second.values, bo.states()[i].second.values,
                                     5e-5F);
                    }
                    if (replay) {
                        MINI_MLA_CHECK(resumed.model->parameters()[i].value.values ==
                                       batched_model->parameters()[i].value.values);
                        MINI_MLA_CHECK(resumed.optimizer->states()[i].first.values ==
                                       bo.states()[i].first.values);
                        MINI_MLA_CHECK(resumed.optimizer->states()[i].second.values ==
                                       bo.states()[i].second.values);
                    }
                }
                MINI_MLA_CHECK(bo.steps() == static_cast<std::uint64_t>(step + 1));
                if (replay)
                    MINI_MLA_CHECK(resumed.optimizer->steps() == bo.steps());
                if (step == 2) {
                    auto path = temp.path / (std::string(precision) + (nesterov ? "-on" : "-off"));
                    MINI_MLA_REQUIRE(save_checkpoint(path, *batched_model, bo, Json::Object{},
                                                     "examples/tokenizer.json"));
                    resumed = take(load_checkpoint(path, true));
                    MINI_MLA_CHECK(resumed.optimizer->config().json().dump() == oc.json().dump());
                    replay = take(make_execution(*resumed.model, resumed.optimizer.get(), "mps",
                                                 precision, {"reference", 4, "batched"}));
                }
            }
        }
    }
}
MINI_MLA_TEST(flash_matches_cpu_and_submission_windows_preserve_every_gradient) {
    for (auto attention : {"reference", "flash"}) {
        auto cpu_model = Model::create(config(), 31);
        auto flash_model = Model::create(config(), 31);
        auto serial_model = Model::create(config(), 31);
        MINI_MLA_REQUIRE(cpu_model && flash_model && serial_model);
        Optimizer co, fo, so;
        MINI_MLA_REQUIRE(co.initialize(*cpu_model.value()));
        MINI_MLA_REQUIRE(fo.initialize(*flash_model.value()));
        MINI_MLA_REQUIRE(so.initialize(*serial_model.value()));
        auto cpu = make_execution(*cpu_model.value(), &co, "cpu");
        auto flash = make_execution(*flash_model.value(), &fo, "mps", "fp32", {attention, 4});
        auto serial = make_execution(*serial_model.value(), &so, "mps", "fp32", {attention, 1});
        require_result(flash);
        MINI_MLA_REQUIRE(cpu && flash && serial);
        std::vector<Batch> batches;
        for (int sequence : {1, 33, 65}) {
            Batch b;
            b.batch = 2;
            b.sequence = sequence;
            b.input.resize(static_cast<std::size_t>(2 * sequence));
            b.target = b.input;
            b.mask.resize(b.input.size());
            for (std::size_t i = 0; i < b.input.size(); ++i) {
                b.input[i] = static_cast<std::uint16_t>((i * 5 + 1) % 32);
                b.target[i] = static_cast<std::uint16_t>((i * 5 + 6) % 32);
                b.mask[i] = i % 3 == 0 ? 0 : 1;
            }
            auto expected = cpu.value()->forward(b.input, 2, sequence);
            auto actual = flash.value()->forward(b.input, 2, sequence);
            require_result(actual);
            MINI_MLA_REQUIRE(expected && actual);
            for (std::size_t i = 0; i < expected.value()->size(); ++i)
                MINI_MLA_CHECK(std::abs(expected.value()->values[i] - actual.value()->values[i]) <
                               1e-4F);
            batches.push_back(std::move(b));
        }
        std::vector<std::vector<float>> coefficients;
        for (int i = 0; i < 9; ++i) {
            auto c = batches[static_cast<std::size_t>(i % 3)].mask;
            for (auto& x : c)
                x *= (i % 2 == 0 ? 1.0F : -0.3F) / 300;
            coefficients.push_back(std::move(c));
        }
        std::vector<GradientBatch> jobs;
        for (std::size_t i = 0; i < coefficients.size(); ++i)
            jobs.push_back({&batches[i % 3], coefficients[i]});
        MINI_MLA_REQUIRE(cpu.value()->zero_grad());
        MINI_MLA_REQUIRE(flash.value()->zero_grad());
        MINI_MLA_REQUIRE(serial.value()->zero_grad());
        auto before = flash.value()->stats();
        auto expected = cpu.value()->step(jobs, 0.001F);
        auto actual = flash.value()->step(jobs, 0.001F);
        auto sequential = serial.value()->step(jobs, 0.001F);
        require_result(actual);
        require_result(sequential);
        MINI_MLA_REQUIRE(expected && actual && sequential);
        MINI_MLA_CHECK(flash.value()->stats().host_waits - before.host_waits == 3);
        MINI_MLA_CHECK(serial.value()->stats().host_waits == 10);
        MINI_MLA_CHECK(std::abs(expected.value().loss - actual.value().loss) < 1e-4);
        MINI_MLA_CHECK(std::abs(expected.value().gradient_norm - actual.value().gradient_norm) <
                       1e-4);
        MINI_MLA_CHECK(std::abs(actual.value().loss - sequential.value().loss) < 1e-5);
        MINI_MLA_REQUIRE(flash.value()->synchronize_to_host());
        MINI_MLA_REQUIRE(serial.value()->synchronize_to_host());
        for (std::size_t p = 0; p < cpu_model.value()->parameters().size(); ++p) {
            const auto& cg = cpu_model.value()->parameters()[p].gradient;
            const auto& fg = flash_model.value()->parameters()[p].gradient;
            for (std::size_t i = 0; i < cg.size(); ++i) {
                MINI_MLA_CHECK_MSG(std::abs(cg.values[i] - fg.values[i]) <
                                       0.0005F + 0.0005F * std::abs(cg.values[i]),
                                   cpu_model.value()->parameters()[p].name);
                MINI_MLA_CHECK(std::abs(cpu_model.value()->parameters()[p].value.values[i] -
                                        flash_model.value()->parameters()[p].value.values[i]) <
                               2e-5F);
                MINI_MLA_CHECK(std::abs(flash_model.value()->parameters()[p].value.values[i] -
                                        serial_model.value()->parameters()[p].value.values[i]) <
                               2e-5F);
                MINI_MLA_CHECK(std::abs(fo.states()[p].first.values[i] -
                                        so.states()[p].first.values[i]) < 2e-5F);
                MINI_MLA_CHECK(std::abs(fo.states()[p].second.values[i] -
                                        so.states()[p].second.values[i]) < 2e-5F);
            }
        }
    }
}
MINI_MLA_TEST(flash_matrix_heads_match_cpu_in_both_precisions) {
    for (int width : {16, 64, 128}) {
        for (auto precision : {"fp32", "fp16"}) {
            auto c = config();
            c.d_model = 32;
            c.head_dim_nope = width - 4;
            c.head_dim_rope = 4;
            c.head_dim_value = width == 128 ? 96 : width;
            c.q_lora_rank = 16;
            c.kv_lora_rank = 8;
            c.d_ff = 48;
            auto cpu_model = Model::create(c, 29);
            auto gpu_model = Model::create(c, 29);
            MINI_MLA_REQUIRE(cpu_model && gpu_model);
            OptimizerConfig oc;
            if (width == 16)
                oc.kind = "hybrid_muon_adamw";
            Optimizer co(oc), go(oc);
            MINI_MLA_REQUIRE(co.initialize(*cpu_model.value()));
            MINI_MLA_REQUIRE(go.initialize(*gpu_model.value()));
            auto cpu = make_execution(*cpu_model.value(), &co, "cpu");
            auto gpu = make_execution(*gpu_model.value(), &go, "mps", precision, {"flash", 4});
            require_result(gpu);
            MINI_MLA_REQUIRE(cpu && gpu);
            Batch b;
            b.batch = 2;
            b.sequence = 33;
            b.input.resize(66);
            b.target.resize(66);
            b.mask.resize(66, 1);
            for (std::size_t i = 0; i < b.input.size(); ++i) {
                b.input[i] = static_cast<std::uint16_t>((i * 7 + 1) % 32);
                b.target[i] = static_cast<std::uint16_t>((i * 7 + 8) % 32);
            }
            auto expected = cpu.value()->forward(b.input, 2, 33);
            auto actual = gpu.value()->forward(b.input, 2, 33);
            require_result(actual);
            MINI_MLA_REQUIRE(expected && actual);
            bool half = std::string_view(precision) == "fp16";
            for (std::size_t i = 0; i < expected.value()->size(); ++i)
                MINI_MLA_CHECK(std::abs(expected.value()->values[i] - actual.value()->values[i]) <
                               (half ? 0.01F : 1e-4F));
            auto coefficients = b.coefficients(0.5F);
            MINI_MLA_REQUIRE(coefficients);
            auto negative = coefficients.value();
            for (auto& x : negative)
                x *= -0.3F;
            std::array<GradientBatch, 2> jobs{{{&b, coefficients.value()}, {&b, negative}}};
            MINI_MLA_REQUIRE(cpu.value()->zero_grad());
            MINI_MLA_REQUIRE(gpu.value()->zero_grad());
            auto expected_step = cpu.value()->step(jobs, 0.0003F);
            auto actual_step = gpu.value()->step(jobs, 0.0003F);
            require_result(actual_step);
            MINI_MLA_REQUIRE(expected_step && actual_step);
            MINI_MLA_CHECK(std::abs(expected_step.value().loss - actual_step.value().loss) <
                           (half ? 0.003 : 1e-4));
            MINI_MLA_REQUIRE(gpu.value()->synchronize_to_host());
            for (std::size_t p = 0; p < cpu_model.value()->parameters().size(); ++p) {
                const auto& cp = cpu_model.value()->parameters()[p];
                const auto& gp = gpu_model.value()->parameters()[p];
                for (std::size_t i = 0; i < cp.value.size(); ++i) {
                    MINI_MLA_CHECK_MSG(std::abs(cp.gradient.values[i] - gp.gradient.values[i]) <
                                           (half ? 0.0008F : 0.0005F) +
                                               (half ? 0.01F : 0.0005F) *
                                                   std::abs(cp.gradient.values[i]),
                                       cp.name);
                    MINI_MLA_CHECK_MSG(std::abs(cp.value.values[i] - gp.value.values[i]) <
                                           (half ? 0.0006F : 2e-5F),
                                       cp.name);
                }
            }
        }
    }
}
MINI_MLA_TEST(mps_invalid_late_job_is_rejected_before_submission) {
    for (auto attention : {"reference", "flash"}) {
        auto model = Model::create(config(), 9);
        MINI_MLA_REQUIRE(model);
        Optimizer optimizer;
        MINI_MLA_REQUIRE(optimizer.initialize(*model.value()));
        auto gpu = make_execution(*model.value(), &optimizer, "mps", "fp32", {attention, 4});
        MINI_MLA_REQUIRE(gpu);
        auto b = batch();
        auto coefficients = b.coefficients();
        MINI_MLA_REQUIRE(coefficients);
        auto bad = coefficients.value();
        bad.back() = std::numeric_limits<float>::quiet_NaN();
        std::array<GradientBatch, 2> jobs{{{&b, coefficients.value()}, {&b, bad}}};
        auto weights = model.value()->parameters().front().value.values;
        MINI_MLA_CHECK(!gpu.value()->step(jobs, 0.001F));
        MINI_MLA_CHECK(gpu.value()->stats().graph_submissions == 0);
        MINI_MLA_CHECK(gpu.value()->stats().host_waits == 0);
        MINI_MLA_REQUIRE(gpu.value()->synchronize_to_host());
        MINI_MLA_CHECK(optimizer.steps() == 0);
        MINI_MLA_CHECK(model.value()->parameters().front().value.values == weights);
        for (const auto& p : model.value()->parameters())
            for (float x : p.gradient.values)
                MINI_MLA_CHECK(x == 0);
    }
}
// Builds `count` jobs over one repeated batch with distinct signed coefficients.
static std::vector<std::vector<float>> job_coefficients(const Batch& b, int count) {
    std::vector<std::vector<float>> all;
    for (int i = 0; i < count; ++i) {
        auto c = b.mask;
        for (auto& x : c)
            x *= (i % 3 == 1 ? -0.3F : 1.0F + 0.1F * static_cast<float>(i)) / 40;
        all.push_back(std::move(c));
    }
    return all;
}
static Batch wide_batch(int sequence) {
    Batch b;
    b.batch = 2;
    b.sequence = sequence;
    b.input.resize(static_cast<std::size_t>(2 * sequence));
    b.target = b.input;
    b.mask.resize(b.input.size());
    for (std::size_t i = 0; i < b.input.size(); ++i) {
        b.input[i] = static_cast<std::uint16_t>((i * 5 + 1) % 32);
        b.target[i] = static_cast<std::uint16_t>((i * 7 + 3) % 32);
        b.mask[i] = i % 4 == 0 ? 0 : 1;
    }
    return b;
}
MINI_MLA_TEST(pipelined_updates_match_synchronous_updates_bit_for_bit) {
    // Pipelining only moves host encoding ahead of the previous update's completion. The device
    // must execute the same work in the same order, so every result has to be identical.
    auto data = wide_batch(9);
    for (auto precision : {"fp32", "fp16"}) {
        for (auto kind : {"adamw", "hybrid_muon_adamw"}) {
            for (auto attention : {"reference", "flash"}) {
                for (int accumulation : {1, 3, 9}) {
                    OptimizerConfig oc;
                    oc.kind = kind;
                    auto serial_model = take(Model::create(config(), 61));
                    auto piped_model = take(Model::create(config(), 61));
                    Optimizer so(oc), po(oc);
                    MINI_MLA_REQUIRE(so.initialize(*serial_model));
                    MINI_MLA_REQUIRE(po.initialize(*piped_model));
                    // A budget of 12 tokens leaves each 18-token job in its own pass, so nine
                    // jobs span three submission windows and exercise both staging banks.
                    ExecutionOptions options{attention, 4, "batched", "gram", 12};
                    auto serial =
                        take(make_execution(*serial_model, &so, "mps", precision, options));
                    auto piped =
                        take(make_execution(*piped_model, &po, "mps", precision, options));
                    auto coefficients = job_coefficients(data, accumulation);
                    std::vector<GradientBatch> jobs;
                    for (const auto& c : coefficients)
                        jobs.push_back({&data, c});
                    std::vector<StepResult> expected, actual;
                    const int steps = 6;
                    for (int step = 0; step < steps; ++step) {
                        float lr = 0.001F * static_cast<float>(step + 1);
                        MINI_MLA_REQUIRE(serial->zero_grad());
                        auto reference = serial->step(jobs, lr);
                        require_result(reference);
                        MINI_MLA_REQUIRE(reference);
                        expected.push_back(reference.value());
                        MINI_MLA_REQUIRE(piped->zero_grad());
                        auto landed = piped->begin_step(jobs, lr);
                        require_result(landed);
                        MINI_MLA_REQUIRE(landed);
                        // The first call has nothing to report; later calls report one behind.
                        MINI_MLA_CHECK(landed.value().has_value() == (step > 0 && step != 4));
                        if (landed.value())
                            actual.push_back(*landed.value());
                        MINI_MLA_CHECK(po.steps() == static_cast<std::uint64_t>(step + 1));
                        if (step == 3) {
                            // Any synchronous call lands the outstanding update first and must
                            // observe its weights. Its result stays available afterwards.
                            auto after_serial = serial->forward(data.input, 2, 9);
                            auto after_piped = piped->forward(data.input, 2, 9);
                            MINI_MLA_REQUIRE(after_serial && after_piped);
                            MINI_MLA_CHECK(after_serial.value()->values ==
                                           after_piped.value()->values);
                            auto held = piped->finish_step();
                            MINI_MLA_REQUIRE(held);
                            MINI_MLA_REQUIRE(held.value().has_value());
                            actual.push_back(*held.value());
                        }
                    }
                    auto last = piped->finish_step();
                    require_result(last);
                    MINI_MLA_REQUIRE(last);
                    MINI_MLA_REQUIRE(last.value().has_value());
                    actual.push_back(*last.value());
                    auto nothing = piped->finish_step();
                    MINI_MLA_REQUIRE(nothing);
                    MINI_MLA_CHECK(!nothing.value().has_value());
                    MINI_MLA_REQUIRE(actual.size() == expected.size());
                    for (std::size_t i = 0; i < expected.size(); ++i) {
                        MINI_MLA_CHECK(expected[i].loss == actual[i].loss);
                        MINI_MLA_CHECK(expected[i].gradient_norm == actual[i].gradient_norm);
                    }
                    MINI_MLA_REQUIRE(serial->synchronize_to_host());
                    MINI_MLA_REQUIRE(piped->synchronize_to_host());
                    for (std::size_t p = 0; p < serial_model->parameters().size(); ++p) {
                        MINI_MLA_CHECK_MSG(serial_model->parameters()[p].value.values ==
                                               piped_model->parameters()[p].value.values,
                                           serial_model->parameters()[p].name);
                        MINI_MLA_CHECK(serial_model->parameters()[p].gradient.values ==
                                       piped_model->parameters()[p].gradient.values);
                        MINI_MLA_CHECK(so.states()[p].first.values == po.states()[p].first.values);
                        MINI_MLA_CHECK(so.states()[p].second.values ==
                                       po.states()[p].second.values);
                    }
                }
            }
        }
    }
}
MINI_MLA_TEST(coalesced_passes_match_separate_passes_and_respect_the_budget) {
    // Accumulating over jobs and running their concatenation are the same gradient. Check the
    // merged pass against one pass per job, and that merging stops at the token budget, at a
    // change of sequence length, and earlier for long sequences.
    auto data = wide_batch(9); // 18 tokens per job
    for (auto attention : {"reference", "flash"}) {
        auto separate_model = take(Model::create(config(), 67));
        auto merged_model = take(Model::create(config(), 67));
        Optimizer so, mo;
        MINI_MLA_REQUIRE(so.initialize(*separate_model));
        MINI_MLA_REQUIRE(mo.initialize(*merged_model));
        auto separate = take(make_execution(*separate_model, &so, "mps", "fp32",
                                            {attention, 4, "batched", "gram", 0}));
        auto merged = take(make_execution(*merged_model, &mo, "mps", "fp32",
                                          {attention, 4, "batched", "gram", 54}));
        auto coefficients = job_coefficients(data, 7);
        std::vector<GradientBatch> jobs;
        for (const auto& c : coefficients)
            jobs.push_back({&data, c});
        MINI_MLA_REQUIRE(separate->zero_grad());
        MINI_MLA_REQUIRE(merged->zero_grad());
        auto before_separate = separate->stats(), before_merged = merged->stats();
        auto expected = separate->step(jobs, 0.001F);
        auto actual = merged->step(jobs, 0.001F);
        require_result(actual);
        MINI_MLA_REQUIRE(expected && actual);
        MINI_MLA_CHECK(std::abs(expected.value().loss - actual.value().loss) < 1e-5);
        MINI_MLA_CHECK(std::abs(expected.value().gradient_norm - actual.value().gradient_norm) <
                       1e-5);
        // Seven 18-token jobs under a 54-token budget are passes of 3, 3 and 1 jobs. Reference
        // attention runs one graph per pass; the Metal AdamW update is not a graph.
        if (std::string_view(attention) == "reference") {
            MINI_MLA_CHECK(separate->stats().graph_submissions -
                               before_separate.graph_submissions ==
                           7);
            MINI_MLA_CHECK(merged->stats().graph_submissions - before_merged.graph_submissions ==
                           3);
        }
        MINI_MLA_CHECK(separate->stats().host_waits - before_separate.host_waits == 2);
        MINI_MLA_CHECK(merged->stats().host_waits - before_merged.host_waits == 1);
        MINI_MLA_REQUIRE(separate->synchronize_to_host());
        MINI_MLA_REQUIRE(merged->synchronize_to_host());
        for (std::size_t p = 0; p < separate_model->parameters().size(); ++p) {
            const auto& a = separate_model->parameters()[p];
            const auto& b = merged_model->parameters()[p];
            for (std::size_t i = 0; i < a.value.size(); ++i) {
                MINI_MLA_CHECK_MSG(std::abs(a.gradient.values[i] - b.gradient.values[i]) <
                                       2e-6F + 2e-5F * std::abs(a.gradient.values[i]),
                                   a.name);
                MINI_MLA_CHECK_MSG(std::abs(a.value.values[i] - b.value.values[i]) < 2e-6F, a.name);
            }
        }
    }
    // Different sequence lengths never share a pass, and sequences longer than 256 tokens shrink
    // the allowance in proportion: at 512 tokens a 2,048 budget admits two sequences, not four.
    auto model = take(Model::create(config(), 71));
    Optimizer optimizer;
    MINI_MLA_REQUIRE(optimizer.initialize(*model));
    auto c = config();
    c.max_sequence_length = 512;
    auto long_model = take(Model::create(c, 71));
    Optimizer long_optimizer;
    MINI_MLA_REQUIRE(long_optimizer.initialize(*long_model));
    auto mixed = take(make_execution(*model, &optimizer, "mps", "fp32",
                                     {"reference", 16, "batched", "gram", 4096}));
    auto short_a = wide_batch(5), short_b = wide_batch(7);
    auto ca = take(short_a.coefficients()), cb = take(short_b.coefficients());
    std::array<GradientBatch, 5> alternating{
        {{&short_a, ca}, {&short_a, ca}, {&short_b, cb}, {&short_a, ca}, {&short_a, ca}}};
    MINI_MLA_REQUIRE(mixed->zero_grad());
    MINI_MLA_REQUIRE(mixed->step(alternating, 0.001F));
    MINI_MLA_CHECK(mixed->stats().graph_submissions == 3);
    auto lengthy = take(make_execution(*long_model, &long_optimizer, "mps", "fp32",
                                       {"reference", 16, "batched", "gram", 2048}));
    Batch one;
    one.batch = 1;
    one.sequence = 512;
    one.input.assign(512, 3);
    one.target.assign(512, 4);
    one.mask.assign(512, 1);
    auto cl = take(one.coefficients());
    std::array<GradientBatch, 4> four{{{&one, cl}, {&one, cl}, {&one, cl}, {&one, cl}}};
    MINI_MLA_REQUIRE(lengthy->zero_grad());
    auto stepped = lengthy->step(four, 0.001F);
    require_result(stepped);
    MINI_MLA_REQUIRE(stepped);
    MINI_MLA_CHECK(lengthy->stats().graph_submissions == 2);
}
MINI_MLA_TEST(pipelined_failure_is_reported_late_and_disables_the_execution) {
    auto model = take(Model::create(config(), 73));
    Optimizer optimizer;
    MINI_MLA_REQUIRE(optimizer.initialize(*model));
    auto gpu = take(make_execution(*model, &optimizer, "mps", "fp32", {"reference", 4}));
    auto data = batch();
    auto fine = take(data.coefficients());
    // Rejected before anything is submitted: the execution stays healthy.
    auto bad = fine;
    bad.back() = std::numeric_limits<float>::quiet_NaN();
    std::array<GradientBatch, 1> invalid{{{&data, bad}}};
    MINI_MLA_CHECK(!gpu->begin_step(invalid, 0.001F));
    MINI_MLA_CHECK(gpu->stats().graph_submissions == 0);
    MINI_MLA_CHECK(optimizer.steps() == 0);
    // Finite inputs whose loss overflows on the device: begin_step cannot know yet.
    auto huge = fine;
    for (auto& x : huge)
        x = x == 0 ? 0 : 3e38F;
    std::array<GradientBatch, 1> overflow{{{&data, huge}}}, healthy{{{&data, fine}}};
    MINI_MLA_REQUIRE(gpu->zero_grad());
    auto begun = gpu->begin_step(overflow, 0.001F);
    MINI_MLA_REQUIRE(begun);
    MINI_MLA_CHECK(!begun.value().has_value());
    auto reported = gpu->finish_step();
    MINI_MLA_CHECK(!reported);
    MINI_MLA_CHECK(!gpu->begin_step(healthy, 0.001F));
    MINI_MLA_CHECK(!gpu->step(healthy, 0.001F));
    MINI_MLA_CHECK(!gpu->forward(data.input, 2, 3));
    MINI_MLA_CHECK(!gpu->synchronize_to_host());
    // The serial comparison path has no pipeline: begin_step there is a synchronous step whose
    // result is handed back one call later, and a failed update leaves the weights untouched.
    auto serial_model = take(Model::create(config(), 73));
    Optimizer serial_optimizer;
    MINI_MLA_REQUIRE(serial_optimizer.initialize(*serial_model));
    auto serial =
        take(make_execution(*serial_model, &serial_optimizer, "mps", "fp32", {"reference", 1}));
    MINI_MLA_REQUIRE(serial->zero_grad());
    auto first = serial->begin_step(healthy, 0.001F);
    MINI_MLA_REQUIRE(first);
    MINI_MLA_CHECK(!first.value().has_value());
    MINI_MLA_CHECK(serial_optimizer.steps() == 1);
    MINI_MLA_REQUIRE(serial->zero_grad());
    auto second = serial->begin_step(healthy, 0.001F);
    MINI_MLA_REQUIRE(second);
    MINI_MLA_CHECK(second.value().has_value());
    auto third = serial->finish_step();
    MINI_MLA_REQUIRE(third);
    MINI_MLA_CHECK(third.value().has_value());
    MINI_MLA_CHECK(serial_optimizer.steps() == 2);
}
MINI_MLA_TEST(muon_newton_schulz_orders_agree_across_matrix_shapes) {
    // Composed, Gram and chain orders are the same polynomial of the Gram matrix. Composed is
    // used only where it is cheaper (wide beyond 1.5x) and falls back to Gram elsewhere, so the
    // shapes here cover wide, tall, barely-wide and square matrices in one model.
    auto c = config();
    c.d_model = 16;
    c.n_heads = 2;
    c.head_dim_nope = 6;
    c.head_dim_rope = 2;
    c.head_dim_value = 8;   // w_o is square 16x16: Gram fallback
    c.q_lora_rank = 12;     // w_dq is 16x12: wide by 1.33x, Gram fallback
    c.kv_lora_rank = 4;     // w_uv is 4x16 and w_dkv is 16x6: composed
    c.d_ff = 40;            // w_gate 16x40 and w_down 40x16: composed
    auto data = wide_batch(9);
    auto coefficients = take(data.coefficients());
    std::array<GradientBatch, 1> jobs{{{&data, coefficients}}};
    for (auto layout : {"batched", "individual"}) {
        for (int ns_steps : {2, 5}) {
            OptimizerConfig oc;
            oc.kind = "hybrid_muon_adamw";
            oc.ns_steps = ns_steps;
            std::vector<std::unique_ptr<Model>> models;
            std::vector<std::unique_ptr<Optimizer>> optimizers;
            std::vector<std::unique_ptr<Execution>> executions;
            for (auto order : {"gram", "composed", "chain"}) {
                models.push_back(take(Model::create(c, 79)));
                optimizers.push_back(std::make_unique<Optimizer>(oc));
                MINI_MLA_REQUIRE(optimizers.back()->initialize(*models.back()));
                executions.push_back(take(make_execution(*models.back(), optimizers.back().get(),
                                                         "mps", "fp32",
                                                         {"reference", 4, layout, order})));
            }
            for (int step = 0; step < 4; ++step)
                for (auto& execution : executions) {
                    MINI_MLA_REQUIRE(execution->zero_grad());
                    auto result = execution->step(jobs, 0.002F);
                    require_result(result);
                    MINI_MLA_REQUIRE(result);
                }
            for (auto& execution : executions)
                MINI_MLA_REQUIRE(execution->synchronize_to_host());
            for (std::size_t p = 0; p < models[0]->parameters().size(); ++p) {
                const auto& gram = models[0]->parameters()[p];
                for (std::size_t other : {std::size_t{1}, std::size_t{2}})
                    for (std::size_t i = 0; i < gram.value.size(); ++i) {
                        float value = models[other]->parameters()[p].value.values[i];
                        MINI_MLA_CHECK_MSG(std::isfinite(value) &&
                                               std::abs(gram.value.values[i] - value) < 2e-5F,
                                           gram.name);
                        MINI_MLA_CHECK(std::abs(optimizers[0]->states()[p].first.values[i] -
                                                optimizers[other]->states()[p].first.values[i]) <
                                       2e-5F);
                    }
            }
        }
    }
}
MINI_MLA_TEST(flash_prefill_cache_decode_and_shorter_replacement_match_cpu) {
    auto made = Model::create(config(), 53, false);
    MINI_MLA_REQUIRE(made);
    for (auto precision : {"fp32", "fp16"}) {
        auto gpu = make_execution(*made.value(), nullptr, "mps", precision, {"flash", 4});
        MINI_MLA_REQUIRE(gpu);
        for (int length : {33, 7}) {
            std::vector<std::uint16_t> ids(static_cast<std::size_t>(length));
            for (int i = 0; i < length; ++i)
                ids[static_cast<std::size_t>(i)] = static_cast<std::uint16_t>((i * 3 + 2) % 32);
            auto actual = gpu.value()->prefill(ids);
            require_result(actual);
            MINI_MLA_REQUIRE(actual);
            auto expected = made.value()->forward(ids, 1, length);
            MINI_MLA_REQUIRE(expected);
            float tolerance = std::string_view(precision) == "fp32" ? 1e-4F : 0.015F;
            for (int v = 0; v < config().vocab_size; ++v)
                MINI_MLA_CHECK(std::abs(actual.value()[static_cast<std::size_t>(v)] -
                                        (*expected.value())(length - 1, v)) < tolerance);
            ids.push_back(4);
            actual = gpu.value()->decode(4);
            require_result(actual);
            MINI_MLA_REQUIRE(actual);
            expected = made.value()->forward(ids, 1, length + 1);
            MINI_MLA_REQUIRE(expected);
            for (int v = 0; v < config().vocab_size; ++v)
                MINI_MLA_CHECK(std::abs(actual.value()[static_cast<std::size_t>(v)] -
                                        (*expected.value())(length, v)) < tolerance);
        }
    }
}
int main() {
    auto info = metal_device_info();
    if (!info) {
        std::fprintf(stderr, "SKIP: %s\n", info.error().message.c_str());
        return 77;
    }
    std::fprintf(stderr, "%s\n", info.value().dump().c_str());
    return mini_mla::testing::run_all();
}
