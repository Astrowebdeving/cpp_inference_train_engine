#include "inference_sessions.hpp"
#include "mini_mla/testing.hpp"
#include "mini_mla/trainer.hpp"

#include <cmath>
#include <limits>

using namespace mini_mla;
namespace {
ModelConfig small() {
    ModelConfig c;
    c.vocab_size = 16;
    c.n_layers = 2;
    c.d_model = 16;
    c.n_heads = 2;
    c.head_dim_nope = 4;
    c.head_dim_rope = 2;
    c.head_dim_value = 6;
    c.q_lora_rank = 8;
    c.kv_lora_rank = 6;
    c.d_ff = 32;
    c.max_sequence_length = 32;
    return c;
}
} // namespace
MINI_MLA_TEST(full_decoder_overfits_a_small_sequence) {
    auto made = Model::create(small(), 23);
    MINI_MLA_REQUIRE(made);
    OptimizerConfig oc;
    oc.weight_decay = 0;
    Optimizer optimizer(oc);
    MINI_MLA_REQUIRE(optimizer.initialize(*made.value()));
    auto execution = make_execution(*made.value(), &optimizer, "cpu");
    MINI_MLA_REQUIRE(execution);
    Batch b{1, 4, {1, 2, 3, 4}, {2, 3, 4, 5}, {1, 1, 1, 1}};
    auto coefficients = b.coefficients();
    MINI_MLA_REQUIRE(coefficients);
    double first = 0, last = 0;
    for (int i = 0; i < 100; ++i) {
        MINI_MLA_REQUIRE(execution.value()->zero_grad());
        auto loss = execution.value()->accumulate(b, coefficients.value());
        MINI_MLA_REQUIRE(loss);
        if (i == 0)
            first = loss.value();
        last = loss.value();
        MINI_MLA_REQUIRE(execution.value()->update(0.008F));
    }
    MINI_MLA_CHECK_MSG(last < 0.08 && last < first * 0.05,
                       "first=" + std::to_string(first) + " last=" + std::to_string(last));
    MINI_MLA_CHECK(made.value()->gradients_finite());
}
MINI_MLA_TEST(compressed_decode_matches_full_recomputation) {
    auto made = Model::create(small(), 21, false);
    MINI_MLA_REQUIRE(made);
    for (bool absorbed : {true, false}) {
        CpuDecoder decoder(*made.value());
        decoder.set_absorbed(absorbed);
        std::vector<std::uint16_t> ids{1, 2, 3};
        MINI_MLA_REQUIRE(decoder.prefill(ids));
        MINI_MLA_CHECK(decoder.cache_bytes() ==
                       static_cast<std::size_t>(small().n_layers * small().max_sequence_length *
                                                small().kv_width() * 4));
        for (int i = 0; i < 20; ++i) {
            auto id = static_cast<std::uint16_t>(i % 16);
            ids.push_back(id);
            auto actual = decoder.decode(id);
            MINI_MLA_REQUIRE(actual);
            auto expected = made.value()->forward(ids, 1, static_cast<int>(ids.size()));
            MINI_MLA_REQUIRE(expected);
            for (int j = 0; j < small().vocab_size; ++j)
                MINI_MLA_CHECK(std::abs(actual.value()[static_cast<std::size_t>(j)] -
                                        (*expected.value())(static_cast<int>(ids.size() - 1), j)) <
                               2e-5F);
        }
    }
}
MINI_MLA_TEST(sampling_is_seeded_and_respects_filters) {
    float scores[] = {2, -1, 1, 0};
    GenerationConfig c;
    c.temperature = 0;
    Rng rng(1);
    auto greedy = sample_token(scores, {}, c, rng);
    MINI_MLA_REQUIRE(greedy);
    MINI_MLA_CHECK(greedy.value() == 0);
    c.repetition_penalty = 3;
    std::uint16_t history[] = {0, 0};
    auto penalized = sample_token(scores, history, c, rng);
    MINI_MLA_REQUIRE(penalized);
    MINI_MLA_CHECK(penalized.value() == 2);
    c.temperature = 0.7F;
    c.top_k = 1;
    c.repetition_penalty = 1;
    for (int i = 0; i < 30; ++i) {
        auto id = sample_token(scores, {}, c, rng);
        MINI_MLA_REQUIRE(id);
        MINI_MLA_CHECK(id.value() == 0);
    }
    c.top_k = 0;
    c.top_p = 1;
    Rng a(99), b(99);
    for (int i = 0; i < 50; ++i) {
        auto x = sample_token(scores, {}, c, a), y = sample_token(scores, {}, c, b);
        MINI_MLA_REQUIRE(x && y);
        MINI_MLA_CHECK(x.value() == y.value());
    }
    std::uint16_t banned[] = {0, 1, 2, 3};
    MINI_MLA_CHECK(!sample_token(scores, {}, c, rng, banned));
    c.top_p = 0;
    MINI_MLA_CHECK(!sample_token(scores, {}, c, rng));
}
MINI_MLA_TEST(weight_changes_invalidate_decode_cache) {
    auto made = Model::create(small(), 9);
    MINI_MLA_REQUIRE(made);
    CpuDecoder decoder(*made.value());
    std::uint16_t ids[] = {1, 2, 3};
    MINI_MLA_REQUIRE(decoder.prefill(ids));
    made.value()->parameters()[0].value(0, 0) += 0.01F;
    made.value()->mark_weights_changed();
    MINI_MLA_CHECK(!decoder.decode(4));
    MINI_MLA_REQUIRE(decoder.prefill(ids));
    MINI_MLA_REQUIRE(decoder.decode(4));
}
MINI_MLA_TEST(dpo_loss_gradient_and_initial_baseline) {
    auto baseline = dpo_objective(-5, -7, -5, -7, 0.1F);
    MINI_MLA_CHECK(std::abs(baseline.loss - std::log(2.0)) < 1e-12);
    double chosen = -3.2, rejected = -6.4, epsilon = 1e-4;
    auto objective = dpo_objective(chosen, rejected, -5, -6, 0.2F);
    double derivative = (dpo_objective(chosen + epsilon, rejected, -5, -6, 0.2F).loss -
                         dpo_objective(chosen - epsilon, rejected, -5, -6, 0.2F).loss) /
                        (2 * epsilon);
    MINI_MLA_CHECK(std::abs(derivative + objective.chosen_nll_coefficient) < 1e-7);
    MINI_MLA_CHECK(objective.chosen_nll_coefficient > 0 && objective.rejected_nll_coefficient < 0);
    MINI_MLA_CHECK(std::isfinite(dpo_objective(10000, -10000, -5, -6, 0.1F).loss));
}
MINI_MLA_TEST(grpo_clipping_and_kl_gradients) {
    for (float advantage : {-1.0F, 1.0F})
        for (float lp : {-2.1F, -1.95F, -1.5F, -2.5F}) {
            auto objective = grpo_objective(lp, -2, -2.2F, advantage, 0.2F, 0.02F);
            MINI_MLA_REQUIRE(objective);
            float epsilon = 0.0005F;
            auto pos = grpo_objective(lp + epsilon, -2, -2.2F, advantage, 0.2F, 0.02F),
                 neg = grpo_objective(lp - epsilon, -2, -2.2F, advantage, 0.2F, 0.02F);
            MINI_MLA_REQUIRE(pos && neg);
            double numeric = (pos.value().loss - neg.value().loss) / (2 * epsilon);
            MINI_MLA_CHECK(std::abs(numeric + objective.value().nll_coefficient) < 0.0003);
        }
    auto clipped = grpo_objective(-1, -2, -1, 1, 0.2F, 0);
    MINI_MLA_REQUIRE(clipped);
    MINI_MLA_CHECK(clipped.value().nll_coefficient == 0);
    MINI_MLA_CHECK(!grpo_objective(0, -100, 0, 1, 0.2F, 0));
    float rewards[] = {0, 1, 1, 0};
    auto advantages = group_advantages(rewards);
    MINI_MLA_REQUIRE(advantages);
    MINI_MLA_CHECK(std::abs(advantages.value()[0] + 1) < 1e-5F &&
                   std::abs(advantages.value()[1] - 1) < 1e-5F);
    float constant[] = {1, 1, 1};
    auto zero = group_advantages(constant);
    MINI_MLA_REQUIRE(zero);
    for (auto x : zero.value())
        MINI_MLA_CHECK(x == 0);
}
MINI_MLA_TEST(completion_masks_and_cosine_schedule) {
    std::uint16_t prompt[] = {1, 2, 3}, response[] = {4, 5};
    auto b = completion_batch(prompt, response, 6, 0);
    MINI_MLA_REQUIRE(b);
    MINI_MLA_CHECK(b.value().mask == std::vector<float>({0, 0, 1, 1, 0, 0}));
    MINI_MLA_CHECK(b.value().target == std::vector<std::uint16_t>({2, 3, 4, 5, 0, 0}));
    MINI_MLA_CHECK(!completion_batch(prompt, response, 3, 0));
    MINI_MLA_CHECK(std::abs(cosine_learning_rate(0, 100, 10, 0.001F, 0.1F) - 0.0001F) < 1e-9F);
    MINI_MLA_CHECK(std::abs(cosine_learning_rate(9, 100, 10, 0.001F, 0.1F) - 0.001F) < 1e-9F);
    MINI_MLA_CHECK(std::abs(cosine_learning_rate(100, 100, 10, 0.001F, 0.1F) - 0.0001F) < 1e-9F);
}
MINI_MLA_TEST(muon_routes_only_hidden_matrices) {
    auto made = Model::create(small(), 3);
    MINI_MLA_REQUIRE(made);
    OptimizerConfig c;
    c.kind = "hybrid_muon_adamw";
    Optimizer o(c);
    MINI_MLA_REQUIRE(o.initialize(*made.value()));
    for (std::size_t i = 0; i < o.states().size(); ++i)
        MINI_MLA_CHECK((o.states()[i].second.size() == 0) ==
                       (made.value()->parameters()[i].role == ParameterRole::hidden_matrix));
    Rng rng(12);
    for (auto& p : made.value()->parameters())
        for (auto& x : p.gradient.values)
            x = rng.normal() * 0.1F;
    MINI_MLA_REQUIRE(o.step(*made.value(), 0.0003F));
    MINI_MLA_CHECK(made.value()->gradients_finite());
    for (const auto& p : made.value()->parameters())
        MINI_MLA_CHECK(ops::finite(p.value.values));
}
MINI_MLA_TEST(malformed_optimizer_state_cannot_mutate_weights) {
    auto made = Model::create(small(), 3);
    MINI_MLA_REQUIRE(made);
    Optimizer optimizer;
    MINI_MLA_REQUIRE(optimizer.initialize(*made.value()));
    auto weights = made.value()->parameters().front().value.values;
    optimizer.states().back().second = Matrix();
    MINI_MLA_CHECK(!optimizer.step(*made.value(), 0.001F));
    MINI_MLA_CHECK(optimizer.steps() == 0);
    MINI_MLA_CHECK(made.value()->parameters().front().value.values == weights);
    MINI_MLA_REQUIRE(optimizer.initialize(*made.value()));
    optimizer.states().back().first.rows += 1;
    MINI_MLA_CHECK(!optimizer.step(*made.value(), 0.001F));
    MINI_MLA_CHECK(optimizer.steps() == 0);
}
MINI_MLA_TEST(execution_options_preserve_legacy_resume_and_reject_invalid_steps) {
    TrainOptions current;
    auto saved = current.json();
    auto restored = TrainOptions::from_json(saved);
    MINI_MLA_REQUIRE(restored);
    MINI_MLA_CHECK(restored.value().attention == "auto");
    MINI_MLA_CHECK(restored.value().submission_window == 4);
    MINI_MLA_CHECK(restored.value().muon_layout == "batched");
    MINI_MLA_CHECK(restored.value().muon_ns_order == "gram");
    auto legacy_object = saved.object();
    legacy_object.erase("attention");
    legacy_object.erase("submission_window");
    legacy_object.erase("muon_layout");
    legacy_object.erase("muon_ns_order");
    saved = Json(std::move(legacy_object));
    auto legacy = TrainOptions::from_json(saved);
    MINI_MLA_REQUIRE(legacy);
    MINI_MLA_CHECK(legacy.value().attention == "reference");
    MINI_MLA_CHECK(legacy.value().submission_window == 1);
    MINI_MLA_CHECK(legacy.value().muon_layout == "individual");
    MINI_MLA_CHECK(legacy.value().muon_ns_order == "chain");
    MINI_MLA_CHECK(legacy.value().json().dump() == saved.dump());
    MINI_MLA_CHECK(!ExecutionOptions({"flash", 0}).validate());
    MINI_MLA_CHECK(!ExecutionOptions({"auto", 17}).validate());
    MINI_MLA_CHECK(!ExecutionOptions({"invalid", 4}).validate());
    MINI_MLA_CHECK(!ExecutionOptions({"auto", 4, "invalid"}).validate());
    MINI_MLA_CHECK(!ExecutionOptions({"auto", 4, "batched", "invalid"}).validate());
    auto model = Model::create(small(), 6);
    MINI_MLA_REQUIRE(model);
    Optimizer optimizer;
    MINI_MLA_REQUIRE(optimizer.initialize(*model.value()));
    MINI_MLA_CHECK(!make_execution(*model.value(), &optimizer, "cpu", "fp32", {"flash", 4}));
    auto cpu = make_execution(*model.value(), &optimizer, "cpu");
    MINI_MLA_REQUIRE(cpu);
    Batch b{1, 2, {1, 2}, {2, 3}, {1, 1}};
    auto good = b.coefficients();
    MINI_MLA_REQUIRE(good);
    auto bad = good.value();
    bad[1] = std::numeric_limits<float>::quiet_NaN();
    std::vector<GradientBatch> jobs{{&b, good.value()}, {&b, bad}};
    MINI_MLA_CHECK(!cpu.value()->step(jobs, 0.001F));
    MINI_MLA_CHECK(optimizer.steps() == 0);
    for (const auto& p : model.value()->parameters())
        for (float x : p.gradient.values)
            MINI_MLA_CHECK(x == 0);
}
MINI_MLA_TEST(muon_settings_round_trip_and_reject_invalid_variants) {
    TrainOptions options;
    options.optimizer.kind = "hybrid_muon_adamw";
    options.optimizer.momentum = 0.8F;
    options.optimizer.nesterov = false;
    options.optimizer.ns_steps = 3;
    auto saved = options.json();
    auto restored = TrainOptions::from_json(saved);
    MINI_MLA_REQUIRE(restored);
    MINI_MLA_CHECK(restored.value().json().dump() == saved.dump());
    auto invalid = options.optimizer.json();
    for (int steps : {0, 21}) {
        invalid["ns_steps"] = steps;
        MINI_MLA_CHECK(!OptimizerConfig::from_json(invalid));
    }
    invalid = options.optimizer.json();
    invalid["momentum"] = 1;
    MINI_MLA_CHECK(!OptimizerConfig::from_json(invalid));
    invalid = options.optimizer.json();
    invalid["muon_variant"] = "unknown";
    MINI_MLA_CHECK(!OptimizerConfig::from_json(invalid));
}
MINI_MLA_TEST(cpu_inference_forks_preserve_independent_request_caches) {
    MINI_MLA_CHECK(check_inference_sessions(small(), "cpu", "fp32"));
}
MINI_MLA_TEST_MAIN()
