#pragma once
#include "mini_mla/execution.hpp"
#include "mini_mla/workflow.hpp"

#include <array>
#include <atomic>
#include <cmath>
#include <thread>

inline bool check_inference_sessions(const mini_mla::ModelConfig& c, std::string_view backend,
                                     std::string_view precision) {
    using namespace mini_mla;
    auto model = take(Model::create(c, 91));
    auto oracle_model = take(Model::create(c, 91));
    auto root = take(make_execution(*model, nullptr, backend, precision));
    auto oracle = take(make_execution(*oracle_model, nullptr, "cpu"));
    const std::array<std::vector<std::uint16_t>, 3> prompts{
        std::vector<std::uint16_t>{1, 2, 3, 4, 5}, std::vector<std::uint16_t>{2, 3, 4, 5, 6, 7, 8},
        std::vector<std::uint16_t>{3, 4, 5, 6, 7, 8, 9, 10, 11}};
    std::array<std::vector<std::vector<float>>, 3> expected;
    for (std::size_t i = 0; i < prompts.size(); ++i) {
        auto logits = take(oracle->prefill(prompts[i]));
        expected[i].emplace_back(logits.begin(), logits.end());
        for (std::uint16_t token : std::array<std::uint16_t, 3>{4, 5, 6}) {
            logits = take(oracle->decode(token));
            expected[i].emplace_back(logits.begin(), logits.end());
        }
    }
    (void)take(root->prefill(prompts[1]));
    (void)take(root->decode(4));
    std::array<std::unique_ptr<Execution>, 3> sessions;
    for (auto& session : sessions)
        session = take(root->fork_inference());
    std::atomic<bool> okay{true};
    std::vector<std::thread> workers;
    for (std::size_t i = 0; i < sessions.size(); ++i)
        workers.emplace_back([&, i] {
            auto match = [&](const auto& actual, std::size_t step) {
                if (!actual || actual.value().size() != expected[i][step].size()) {
                    okay = false;
                    return;
                }
                float tolerance = precision == "fp16" ? 0.005F : 0.0001F;
                for (std::size_t j = 0; j < actual.value().size(); ++j)
                    if (!std::isfinite(actual.value()[j]) ||
                        std::abs(actual.value()[j] - expected[i][step][j]) > tolerance)
                        okay = false;
            };
            for (int repeat = 0; repeat < 3; ++repeat) {
                match(sessions[i]->prefill(prompts[i]), 0);
                for (int step = 0; step < 3; ++step)
                    match(sessions[i]->decode(static_cast<std::uint16_t>(step + 4)),
                          static_cast<std::size_t>(step + 1));
            }
        });
    for (auto& worker : workers)
        worker.join();
    return okay;
}
