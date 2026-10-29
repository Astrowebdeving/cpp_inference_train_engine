// Phase breakdown through the real engine: forward, forward+loss, forward+backward, update.
#include "mini_mla/execution.hpp"
#include "mini_mla/workflow.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

using namespace mini_mla;

static double median_ms(int warmup, int reps, const std::function<void()>& fn) {
    std::vector<double> s;
    for (int i = -warmup; i < reps; ++i) {
        auto t = std::chrono::steady_clock::now();
        fn();
        double ms =
            std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
        if (i >= 0)
            s.push_back(ms);
    }
    std::sort(s.begin(), s.end());
    return s[s.size() / 2];
}

int main(int argc, char** argv) {
    std::string precision = argc > 1 ? argv[1] : "fp16";
    int batch = argc > 2 ? std::atoi(argv[2]) : 8, sequence = argc > 3 ? std::atoi(argv[3]) : 128;
    std::string kind = argc > 4 ? argv[4] : "adamw";
    auto c = take(ModelConfig::load(std::getenv("CFG") ? std::getenv("CFG") : "/Users/tu15/alex_cpp/CPP_Completed/config/mla36m-v1.json"));
    auto model = take(Model::create(c, 42, true));
    OptimizerConfig oc;
    oc.kind = kind == "muon" ? "hybrid_muon_adamw" : "adamw";
    Optimizer optimizer(oc);
    check(optimizer.initialize(*model));
    auto e = take(make_execution(*model, &optimizer, "mps", precision, {"reference", 4}));
    Batch b;
    b.batch = batch;
    b.sequence = sequence;
    b.input.resize(static_cast<std::size_t>(batch * sequence));
    b.target = b.input;
    b.mask.resize(b.input.size(), 1);
    for (std::size_t i = 0; i < b.input.size(); ++i) {
        b.input[i] = static_cast<std::uint16_t>(i % static_cast<std::size_t>(c.vocab_size));
        b.target[i] = static_cast<std::uint16_t>((i + 1) % static_cast<std::size_t>(c.vocab_size));
    }
    auto coefficients = take(b.coefficients());
    std::vector<GradientBatch> jobs{GradientBatch{&b, coefficients}};
    int w = 3, r = 20;
    double forward = median_ms(w, r, [&] { (void)take(e->forward(b.input, batch, sequence)); });
    double logp = median_ms(w, r, [&] { (void)take(e->log_probabilities(b)); });
    double backward = median_ms(w, r, [&] {
        check(e->zero_grad());
        (void)take(e->accumulate(b, coefficients));
    });
    double accumulate2 = median_ms(w, r, [&] { (void)take(e->accumulate(b, coefficients)); });
    double update = median_ms(w, r, [&] { (void)take(e->update(0.0F)); });
    double step = median_ms(w, r, [&] {
        check(e->zero_grad());
        (void)take(e->step(jobs, 0.0F));
    });
    std::printf("%s b=%d t=%d %s\n", precision.c_str(), batch, sequence, kind.c_str());
    std::printf("  forward (logits to host)      %8.3f ms\n", forward);
    std::printf("  forward + selected log-probs  %8.3f ms\n", logp);
    std::printf("  forward + backward (fresh)    %8.3f ms\n", backward);
    std::printf("  forward + backward (accum)    %8.3f ms\n", accumulate2);
    std::printf("  optimizer update only         %8.3f ms\n", update);
    std::printf("  full step                     %8.3f ms\n", step);
    return 0;
}
