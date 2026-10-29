#include "app_utils.hpp"

#include <fstream>

int main(int argc, char** argv) {
    return mini_mla::app_main(argc, argv, [](const mini_mla::Args& a) {
        a.known({"model", "export", "dtype", "tokens", "targets", "mask", "debug-output",
                 "debug-update", "learning-rate", "backend", "precision", "attention", "help"});
        if (a.has("help")) {
            std::puts("mini_mla_inspect --model DIR [--export DIR --dtype f16|f32]\n  --tokens "
                      "1,2,3 --targets 2,3,4 --debug-output DIR [--debug-update]\n  "
                      "--backend cpu|mps --precision fp32|fp16 --attention auto|reference|flash");
            return 0;
        }
        auto loaded = mini_mla::take(mini_mla::load_checkpoint(a.required("model")));
        auto& m = *loaded.model;
        if (a.has("export")) {
            mini_mla::check(mini_mla::export_model(
                a.get("export"), m, std::filesystem::path(a.get("model")) / "tokenizer.json",
                a.get("dtype", "f16")));
            return 0;
        }
        mini_mla::Json::Array parameters;
        for (const auto& p : m.parameters())
            parameters.emplace_back(
                mini_mla::Json::Object{{"name", p.name},
                                       {"role", std::string(mini_mla::role_name(p.role))},
                                       {"shape", mini_mla::Json::Array{p.value.rows, p.value.cols}},
                                       {"elements", p.value.size()}});
        mini_mla::Json report =
            mini_mla::Json::Object{{"config", m.config().json()},
                                   {"parameters", std::move(parameters)},
                                   {"parameter_count", m.config().parameter_count()},
                                   {"tokenizer_sha256", loaded.tokenizer.hash()}};
        if (a.has("tokens")) {
            auto ids = parse_tokens(a.get("tokens"));
            if (ids.empty())
                throw std::invalid_argument("Empty input tokens");
            mini_mla::Optimizer optimizer;
            m.zero_grad();
            mini_mla::check(optimizer.initialize(m));
            auto execution = mini_mla::take(mini_mla::make_execution(
                m, &optimizer, a.get("backend", "cpu"), a.get("precision", "fp32"),
                {a.get("attention", "auto"), 4}));
            auto logits = *mini_mla::take(execution->forward(ids, 1, static_cast<int>(ids.size())));
            mini_mla::Json::Array output;
            for (float x : logits.values)
                output.emplace_back(static_cast<double>(x));
            report["logits"] = std::move(output);
            if (a.has("targets")) {
                auto targets = parse_tokens(a.get("targets"));
                auto masks = a.has("mask") ? parse_tokens(a.get("mask"))
                                           : std::vector<std::uint16_t>(ids.size(), 1);
                mini_mla::Batch batch{1, static_cast<int>(ids.size()), ids, targets, {}};
                for (auto bit : masks)
                    batch.mask.push_back(static_cast<float>(bit));
                auto coefficients = mini_mla::take(batch.coefficients());
                mini_mla::check(execution->zero_grad());
                report["loss"] = mini_mla::take(execution->accumulate(batch, coefficients));
                mini_mla::check(execution->synchronize_to_host());
                if (a.has("debug-output")) {
                    std::filesystem::path path = a.get("debug-output");
                    std::filesystem::create_directories(path);
                    std::ofstream data(path / "tensors.bin", std::ios::binary);
                    if (!data)
                        throw std::runtime_error("Cannot open debug output");
                    mini_mla::Json::Array entries;
                    std::size_t offset = 0;
                    auto write = [&](std::string name, const mini_mla::Matrix& matrix) {
                        auto bytes = matrix.size() * 4;
                        data.write(reinterpret_cast<const char*>(matrix.data()),
                                   static_cast<std::streamsize>(bytes));
                        entries.emplace_back(mini_mla::Json::Object{
                            {"name", std::move(name)},
                            {"shape", mini_mla::Json::Array{matrix.rows, matrix.cols}},
                            {"offset", offset},
                            {"nbytes", bytes}});
                        offset += bytes;
                    };
                    write("logits", logits);
                    for (const auto& p : m.parameters()) {
                        write(p.name + ".weight", p.value);
                        write(p.name + ".gradient", p.gradient);
                    }
                    if (a.has("debug-update")) {
                        mini_mla::take(execution->update(a.real("learning-rate", 0.001F)));
                        mini_mla::check(execution->synchronize_to_host());
                        for (const auto& p : m.parameters())
                            write(p.name + ".updated", p.value);
                    }
                    data.flush();
                    if (!data)
                        throw std::runtime_error("Debug write failed");
                    data.close();
                    report["tensors"] = std::move(entries);
                    report["input"] = a.get("tokens");
                    report["targets"] = a.get("targets");
                    report["mask"] = a.get("mask", "");
                    report["optimizer"] = optimizer.config().json();
                    report["learning_rate"] = static_cast<double>(a.real("learning-rate", 0.001F));
                    mini_mla::check(mini_mla::write_json(path / "debug.json", report));
                }
            } else if (a.has("debug-output"))
                throw std::invalid_argument("Debug gradients require --targets");
        }
        std::puts(report.dump().c_str());
        return 0;
    });
}
