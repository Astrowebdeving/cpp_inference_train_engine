#include "mini_mla/backend.hpp"
#include "mini_mla/cli.hpp"
#include "mini_mla/execution.hpp"

int main(int argc, char** argv) {
    return mini_mla::app_main(argc, argv, [](const mini_mla::Args& a) {
        a.known({"config", "help"});
        if (a.has("help")) {
            std::puts("mini_mla_info [--config config/mla36m-v1.json]");
            return 0;
        }
        auto cpu = mini_mla::take(mini_mla::make_cpu_backend());
        auto c =
            mini_mla::take(mini_mla::ModelConfig::load(a.get("config", "config/mla36m-v1.json")));
        auto metal = mini_mla::metal_device_info();
        mini_mla::Json report = mini_mla::Json::Object{
            {"cpu", cpu->device_info().name},
            {"physical_memory_bytes",
             static_cast<std::size_t>(cpu->device_info().total_memory_bytes)},
            {"parameters", c.parameter_count()},
            {"fp32_weights_bytes", c.parameter_count() * 4},
            {"fp16_cache_bytes", static_cast<std::size_t>(c.n_layers) *
                                     static_cast<std::size_t>(c.max_sequence_length) *
                                     static_cast<std::size_t>(c.kv_width()) * 2}};
        report["metal"] = metal ? metal.value()
                                : mini_mla::Json(mini_mla::Json::Object{
                                      {"available", false}, {"reason", metal.error().message}});
        std::puts(report.dump().c_str());
        return 0;
    });
}
