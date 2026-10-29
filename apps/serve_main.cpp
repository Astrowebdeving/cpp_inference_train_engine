#include "mini_mla/cli.hpp"
#include "mini_mla/server.hpp"

#include <csignal>

namespace {
volatile std::sig_atomic_t interrupted = 0;
void stop(int) {
    interrupted = 1;
}
} // namespace
int main(int argc, char** argv) {
    return mini_mla::app_main(argc, argv, [](const mini_mla::Args& a) {
        a.known({"model", "backend", "precision", "host", "port", "max-context", "max-new-tokens",
                 "max-queue", "workers", "request-timeout", "attention", "help"});
        if (a.has("help")) {
            std::puts("mini_mla_serve --model DIR [--host 127.0.0.1 --port 8080]\n  --backend "
                      "cpu|mps --precision fp32|fp16 --max-queue 16 --workers 4\n  "
                      "--attention auto|reference|flash");
            return 0;
        }
        auto engine = mini_mla::take(mini_mla::InferenceEngine::load(
            a.required("model"), a.get("backend", "cpu"), a.get("precision", "fp32"),
            {a.get("attention", "auto"), 4}));
        mini_mla::ServerConfig c;
        c.host = a.get("host", c.host);
        c.port = a.integer("port", c.port);
        c.max_context = a.integer("max-context", engine->config().max_sequence_length);
        c.max_new_tokens = a.integer("max-new-tokens", c.max_new_tokens);
        c.max_queue = a.integer("max-queue", c.max_queue);
        c.workers = a.integer("workers", c.workers);
        c.request_timeout_seconds = a.integer("request-timeout", c.request_timeout_seconds);
        std::signal(SIGINT, stop);
        std::signal(SIGTERM, stop);
        std::signal(SIGPIPE, SIG_IGN);
        mini_mla::check(mini_mla::serve(*engine, c, [] { return interrupted != 0; }));
        return 0;
    });
}
