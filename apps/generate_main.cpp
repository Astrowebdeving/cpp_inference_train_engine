#include "app_utils.hpp"

#include <csignal>

int main(int argc, char** argv) {
    return mini_mla::app_main(argc, argv, [](const mini_mla::Args& a) {
        a.known({"model", "prompt", "prompt-file", "chat", "chat-json", "backend", "precision",
                 "max-new-tokens", "temperature", "top-k", "top-p", "repetition-penalty", "seed",
                 "stop", "stop-tokens", "json", "attention", "help"});
        if (a.has("help")) {
            std::puts("mini_mla_generate --model DIR --prompt TEXT [--backend cpu|mps] "
                      "[--precision fp32|fp16 --attention auto|reference|flash]\n  "
                      "--max-new-tokens 128 --temperature 0.8 --top-k 0 "
                      "--top-p 0.95 --seed 42\n  --chat formats a user message; --chat-json FILE "
                      "accepts a messages array.");
            return 0;
        }
        std::signal(SIGPIPE, SIG_IGN);
        auto engine = mini_mla::take(mini_mla::InferenceEngine::load(
            a.required("model"), a.get("backend", "cpu"), a.get("precision", "fp32"),
            {a.get("attention", "auto"), 4}));
        auto config = generation_options(a);
        auto callback = [&](std::string_view text) {
            if (a.has("json"))
                return true;
            auto written = std::fwrite(text.data(), 1, text.size(), stdout);
            std::fflush(stdout);
            return written == text.size();
        };
        mini_mla::Generation result;
        if (a.has("chat-json")) {
            auto messages = mini_mla::take(mini_mla::read_json(a.get("chat-json")));
            auto ids = mini_mla::take(engine->tokenizer().chat_prompt(messages.array()));
            result = mini_mla::take(engine->generate_tokens(ids, config, callback));
        } else {
            auto prompt = a.has("prompt-file")
                              ? mini_mla::take(mini_mla::read_text(a.get("prompt-file")))
                              : a.required("prompt");
            if (a.has("chat")) {
                auto ids = mini_mla::take(engine->tokenizer().chat_prompt(mini_mla::Json::Array{
                    mini_mla::Json::Object{{"role", "user"}, {"content", prompt}}}));
                result = mini_mla::take(engine->generate_tokens(ids, config, callback));
            } else
                result = mini_mla::take(engine->generate(prompt, config, callback));
        }
        mini_mla::Json stats = mini_mla::Json::Object{
            {"prompt_tokens", result.prompt_tokens},
            {"completion_tokens", result.tokens.size()},
            {"finish_reason", result.finish_reason},
            {"prefill_ms", result.prefill_ms},
            {"decode_ms", result.decode_ms},
            {"time_to_first_token_ms", result.time_to_first_token_ms},
            {"prefill_tokens_per_second",
             result.prompt_tokens * 1000.0 / std::max(result.prefill_ms, 1e-9)},
            {"decode_tokens_per_second", (result.tokens.empty() ? 0 : result.tokens.size() - 1) *
                                             1000.0 / std::max(result.decode_ms, 1e-9)}};
        if (a.has("json")) {
            stats["text"] = result.text;
            std::puts(stats.dump().c_str());
        } else {
            std::putchar('\n');
            std::fprintf(stderr, "%s\n", stats.dump().c_str());
        }
        return 0;
    });
}
