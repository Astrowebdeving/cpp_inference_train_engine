#include "mini_mla/cli.hpp"
#include "mini_mla/gemma.hpp"

#include <iostream>

int main(int argc, char** argv) {
    try {
        mini_mla::Args args(argc, argv);
        args.known(
            {"model", "context-limit", "sessions", "full-prefill", "independent-kv", "help"});
        if (args.has("help")) {
            std::puts("mini_mla_gemma --model MLX_CHECKPOINT --context-limit 4096 --sessions 16\n"
                      "Reads one JSON request per line; commands: prefill, decode, release, info.\n"
                      "--independent-kv disables packed-batch cache reuse for comparisons.");
            return 0;
        }
        mini_mla::GemmaEngine engine(args.required("model"), args.integer("context-limit", 4096),
                                     args.integer("sessions", 16), !args.has("full-prefill"),
                                     !args.has("independent-kv"));
        std::cout << engine.info().dump() << std::endl;
        std::string line;
        while (std::getline(std::cin, line)) {
            try {
                if (line.size() > 4 * 1024 * 1024)
                    throw std::invalid_argument("Gemma command exceeds input limit");
                auto request = mini_mla::Json::parse(line);
                if (!request)
                    throw std::invalid_argument(request.error().message);
                const auto& r = request.value();
                auto command = r.at("command").string();
                mini_mla::Json result;
                if (command == "prefill") {
                    std::vector<std::uint32_t> tokens;
                    for (const auto& token : r.at("tokens").array()) {
                        auto id = token.size_integer();
                        if (id > UINT32_MAX)
                            throw std::invalid_argument("Token ID exceeds uint32");
                        tokens.push_back(static_cast<std::uint32_t>(id));
                    }
                    result = engine.prefill(r.at("session").integer(), tokens);
                } else if (command == "prefill_batch") {
                    std::vector<int> sessions;
                    std::vector<std::uint32_t> tokens;
                    for (const auto& session : r.at("sessions").array())
                        sessions.push_back(session.integer());
                    int length = -1;
                    for (const auto& prompt : r.at("prompts").array()) {
                        if (prompt.array().size() > static_cast<std::size_t>(INT_MAX))
                            throw std::invalid_argument("Gemma prompt is too large");
                        int size = static_cast<int>(prompt.array().size());
                        if (length != -1 && size != length)
                            throw std::invalid_argument(
                                "Prefill batches require equal prompt lengths");
                        length = size;
                        for (const auto& token : prompt.array()) {
                            auto id = token.size_integer();
                            if (id > UINT32_MAX)
                                throw std::invalid_argument("Token ID exceeds uint32");
                            tokens.push_back(static_cast<std::uint32_t>(id));
                        }
                    }
                    if (r.at("prompts").array().size() != sessions.size())
                        throw std::invalid_argument("Prefill prompts do not match sessions");
                    result = engine.prefill_batch(sessions, tokens, length);
                } else if (command == "decode") {
                    std::vector<int> sessions;
                    std::vector<std::uint32_t> tokens;
                    for (const auto& session : r.at("sessions").array())
                        sessions.push_back(session.integer());
                    for (const auto& token : r.at("tokens").array()) {
                        auto id = token.size_integer();
                        if (id > UINT32_MAX)
                            throw std::invalid_argument("Token ID exceeds uint32");
                        tokens.push_back(static_cast<std::uint32_t>(id));
                    }
                    result = engine.decode(sessions, tokens);
                } else if (command == "release") {
                    engine.release(r.at("session").integer());
                    result = mini_mla::Json::Object{{"ok", true}};
                } else if (command == "info")
                    result = engine.info();
                else
                    throw std::invalid_argument("Unknown Gemma command");
                if (r.contains("logits"))
                    engine.save_logits(r.at("logits").string());
                if (command != "info" && r.bool_or("include_info", false))
                    result["info"] = engine.info();
                std::cout << result.dump() << std::endl;
            } catch (const std::exception& e) {
                std::cout << mini_mla::Json(mini_mla::Json::Object{{"error", e.what()}}).dump()
                          << std::endl;
            }
        }
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "%s\n", e.what());
        return 1;
    }
}
