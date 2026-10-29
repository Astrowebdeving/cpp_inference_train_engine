#pragma once
#include "mini_mla/cli.hpp"
#include "mini_mla/inference.hpp"

inline std::vector<std::uint16_t> parse_tokens(std::string_view text);
inline mini_mla::OptimizerConfig optimizer_options(const mini_mla::Args& args,
                                                   mini_mla::OptimizerConfig c = {}) {
    c.kind = args.get("optimizer", c.kind);
    if (c.kind == "muon")
        c.kind = "hybrid_muon_adamw";
    c.weight_decay = args.real("weight-decay", c.weight_decay);
    c.clip_norm = args.real("grad-clip", c.clip_norm);
    c.momentum = args.real("muon-momentum", c.momentum);
    c.ns_steps = args.integer("muon-ns-steps", c.ns_steps);
    if (args.has("muon-nesterov")) {
        auto value = args.get("muon-nesterov");
        if (value != "true" && value != "false")
            throw std::invalid_argument("--muon-nesterov must be true or false");
        c.nesterov = value == "true";
    }
    if (c.kind != "hybrid_muon_adamw" &&
        (args.has("muon-momentum") || args.has("muon-ns-steps") || args.has("muon-nesterov") ||
         args.has("muon-layout") || args.has("muon-ns-order")))
        throw std::invalid_argument("Muon options require --optimizer muon");
    mini_mla::check(c.validate());
    return c;
}
inline mini_mla::GenerationConfig generation_options(const mini_mla::Args& args) {
    mini_mla::GenerationConfig c;
    c.max_new_tokens = args.integer("max-new-tokens", c.max_new_tokens);
    c.temperature = args.real("temperature", c.temperature);
    c.top_k = args.integer("top-k", c.top_k);
    c.top_p = args.real("top-p", c.top_p);
    c.repetition_penalty = args.real("repetition-penalty", c.repetition_penalty);
    c.seed = args.u64("seed", c.seed);
    c.stop = args.strings("stop");
    for (const auto& sequence : args.strings("stop-tokens"))
        c.stop_tokens.push_back(parse_tokens(sequence));
    mini_mla::check(c.validate());
    return c;
}
inline std::vector<std::uint16_t> parse_tokens(std::string_view text) {
    std::vector<std::uint16_t> ids;
    while (!text.empty()) {
        auto p = text.find(',');
        auto part = text.substr(0, p);
        unsigned v;
        auto r = std::from_chars(part.data(), part.data() + part.size(), v);
        if (r.ec != std::errc{} || r.ptr != part.data() + part.size() || v > 65535)
            throw std::invalid_argument("Tokens must be comma-separated uint16 values");
        ids.push_back(static_cast<std::uint16_t>(v));
        if (p == std::string_view::npos)
            break;
        text.remove_prefix(p + 1);
    }
    return ids;
}
