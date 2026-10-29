#pragma once
#include "mini_mla/inference.hpp"

namespace mini_mla {
struct ServerConfig {
    std::string host = "127.0.0.1";
    int port = 8080, max_context = 2048, max_new_tokens = 128, max_queue = 16, workers = 4;
    int request_timeout_seconds = 60;
    std::size_t max_request_bytes = 1024 * 1024;
};
[[nodiscard]] Status serve(InferenceEngine&, const ServerConfig&,
                           std::function<bool()> stop_requested);
} // namespace mini_mla
