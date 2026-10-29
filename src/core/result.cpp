#include "mini_mla/result.hpp"

#include <cstdio>
#include <cstdlib>

namespace mini_mla {

std::string_view to_string(ErrorCode code) noexcept {
    switch (code) {
    case ErrorCode::invalid_argument:
        return "invalid_argument";
    case ErrorCode::io_error:
        return "io_error";
    case ErrorCode::unsupported:
        return "unsupported";
    case ErrorCode::out_of_memory:
        return "out_of_memory";
    case ErrorCode::gpu_error:
        return "gpu_error";
    case ErrorCode::numerical_error:
        return "numerical_error";
    }

    return "unknown_error";
}

Error make_error(ErrorCode code, std::string message) {
    return Error{code, std::move(message)};
}

namespace detail {

void fatal(std::string_view message, std::source_location location) {

    std::fprintf(stderr, "fatal: %.*s\n  at %s:%u in %s\n", static_cast<int>(message.size()),
                 message.data(), location.file_name(), location.line(), location.function_name());
    std::fflush(stderr);
    std::abort();
}

} // namespace detail

} // namespace mini_mla
