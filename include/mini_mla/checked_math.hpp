#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace mini_mla {

[[nodiscard]] constexpr std::optional<std::int64_t> checked_mul(std::int64_t a,
                                                                std::int64_t b) noexcept {
    std::int64_t result = 0;
    if (__builtin_mul_overflow(a, b, &result)) {
        return std::nullopt;
    }
    return result;
}

[[nodiscard]] constexpr std::optional<std::int64_t> checked_add(std::int64_t a,
                                                                std::int64_t b) noexcept {
    std::int64_t result = 0;
    if (__builtin_add_overflow(a, b, &result)) {
        return std::nullopt;
    }
    return result;
}

[[nodiscard]] constexpr std::optional<std::size_t> to_size(std::int64_t value) noexcept {
    if (value < 0) {
        return std::nullopt;
    }
    using Limit = std::numeric_limits<std::size_t>;
    if (static_cast<std::uint64_t>(value) > Limit::max()) {
        return std::nullopt;
    }
    return static_cast<std::size_t>(value);
}

[[nodiscard]] constexpr std::optional<std::size_t> checked_mul_size(std::size_t a,
                                                                    std::size_t b) noexcept {
    std::size_t result = 0;
    if (__builtin_mul_overflow(a, b, &result)) {
        return std::nullopt;
    }
    return result;
}

[[nodiscard]] constexpr std::optional<std::size_t> checked_add_size(std::size_t a,
                                                                    std::size_t b) noexcept {
    std::size_t result = 0;
    if (__builtin_add_overflow(a, b, &result)) {
        return std::nullopt;
    }
    return result;
}

static_assert(checked_mul(1'000'000, 1'000'000).value() == 1'000'000'000'000);
static_assert(!checked_mul(std::numeric_limits<std::int64_t>::max(), 2).has_value());
static_assert(!to_size(-1).has_value());

} // namespace mini_mla
