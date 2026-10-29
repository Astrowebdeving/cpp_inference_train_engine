#pragma once

#include <cstdint>
#include <source_location>
#include <string>
#include <string_view>
#include <utility>
#include <variant>

namespace mini_mla {

enum class ErrorCode : std::uint8_t {
    invalid_argument,
    io_error,
    unsupported,
    out_of_memory,
    gpu_error,
    numerical_error,
};

[[nodiscard]] std::string_view to_string(ErrorCode code) noexcept;

struct Error {
    ErrorCode code{ErrorCode::invalid_argument};
    std::string message;
};

[[nodiscard]] Error make_error(ErrorCode code, std::string message);

namespace detail {

[[noreturn]] void fatal(std::string_view message,
                        std::source_location location = std::source_location::current());

}

template <typename T> class [[nodiscard]] Result {
  public:
    using value_type = T;

    [[nodiscard]] static Result success(T value) {
        return Result(std::in_place_index<0>, std::move(value));
    }

    [[nodiscard]] static Result failure(Error error) {
        return Result(std::in_place_index<1>, std::move(error));
    }

    [[nodiscard]] static Result failure(ErrorCode code, std::string message) {
        return failure(make_error(code, std::move(message)));
    }

    [[nodiscard]] bool has_value() const noexcept { return storage_.index() == 0; }

    [[nodiscard]] explicit operator bool() const noexcept { return has_value(); }

    [[nodiscard]] T& value() & {
        require_value();
        return std::get<0>(storage_);
    }
    [[nodiscard]] const T& value() const& {
        require_value();
        return std::get<0>(storage_);
    }
    [[nodiscard]] T&& value() && {
        require_value();
        return std::get<0>(std::move(storage_));
    }

    [[nodiscard]] const Error& error() const& {
        require_error();
        return std::get<1>(storage_);
    }
    [[nodiscard]] Error&& error() && {
        require_error();
        return std::get<1>(std::move(storage_));
    }

    template <typename U> [[nodiscard]] static Result propagate(Result<U>&& other) {
        return failure(std::move(other).error());
    }

  private:
    explicit Result(std::in_place_index_t<0> tag, T value) : storage_(tag, std::move(value)) {}
    explicit Result(std::in_place_index_t<1> tag, Error error) : storage_(tag, std::move(error)) {}

    void require_value() const {
        if (!has_value()) {
            detail::fatal("Result::value() called on a failed Result");
        }
    }
    void require_error() const {
        if (has_value()) {
            detail::fatal("Result::error() called on a successful Result");
        }
    }

    std::variant<T, Error> storage_;
};

using Status = Result<std::monostate>;

[[nodiscard]] inline Status ok() {
    return Status::success(std::monostate{});
}

} // namespace mini_mla
