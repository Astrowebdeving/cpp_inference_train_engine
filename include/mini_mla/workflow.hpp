#pragma once
#include "mini_mla/result.hpp"

#include <stdexcept>

namespace mini_mla {
class Failure : public std::runtime_error {
  public:
    explicit Failure(Error error) : std::runtime_error(error.message), error_(std::move(error)) {}
    [[nodiscard]] const Error& error() const { return error_; }

  private:
    Error error_;
};
template <class T> T take(Result<T> result) {
    if (!result)
        throw Failure(std::move(result).error());
    return std::move(result).value();
}
inline void check(Status status) {
    if (!status)
        throw Failure(std::move(status).error());
}
} // namespace mini_mla
