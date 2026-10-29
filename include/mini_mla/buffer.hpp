#pragma once

#include "mini_mla/dtype.hpp"
#include "mini_mla/result.hpp"

#include <cstddef>
#include <memory>
#include <span>

namespace mini_mla {

class Buffer {
  public:
    Buffer() noexcept;
    ~Buffer();

    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;

    Buffer(Buffer&&) noexcept;
    Buffer& operator=(Buffer&&) noexcept;

    [[nodiscard]] static Result<Buffer> allocate_cpu(std::size_t bytes, std::size_t alignment = 64);

    [[nodiscard]] bool empty() const noexcept;
    [[nodiscard]] std::size_t size_bytes() const noexcept;
    [[nodiscard]] DeviceType device() const noexcept;

    [[nodiscard]] std::span<std::byte> bytes() noexcept;
    [[nodiscard]] std::span<const std::byte> bytes() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace mini_mla
