#pragma once

#include "mini_mla/dtype.hpp"
#include "mini_mla/result.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace mini_mla {

struct Shape {
    static constexpr std::size_t max_rank = 4;

    std::array<std::int64_t, max_rank> dims{};
    std::uint8_t rank{0};

    [[nodiscard]] static Result<Shape> make(std::span<const std::int64_t> dims);

    [[nodiscard]] Result<std::size_t> numel() const;

    [[nodiscard]] std::int64_t dim(std::size_t index) const;

    [[nodiscard]] bool operator==(const Shape& other) const noexcept = default;

    [[nodiscard]] std::string to_string() const;
};

[[nodiscard]] Result<std::array<std::int64_t, Shape::max_rank>>
contiguous_strides(const Shape& shape);

struct TensorDesc {
    Shape shape{};
    std::array<std::int64_t, Shape::max_rank> strides{};
    DType dtype{DType::f32};
    DeviceType device{DeviceType::cpu};
    std::size_t byte_offset{0};

    [[nodiscard]] static Result<TensorDesc>
    make_contiguous(Shape shape, DType dtype, DeviceType device, std::size_t byte_offset = 0);

    [[nodiscard]] Result<std::size_t> required_bytes() const;

    [[nodiscard]] bool is_contiguous() const;

    [[nodiscard]] std::string to_string() const;
};

} // namespace mini_mla
