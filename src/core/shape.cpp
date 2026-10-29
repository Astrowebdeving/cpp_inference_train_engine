#include "mini_mla/shape.hpp"

#include "mini_mla/checked_math.hpp"

#include <limits>

namespace mini_mla {
Result<Shape> Shape::make(std::span<const std::int64_t> dimensions) {
    if (dimensions.size() > max_rank)
        return Result<Shape>::failure(ErrorCode::invalid_argument, "Tensor rank exceeds four");
    Shape s;
    s.rank = static_cast<std::uint8_t>(dimensions.size());
    for (std::size_t i = 0; i < dimensions.size(); ++i) {
        if (dimensions[i] < 0)
            return Result<Shape>::failure(ErrorCode::invalid_argument, "Negative tensor dimension");
        s.dims[i] = dimensions[i];
    }
    return Result<Shape>::success(s);
}

Result<std::size_t> Shape::numel() const {
    if (rank > max_rank)
        return Result<std::size_t>::failure(ErrorCode::invalid_argument, "Invalid tensor rank");
    bool empty = false;
    for (std::size_t i = 0; i < max_rank; ++i) {
        if ((i < rank && dims[i] < 0) || (i >= rank && dims[i] != 0))
            return Result<std::size_t>::failure(ErrorCode::invalid_argument,
                                                "Invalid tensor dimension");
        if (i < rank && dims[i] == 0)
            empty = true;
    }
    if (empty)
        return Result<std::size_t>::success(0);
    std::size_t n = 1;
    for (std::size_t i = 0; i < rank; ++i) {
        const auto d = static_cast<std::size_t>(dims[i]);
        if (n > std::numeric_limits<std::size_t>::max() / d)
            return Result<std::size_t>::failure(ErrorCode::invalid_argument,
                                                "Tensor element count overflows");
        n *= d;
    }
    return Result<std::size_t>::success(n);
}

std::int64_t Shape::dim(std::size_t index) const {
    if (index >= rank || rank > max_rank)
        detail::fatal("Shape index out of range");
    return dims[index];
}

std::string Shape::to_string() const {
    std::string s = "[";
    for (std::size_t i = 0; i < rank && i < max_rank; ++i) {
        if (i)
            s += ", ";
        s += std::to_string(dims[i]);
    }
    return s + "]";
}

Result<std::array<std::int64_t, Shape::max_rank>> contiguous_strides(const Shape& shape) {
    using Strides = std::array<std::int64_t, Shape::max_rank>;
    auto n = shape.numel();
    if (!n)
        return Result<Strides>::failure(n.error());
    Strides strides{};
    std::int64_t stride = 1;
    for (std::size_t i = shape.rank; i > 0; --i) {
        strides[i - 1] = stride;
        if (shape.dims[i - 1] > 0 && stride > INT64_MAX / shape.dims[i - 1])
            return Result<Strides>::failure(ErrorCode::invalid_argument, "Tensor strides overflow");
        stride *= shape.dims[i - 1];
    }
    return Result<Strides>::success(strides);
}

Result<TensorDesc> TensorDesc::make_contiguous(Shape shape, DType dtype, DeviceType device,
                                               std::size_t byte_offset) {
    auto strides = contiguous_strides(shape);
    if (!strides)
        return Result<TensorDesc>::failure(strides.error());
    TensorDesc desc{shape, strides.value(), dtype, device, byte_offset};
    auto bytes = desc.required_bytes();
    if (!bytes)
        return Result<TensorDesc>::failure(bytes.error());
    return Result<TensorDesc>::success(desc);
}

Result<std::size_t> TensorDesc::required_bytes() const {
    auto n = shape.numel();
    if (!n)
        return Result<std::size_t>::failure(n.error());
    const auto item = dtype_size_bytes(dtype);
    if (item == 0 || (device != DeviceType::cpu && device != DeviceType::metal) ||
        byte_offset % item)
        return Result<std::size_t>::failure(ErrorCode::invalid_argument,
                                            "Invalid dtype, device, or byte alignment");
    for (std::size_t i = 0; i < Shape::max_rank; ++i)
        if (strides[i] < 0 || (i >= shape.rank && strides[i] != 0))
            return Result<std::size_t>::failure(ErrorCode::invalid_argument,
                                                "Invalid tensor stride");
    if (n.value() == 0)
        return Result<std::size_t>::success(byte_offset);
    std::size_t extent = 0;
    for (std::size_t i = 0; i < shape.rank; ++i) {
        std::size_t term;
        if (__builtin_mul_overflow(static_cast<std::size_t>(shape.dims[i] - 1),
                                   static_cast<std::size_t>(strides[i]), &term) ||
            __builtin_add_overflow(extent, term, &extent))
            return Result<std::size_t>::failure(ErrorCode::invalid_argument,
                                                "Tensor extent overflows");
    }
    if (__builtin_add_overflow(extent, std::size_t{1}, &extent) ||
        __builtin_mul_overflow(extent, item, &extent) ||
        __builtin_add_overflow(extent, byte_offset, &extent))
        return Result<std::size_t>::failure(ErrorCode::invalid_argument,
                                            "Tensor byte count overflows");
    return Result<std::size_t>::success(extent);
}

bool TensorDesc::is_contiguous() const {
    auto n = shape.numel();
    if (!n || !required_bytes())
        return false;
    if (n.value() == 0)
        return true;
    std::int64_t expected = 1;
    for (std::size_t i = shape.rank; i > 0; --i) {
        if (shape.dims[i - 1] != 1 && strides[i - 1] != expected)
            return false;
        if (__builtin_mul_overflow(expected, shape.dims[i - 1], &expected))
            return false;
    }
    return true;
}
std::string TensorDesc::to_string() const {
    return shape.to_string() + " " + std::string(mini_mla::to_string(dtype));
}
} // namespace mini_mla
