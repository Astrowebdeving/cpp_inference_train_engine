#pragma once

#include <cstddef>
#include <cstdint>
#include <string_view>

namespace mini_mla {

enum class DType : std::uint8_t {
    f32,
    f16,
    bf16,
    i32,
    u16,
    u8,
};

enum class DeviceType : std::uint8_t {
    cpu,
    metal,
};

[[nodiscard]] constexpr std::size_t dtype_size_bytes(DType dtype) noexcept {
    switch (dtype) {
    case DType::f32:
        return 4;
    case DType::i32:
        return 4;
    case DType::f16:
        return 2;
    case DType::bf16:
        return 2;
    case DType::u16:
        return 2;
    case DType::u8:
        return 1;
    }
    return 0;
}

[[nodiscard]] constexpr bool is_float_dtype(DType dtype) noexcept {
    return dtype == DType::f32 || dtype == DType::f16 || dtype == DType::bf16;
}

static_assert(dtype_size_bytes(DType::f32) == 4);
static_assert(dtype_size_bytes(DType::f16) == 2);
static_assert(dtype_size_bytes(DType::u8) == 1);
static_assert(is_float_dtype(DType::bf16));
static_assert(!is_float_dtype(DType::i32));

[[nodiscard]] std::string_view to_string(DType dtype) noexcept;
[[nodiscard]] std::string_view to_string(DeviceType device) noexcept;

template <typename T> struct DTypeOf;

template <> struct DTypeOf<float> {
    static constexpr DType value = DType::f32;
};
template <> struct DTypeOf<std::int32_t> {
    static constexpr DType value = DType::i32;
};
template <> struct DTypeOf<std::uint16_t> {
    static constexpr DType value = DType::u16;
};
template <> struct DTypeOf<std::uint8_t> {
    static constexpr DType value = DType::u8;
};

template <typename T> inline constexpr DType dtype_of = DTypeOf<T>::value;

static_assert(dtype_of<float> == DType::f32);

} // namespace mini_mla
