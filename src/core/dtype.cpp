#include "mini_mla/dtype.hpp"

namespace mini_mla {

std::string_view to_string(DType dtype) noexcept {
    switch (dtype) {
    case DType::f32:
        return "f32";
    case DType::f16:
        return "f16";
    case DType::bf16:
        return "bf16";
    case DType::i32:
        return "i32";
    case DType::u16:
        return "u16";
    case DType::u8:
        return "u8";
    }
    __builtin_unreachable();
}

std::string_view to_string(DeviceType device) noexcept {
    switch (device) {
    case DeviceType::cpu:
        return "cpu";
    case DeviceType::metal:
        return "metal";
    }
    __builtin_unreachable();
}

} // namespace mini_mla
