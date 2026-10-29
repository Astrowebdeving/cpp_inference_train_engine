#include "mini_mla/buffer.hpp"

#include <cstdlib>
#include <new>

namespace mini_mla {
struct Buffer::Impl {
    struct Deleter {
        void operator()(std::byte* p) const noexcept { std::free(p); }
    };
    std::unique_ptr<std::byte, Deleter> memory;
    std::size_t size = 0;
};
Buffer::Buffer() noexcept = default;
Buffer::~Buffer() = default;
Buffer::Buffer(Buffer&&) noexcept = default;
Buffer& Buffer::operator=(Buffer&&) noexcept = default;
Result<Buffer> Buffer::allocate_cpu(std::size_t bytes, std::size_t alignment) {
    if (alignment < alignof(std::max_align_t) || (alignment & (alignment - 1)) != 0)
        return Result<Buffer>::failure(ErrorCode::invalid_argument,
                                       "Alignment must be a power of two >= max_align_t");
    Buffer b;
    if (bytes == 0)
        return Result<Buffer>::success(std::move(b));
    try {
        b.impl_ = std::make_unique<Impl>();
    } catch (const std::bad_alloc&) {
        return Result<Buffer>::failure(ErrorCode::out_of_memory,
                                       "Buffer metadata allocation failed");
    }
    void* p = nullptr;
    if (::posix_memalign(&p, alignment, bytes) != 0)
        return Result<Buffer>::failure(ErrorCode::out_of_memory, "Aligned allocation failed");
    b.impl_->memory.reset(static_cast<std::byte*>(p));
    b.impl_->size = bytes;
    return Result<Buffer>::success(std::move(b));
}
bool Buffer::empty() const noexcept {
    return size_bytes() == 0;
}
std::size_t Buffer::size_bytes() const noexcept {
    return impl_ ? impl_->size : 0;
}
DeviceType Buffer::device() const noexcept {
    return DeviceType::cpu;
}
std::span<std::byte> Buffer::bytes() noexcept {
    return impl_ ? std::span{impl_->memory.get(), impl_->size} : std::span<std::byte>{};
}
std::span<const std::byte> Buffer::bytes() const noexcept {
    return impl_ ? std::span<const std::byte>{impl_->memory.get(), impl_->size}
                 : std::span<const std::byte>{};
}
} // namespace mini_mla
