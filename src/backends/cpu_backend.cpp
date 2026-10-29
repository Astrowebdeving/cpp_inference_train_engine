#include "mini_mla/backend.hpp"

#include <cstring>
#include <sys/sysctl.h>

namespace mini_mla {
namespace {
class CpuBackend final : public Backend {
  public:
    DeviceInfo device_info() const override {
        DeviceInfo info;
        info.name = "CPU / Accelerate";
        std::size_t n = sizeof(info.total_memory_bytes);
        ::sysctlbyname("hw.memsize", &info.total_memory_bytes, &n, nullptr, 0);
        info.unified_memory = true;
        return info;
    }
    Result<Buffer> allocate(std::size_t bytes, BufferUsage) override {
        return Buffer::allocate_cpu(bytes);
    }
    Status upload(Buffer& dst, std::span<const std::byte> src) override {
        if (dst.device() != DeviceType::cpu || src.size() > dst.size_bytes())
            return Status::failure(ErrorCode::invalid_argument, "Upload exceeds CPU buffer");
        if (!src.empty())
            std::memmove(dst.bytes().data(), src.data(), src.size());
        return ok();
    }
    Status download(const Buffer& src, std::span<std::byte> dst) override {
        if (src.device() != DeviceType::cpu || dst.size() != src.size_bytes())
            return Status::failure(ErrorCode::invalid_argument,
                                   "Download size must equal buffer size");
        if (!dst.empty())
            std::memmove(dst.data(), src.bytes().data(), dst.size());
        return ok();
    }
};
} // namespace
Result<std::unique_ptr<Backend>> make_cpu_backend() {
    return Result<std::unique_ptr<Backend>>::success(std::make_unique<CpuBackend>());
}
std::string describe(const DeviceInfo& d) {
    return d.name + "\nMemory: " + std::to_string(d.total_memory_bytes) + " bytes\n";
}
} // namespace mini_mla
