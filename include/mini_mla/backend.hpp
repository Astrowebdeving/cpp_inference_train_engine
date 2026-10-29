#pragma once

#include "mini_mla/buffer.hpp"
#include "mini_mla/dtype.hpp"
#include "mini_mla/result.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <vector>

namespace mini_mla {

struct DeviceInfo {
    DeviceType type{DeviceType::cpu};
    std::string name;
    std::string os_version;
    std::uint64_t total_memory_bytes{0};
    std::uint64_t recommended_max_working_set_bytes{0};
    bool unified_memory{false};
    std::vector<std::string> feature_notes;
};

[[nodiscard]] std::string describe(const DeviceInfo& info);

enum class BufferUsage : std::uint8_t {
    storage,
    staging,
};

class Backend {
  public:
    Backend() = default;

    virtual ~Backend() = default;

    Backend(const Backend&) = delete;
    Backend& operator=(const Backend&) = delete;
    Backend(Backend&&) = delete;
    Backend& operator=(Backend&&) = delete;

    [[nodiscard]] virtual DeviceInfo device_info() const = 0;

    [[nodiscard]] virtual Result<Buffer> allocate(std::size_t bytes, BufferUsage usage) = 0;

    [[nodiscard]] virtual Status upload(Buffer& destination, std::span<const std::byte> source) = 0;

    [[nodiscard]] virtual Status download(const Buffer& source,
                                          std::span<std::byte> destination) = 0;
};

[[nodiscard]] Result<std::unique_ptr<Backend>> make_cpu_backend();

} // namespace mini_mla
