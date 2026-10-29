#pragma once
#include "mini_mla/result.hpp"

#include <array>
#include <filesystem>
#include <memory>
#include <span>

namespace mini_mla {
class MappedFile {
  public:
    MappedFile() noexcept;
    ~MappedFile();
    MappedFile(MappedFile&&) noexcept;
    MappedFile& operator=(MappedFile&&) noexcept;
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    [[nodiscard]] static Result<MappedFile> open(const std::filesystem::path&);
    [[nodiscard]] std::span<const std::byte> bytes() const noexcept;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
[[nodiscard]] Result<std::string> read_text(const std::filesystem::path&,
                                            std::size_t limit = 64 * 1024 * 1024);
[[nodiscard]] Status write_bytes(const std::filesystem::path&, std::span<const std::byte>);
[[nodiscard]] Status write_text(const std::filesystem::path&, std::string_view);
[[nodiscard]] Status sync_directory(const std::filesystem::path&);
[[nodiscard]] Status sync_file(const std::filesystem::path&);
[[nodiscard]] std::array<std::uint8_t, 32> sha256(std::span<const std::byte>);
[[nodiscard]] std::string hex(std::span<const std::uint8_t>);
[[nodiscard]] std::string hash_text(std::string_view);
[[nodiscard]] std::uint32_t crc32(std::span<const std::byte>);
[[nodiscard]] std::uint16_t to_half(float) noexcept;
[[nodiscard]] float from_half(std::uint16_t) noexcept;
} // namespace mini_mla
