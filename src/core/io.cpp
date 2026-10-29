#include "mini_mla/io.hpp"

#include <CommonCrypto/CommonDigest.h>
#include <algorithm>
#include <bit>
#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace mini_mla {
namespace {
class FileDescriptor {
  public:
    explicit FileDescriptor(int fd) : fd_(fd) {}
    ~FileDescriptor() {
        if (fd_ >= 0)
            ::close(fd_);
    }
    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;
    [[nodiscard]] int get() const { return fd_; }

  private:
    int fd_;
};
Error io_error(const std::filesystem::path& p) {
    return make_error(ErrorCode::io_error, p.string() + ": " + std::strerror(errno));
}
} // namespace
struct MappedFile::Impl {
    void* ptr = MAP_FAILED;
    std::size_t size = 0;
    ~Impl() {
        if (ptr != MAP_FAILED)
            ::munmap(ptr, size);
    }
};
MappedFile::MappedFile() noexcept = default;
MappedFile::~MappedFile() = default;
MappedFile::MappedFile(MappedFile&&) noexcept = default;
MappedFile& MappedFile::operator=(MappedFile&&) noexcept = default;
Result<MappedFile> MappedFile::open(const std::filesystem::path& path) {
    FileDescriptor fd(::open(path.c_str(), O_RDONLY | O_CLOEXEC));
    if (fd.get() < 0)
        return Result<MappedFile>::failure(io_error(path));
    struct stat st{};
    if (::fstat(fd.get(), &st) != 0)
        return Result<MappedFile>::failure(io_error(path));
    if (!S_ISREG(st.st_mode) || st.st_size < 0)
        return Result<MappedFile>::failure(ErrorCode::invalid_argument,
                                           "Expected regular file: " + path.string());
    MappedFile f;
    f.impl_ = std::make_unique<Impl>();
    f.impl_->size = static_cast<std::size_t>(st.st_size);
    if (f.impl_->size) {
        f.impl_->ptr = ::mmap(nullptr, f.impl_->size, PROT_READ, MAP_PRIVATE, fd.get(), 0);
        if (f.impl_->ptr == MAP_FAILED)
            return Result<MappedFile>::failure(io_error(path));
    }
    return Result<MappedFile>::success(std::move(f));
}
std::span<const std::byte> MappedFile::bytes() const noexcept {
    return impl_ && impl_->size ? std::span{static_cast<const std::byte*>(impl_->ptr), impl_->size}
                                : std::span<const std::byte>{};
}
Result<std::string> read_text(const std::filesystem::path& p, std::size_t limit) {
    auto f = MappedFile::open(p);
    if (!f)
        return Result<std::string>::failure(f.error());
    auto b = f.value().bytes();
    if (b.size() > limit)
        return Result<std::string>::failure(ErrorCode::invalid_argument,
                                            "Text file exceeds size limit: " + p.string());
    return Result<std::string>::success(
        b.empty() ? std::string{} : std::string(reinterpret_cast<const char*>(b.data()), b.size()));
}
Status write_bytes(const std::filesystem::path& p, std::span<const std::byte> data) {
    FileDescriptor fd(::open(p.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0600));
    if (fd.get() < 0)
        return Status::failure(io_error(p));
    while (!data.empty()) {
        auto n = ::write(fd.get(), data.data(), std::min(data.size(), std::size_t{1} << 30));
        if (n < 0 && errno == EINTR)
            continue;
        if (n <= 0)
            return Status::failure(io_error(p));
        data = data.subspan(static_cast<std::size_t>(n));
    }
    if (::fsync(fd.get()) != 0)
        return Status::failure(io_error(p));
    return ok();
}
Status write_text(const std::filesystem::path& p, std::string_view s) {
    return write_bytes(p, std::as_bytes(std::span{s.data(), s.size()}));
}
Status sync_directory(const std::filesystem::path& p) {
    FileDescriptor fd(::open(p.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC));
    if (fd.get() < 0 || ::fsync(fd.get()) != 0)
        return Status::failure(io_error(p));
    return ok();
}
Status sync_file(const std::filesystem::path& p) {
    FileDescriptor fd(::open(p.c_str(), O_RDONLY | O_CLOEXEC));
    if (fd.get() < 0 || ::fsync(fd.get()) != 0)
        return Status::failure(io_error(p));
    return ok();
}
std::array<std::uint8_t, 32> sha256(std::span<const std::byte> bytes) {
    CC_SHA256_CTX ctx;
    CC_SHA256_Init(&ctx);
    while (!bytes.empty()) {
        auto n = std::min(bytes.size(), std::size_t{1} << 30);
        CC_SHA256_Update(&ctx, bytes.data(), static_cast<CC_LONG>(n));
        bytes = bytes.subspan(n);
    }
    std::array<std::uint8_t, 32> result{};
    CC_SHA256_Final(result.data(), &ctx);
    return result;
}
std::string hex(std::span<const std::uint8_t> b) {
    const char* digits = "0123456789abcdef";
    std::string s;
    s.reserve(b.size() * 2);
    for (auto v : b) {
        s += digits[v >> 4];
        s += digits[v & 15];
    }
    return s;
}
std::string hash_text(std::string_view s) {
    return hex(sha256(std::as_bytes(std::span{s.data(), s.size()})));
}
std::uint32_t crc32(std::span<const std::byte> data) {
    static const auto table = [] {
        std::array<std::uint32_t, 256> t{};
        for (std::uint32_t i = 0; i < 256; ++i) {
            auto x = i;
            for (int k = 0; k < 8; ++k)
                x = (x >> 1) ^ ((x & 1) ? 0xedb88320U : 0);
            t[i] = x;
        }
        return t;
    }();
    std::uint32_t c = 0xffffffffU;
    for (auto x : data)
        c = table[(c ^ std::to_integer<std::uint32_t>(x)) & 255] ^ (c >> 8);
    return c ^ 0xffffffffU;
}
std::uint16_t to_half(float f) noexcept {
    _Float16 h = static_cast<_Float16>(f);
    return std::bit_cast<std::uint16_t>(h);
}
float from_half(std::uint16_t h) noexcept {
    return static_cast<float>(std::bit_cast<_Float16>(h));
}
} // namespace mini_mla
