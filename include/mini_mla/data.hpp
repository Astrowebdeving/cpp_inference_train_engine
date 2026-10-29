#pragma once
#include "mini_mla/io.hpp"
#include "mini_mla/json.hpp"
#include "mini_mla/rng.hpp"

#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>

namespace mini_mla {
struct Batch {
    int batch = 0, sequence = 0;
    std::vector<std::uint16_t> input, target;
    std::vector<float> mask;
    [[nodiscard]] Result<std::vector<float>> coefficients(float scale = 1) const;
    [[nodiscard]] std::size_t supervised_tokens() const;
};
class TokenShard {
  public:
    [[nodiscard]] static Result<TokenShard> open(const std::filesystem::path&, int vocab_size,
                                                 std::string_view tokenizer_hash);
    [[nodiscard]] std::span<const std::uint16_t> tokens() const { return tokens_; }
    [[nodiscard]] const std::string& payload_hash() const { return payload_hash_; }

  private:
    MappedFile file_;
    std::span<const std::uint16_t> tokens_;
    std::string payload_hash_;
};
class DataLoader {
  public:
    [[nodiscard]] static Result<std::unique_ptr<DataLoader>>
    tokens(std::span<const std::filesystem::path> paths, int sequence, int vocab,
           std::string_view tokenizer_hash, std::uint64_t seed);
    [[nodiscard]] static Result<std::unique_ptr<DataLoader>>
    sft(std::span<const std::filesystem::path> paths, int sequence, int vocab,
        std::string_view tokenizer_hash, std::uint64_t seed);
    [[nodiscard]] Result<Batch> next(int batch);
    [[nodiscard]] Json state() const;
    [[nodiscard]] Status restore(const Json&);
    [[nodiscard]] const std::string& fingerprint() const { return fingerprint_; }

  private:
    struct Sample {
        std::vector<std::uint16_t> tokens;
        std::vector<float> mask;
    };
    explicit DataLoader(int sequence, std::uint64_t seed) : sequence_(sequence), rng_(seed) {}
    void begin_epoch();
    std::vector<TokenShard> shards_;
    std::vector<std::size_t> ends_;
    std::vector<Sample> samples_;
    int sequence_, pad_ = 0;
    std::uint64_t count_ = 0, position_ = 0, epoch_ = 0, multiplier_ = 1, offset_ = 0;
    Rng rng_;
    std::string fingerprint_;
};
struct PreparedBatch {
    Batch batch;
    Json state_after;
};
class PrefetchLoader {
  public:
    PrefetchLoader(std::unique_ptr<DataLoader>, int batch, std::size_t capacity = 2);
    ~PrefetchLoader();
    PrefetchLoader(const PrefetchLoader&) = delete;
    PrefetchLoader& operator=(const PrefetchLoader&) = delete;
    [[nodiscard]] Result<Batch> next();
    [[nodiscard]] const Json& consumed_state() const { return consumed_; }

  private:
    void run();
    std::unique_ptr<DataLoader> loader_;
    int batch_;
    std::size_t capacity_;
    Json consumed_;
    std::mutex mutex_;
    std::condition_variable readable_, writable_;
    std::deque<Result<PreparedBatch>> queue_;
    bool stopping_ = false;
    std::thread worker_;
};
} // namespace mini_mla
