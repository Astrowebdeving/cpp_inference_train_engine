#pragma once
#include "mini_mla/json.hpp"

#include <memory>
#include <span>

namespace mini_mla {
class GemmaEngine {
  public:
    explicit GemmaEngine(const std::filesystem::path& model, int context_limit = 4096,
                         int session_limit = 16, bool reduce_prefill_queries = true);
    ~GemmaEngine();
    GemmaEngine(const GemmaEngine&) = delete;
    GemmaEngine& operator=(const GemmaEngine&) = delete;
    [[nodiscard]] Json prefill(int session, std::span<const std::uint32_t> ids);
    [[nodiscard]] Json prefill_batch(std::span<const int> sessions,
                                     std::span<const std::uint32_t> ids, int length);
    [[nodiscard]] Json decode(std::span<const int> sessions, std::span<const std::uint32_t> tokens);
    void release(int session);
    void save_logits(const std::filesystem::path&) const;
    [[nodiscard]] Json info() const;

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};
} // namespace mini_mla
