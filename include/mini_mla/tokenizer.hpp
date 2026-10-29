#pragma once
#include "mini_mla/json.hpp"

#include <span>
#include <unordered_map>

namespace mini_mla {
class Tokenizer {
  public:
    [[nodiscard]] static Result<Tokenizer> load(const std::filesystem::path&);
    [[nodiscard]] static Tokenizer bytes();
    [[nodiscard]] Result<std::vector<std::uint16_t>> encode(std::string_view,
                                                            bool allow_special = false) const;
    [[nodiscard]] Result<std::string> decode(std::span<const std::uint16_t>,
                                             bool skip_special = false) const;
    [[nodiscard]] Result<std::vector<std::uint16_t>> chat_prompt(const Json::Array& messages) const;
    [[nodiscard]] int vocab_size() const { return static_cast<int>(vocabulary_.size()); }
    [[nodiscard]] const std::string& hash() const { return hash_; }
    [[nodiscard]] std::uint16_t special(std::string_view name) const;
    [[nodiscard]] bool is_special(std::uint16_t id) const;
    [[nodiscard]] std::string_view token_bytes(std::uint16_t id) const;

  private:
    void encode_bytes(std::string_view, std::vector<std::uint16_t>&) const;
    std::vector<std::string> vocabulary_;
    std::unordered_map<std::uint32_t, std::uint16_t> merges_;
    std::map<std::string, std::uint16_t, std::less<>> specials_;
    std::string hash_;
};
[[nodiscard]] bool valid_utf8(std::string_view);
[[nodiscard]] std::string clean_utf8(std::string_view);
} // namespace mini_mla
