#pragma once
#include "mini_mla/result.hpp"

#include <filesystem>
#include <map>
#include <vector>

namespace mini_mla {
class Json {
  public:
    using Array = std::vector<Json>;
    using Object = std::map<std::string, Json, std::less<>>;
    Json() = default;
    Json(bool v) : value_(v) {}
    Json(int v) : value_(static_cast<double>(v)) {}
    Json(std::size_t v) : value_(static_cast<double>(v)) {}
    Json(double v) : value_(v) {}
    Json(const char* v) : value_(std::string(v)) {}
    Json(std::string v) : value_(std::move(v)) {}
    Json(Array v) : value_(std::move(v)) {}
    Json(Object v) : value_(std::move(v)) {}
    [[nodiscard]] static Result<Json> parse(std::string_view text);
    [[nodiscard]] std::string dump() const;
    [[nodiscard]] const Json& at(std::string_view key) const;
    [[nodiscard]] bool contains(std::string_view key) const;
    Json& operator[](std::string key);
    [[nodiscard]] const Array& array() const;
    [[nodiscard]] const Object& object() const;
    [[nodiscard]] const std::string& string() const;
    [[nodiscard]] double number() const;
    [[nodiscard]] int integer() const;
    [[nodiscard]] std::size_t size_integer() const;
    [[nodiscard]] bool boolean() const;
    [[nodiscard]] bool is_null() const { return std::holds_alternative<std::monostate>(value_); }
    [[nodiscard]] bool is_string() const { return std::holds_alternative<std::string>(value_); }
    [[nodiscard]] std::string str_or(std::string_view key, std::string fallback) const;
    [[nodiscard]] double num_or(std::string_view key, double fallback) const;
    [[nodiscard]] bool bool_or(std::string_view key, bool fallback) const;

  private:
    std::variant<std::monostate, bool, double, std::string, Array, Object> value_;
};
[[nodiscard]] Result<Json> read_json(const std::filesystem::path&);
[[nodiscard]] Status write_json(const std::filesystem::path&, const Json&);
} // namespace mini_mla
