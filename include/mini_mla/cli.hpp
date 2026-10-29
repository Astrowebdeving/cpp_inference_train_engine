#pragma once
#include "mini_mla/workflow.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstdio>
#include <filesystem>
#include <glob.h>
#include <map>
#include <span>
#include <vector>

namespace mini_mla {
class Args {
  public:
    Args(int argc, char** argv) {
        for (int i = 1; i < argc; ++i) {
            std::string key = argv[i];
            if (!key.starts_with("--"))
                throw std::invalid_argument("Expected --option, got " + key);
            key.erase(0, 2);
            auto equals = key.find('=');
            if (equals != std::string::npos) {
                auto value = key.substr(equals + 1);
                key.resize(equals);
                values_[key].push_back(value);
                continue;
            }
            auto& values = values_[key];
            while (i + 1 < argc && !std::string_view(argv[i + 1]).starts_with("--"))
                values.emplace_back(argv[++i]);
        }
    }
    void known(std::initializer_list<std::string_view> names) const {
        for (const auto& [key, v] : values_) {
            (void)v;
            if (std::find(names.begin(), names.end(), key) == names.end())
                throw std::invalid_argument("Unknown option --" + key);
        }
    }
    [[nodiscard]] bool has(std::string_view key) const { return values_.contains(key); }
    [[nodiscard]] std::string get(std::string_view key, std::string fallback = {}) const {
        auto it = values_.find(key);
        if (it == values_.end())
            return fallback;
        if (it->second.size() != 1)
            throw std::invalid_argument("--" + std::string(key) + " needs exactly one value");
        return it->second[0];
    }
    [[nodiscard]] std::string required(std::string_view key) const {
        if (!has(key))
            throw std::invalid_argument("Missing --" + std::string(key));
        return get(key);
    }
    [[nodiscard]] std::vector<std::filesystem::path> paths(std::string_view key) const {
        std::vector<std::filesystem::path> paths;
        auto it = values_.find(key);
        if (it == values_.end())
            return paths;
        if (it->second.empty())
            throw std::invalid_argument("--" + std::string(key) + " needs file paths");
        for (const auto& v : it->second) {
            if (v.find_first_of("*?[") == std::string::npos) {
                paths.emplace_back(v);
                continue;
            }
            struct Matches {
                glob_t value{};
                ~Matches() { ::globfree(&value); }
            } matches;
            if (::glob(v.c_str(), GLOB_NOSORT, nullptr, &matches.value) != 0)
                throw std::invalid_argument("Cannot expand file pattern: " + v);
            std::vector<std::filesystem::path> expanded;
            for (std::size_t i = 0; i < matches.value.gl_pathc; ++i)
                expanded.emplace_back(matches.value.gl_pathv[i]);
            std::sort(expanded.begin(), expanded.end());
            paths.insert(paths.end(), expanded.begin(), expanded.end());
        }
        return paths;
    }
    [[nodiscard]] std::vector<std::string> strings(std::string_view key) const {
        auto it = values_.find(key);
        return it == values_.end() ? std::vector<std::string>{} : it->second;
    }
    [[nodiscard]] int integer(std::string_view key, int fallback) const {
        if (!has(key))
            return fallback;
        auto s = get(key);
        int n;
        auto r = std::from_chars(s.data(), s.data() + s.size(), n);
        if (r.ec != std::errc{} || r.ptr != s.data() + s.size())
            throw std::invalid_argument("Invalid integer --" + std::string(key));
        return n;
    }
    [[nodiscard]] std::uint64_t u64(std::string_view key, std::uint64_t fallback) const {
        if (!has(key))
            return fallback;
        auto s = get(key);
        std::uint64_t n;
        auto r = std::from_chars(s.data(), s.data() + s.size(), n);
        if (r.ec != std::errc{} || r.ptr != s.data() + s.size())
            throw std::invalid_argument("Invalid integer --" + std::string(key));
        return n;
    }
    [[nodiscard]] float real(std::string_view key, float fallback) const {
        if (!has(key))
            return fallback;
        auto s = get(key);
        float n;
        auto r = std::from_chars(s.data(), s.data() + s.size(), n);
        if (r.ec != std::errc{} || r.ptr != s.data() + s.size() || !std::isfinite(n))
            throw std::invalid_argument("Invalid number --" + std::string(key));
        return n;
    }

  private:
    std::map<std::string, std::vector<std::string>, std::less<>> values_;
};
template <class F> int app_main(int argc, char** argv, F function) {
    try {
        return function(Args(argc, argv));
    } catch (const Failure& e) {
        std::fprintf(stderr, "%.*s: %s\n", static_cast<int>(to_string(e.error().code).size()),
                     to_string(e.error().code).data(), e.what());
    } catch (const std::exception& e) {
        std::fprintf(stderr, "error: %s\n", e.what());
    }
    return 1;
}
} // namespace mini_mla
