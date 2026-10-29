#pragma once
#include <cmath>
#include <cstdint>
#include <numbers>

namespace mini_mla {
class Rng {
  public:
    explicit Rng(std::uint64_t seed = 42) : state_(seed) {}
    [[nodiscard]] std::uint64_t next() noexcept {
        auto z = (state_ += 0x9e3779b97f4a7c15ULL);
        z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
        z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
        return z ^ (z >> 31);
    }
    [[nodiscard]] float uniform() noexcept {
        return (static_cast<float>(next() >> 40) + 0.5F) / 16777216.0F;
    }
    [[nodiscard]] std::uint64_t bounded(std::uint64_t bound) noexcept {
        if (bound == 0)
            return 0;
        const auto threshold = (0ULL - bound) % bound;
        for (;;) {
            auto r = next();
            if (r >= threshold)
                return r % bound;
        }
    }
    [[nodiscard]] float normal() noexcept {
        return std::sqrt(-2.0F * std::log(uniform())) *
               std::cos(2.0F * std::numbers::pi_v<float> * uniform());
    }
    [[nodiscard]] std::uint64_t state() const noexcept { return state_; }
    void restore(std::uint64_t state) noexcept { state_ = state; }

  private:
    std::uint64_t state_;
};
} // namespace mini_mla
