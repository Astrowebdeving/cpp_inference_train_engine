#pragma once
#include "mini_mla/result.hpp"

#include <algorithm>
#include <span>
#include <vector>

namespace mini_mla {
struct Matrix {
    int rows = 0, cols = 0;
    std::vector<float> values;
    Matrix() = default;
    Matrix(int r, int c) { resize(r, c); }
    void resize(int r, int c) {
        if (r < 0 || c < 0)
            detail::fatal("Negative matrix extent");
        rows = r;
        cols = c;
        values.resize(static_cast<std::size_t>(r) * static_cast<std::size_t>(c));
    }
    void zero() { std::fill(values.begin(), values.end(), 0.0F); }
    [[nodiscard]] float* data() { return values.data(); }
    [[nodiscard]] const float* data() const { return values.data(); }
    [[nodiscard]] std::size_t size() const { return values.size(); }
    float& operator()(int r, int c) {
        return values[static_cast<std::size_t>(r) * static_cast<std::size_t>(cols) +
                      static_cast<std::size_t>(c)];
    }
    float operator()(int r, int c) const {
        return values[static_cast<std::size_t>(r) * static_cast<std::size_t>(cols) +
                      static_cast<std::size_t>(c)];
    }
};
namespace ops {
void matmul(const Matrix& a, const Matrix& b, Matrix& out, bool transpose_a = false,
            bool transpose_b = false, bool accumulate = false);
void matmul_naive(const Matrix& a, const Matrix& b, Matrix& out);
void linear_backward(const Matrix& x, const Matrix& w, const Matrix& dy, Matrix& dx, Matrix& dw);
void rmsnorm(const Matrix& x, const Matrix& w, Matrix& out, float epsilon);
void rmsnorm_backward(const Matrix& x, const Matrix& w, const Matrix& dy, Matrix& dx, Matrix& dw,
                      float epsilon);
[[nodiscard]] float sigmoid(float x) noexcept;
void swiglu(const Matrix& gate, const Matrix& up, Matrix& out);
void swiglu_backward(const Matrix& gate, const Matrix& up, const Matrix& dy, Matrix& dg,
                     Matrix& du);
void rope(Matrix& x, int batch, int sequence, int heads, int head_width, int begin, int rope_width,
          float theta, int position_offset = 0, bool inverse = false);
void attention(const Matrix& q, const Matrix& k, const Matrix& v, Matrix& probabilities,
               Matrix& out, int batch, int sequence, int heads, int qk_width, int value_width);
void attention_backward(const Matrix& q, const Matrix& k, const Matrix& v,
                        const Matrix& probabilities, const Matrix& dy, Matrix& dq, Matrix& dk,
                        Matrix& dv, Matrix& scratch, int batch, int sequence, int heads,
                        int qk_width, int value_width);
[[nodiscard]] Result<double> weighted_nll(const Matrix& logits,
                                          std::span<const std::uint16_t> targets,
                                          std::span<const float> coefficients,
                                          Matrix* gradient = nullptr,
                                          std::vector<float>* log_probs = nullptr);
void add_inplace(Matrix& dst, const Matrix& src);
[[nodiscard]] bool finite(std::span<const float> x);
} // namespace ops
} // namespace mini_mla
