#include "mini_mla/ops.hpp"

#include <Accelerate/Accelerate.h>
#include <cmath>
#include <limits>

namespace mini_mla::ops {
void matmul(const Matrix& a, const Matrix& b, Matrix& out, bool ta, bool tb, bool accumulate) {
    int m = ta ? a.cols : a.rows, n = tb ? b.rows : b.cols, inner = ta ? a.rows : a.cols;
    if (inner != (tb ? b.cols : b.rows) || m < 1 || n < 1 || inner < 1 || &out == &a || &out == &b)
        detail::fatal("Invalid or aliased matmul");
    if (accumulate && (out.rows != m || out.cols != n))
        detail::fatal("Matmul accumulation shape mismatch");
    out.resize(m, n);
    cblas_sgemm(CblasRowMajor, ta ? CblasTrans : CblasNoTrans, tb ? CblasTrans : CblasNoTrans, m, n,
                inner, 1.0F, a.data(), a.cols, b.data(), b.cols, accumulate ? 1.0F : 0.0F,
                out.data(), n);
}
void matmul_naive(const Matrix& a, const Matrix& b, Matrix& out) {
    if (a.cols != b.rows || &out == &a || &out == &b)
        detail::fatal("Invalid naive matmul");
    out.resize(a.rows, b.cols);
    out.zero();
    for (int i = 0; i < a.rows; ++i)
        for (int k = 0; k < a.cols; ++k)
            for (int j = 0; j < b.cols; ++j)
                out(i, j) += a(i, k) * b(k, j);
}
void linear_backward(const Matrix& x, const Matrix& w, const Matrix& dy, Matrix& dx, Matrix& dw) {
    matmul(dy, w, dx, false, true);
    matmul(x, dy, dw, true, false, true);
}
void rmsnorm(const Matrix& x, const Matrix& w, Matrix& out, float eps) {
    if (w.rows != 1 || w.cols != x.cols || x.cols < 1 || eps <= 0)
        detail::fatal("Invalid RMSNorm shape");
    out.resize(x.rows, x.cols);
    for (int i = 0; i < x.rows; ++i) {
        float sum = 0;
        for (int j = 0; j < x.cols; ++j)
            sum += x(i, j) * x(i, j);
        float inv = 1.0F / std::sqrt(sum / static_cast<float>(x.cols) + eps);
        for (int j = 0; j < x.cols; ++j)
            out(i, j) = x(i, j) * inv * w(0, j);
    }
}
void rmsnorm_backward(const Matrix& x, const Matrix& w, const Matrix& dy, Matrix& dx, Matrix& dw,
                      float eps) {
    if (x.rows != dy.rows || x.cols != dy.cols || dw.cols != w.cols)
        detail::fatal("Invalid RMSNorm gradient shape");
    dx.resize(x.rows, x.cols);
    for (int i = 0; i < x.rows; ++i) {
        float sq = 0, dot = 0;
        for (int j = 0; j < x.cols; ++j) {
            sq += x(i, j) * x(i, j);
            dot += dy(i, j) * w(0, j) * x(i, j);
        }
        float inv = 1.0F / std::sqrt(sq / static_cast<float>(x.cols) + eps);
        float correction = dot * inv * inv * inv / static_cast<float>(x.cols);
        for (int j = 0; j < x.cols; ++j) {
            dx(i, j) = dy(i, j) * w(0, j) * inv - x(i, j) * correction;
            dw(0, j) += dy(i, j) * x(i, j) * inv;
        }
    }
}
float sigmoid(float x) noexcept {
    if (x >= 0)
        return 1.0F / (1.0F + std::exp(-x));
    float e = std::exp(x);
    return e / (1.0F + e);
}
void swiglu(const Matrix& gate, const Matrix& up, Matrix& out) {
    if (gate.rows != up.rows || gate.cols != up.cols)
        detail::fatal("Invalid SwiGLU shape");
    out.resize(gate.rows, gate.cols);
    for (std::size_t i = 0; i < gate.size(); ++i)
        out.values[i] = gate.values[i] * sigmoid(gate.values[i]) * up.values[i];
}
void swiglu_backward(const Matrix& gate, const Matrix& up, const Matrix& dy, Matrix& dg,
                     Matrix& du) {
    dg.resize(gate.rows, gate.cols);
    du.resize(gate.rows, gate.cols);
    for (std::size_t i = 0; i < gate.size(); ++i) {
        float x = gate.values[i], s = sigmoid(x);
        dg.values[i] = dy.values[i] * up.values[i] * (s + x * s * (1 - s));
        du.values[i] = dy.values[i] * x * s;
    }
}
void rope(Matrix& x, int batch, int sequence, int heads, int width, int begin, int rope_width,
          float theta, int offset, bool inverse) {
    if (x.rows != batch * sequence || x.cols != heads * width || begin < 0 || rope_width % 2 ||
        begin + rope_width > width)
        detail::fatal("Invalid RoPE shape");
    for (int t = 0; t < sequence; ++t)
        for (int j = 0; j < rope_width; j += 2) {
            float angle = static_cast<float>(t + offset) *
                          std::pow(theta, -static_cast<float>(j) / static_cast<float>(rope_width));
            if (inverse)
                angle = -angle;
            float cs = std::cos(angle), sn = std::sin(angle);
            for (int b = 0; b < batch; ++b)
                for (int h = 0; h < heads; ++h) {
                    int row = b * sequence + t, col = h * width + begin + j;
                    float a = x(row, col), v = x(row, col + 1);
                    x(row, col) = a * cs - v * sn;
                    x(row, col + 1) = a * sn + v * cs;
                }
        }
}
void attention(const Matrix& q, const Matrix& k, const Matrix& v, Matrix& a, Matrix& out, int batch,
               int t, int h, int d, int dv) {
    a.resize(batch * h * t, t);
    out.resize(batch * t, h * dv);
    float scale = 1.0F / std::sqrt(static_cast<float>(d));
    for (int b = 0; b < batch; ++b)
        for (int head = 0; head < h; ++head) {
            auto base = static_cast<std::size_t>(b * t);
            float* p = a.data() +
                       static_cast<std::size_t>((b * h + head) * t) * static_cast<std::size_t>(t);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, t, t, d, scale,
                        q.data() + base * static_cast<std::size_t>(h * d) + head * d, h * d,
                        k.data() + base * static_cast<std::size_t>(h * d) + head * d, h * d, 0, p,
                        t);
            for (int i = 0; i < t; ++i) {
                float* row = p + static_cast<std::size_t>(i * t);
                float max = row[0];
                for (int j = 1; j <= i; ++j)
                    max = std::max(max, row[j]);
                float sum = 0;
                for (int j = 0; j <= i; ++j) {
                    row[j] = std::exp(row[j] - max);
                    sum += row[j];
                }
                for (int j = 0; j <= i; ++j)
                    row[j] /= sum;
                for (int j = i + 1; j < t; ++j)
                    row[j] = 0;
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, t, dv, t, 1, p, t,
                        v.data() + base * static_cast<std::size_t>(h * dv) + head * dv, h * dv, 0,
                        out.data() + base * static_cast<std::size_t>(h * dv) + head * dv, h * dv);
        }
}
void attention_backward(const Matrix& q, const Matrix& k, const Matrix& v, const Matrix& a,
                        const Matrix& dy, Matrix& dq, Matrix& dk, Matrix& dvout, Matrix& s,
                        int batch, int t, int h, int d, int dv) {
    dq.resize(batch * t, h * d);
    dk.resize(batch * t, h * d);
    dvout.resize(batch * t, h * dv);
    s.resize(t, t);
    float scale = 1.0F / std::sqrt(static_cast<float>(d));
    for (int b = 0; b < batch; ++b)
        for (int head = 0; head < h; ++head) {
            auto qi = static_cast<std::size_t>(b * t * h * d + head * d),
                 vi = static_cast<std::size_t>(b * t * h * dv + head * dv);
            const float* p = a.data() + static_cast<std::size_t>((b * h + head) * t * t);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, t, t, dv, 1, dy.data() + vi,
                        h * dv, v.data() + vi, h * dv, 0, s.data(), t);
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, t, dv, t, 1, p, t, dy.data() + vi,
                        h * dv, 0, dvout.data() + vi, h * dv);
            for (int i = 0; i < t; ++i) {
                float dot = 0;
                for (int j = 0; j <= i; ++j)
                    dot += s(i, j) * p[i * t + j];
                for (int j = 0; j < t; ++j)
                    s(i, j) = j <= i ? p[i * t + j] * (s(i, j) - dot) : 0;
            }
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, t, d, t, scale, s.data(), t,
                        k.data() + qi, h * d, 0, dq.data() + qi, h * d);
            cblas_sgemm(CblasRowMajor, CblasTrans, CblasNoTrans, t, d, t, scale, s.data(), t,
                        q.data() + qi, h * d, 0, dk.data() + qi, h * d);
        }
}
Result<double> weighted_nll(const Matrix& logits, std::span<const std::uint16_t> targets,
                            std::span<const float> coeff, Matrix* grad,
                            std::vector<float>* log_probs) {
    if (logits.rows < 1 || logits.cols < 1 ||
        targets.size() != static_cast<std::size_t>(logits.rows) || coeff.size() != targets.size())
        return Result<double>::failure(ErrorCode::invalid_argument, "Loss shapes disagree");
    if (grad)
        grad->resize(logits.rows, logits.cols);
    if (log_probs)
        log_probs->resize(targets.size());
    double loss = 0;
    for (int i = 0; i < logits.rows; ++i) {
        auto index = static_cast<std::size_t>(i);
        if (targets[index] >= logits.cols || !std::isfinite(coeff[index]))
            return Result<double>::failure(ErrorCode::invalid_argument,
                                           "Invalid target or loss coefficient");
        float max = logits(i, 0);
        for (int j = 1; j < logits.cols; ++j)
            max = std::max(max, logits(i, j));
        double z = 0;
        for (int j = 0; j < logits.cols; ++j)
            z += std::exp(static_cast<double>(logits(i, j) - max));
        double nll =
            std::log(z) + static_cast<double>(max) - static_cast<double>(logits(i, targets[index]));
        if (!std::isfinite(nll))
            return Result<double>::failure(ErrorCode::numerical_error, "Nonfinite logits or loss");
        loss += static_cast<double>(coeff[index]) * nll;
        if (log_probs)
            (*log_probs)[index] = static_cast<float>(-nll);
        if (grad)
            for (int j = 0; j < logits.cols; ++j)
                (*grad)(i, j) =
                    coeff[index] *
                    (static_cast<float>(std::exp(static_cast<double>(logits(i, j) - max)) / z) -
                     (j == targets[index] ? 1.0F : 0.0F));
    }
    return Result<double>::success(loss);
}
void add_inplace(Matrix& dst, const Matrix& src) {
    if (dst.rows != src.rows || dst.cols != src.cols)
        detail::fatal("Add shape mismatch");
    for (std::size_t i = 0; i < dst.size(); ++i)
        dst.values[i] += src.values[i];
}
bool finite(std::span<const float> x) {
    return std::all_of(x.begin(), x.end(), [](float v) { return std::isfinite(v); });
}
} // namespace mini_mla::ops
