#include "mini_mla/ops.hpp"
#include "mini_mla/testing.hpp"

#include <cmath>

using namespace mini_mla;
MINI_MLA_TEST(blas_matches_naive_and_transposes) {
    Matrix a(3, 4), b(4, 5), x, y, at(4, 3), bt(5, 4);
    for (std::size_t i = 0; i < a.size(); ++i)
        a.values[i] = static_cast<float>(i) / 7;
    for (std::size_t i = 0; i < b.size(); ++i)
        b.values[i] = static_cast<float>(i) / 13;
    ops::matmul(a, b, x);
    ops::matmul_naive(a, b, y);
    for (std::size_t i = 0; i < x.size(); ++i)
        MINI_MLA_CHECK(std::abs(x.values[i] - y.values[i]) < 1e-5F);
    for (int i = 0; i < a.rows; ++i)
        for (int j = 0; j < a.cols; ++j)
            at(j, i) = a(i, j);
    for (int i = 0; i < b.rows; ++i)
        for (int j = 0; j < b.cols; ++j)
            bt(j, i) = b(i, j);
    ops::matmul(at, bt, y, true, true);
    for (std::size_t i = 0; i < x.size(); ++i)
        MINI_MLA_CHECK(std::abs(x.values[i] - y.values[i]) < 1e-5F);
}
MINI_MLA_TEST(rope_inverse_preserves_vectors) {
    Matrix x(6, 12);
    for (std::size_t i = 0; i < x.size(); ++i)
        x.values[i] = std::sin(static_cast<float>(i));
    auto original = x;
    ops::rope(x, 2, 3, 2, 6, 2, 4, 10000);
    ops::rope(x, 2, 3, 2, 6, 2, 4, 10000, 0, true);
    for (std::size_t i = 0; i < x.size(); ++i)
        MINI_MLA_CHECK(std::abs(x.values[i] - original.values[i]) < 2e-7F);
}
MINI_MLA_TEST(stable_cross_entropy_and_prompt_mask) {
    Matrix logits(2, 3), g;
    logits.values = {1000, 999, 998, -999, -1000, -1001};
    std::uint16_t target[] = {0, 1};
    float coeff[] = {0, 1};
    auto loss = ops::weighted_nll(logits, target, coeff, &g);
    MINI_MLA_REQUIRE(loss);
    MINI_MLA_CHECK(std::abs(loss.value() - 1.407605964) < 1e-6);
    MINI_MLA_CHECK(g(0, 0) == 0 && g(0, 1) == 0 && g(0, 2) == 0);
    MINI_MLA_CHECK(std::abs(g(1, 0) + g(1, 1) + g(1, 2)) < 1e-7F);
}
MINI_MLA_TEST_MAIN()
