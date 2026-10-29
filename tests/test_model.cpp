#include "mini_mla/model.hpp"
#include "mini_mla/testing.hpp"

#include <cmath>

using namespace mini_mla;
static ModelConfig small() {
    ModelConfig c;
    c.vocab_size = 16;
    c.n_layers = 1;
    c.d_model = 8;
    c.n_heads = 2;
    c.head_dim_nope = 2;
    c.head_dim_rope = 2;
    c.head_dim_value = 3;
    c.q_lora_rank = 4;
    c.kv_lora_rank = 3;
    c.d_ff = 12;
    return c;
}
MINI_MLA_TEST(exact_full_parameter_count) {
    auto c = ModelConfig::load("config/mla36m-v1.json");
    MINI_MLA_REQUIRE(c);
    MINI_MLA_CHECK(c.value().parameter_count() == 35763840);
    auto j = c.value().json();
    j["head_dim_rope"] = 3;
    MINI_MLA_CHECK(!ModelConfig::from_json(j));
}
MINI_MLA_TEST(full_model_analytical_gradients_match_finite_differences) {
    auto made = Model::create(small(), 5);
    MINI_MLA_REQUIRE(made);
    auto& m = *made.value();
    std::uint16_t ids[] = {1, 3, 5, 2, 4, 6}, targets[] = {3, 5, 7, 4, 6, 8};
    float coeff[] = {0, 0.2F, 0.2F, 0.2F, 0.2F, 0.2F};
    Matrix g;
    auto logits = m.forward(ids, 2, 3);
    MINI_MLA_REQUIRE(logits);
    auto loss = ops::weighted_nll(*logits.value(), targets, coeff, &g);
    MINI_MLA_REQUIRE(loss);
    m.zero_grad();
    MINI_MLA_REQUIRE(m.backward(g));
    for (auto& p : m.parameters()) {
        for (std::size_t i : {std::size_t{0}, p.value.size() / 3, p.value.size() - 1}) {
            float original = p.value.values[i];
            float eps = p.role == ParameterRole::norm_scale ? 0.002F : 0.0002F;
            p.value.values[i] = original + eps;
            auto pos = m.forward(ids, 2, 3);
            MINI_MLA_REQUIRE(pos);
            auto lp = ops::weighted_nll(*pos.value(), targets, coeff);
            MINI_MLA_REQUIRE(lp);
            p.value.values[i] = original - eps;
            auto neg = m.forward(ids, 2, 3);
            MINI_MLA_REQUIRE(neg);
            auto ln = ops::weighted_nll(*neg.value(), targets, coeff);
            MINI_MLA_REQUIRE(ln);
            p.value.values[i] = original;
            double numeric = (lp.value() - ln.value()) / (2 * eps),
                   analytical = p.gradient.values[i];
            MINI_MLA_CHECK_MSG(std::abs(numeric - analytical) < 0.0002 + 0.003 * std::abs(numeric),
                               p.name + " analytical=" + std::to_string(analytical) +
                                   " numeric=" + std::to_string(numeric));
        }
    }
}
MINI_MLA_TEST(causal_attention_has_no_future_leak) {
    auto made = Model::create(small(), 1);
    MINI_MLA_REQUIRE(made);
    std::uint16_t a[] = {1, 2, 3}, b[] = {1, 2, 9};
    auto x = made.value()->forward(a, 1, 3);
    MINI_MLA_REQUIRE(x);
    auto original = *x.value();
    auto y = made.value()->forward(b, 1, 3);
    MINI_MLA_REQUIRE(y);
    for (int i = 0; i < 2; ++i)
        for (int j = 0; j < original.cols; ++j)
            MINI_MLA_CHECK(original(i, j) == (*y.value())(i, j));
}
MINI_MLA_TEST_MAIN()
