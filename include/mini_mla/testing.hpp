#pragma once

#include <cstdio>
#include <exception>
#include <source_location>
#include <string>
#include <string_view>
#include <vector>

namespace mini_mla::testing {

struct TestCase {
    std::string_view name;
    void (*fn)();
};

inline std::vector<TestCase>& registry() {
    static std::vector<TestCase> tests;
    return tests;
}

inline int& failure_count() {
    static int count = 0;
    return count;
}

inline void report_failure(std::string_view expression, std::string_view detail,
                           const std::source_location& location) {
    ++failure_count();
    std::fprintf(stderr, "  FAIL %s:%u\n       check: %.*s\n", location.file_name(),
                 location.line(), static_cast<int>(expression.size()), expression.data());
    if (!detail.empty()) {
        std::fprintf(stderr, "       %.*s\n", static_cast<int>(detail.size()), detail.data());
    }
}

struct Registrar {
    Registrar(std::string_view name, void (*fn)()) { registry().push_back(TestCase{name, fn}); }
};

inline int run_all() {
    int failed_tests = 0;
    for (const TestCase& test : registry()) {
        const int before = failure_count();
        std::fprintf(stderr, "RUN  %.*s\n", static_cast<int>(test.name.size()), test.name.data());
        test.fn();
        if (failure_count() != before) {
            ++failed_tests;
            std::fprintf(stderr, "FAIL %.*s\n", static_cast<int>(test.name.size()),
                         test.name.data());
        } else {
            std::fprintf(stderr, "ok   %.*s\n", static_cast<int>(test.name.size()),
                         test.name.data());
        }
    }
    std::fprintf(stderr, "\n%d/%zu tests passed\n",
                 static_cast<int>(registry().size()) - failed_tests, registry().size());
    return failed_tests == 0 ? 0 : 1;
}

} // namespace mini_mla::testing

#define MINI_MLA_TEST(test_name)                                                                   \
    static void test_name();                                                                       \
    static const ::mini_mla::testing::Registrar mini_mla_registrar_##test_name{#test_name,         \
                                                                               &test_name};        \
    static void test_name()

#define MINI_MLA_CHECK(expr)                                                                       \
    do {                                                                                           \
        if (!(expr)) {                                                                             \
            ::mini_mla::testing::report_failure(#expr, {}, std::source_location::current());       \
        }                                                                                          \
    } while (false)

#define MINI_MLA_CHECK_MSG(expr, message)                                                          \
    do {                                                                                           \
        if (!(expr)) {                                                                             \
            ::mini_mla::testing::report_failure(#expr, (message),                                  \
                                                std::source_location::current());                  \
        }                                                                                          \
    } while (false)

#define MINI_MLA_REQUIRE(expr)                                                                     \
    do {                                                                                           \
        if (!(expr)) {                                                                             \
            ::mini_mla::testing::report_failure(#expr, "(required)",                               \
                                                std::source_location::current());                  \
            return;                                                                                \
        }                                                                                          \
    } while (false)

#define MINI_MLA_TEST_MAIN()                                                                       \
    int main() {                                                                                   \
        return ::mini_mla::testing::run_all();                                                     \
    }
