#include "app_utils.hpp"

int main(int argc, char** argv) {
    return mini_mla::app_main(argc, argv, [](const mini_mla::Args& a) {
        a.known({"tokenizer", "text", "input", "decode", "allow-special", "json", "help"});
        if (a.has("help")) {
            std::puts("mini_mla_tokenize --tokenizer tokenizer.json --text TEXT "
                      "[--json]\nmini_mla_tokenize --tokenizer tokenizer.json --decode 1,2,3");
            return 0;
        }
        auto t = mini_mla::take(mini_mla::Tokenizer::load(a.required("tokenizer")));
        if (a.has("decode")) {
            auto text = mini_mla::take(t.decode(parse_tokens(a.get("decode"))));
            std::fwrite(text.data(), 1, text.size(), stdout);
            return 0;
        }
        std::string text = a.has("input") ? mini_mla::take(mini_mla::read_text(a.get("input")))
                                          : a.required("text");
        auto ids = mini_mla::take(t.encode(text, a.has("allow-special")));
        mini_mla::Json::Array values;
        for (auto id : ids)
            values.emplace_back(static_cast<int>(id));
        if (a.has("json"))
            std::puts(
                mini_mla::Json(mini_mla::Json::Object{
                                   {"tokens", std::move(values)},
                                   {"decoded", mini_mla::clean_utf8(mini_mla::take(t.decode(ids)))},
                                   {"tokenizer_sha256", t.hash()}})
                    .dump()
                    .c_str());
        else
            std::puts(mini_mla::Json(std::move(values)).dump().c_str());
        return 0;
    });
}
