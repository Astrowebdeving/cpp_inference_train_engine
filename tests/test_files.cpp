#include "mini_mla/checkpoint.hpp"
#include "mini_mla/data.hpp"
#include "mini_mla/testing.hpp"

#include <cstring>
#include <fstream>
#include <numeric>
#include <unistd.h>

using namespace mini_mla;
namespace {
struct Temporary {
    std::filesystem::path path;
    Temporary() {
        static unsigned counter = 0;
        path = std::filesystem::temp_directory_path() /
               ("mla-files-" + std::to_string(getpid()) + "-" + std::to_string(counter++));
        std::filesystem::create_directories(path);
    }
    ~Temporary() {
        std::error_code e;
        std::filesystem::remove_all(path, e);
    }
};
ModelConfig small() {
    auto c = ModelConfig::load("config/tiny-test.json").value();
    c.n_layers = 1;
    c.d_model = 8;
    c.head_dim_nope = 2;
    c.head_dim_rope = 2;
    c.head_dim_value = 3;
    c.q_lora_rank = 4;
    c.kv_lora_rank = 3;
    c.d_ff = 12;
    return c;
}
Status shard(const std::filesystem::path& path, std::span<const std::uint16_t> tokens,
             const Tokenizer& tokenizer) {
    std::vector<std::byte> bytes(160 + tokens.size_bytes());
    auto set = [&](std::size_t offset, auto value) {
        std::memcpy(bytes.data() + offset, &value, sizeof(value));
    };
    std::memcpy(bytes.data(), "MLATOK01", 8);
    set(8, std::uint32_t{1});
    set(12, std::uint32_t{160});
    set(16, std::uint32_t{1});
    set(20, static_cast<std::uint32_t>(tokenizer.vocab_size()));
    set(24, static_cast<std::uint64_t>(tokens.size()));
    auto source = read_text("examples/tokenizer.json").value();
    auto hash = sha256(std::as_bytes(std::span{source.data(), source.size()}));
    std::memcpy(bytes.data() + 32, hash.data(), 32);
    std::memcpy(bytes.data() + 160, tokens.data(), tokens.size_bytes());
    auto payload = sha256(std::span{bytes}.subspan(160));
    std::memcpy(bytes.data() + 64, payload.data(), 32);
    return write_bytes(path, bytes);
}
} // namespace
MINI_MLA_TEST(json_rejects_malformed_and_round_trips_unicode) {
    for (auto text : {"{\"a\":1,\"a\":2}", "[1,]", "01", "1e309", "\"\\ud800\"", "true false"})
        MINI_MLA_CHECK(!Json::parse(text));
    auto parsed = Json::parse("{\"s\":\"\\ud83d\\ude80\",\"array\":[null,true,-2.5]}");
    MINI_MLA_REQUIRE(parsed);
    MINI_MLA_CHECK(parsed.value().at("s").string() == "🚀");
    auto copy = Json::parse(parsed.value().dump());
    MINI_MLA_REQUIRE(copy);
    MINI_MLA_CHECK(copy.value().dump() == parsed.value().dump());
}
MINI_MLA_TEST(tokenizer_raw_bytes_and_specials_are_unambiguous) {
    auto t = Tokenizer::load("examples/tokenizer.json");
    MINI_MLA_REQUIRE(t);
    std::string text = "Hello, café 世界 🚀\n<|eos|>";
    text.push_back('\0');
    auto ids = t.value().encode(text);
    MINI_MLA_REQUIRE(ids);
    auto decoded = t.value().decode(ids.value());
    MINI_MLA_REQUIRE(decoded);
    MINI_MLA_CHECK(decoded.value() == text);
    auto special = t.value().encode("<|eos|>", true);
    MINI_MLA_REQUIRE(special);
    MINI_MLA_CHECK(special.value() == std::vector<std::uint16_t>{t.value().special("<|eos|>")});
    MINI_MLA_CHECK(valid_utf8("café 🚀"));
    MINI_MLA_CHECK(!valid_utf8(std::string("\xff", 1)));
    MINI_MLA_CHECK(valid_utf8(clean_utf8(std::string("\xff", 1))));
}
MINI_MLA_TEST(bpe_merge_priority_and_invalid_tables) {
    Temporary temp;
    auto j = read_json("examples/tokenizer.json");
    MINI_MLA_REQUIRE(j);
    j.value()["vocab_size"] = 266;
    j.value()["merges"] = Json::Array{Json::Array{97, 98, 256}, Json::Array{256, 97, 257}};
    auto specials = j.value()["specials"].object();
    for (auto& [name, id] : specials) {
        (void)name;
        id = id.integer() + 2;
    }
    j.value()["specials"] = std::move(specials);
    MINI_MLA_REQUIRE(write_json(temp.path / "tokenizer.json", j.value()));
    auto t = Tokenizer::load(temp.path / "tokenizer.json");
    MINI_MLA_REQUIRE(t);
    auto ids = t.value().encode("ababa");
    MINI_MLA_REQUIRE(ids);
    MINI_MLA_CHECK(ids.value() == std::vector<std::uint16_t>({256, 257}));
    j.value()["merges"] = Json::Array{Json::Array{257, 98, 256}};
    MINI_MLA_REQUIRE(write_json(temp.path / "bad.json", j.value()));
    MINI_MLA_CHECK(!Tokenizer::load(temp.path / "bad.json"));
}
MINI_MLA_TEST(checkpoints_restore_weights_optimizer_and_state_exactly) {
    Temporary temp;
    auto m = Model::create(small(), 7);
    MINI_MLA_REQUIRE(m);
    Optimizer optimizer;
    MINI_MLA_REQUIRE(optimizer.initialize(*m.value()));
    for (auto& p : m.value()->parameters())
        for (std::size_t i = 0; i < p.gradient.size(); ++i)
            p.gradient.values[i] = static_cast<float>(i % 7) * 0.01F;
    MINI_MLA_REQUIRE(optimizer.step(*m.value(), 0.001F));
    Json state = Json::Object{{"rng", "18446744073709551615"}, {"step", 1}};
    MINI_MLA_REQUIRE(save_checkpoint(temp.path / "checkpoint", *m.value(), optimizer, state,
                                     "examples/tokenizer.json"));
    auto loaded = load_checkpoint(temp.path / "checkpoint", true);
    MINI_MLA_REQUIRE(loaded);
    MINI_MLA_CHECK(loaded.value().trainer_state.dump() == state.dump());
    MINI_MLA_CHECK(loaded.value().optimizer->steps() == 1);
    for (std::size_t i = 0; i < m.value()->parameters().size(); ++i) {
        MINI_MLA_CHECK(loaded.value().model->parameters()[i].value.values ==
                       m.value()->parameters()[i].value.values);
        MINI_MLA_CHECK(loaded.value().optimizer->states()[i].first.values ==
                       optimizer.states()[i].first.values);
        MINI_MLA_CHECK(loaded.value().optimizer->states()[i].second.values ==
                       optimizer.states()[i].second.values);
        loaded.value().model->parameters()[i].gradient = m.value()->parameters()[i].gradient;
    }
    MINI_MLA_REQUIRE(optimizer.step(*m.value(), 0.001F));
    MINI_MLA_REQUIRE(loaded.value().optimizer->step(*loaded.value().model, 0.001F));
    for (std::size_t i = 0; i < m.value()->parameters().size(); ++i)
        MINI_MLA_CHECK(loaded.value().model->parameters()[i].value.values ==
                       m.value()->parameters()[i].value.values);
    MINI_MLA_CHECK(!save_checkpoint(temp.path / "checkpoint", *m.value(), optimizer, state,
                                    "examples/tokenizer.json"));
    MINI_MLA_REQUIRE(update_latest_link(temp.path / "latest", temp.path / "checkpoint"));
    MINI_MLA_REQUIRE(load_checkpoint(temp.path / "latest"));
}
MINI_MLA_TEST(checkpoint_checksum_completion_and_half_export) {
    Temporary temp;
    auto m = Model::create(small(), 11, false);
    MINI_MLA_REQUIRE(m);
    MINI_MLA_REQUIRE(
        export_model(temp.path / "model", *m.value(), "examples/tokenizer.json", "f16"));
    auto half = load_checkpoint(temp.path / "model");
    MINI_MLA_REQUIRE(half);
    for (std::size_t p = 0; p < m.value()->parameters().size(); ++p)
        for (std::size_t i = 0; i < m.value()->parameters()[p].value.size(); ++i)
            MINI_MLA_CHECK(std::abs(m.value()->parameters()[p].value.values[i] -
                                    half.value().model->parameters()[p].value.values[i]) < 0.0001F);
    MINI_MLA_CHECK(!load_checkpoint(temp.path / "model", true));
    std::fstream f(temp.path / "model" / "weights.bin",
                   std::ios::binary | std::ios::in | std::ios::out);
    char c;
    f.read(&c, 1);
    c ^= 1;
    f.seekp(0);
    f.write(&c, 1);
    f.close();
    MINI_MLA_CHECK(!load_checkpoint(temp.path / "model"));
    std::filesystem::remove(temp.path / "model" / "COMPLETE");
    MINI_MLA_CHECK(!load_checkpoint(temp.path / "model"));
}
MINI_MLA_TEST(data_permutation_prefetch_and_resume_are_exact) {
    Temporary temp;
    auto t = Tokenizer::load("examples/tokenizer.json");
    MINI_MLA_REQUIRE(t);
    std::vector<std::uint16_t> tokens(41);
    std::iota(tokens.begin(), tokens.end(), std::uint16_t{0});
    auto path = temp.path / "data.tok";
    MINI_MLA_REQUIRE(shard(path, tokens, t.value()));
    std::filesystem::path paths[] = {path};
    auto first = DataLoader::tokens(paths, 4, 264, t.value().hash(), 99);
    MINI_MLA_REQUIRE(first);
    Json state;
    std::vector<Batch> expected;
    {
        PrefetchLoader loader(std::move(first).value(), 3, 4);
        for (int i = 0; i < 7; ++i)
            MINI_MLA_REQUIRE(loader.next());
        state = loader.consumed_state();
        for (int i = 0; i < 10; ++i) {
            auto b = loader.next();
            MINI_MLA_REQUIRE(b);
            expected.push_back(std::move(b).value());
        }
    }
    auto second = DataLoader::tokens(paths, 4, 264, t.value().hash(), 99);
    MINI_MLA_REQUIRE(second);
    MINI_MLA_REQUIRE(second.value()->restore(state));
    for (const auto& b : expected) {
        auto next = second.value()->next(3);
        MINI_MLA_REQUIRE(next);
        MINI_MLA_CHECK(next.value().input == b.input);
        MINI_MLA_CHECK(next.value().target == b.target);
    }
    tokens[0] = 99;
    MINI_MLA_REQUIRE(shard(path, tokens, t.value()));
    auto changed = DataLoader::tokens(paths, 4, 264, t.value().hash(), 99);
    MINI_MLA_REQUIRE(changed);
    MINI_MLA_CHECK(!changed.value()->restore(state));
    std::filesystem::resize_file(path, 161);
    MINI_MLA_CHECK(!TokenShard::open(path, 264, t.value().hash()));
}
MINI_MLA_TEST(sft_targets_have_assistant_and_eos_loss_only) {
    Temporary temp;
    auto t = Tokenizer::load("examples/tokenizer.json");
    MINI_MLA_REQUIRE(t);
    auto path = temp.path / "sample.sft.jsonl";
    std::string text =
        Json(Json::Object{{"tokens", Json::Array{257, 259, 10, 65, 258, 260, 10, 72, 258}},
                          {"mask", Json::Array{0, 0, 0, 0, 0, 0, 0, 1, 1}}})
            .dump() +
        "\n";
    MINI_MLA_REQUIRE(write_text(path, text));
    MINI_MLA_REQUIRE(write_json(path.string() + ".manifest.json",
                                Json::Object{{"format", "mini-mla-sft-v1"},
                                             {"tokenizer_sha256", t.value().hash()},
                                             {"payload_sha256", hash_text(text)}}));
    std::filesystem::path paths[] = {path};
    auto loader = DataLoader::sft(paths, 16, 264, t.value().hash(), 1);
    MINI_MLA_REQUIRE(loader);
    auto b = loader.value()->next(1);
    MINI_MLA_REQUIRE(b);
    MINI_MLA_CHECK(b.value().supervised_tokens() == 2);
    MINI_MLA_CHECK(b.value().mask[6] == 1 && b.value().mask[7] == 1 && b.value().mask[8] == 0);
    MINI_MLA_CHECK(b.value().target[6] == 72 && b.value().target[7] == 258 &&
                   b.value().target[8] == 256);
}
MINI_MLA_TEST_MAIN()
