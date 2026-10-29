#include "mini_mla/tokenizer.hpp"

#include "mini_mla/io.hpp"

#include <limits>
#include <queue>
#include <set>

namespace mini_mla {
namespace {
constexpr std::string_view special_names[] = {"<|pad|>",       "<|bos|>",    "<|eos|>",  "<|user|>",
                                              "<|assistant|>", "<|system|>", "<|tool|>", "<|sep|>"};
std::size_t utf8_length(std::string_view s, std::size_t i) {
    auto c = static_cast<unsigned char>(s[i]);
    if (c < 0x80)
        return 1;
    std::size_t n = c >= 0xc2 && c <= 0xdf   ? 2
                    : c >= 0xe0 && c <= 0xef ? 3
                    : c >= 0xf0 && c <= 0xf4 ? 4
                                             : 0;
    if (!n || i + n > s.size())
        return 0;
    for (std::size_t k = 1; k < n; ++k) {
        auto b = static_cast<unsigned char>(s[i + k]);
        if (b < 0x80 || b > 0xbf)
            return 0;
    }
    auto b = static_cast<unsigned char>(s[i + 1]);
    if ((c == 0xe0 && b < 0xa0) || (c == 0xed && b >= 0xa0) || (c == 0xf0 && b < 0x90) ||
        (c == 0xf4 && b >= 0x90))
        return 0;
    return n;
}
} // namespace
bool valid_utf8(std::string_view s) {
    for (std::size_t i = 0; i < s.size();) {
        auto n = utf8_length(s, i);
        if (!n)
            return false;
        i += n;
    }
    return true;
}
std::string clean_utf8(std::string_view s) {
    std::string out;
    for (std::size_t i = 0; i < s.size();) {
        auto n = utf8_length(s, i);
        if (n) {
            out.append(s.substr(i, n));
            i += n;
        } else {
            out += "\xef\xbf\xbd";
            ++i;
        }
    }
    return out;
}
Tokenizer Tokenizer::bytes() {
    Tokenizer t;
    t.vocabulary_.resize(264);
    for (int i = 0; i < 256; ++i)
        t.vocabulary_[static_cast<std::size_t>(i)] = std::string(1, static_cast<char>(i));
    for (std::size_t i = 0; i < 8; ++i) {
        t.specials_.emplace(std::string(special_names[i]), static_cast<std::uint16_t>(256 + i));
        t.vocabulary_[256 + i] = special_names[i];
    }
    t.hash_ = hash_text("mini-mla-byte-tokenizer-v1");
    return t;
}
Result<Tokenizer> Tokenizer::load(const std::filesystem::path& p) {
    auto text = read_text(p);
    if (!text)
        return Result<Tokenizer>::failure(text.error());
    auto parsed = Json::parse(text.value());
    if (!parsed)
        return Result<Tokenizer>::failure(parsed.error());
    try {
        const auto& j = parsed.value();
        if (j.at("format").string() != "mini-mla-tokenizer" || j.at("version").integer() != 1 ||
            j.at("pretokenizer").string() != "bytes-v1")
            return Result<Tokenizer>::failure(ErrorCode::unsupported,
                                              "Unsupported tokenizer format or pretokenizer");
        int vocab = j.at("vocab_size").integer();
        if (vocab < 264 || vocab > 65535)
            return Result<Tokenizer>::failure(ErrorCode::invalid_argument,
                                              "Tokenizer vocabulary must be in [264,65535]");
        Tokenizer t;
        t.vocabulary_.resize(static_cast<std::size_t>(vocab));
        for (int i = 0; i < 256; ++i)
            t.vocabulary_[static_cast<std::size_t>(i)] = std::string(1, static_cast<char>(i));
        std::size_t id = 256;
        for (const auto& merge : j.at("merges").array()) {
            const auto& a = merge.array();
            if (a.size() != 3 || id >= static_cast<std::size_t>(vocab - 8) ||
                a[2].integer() != static_cast<int>(id))
                throw std::invalid_argument("Malformed BPE merge");
            int left = a[0].integer(), right = a[1].integer();
            if (left < 0 || right < 0 || static_cast<std::size_t>(left) >= id ||
                static_cast<std::size_t>(right) >= id)
                throw std::invalid_argument("BPE merge references a future token");
            auto key = (static_cast<std::uint32_t>(left) << 16) | static_cast<std::uint32_t>(right);
            if (!t.merges_.emplace(key, static_cast<std::uint16_t>(id)).second)
                throw std::invalid_argument("Duplicate BPE pair");
            auto& word = t.vocabulary_[id];
            word = t.vocabulary_[static_cast<std::size_t>(left)] +
                   t.vocabulary_[static_cast<std::size_t>(right)];
            if (word.size() > 1024 * 1024)
                throw std::invalid_argument("BPE token exceeds size limit");
            ++id;
        }
        if (j.at("specials").object().size() != 8)
            throw std::invalid_argument("Tokenizer must define exactly eight special tokens");
        std::size_t special_index = 0;
        for (auto name : special_names) {
            int special_id = j.at("specials").at(name).integer();
            if (special_id != vocab - 8 + static_cast<int>(special_index++))
                throw std::invalid_argument(
                    "Special token IDs disagree with the native chat template");
            t.specials_.emplace(name, static_cast<std::uint16_t>(special_id));
            t.vocabulary_[static_cast<std::size_t>(special_id)] = name;
        }
        t.hash_ = hash_text(text.value());
        return Result<Tokenizer>::success(std::move(t));
    } catch (const std::exception& e) {
        return Result<Tokenizer>::failure(ErrorCode::invalid_argument, e.what());
    }
}
void Tokenizer::encode_bytes(std::string_view s, std::vector<std::uint16_t>& out) const {
    if (merges_.empty()) {
        for (char c : s)
            out.push_back(static_cast<unsigned char>(c));
        return;
    }
    struct Node {
        std::uint16_t id;
        std::size_t prev, next, version = 0;
        bool alive = true;
    };
    struct Pair {
        std::uint16_t rank;
        std::size_t left, right, lv, rv;
        bool operator>(const Pair& p) const {
            return rank > p.rank || (rank == p.rank && left > p.left);
        }
    };
    const auto end = std::numeric_limits<std::size_t>::max();
    std::vector<Node> nodes;
    nodes.reserve(s.size());
    for (std::size_t i = 0; i < s.size(); ++i)
        nodes.push_back(Node{static_cast<unsigned char>(s[i]), i == 0 ? end : i - 1,
                             i + 1 == s.size() ? end : i + 1});
    std::priority_queue<Pair, std::vector<Pair>, std::greater<>> heap;
    auto push = [&](std::size_t left) {
        if (left == end || !nodes[left].alive || nodes[left].next == end)
            return;
        auto right = nodes[left].next;
        auto key = (static_cast<std::uint32_t>(nodes[left].id) << 16) | nodes[right].id;
        auto it = merges_.find(key);
        if (it != merges_.end())
            heap.push(Pair{it->second, left, right, nodes[left].version, nodes[right].version});
    };
    for (std::size_t i = 0; i < nodes.size(); ++i)
        push(i);
    while (!heap.empty()) {
        auto p = heap.top();
        heap.pop();
        auto& l = nodes[p.left];
        auto& r = nodes[p.right];
        if (!l.alive || !r.alive || l.next != p.right || l.version != p.lv || r.version != p.rv)
            continue;
        l.id = p.rank;
        ++l.version;
        l.next = r.next;
        r.alive = false;
        if (r.next != end)
            nodes[r.next].prev = p.left;
        push(l.prev);
        push(p.left);
    }
    for (const auto& n : nodes)
        if (n.alive)
            out.push_back(n.id);
}
Result<std::vector<std::uint16_t>> Tokenizer::encode(std::string_view s, bool allow_special) const {
    std::vector<std::uint16_t> result;
    if (!allow_special) {
        encode_bytes(s, result);
        return Result<std::vector<std::uint16_t>>::success(std::move(result));
    }
    std::size_t begin = 0;
    for (std::size_t i = 0; i < s.size();) {
        bool matched = false;
        if (s[i] == '<')
            for (const auto& [name, id] : specials_)
                if (s.substr(i, name.size()) == name) {
                    encode_bytes(s.substr(begin, i - begin), result);
                    result.push_back(id);
                    i += name.size();
                    begin = i;
                    matched = true;
                    break;
                }
        if (!matched)
            ++i;
    }
    encode_bytes(s.substr(begin), result);
    return Result<std::vector<std::uint16_t>>::success(std::move(result));
}
Result<std::string> Tokenizer::decode(std::span<const std::uint16_t> ids, bool skip_special) const {
    std::string s;
    for (auto id : ids) {
        if (id >= vocabulary_.size() || vocabulary_[id].empty())
            return Result<std::string>::failure(ErrorCode::invalid_argument, "Unknown token ID");
        if (!skip_special || !is_special(id))
            s += vocabulary_[id];
    }
    return Result<std::string>::success(std::move(s));
}
std::uint16_t Tokenizer::special(std::string_view name) const {
    auto it = specials_.find(name);
    if (it == specials_.end())
        detail::fatal("Unknown special token");
    return it->second;
}
bool Tokenizer::is_special(std::uint16_t id) const {
    for (const auto& [name, x] : specials_) {
        (void)name;
        if (id == x)
            return true;
    }
    return false;
}
std::string_view Tokenizer::token_bytes(std::uint16_t id) const {
    return id < vocabulary_.size() ? vocabulary_[id] : std::string_view{};
}
Result<std::vector<std::uint16_t>> Tokenizer::chat_prompt(const Json::Array& messages) const {
    try {
        if (messages.empty())
            return Result<std::vector<std::uint16_t>>::failure(ErrorCode::invalid_argument,
                                                               "Messages must not be empty");
        std::vector<std::uint16_t> ids{special("<|bos|>")};
        for (const auto& message : messages) {
            auto role = message.at("role").string();
            auto content = message.at("content").string();
            if (role != "system" && role != "user" && role != "assistant" && role != "tool")
                throw std::invalid_argument("Unsupported chat role");
            ids.push_back(special("<|" + role + "|>"));
            ids.push_back(10);
            auto text = encode(content);
            if (!text)
                return Result<std::vector<std::uint16_t>>::failure(text.error());
            ids.insert(ids.end(), text.value().begin(), text.value().end());
            ids.push_back(special("<|eos|>"));
            ids.push_back(10);
        }
        ids.push_back(special("<|assistant|>"));
        ids.push_back(10);
        return Result<std::vector<std::uint16_t>>::success(std::move(ids));
    } catch (const std::exception& e) {
        return Result<std::vector<std::uint16_t>>::failure(ErrorCode::invalid_argument, e.what());
    }
}
} // namespace mini_mla
