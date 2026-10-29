#include "mini_mla/json.hpp"

#include "mini_mla/io.hpp"

#include <charconv>
#include <climits>
#include <cmath>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>

namespace mini_mla {
namespace {
class Parser {
  public:
    explicit Parser(std::string_view t) : text(t) {}
    Json parse() {
        auto v = value(0);
        space();
        if (pos != text.size())
            fail("Trailing data");
        return v;
    }

  private:
    std::string_view text;
    std::size_t pos = 0;
    [[noreturn]] void fail(std::string_view why) const {
        throw std::invalid_argument(std::string(why) + " at JSON byte " + std::to_string(pos));
    }
    void space() {
        while (pos < text.size() &&
               (text[pos] == ' ' || text[pos] == '\n' || text[pos] == '\r' || text[pos] == '\t'))
            ++pos;
    }
    bool take(char c) {
        space();
        if (pos < text.size() && text[pos] == c) {
            ++pos;
            return true;
        }
        return false;
    }
    void expect(char c) {
        if (!take(c))
            fail("Unexpected character");
    }
    unsigned hex4() {
        unsigned n = 0;
        for (int i = 0; i < 4; ++i) {
            if (pos == text.size())
                fail("Truncated Unicode escape");
            char c = text[pos++];
            unsigned d;
            if (c >= '0' && c <= '9')
                d = static_cast<unsigned>(c - '0');
            else if (c >= 'a' && c <= 'f')
                d = static_cast<unsigned>(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F')
                d = static_cast<unsigned>(c - 'A' + 10);
            else
                fail("Invalid Unicode escape");
            n = n * 16 + d;
        }
        return n;
    }
    void utf8(std::string& s, unsigned c) {
        if (c < 0x80)
            s += static_cast<char>(c);
        else if (c < 0x800) {
            s += static_cast<char>(0xc0 | (c >> 6));
            s += static_cast<char>(0x80 | (c & 63));
        } else if (c < 0x10000) {
            s += static_cast<char>(0xe0 | (c >> 12));
            s += static_cast<char>(0x80 | ((c >> 6) & 63));
            s += static_cast<char>(0x80 | (c & 63));
        } else {
            s += static_cast<char>(0xf0 | (c >> 18));
            s += static_cast<char>(0x80 | ((c >> 12) & 63));
            s += static_cast<char>(0x80 | ((c >> 6) & 63));
            s += static_cast<char>(0x80 | (c & 63));
        }
    }
    std::string string() {
        expect('"');
        std::string s;
        while (pos < text.size()) {
            unsigned char c = static_cast<unsigned char>(text[pos++]);
            if (c == '"')
                return s;
            if (c < 32)
                fail("Control character in string");
            if (c != '\\') {
                s += static_cast<char>(c);
                continue;
            }
            if (pos == text.size())
                fail("Truncated escape");
            switch (text[pos++]) {
            case '"':
                s += '"';
                break;
            case '\\':
                s += '\\';
                break;
            case '/':
                s += '/';
                break;
            case 'b':
                s += '\b';
                break;
            case 'f':
                s += '\f';
                break;
            case 'n':
                s += '\n';
                break;
            case 'r':
                s += '\r';
                break;
            case 't':
                s += '\t';
                break;
            case 'u': {
                unsigned code = hex4();
                if (code >= 0xd800 && code <= 0xdbff) {
                    if (pos + 2 > text.size() || text.substr(pos, 2) != "\\u")
                        fail("Missing low surrogate");
                    pos += 2;
                    unsigned low = hex4();
                    if (low < 0xdc00 || low > 0xdfff)
                        fail("Invalid low surrogate");
                    code = 0x10000 + ((code - 0xd800) << 10) + low - 0xdc00;
                } else if (code >= 0xdc00 && code <= 0xdfff)
                    fail("Unpaired surrogate");
                utf8(s, code);
                break;
            }
            default:
                fail("Invalid escape");
            }
        }
        fail("Unterminated string");
    }
    Json value(int depth) {
        if (depth > 64)
            fail("JSON nesting exceeds 64");
        space();
        if (pos == text.size())
            fail("Missing value");
        if (text[pos] == '"')
            return Json(string());
        if (take('{')) {
            Json::Object obj;
            if (take('}'))
                return Json(std::move(obj));
            do {
                auto key = string();
                expect(':');
                if (!obj.emplace(key, value(depth + 1)).second)
                    fail("Duplicate object key");
            } while (take(','));
            expect('}');
            return Json(std::move(obj));
        }
        if (take('[')) {
            Json::Array arr;
            if (take(']'))
                return Json(std::move(arr));
            do {
                arr.push_back(value(depth + 1));
            } while (take(','));
            expect(']');
            return Json(std::move(arr));
        }
        for (auto literal :
             {std::string_view("true"), std::string_view("false"), std::string_view("null")})
            if (text.substr(pos, literal.size()) == literal) {
                pos += literal.size();
                if (literal == "null")
                    return Json();
                return Json(literal == "true");
            }
        std::size_t begin = pos;
        if (text[pos] == '-')
            ++pos;
        if (pos == text.size())
            fail("Invalid number");
        if (text[pos] == '0')
            ++pos;
        else {
            if (text[pos] < '1' || text[pos] > '9')
                fail("Invalid number");
            while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9')
                ++pos;
        }
        if (pos < text.size() && text[pos] == '.') {
            ++pos;
            auto first = pos;
            while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9')
                ++pos;
            if (pos == first)
                fail("Invalid fraction");
        }
        if (pos < text.size() && (text[pos] == 'e' || text[pos] == 'E')) {
            ++pos;
            if (pos < text.size() && (text[pos] == '+' || text[pos] == '-'))
                ++pos;
            auto first = pos;
            while (pos < text.size() && text[pos] >= '0' && text[pos] <= '9')
                ++pos;
            if (pos == first)
                fail("Invalid exponent");
        }
        double n;
        auto r = std::from_chars(text.data() + begin, text.data() + pos, n);
        if (r.ec != std::errc{} || !std::isfinite(n))
            fail("Number out of range");
        return Json(n);
    }
};
std::string quote(std::string_view s) {
    std::string out = "\"";
    const char* hex = "0123456789abcdef";
    for (char raw : s) {
        auto c = static_cast<unsigned char>(raw);
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (c < 32) {
                out += "\\u00";
                out += hex[c >> 4];
                out += hex[c & 15];
            } else
                out += static_cast<char>(c);
        }
    }
    return out + '"';
}
} // namespace
Result<Json> Json::parse(std::string_view text) {
    try {
        return Result<Json>::success(Parser(text).parse());
    } catch (const std::exception& e) {
        return Result<Json>::failure(ErrorCode::invalid_argument, e.what());
    }
}
std::string Json::dump() const {
    if (is_null())
        return "null";
    if (auto v = std::get_if<bool>(&value_))
        return *v ? "true" : "false";
    if (auto v = std::get_if<double>(&value_)) {
        if (!std::isfinite(*v))
            throw std::invalid_argument("Nonfinite JSON number");
        std::ostringstream s;
        s << std::setprecision(17) << *v;
        return s.str();
    }
    if (auto v = std::get_if<std::string>(&value_))
        return quote(*v);
    if (auto v = std::get_if<Array>(&value_)) {
        std::string s = "[";
        for (const auto& x : *v) {
            if (s.size() > 1)
                s += ',';
            s += x.dump();
        }
        return s + ']';
    }
    std::string s = "{";
    for (const auto& [key, v] : std::get<Object>(value_)) {
        if (s.size() > 1)
            s += ',';
        s += quote(key) + ':' + v.dump();
    }
    return s + '}';
}
const Json& Json::at(std::string_view key) const {
    auto& o = object();
    auto it = o.find(key);
    if (it == o.end())
        throw std::invalid_argument("Missing JSON field: " + std::string(key));
    return it->second;
}
bool Json::contains(std::string_view key) const {
    auto o = std::get_if<Object>(&value_);
    return o && o->contains(key);
}
Json& Json::operator[](std::string key) {
    if (is_null())
        value_ = Object{};
    return std::get<Object>(value_)[std::move(key)];
}
const Json::Array& Json::array() const {
    if (auto p = std::get_if<Array>(&value_))
        return *p;
    throw std::invalid_argument("Expected JSON array");
}
const Json::Object& Json::object() const {
    if (auto p = std::get_if<Object>(&value_))
        return *p;
    throw std::invalid_argument("Expected JSON object");
}
const std::string& Json::string() const {
    if (auto p = std::get_if<std::string>(&value_))
        return *p;
    throw std::invalid_argument("Expected JSON string");
}
double Json::number() const {
    if (auto p = std::get_if<double>(&value_))
        return *p;
    throw std::invalid_argument("Expected JSON number");
}
int Json::integer() const {
    double v = number();
    if (v != std::floor(v) || v < INT_MIN || v > INT_MAX)
        throw std::invalid_argument("Expected bounded integer");
    return static_cast<int>(v);
}
std::size_t Json::size_integer() const {
    double v = number();
    if (v != std::floor(v) || v < 0 || v > 9007199254740991.0)
        throw std::invalid_argument("Expected exact nonnegative integer");
    return static_cast<std::size_t>(v);
}
bool Json::boolean() const {
    if (auto p = std::get_if<bool>(&value_))
        return *p;
    throw std::invalid_argument("Expected JSON boolean");
}
std::string Json::str_or(std::string_view k, std::string fallback) const {
    return contains(k) ? at(k).string() : std::move(fallback);
}
double Json::num_or(std::string_view k, double fallback) const {
    return contains(k) ? at(k).number() : fallback;
}
bool Json::bool_or(std::string_view k, bool fallback) const {
    return contains(k) ? at(k).boolean() : fallback;
}
Result<Json> read_json(const std::filesystem::path& p) {
    auto t = read_text(p);
    if (!t)
        return Result<Json>::failure(t.error());
    return Json::parse(t.value());
}
Status write_json(const std::filesystem::path& p, const Json& j) {
    return write_text(p, j.dump() + "\n");
}
} // namespace mini_mla
