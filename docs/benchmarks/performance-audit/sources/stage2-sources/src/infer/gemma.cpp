#include "mini_mla/gemma.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <mlx/mlx.h>
#include <numbers>
#include <optional>
#include <set>
#include <unordered_map>

namespace mini_mla {
namespace mx = mlx::core;
namespace {
using Clock = std::chrono::steady_clock;
using Weights = std::unordered_map<std::string, mx::array>;
struct KV {
    mx::array keys, values;
    int cursor = 0;
};
struct Session {
    int position = 0;
    std::vector<std::optional<KV>> cache;
};
struct BatchCache {
    std::vector<int> handles;
    int position;
    std::vector<std::optional<KV>> cache;
};
mx::array gelu(const mx::array& x) {
    auto scalar = [&](float value) { return mx::array(value, x.dtype()); };
    return scalar(0.5F) * x *
           (scalar(1.0F) + mx::tanh(scalar(std::sqrt(2.0F / std::numbers::pi_v<float>)) *
                                    (x + scalar(0.044715F) * mx::power(x, mx::array(3)))));
}
double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
} // namespace
struct GemmaEngine::Impl {
    int layers, hidden, heads, kv_heads, head_dim, global_dim, shared_start, ple_width, window,
        vocab, context_limit, session_limit, bits, group_size;
    float epsilon, softcap;
    bool reduce_prefill_queries, batch_kv;
    std::vector<bool> sliding;
    std::vector<int> source_layer;
    Weights weights;
    std::map<int, Session> sessions;
    std::optional<BatchCache> batch_cache;
    std::uint64_t batch_cache_reuses = 0;
    std::optional<mx::array> last_logits, global_freqs;
    std::function<std::vector<mx::array>(const std::vector<mx::array>&)> geglu;
    std::function<std::vector<mx::array>(const std::vector<mx::array>&)> cap;
    double load_ms = 0;
    std::size_t weight_bytes = 0;

    Impl(const std::filesystem::path& path, int requested_context, int requested_sessions,
         bool reduce, bool batch)
        : reduce_prefill_queries(reduce), batch_kv(batch) {
        auto start = Clock::now();
        if (!mx::is_available(mx::Device::gpu))
            throw std::runtime_error("MLX Metal device is unavailable");
        mx::set_default_device(mx::Device::gpu);
        mx::set_cache_limit(256ULL << 20);
        auto top = read_json(path / "config.json");
        if (!top)
            throw std::invalid_argument(top.error().message);
        const auto& root = top.value();
        if (root.str_or("model_type", "") != "gemma4" &&
            root.str_or("model_type", "") != "gemma4_text")
            throw std::invalid_argument("Expected a Gemma 4 text checkpoint");
        const auto& c = root.contains("text_config") ? root.at("text_config") : root;
        layers = c.at("num_hidden_layers").integer();
        hidden = c.at("hidden_size").integer();
        heads = c.at("num_attention_heads").integer();
        kv_heads = c.at("num_key_value_heads").integer();
        head_dim = c.at("head_dim").integer();
        global_dim = c.at("global_head_dim").integer();
        shared_start = layers - c.at("num_kv_shared_layers").integer();
        ple_width = c.at("hidden_size_per_layer_input").integer();
        window = c.at("sliding_window").integer();
        vocab = c.at("vocab_size").integer();
        epsilon = static_cast<float>(c.at("rms_norm_eps").number());
        softcap = static_cast<float>(c.at("final_logit_softcapping").number());
        context_limit = requested_context;
        session_limit = requested_sessions;
        const auto& q = root.at("quantization");
        bits = q.at("bits").integer();
        group_size = q.at("group_size").integer();
        if (bits != 4 || group_size != 64 || q.str_or("mode", "affine") != "affine" || layers < 1 ||
            layers > 64 || hidden < 1 || hidden > 8192 || heads < 1 || heads > 64 ||
            kv_heads != 1 || head_dim != 256 || global_dim != 512 || ple_width != 256 ||
            shared_start < 1 || shared_start > layers || window != 512 || vocab != 262144 ||
            !c.bool_or("tie_word_embeddings", false) || c.bool_or("enable_moe_block", false) ||
            c.bool_or("attention_k_eq_v", false) || c.bool_or("rope_traditional", false) ||
            !c.bool_or("use_double_wide_mlp", true) || !std::isfinite(epsilon) || epsilon <= 0 ||
            !std::isfinite(softcap) || softcap <= 0 || context_limit < 1 ||
            context_limit > c.at("max_position_embeddings").integer() || session_limit < 1 ||
            session_limit > 64)
            throw std::invalid_argument("Unsupported Gemma 4 configuration or execution limits");
        std::map<bool, int> previous;
        for (const auto& type : c.at("layer_types").array()) {
            if (type.string() != "sliding_attention" && type.string() != "full_attention")
                throw std::invalid_argument("Unknown Gemma attention type");
            auto i = static_cast<int>(sliding.size());
            bool local = type.string() == "sliding_attention";
            sliding.push_back(local);
            if (i < shared_start)
                previous[local] = i;
            if (!previous.contains(local))
                throw std::invalid_argument("Missing source layer for shared Gemma KV");
            source_layer.push_back(i < shared_start ? i : previous.at(local));
        }
        if (sliding.size() != static_cast<std::size_t>(layers))
            throw std::invalid_argument("Gemma layer types do not match layer count");
        const auto& rope = c.at("rope_parameters");
        const auto& full = rope.at("full_attention");
        const auto& local = rope.at("sliding_attention");
        if (full.str_or("rope_type", "") != "proportional" ||
            full.num_or("partial_rotary_factor", 0) != 0.25 ||
            full.num_or("rope_theta", 0) != 1000000 || local.str_or("rope_type", "") != "default" ||
            local.num_or("rope_theta", 0) != 10000)
            throw std::invalid_argument("Unsupported Gemma rotary configuration");
        auto exponents = mx::arange(0.0, static_cast<double>(global_dim / 4), 2.0, mx::float32) /
                         mx::array(static_cast<float>(global_dim));
        global_freqs = mx::concatenate(
            {mx::power(mx::array(1000000.0F), exponents),
             mx::full({global_dim * 3 / 8}, std::numeric_limits<float>::infinity(), mx::float32)});
        auto loaded = mx::load_safetensors((path / "model.safetensors").string());
        for (auto& [name, tensor] : loaded.first) {
            auto key = name;
            if (key.starts_with("language_model."))
                key.erase(0, std::string("language_model.").size());
            else if (!key.starts_with("model."))
                continue;
            if (key.find("layers.") != std::string::npos &&
                (key.find(".self_attn.k_proj") != std::string::npos ||
                 key.find(".self_attn.v_proj") != std::string::npos ||
                 key.find(".self_attn.k_norm") != std::string::npos)) {
                auto begin = key.find("layers.") + 7;
                auto end = key.find('.', begin);
                int layer = std::stoi(key.substr(begin, end - begin));
                if (layer >= shared_start)
                    continue;
            }
            if (key.find("input_max") != std::string::npos ||
                key.find("input_min") != std::string::npos ||
                key.find("output_max") != std::string::npos ||
                key.find("output_min") != std::string::npos)
                continue;
            weight_bytes += tensor.nbytes();
            if (!weights.emplace(std::move(key), std::move(tensor)).second)
                throw std::invalid_argument("Duplicate Gemma tensor name");
        }
        loaded.first.clear();
        check_linear("model.embed_tokens", hidden, vocab);
        check_linear("model.embed_tokens_per_layer", layers * ple_width, vocab);
        check_linear("model.per_layer_model_projection", hidden, layers * ple_width);
        check_shape("model.per_layer_projection_norm.weight", {ple_width});
        check_shape("model.norm.weight", {hidden});
        int ff = c.at("intermediate_size").integer();
        if (ff < 1 || ff > 32768)
            throw std::invalid_argument("Invalid Gemma feedforward width");
        for (int l = 0; l < layers; ++l) {
            auto p = prefix(l);
            int d = sliding[static_cast<std::size_t>(l)] ? head_dim : global_dim;
            int width = l < shared_start ? ff : ff * 2;
            for (auto norm :
                 {"input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm",
                  "post_feedforward_layernorm", "post_per_layer_input_norm"})
                check_shape(p + norm + ".weight", {hidden});
            check_shape(p + "layer_scalar", {1});
            check_shape(p + "self_attn.q_norm.weight", {d});
            check_linear(p + "self_attn.q_proj", hidden, heads * d);
            check_linear(p + "self_attn.o_proj", heads * d, hidden);
            if (l < shared_start) {
                check_linear(p + "self_attn.k_proj", hidden, kv_heads * d);
                check_linear(p + "self_attn.v_proj", hidden, kv_heads * d);
                check_shape(p + "self_attn.k_norm.weight", {d});
            }
            check_linear(p + "mlp.gate_proj", hidden, width);
            check_linear(p + "mlp.up_proj", hidden, width);
            check_linear(p + "mlp.down_proj", width, hidden);
            check_linear(p + "per_layer_input_gate", hidden, ple_width);
            check_linear(p + "per_layer_projection", ple_width, hidden);
        }
        geglu = mx::compile(
            [](const std::vector<mx::array>& v) {
                return std::vector<mx::array>{gelu(v[0]) * v[1]};
            },
            true);
        cap = mx::compile(
            [](const std::vector<mx::array>& v) {
                return std::vector<mx::array>{mx::tanh(v[0] / v[1]) * v[1]};
            },
            true);
        std::vector<mx::array> tensors;
        for (const auto& [name, tensor] : weights) {
            (void)name;
            tensors.push_back(tensor);
        }
        tensors.push_back(*global_freqs);
        mx::eval(std::move(tensors));
        load_ms = milliseconds(start);
    }
    static std::string prefix(int layer) { return "model.layers." + std::to_string(layer) + "."; }
    const mx::array& w(const std::string& name) const {
        auto it = weights.find(name);
        if (it == weights.end())
            throw std::invalid_argument("Missing Gemma tensor: " + name);
        return it->second;
    }
    void check_shape(const std::string& name, mx::Shape expected) const {
        if (w(name).shape() != expected)
            throw std::invalid_argument("Gemma tensor shape mismatch: " + name);
    }
    void check_linear(const std::string& name, int input, int output) const {
        if (w(name + ".weight").dtype() != mx::uint32)
            throw std::invalid_argument("Expected a packed 4-bit Gemma weight: " + name);
        check_shape(name + ".weight", {output, input / 8});
        check_shape(name + ".scales", {output, input / group_size});
        check_shape(name + ".biases", {output, input / group_size});
    }
    mx::array linear(const std::string& name, const mx::array& x) const {
        return mx::quantized_matmul(x, w(name + ".weight"), w(name + ".scales"),
                                    w(name + ".biases"), true, group_size, bits);
    }
    mx::array embedding(const std::string& name, const mx::array& ids) const {
        auto flat = mx::reshape(ids, {-1});
        auto values = mx::dequantize(mx::take(w(name + ".weight"), flat, 0),
                                     mx::take(w(name + ".scales"), flat, 0),
                                     mx::take(w(name + ".biases"), flat, 0), group_size, bits);
        return mx::reshape(values, {ids.shape(0), ids.shape(1), values.shape(-1)});
    }
    mx::array norm(const std::string& name, const mx::array& x) const {
        return mx::fast::rms_norm(x, w(name + ".weight"), epsilon);
    }
    mx::array rotate(const mx::array& x, bool local, const mx::array& offsets) const {
        return local ? mx::fast::rope(x, head_dim, false, 10000.0F, 1.0F, offsets)
                     : mx::fast::rope(x, global_dim, false, std::nullopt, 1.0F, offsets,
                                      global_freqs);
    }
    KV append_cache(const std::optional<KV>& old, mx::array key, mx::array value, bool local,
                    int time) const {
        int count = key.shape(0), d = key.shape(3), cursor = time;
        if (!old)
            return {std::move(key), std::move(value), cursor};
        auto cached_keys = old->keys, cached_values = old->values;
        int length = cached_keys.shape(2);
        cursor = old->cursor;
        if (local && time == 1 && length >= window) {
            if (length > window) {
                cached_keys = mx::slice(cached_keys, {0, 0, length - window, 0},
                                        {count, kv_heads, length, d});
                cached_values = mx::slice(cached_values, {0, 0, length - window, 0},
                                          {count, kv_heads, length, d});
                cursor = window;
            }
            if (cursor == window)
                cursor = 0;
            key = mx::slice_update(cached_keys, key, {0, 0, cursor, 0},
                                   {count, kv_heads, cursor + 1, d});
            value = mx::slice_update(cached_values, value, {0, 0, cursor, 0},
                                     {count, kv_heads, cursor + 1, d});
            ++cursor;
        } else {
            if (local && time > 1) {
                if (cursor < length) {
                    cached_keys = mx::concatenate(
                        {mx::slice(cached_keys, {0, 0, cursor, 0}, {count, kv_heads, length, d}),
                         mx::slice(cached_keys, {0, 0, 0, 0}, {count, kv_heads, cursor, d})},
                        2);
                    cached_values = mx::concatenate(
                        {mx::slice(cached_values, {0, 0, cursor, 0}, {count, kv_heads, length, d}),
                         mx::slice(cached_values, {0, 0, 0, 0}, {count, kv_heads, cursor, d})},
                        2);
                }
                int trim = std::max(0, length - window + 1);
                cached_keys = mx::slice(cached_keys, {0, 0, trim, 0}, {count, kv_heads, length, d});
                cached_values =
                    mx::slice(cached_values, {0, 0, trim, 0}, {count, kv_heads, length, d});
            }
            key = mx::concatenate({cached_keys, key}, 2);
            value = mx::concatenate({cached_values, value}, 2);
            cursor = key.shape(2);
        }
        return {std::move(key), std::move(value), cursor};
    }
    Json run(std::span<const int> handles, std::span<const std::uint32_t> ids, int time) {
        auto start = Clock::now();
        int batch = static_cast<int>(handles.size());
        std::vector<int> offsets;
        std::vector<Session*> states;
        std::set<int> distinct;
        for (int handle : handles) {
            if (!distinct.insert(handle).second || !sessions.contains(handle))
                throw std::invalid_argument("Unknown or duplicate Gemma session");
            auto& state = sessions.at(handle);
            if (state.position > context_limit - time)
                throw std::invalid_argument("Gemma context limit exceeded");
            states.push_back(&state);
            offsets.push_back(state.position);
        }
        for (auto token : ids)
            if (token >= static_cast<std::uint32_t>(vocab))
                throw std::invalid_argument("Gemma token exceeds vocabulary");
        bool same_position = std::all_of(offsets.begin(), offsets.end(),
                                         [&](int offset) { return offset == offsets.front(); });
        bool cache_batch = batch_kv && batch > 1 && same_position;
        bool reuse_batch = cache_batch && time == 1 && batch_cache &&
                           batch_cache->position == offsets.front() &&
                           batch_cache->handles.size() == handles.size() &&
                           std::equal(handles.begin(), handles.end(), batch_cache->handles.begin());
        auto input = mx::array(ids.begin(), {batch, time}, mx::uint32);
        auto position = mx::array(offsets.begin(), {batch});
        auto query_position = position;
        int query_time = time;
        const char* trace = std::getenv("MINI_MLA_GEMMA_TRACE");
        auto h = embedding("model.embed_tokens", input);
        h = h * mx::array(std::sqrt(static_cast<float>(hidden)), h.dtype());
        auto ple = embedding("model.embed_tokens_per_layer", input);
        ple = mx::reshape(ple * mx::array(std::sqrt(static_cast<float>(ple_width)), ple.dtype()),
                          {batch, time, layers, ple_width});
        auto projected = linear("model.per_layer_model_projection", h);
        projected = mx::reshape(
            projected * mx::array(1.0F / std::sqrt(static_cast<float>(hidden)), projected.dtype()),
            {batch, time, layers, ple_width});
        ple = norm("model.per_layer_projection_norm", projected) + ple;
        ple = ple * mx::array(std::sqrt(0.5F), ple.dtype());
        if (trace) {
            mx::save((std::filesystem::path(trace) / "embedding.npy").string(),
                     mx::astype(h, mx::float32));
            mx::save((std::filesystem::path(trace) / "ple.npy").string(),
                     mx::astype(ple, mx::float32));
        }
        std::vector<std::optional<KV>> packed(static_cast<std::size_t>(shared_start));
        std::vector<std::vector<std::optional<KV>>> staged(
            static_cast<std::size_t>(batch),
            std::vector<std::optional<KV>>(static_cast<std::size_t>(shared_start)));
        std::vector<std::optional<mx::array>> masks(static_cast<std::size_t>(shared_start));
        for (int l = 0; l < layers; ++l) {
            if (l == shared_start && time > 1 && reduce_prefill_queries) {
                h = mx::slice(h, {0, time - 1, 0}, {batch, time, hidden});
                ple = mx::slice(ple, {0, time - 1, 0, 0}, {batch, time, layers, ple_width});
                query_time = 1;
                query_position = position + mx::array(time - 1);
                for (auto& mask : masks)
                    if (mask)
                        mask = mx::slice(*mask, {0, 0, time - 1, 0},
                                         {batch, 1, time, mask->shape(-1)});
            }
            auto index = static_cast<std::size_t>(l);
            auto p = prefix(l);
            bool local = sliding[index];
            int d = local ? head_dim : global_dim;
            auto x = norm(p + "input_layernorm", h);
            auto q = norm(p + "self_attn.q_norm", mx::reshape(linear(p + "self_attn.q_proj", x),
                                                              {batch, query_time, heads, d}));
            q = rotate(mx::transpose(q, {0, 2, 1, 3}), local, query_position);
            if (l < shared_start) {
                auto k = norm(p + "self_attn.k_norm", mx::reshape(linear(p + "self_attn.k_proj", x),
                                                                  {batch, time, kv_heads, d}));
                k = rotate(mx::transpose(k, {0, 2, 1, 3}), local, position);
                auto v = mx::fast::rms_norm(
                    mx::reshape(linear(p + "self_attn.v_proj", x), {batch, time, kv_heads, d}),
                    std::nullopt, epsilon);
                v = mx::transpose(v, {0, 2, 1, 3});
                std::vector<int> lengths;
                int longest = 0;
                if (reuse_batch) {
                    auto combined = append_cache(batch_cache->cache[index], k, v, local, time);
                    longest = combined.keys.shape(2);
                    lengths.assign(static_cast<std::size_t>(batch), longest);
                    for (int b = 0; b < batch; ++b)
                        staged[static_cast<std::size_t>(b)][index] = KV{
                            mx::slice(combined.keys, {b, 0, 0, 0}, {b + 1, kv_heads, longest, d}),
                            mx::slice(combined.values, {b, 0, 0, 0}, {b + 1, kv_heads, longest, d}),
                            combined.cursor};
                    packed[index] = std::move(combined);
                } else {
                    std::vector<mx::array> all_k, all_v;
                    for (int b = 0; b < batch; ++b) {
                        auto bi = static_cast<std::size_t>(b);
                        auto combined = append_cache(
                            states[bi]->cache[index],
                            mx::slice(k, {b, 0, 0, 0}, {b + 1, kv_heads, time, d}),
                            mx::slice(v, {b, 0, 0, 0}, {b + 1, kv_heads, time, d}), local, time);
                        int length = combined.keys.shape(2);
                        lengths.push_back(length);
                        longest = std::max(longest, length);
                        staged[bi][index] = combined;
                        all_k.push_back(std::move(combined.keys));
                        all_v.push_back(std::move(combined.values));
                    }
                    for (int b = 0; b < batch; ++b) {
                        auto bi = static_cast<std::size_t>(b);
                        if (lengths[bi] < longest) {
                            auto padding =
                                mx::zeros({1, kv_heads, longest - lengths[bi], d}, k.dtype());
                            all_k[bi] = mx::concatenate({all_k[bi], padding}, 2);
                            all_v[bi] = mx::concatenate({all_v[bi], padding}, 2);
                        }
                    }
                    packed[index] = KV{mx::concatenate(all_k, 0), mx::concatenate(all_v, 0)};
                }
                if (trace) {
                    mx::save((std::filesystem::path(trace) / ("keys-" + std::to_string(l) + ".npy"))
                                 .string(),
                             mx::astype(packed[index]->keys, mx::float32));
                    mx::save(
                        (std::filesystem::path(trace) / ("values-" + std::to_string(l) + ".npy"))
                            .string(),
                        mx::astype(packed[index]->values, mx::float32));
                }
                if (time > 1 || *std::min_element(lengths.begin(), lengths.end()) != longest ||
                    (local && longest > window)) {
                    std::vector<std::uint8_t> allowed(static_cast<std::size_t>(batch) *
                                                      static_cast<std::size_t>(time) *
                                                      static_cast<std::size_t>(longest));
                    std::size_t i = 0;
                    for (int b = 0; b < batch; ++b)
                        for (int t = 0; t < time; ++t) {
                            int current = lengths[static_cast<std::size_t>(b)] - time + t;
                            for (int j = 0; j < longest; ++j)
                                allowed[i++] = static_cast<std::uint8_t>(
                                    j <= current && (!local || j > current - window));
                        }
                    masks[index] = mx::array(allowed.begin(), {batch, 1, time, longest}, mx::bool_);
                }
            }
            auto source = static_cast<std::size_t>(source_layer[index]);
            const auto& kv = *packed[source];
            auto attention = mx::fast::scaled_dot_product_attention(q, kv.keys, kv.values, 1.0F, "",
                                                                    masks[source]);
            attention =
                mx::reshape(mx::transpose(attention, {0, 2, 1, 3}), {batch, query_time, heads * d});
            h = h + norm(p + "post_attention_layernorm", linear(p + "self_attn.o_proj", attention));
            auto f = norm(p + "pre_feedforward_layernorm", h);
            f = geglu({linear(p + "mlp.gate_proj", f), linear(p + "mlp.up_proj", f)})[0];
            h = h + norm(p + "post_feedforward_layernorm", linear(p + "mlp.down_proj", f));
            auto layer_input =
                mx::reshape(mx::slice(ple, {0, 0, l, 0}, {batch, query_time, l + 1, ple_width}),
                            {batch, query_time, ple_width});
            auto gate = geglu({linear(p + "per_layer_input_gate", h), layer_input})[0];
            h = (h +
                 norm(p + "post_per_layer_input_norm", linear(p + "per_layer_projection", gate))) *
                w(p + "layer_scalar");
            if (trace)
                mx::save((std::filesystem::path(trace) / ("layer-" + std::to_string(l) + ".npy"))
                             .string(),
                         mx::astype(h, mx::float32));
        }
        h = mx::reshape(mx::slice(h, {0, query_time - 1, 0}, {batch, query_time, hidden}),
                        {batch, hidden});
        auto logits = linear("model.embed_tokens", norm("model.norm", h));
        logits = cap({logits, mx::array(softcap, logits.dtype())})[0];
        auto tokens = mx::argmax(logits, -1);
        auto finite = mx::all(mx::isfinite(logits));
        std::vector<mx::array> evaluate{tokens, logits, finite};
        for (const auto& row : staged)
            for (const auto& kv : row) {
                evaluate.push_back(kv->keys);
                evaluate.push_back(kv->values);
            }
        mx::eval(std::move(evaluate));
        if (!finite.item<bool>())
            throw std::runtime_error("Nonfinite Gemma logits");
        bool keep_batch = cache_batch;
        for (std::size_t l = 0; keep_batch && l < static_cast<std::size_t>(shared_start); ++l) {
            const auto& first = *staged.front()[l];
            for (const auto& row : staged)
                if (row[l]->cursor != first.cursor || row[l]->keys.shape(2) != first.keys.shape(2))
                    keep_batch = false;
            packed[l]->cursor = first.cursor;
        }
        if (keep_batch)
            batch_cache = BatchCache{std::vector<int>(handles.begin(), handles.end()),
                                     offsets.front() + time, std::move(packed)};
        else
            batch_cache.reset();
        if (reuse_batch)
            ++batch_cache_reuses;
        Json::Array output;
        auto chosen = tokens.data<std::uint32_t>();
        for (int b = 0; b < batch; ++b) {
            auto bi = static_cast<std::size_t>(b);
            states[bi]->cache = std::move(staged[bi]);
            states[bi]->position += time;
            output.emplace_back(static_cast<std::size_t>(chosen[b]));
        }
        last_logits = std::move(logits);
        return Json::Object{
            {"tokens", std::move(output)}, {"elapsed_ms", milliseconds(start)}, {"batch", batch}};
    }
};
GemmaEngine::GemmaEngine(const std::filesystem::path& model, int context, int sessions, bool reduce,
                         bool batch_kv)
    : impl_(std::make_unique<Impl>(model, context, sessions, reduce, batch_kv)) {}
GemmaEngine::~GemmaEngine() = default;
Json GemmaEngine::prefill(int session, std::span<const std::uint32_t> ids) {
    if (ids.size() > static_cast<std::size_t>(impl_->context_limit))
        throw std::invalid_argument("Gemma prompt exceeds context limit");
    return prefill_batch(std::span<const int>{&session, 1}, ids, static_cast<int>(ids.size()));
}
Json GemmaEngine::prefill_batch(std::span<const int> handles, std::span<const std::uint32_t> ids,
                                int length) {
    if (handles.empty() || length < 1 || length > impl_->context_limit ||
        handles.size() > static_cast<std::size_t>(impl_->session_limit) ||
        ids.size() != handles.size() * static_cast<std::size_t>(length))
        throw std::invalid_argument("Invalid Gemma prefill batch");
    std::set<int> distinct;
    std::size_t added = 0;
    for (int handle : handles) {
        if (handle < 0 || !distinct.insert(handle).second)
            throw std::invalid_argument("Invalid or duplicate Gemma session handle");
        added += !impl_->sessions.contains(handle);
    }
    if (impl_->sessions.size() + added > static_cast<std::size_t>(impl_->session_limit))
        throw std::invalid_argument("Gemma session limit exceeded");
    for (auto token : ids)
        if (token >= static_cast<std::uint32_t>(impl_->vocab))
            throw std::invalid_argument("Gemma token exceeds vocabulary");
    impl_->batch_cache.reset();
    auto start = Clock::now();
    for (int handle : handles)
        impl_->sessions[handle] = Session{
            0, std::vector<std::optional<KV>>(static_cast<std::size_t>(impl_->shared_start))};
    try {
        Json result;
        for (int offset = 0; offset < length; offset += 512) {
            int count = std::min(512, length - offset);
            std::vector<std::uint32_t> chunk;
            chunk.reserve(handles.size() * static_cast<std::size_t>(count));
            for (std::size_t b = 0; b < handles.size(); ++b) {
                auto row = b * static_cast<std::size_t>(length) + static_cast<std::size_t>(offset);
                chunk.insert(chunk.end(), ids.begin() + static_cast<std::ptrdiff_t>(row),
                             ids.begin() + static_cast<std::ptrdiff_t>(
                                               row + static_cast<std::size_t>(count)));
            }
            result = impl_->run(handles, chunk, count);
        }
        result["elapsed_ms"] = milliseconds(start);
        return result;
    } catch (...) {
        for (int handle : handles)
            impl_->sessions.erase(handle);
        throw;
    }
}
Json GemmaEngine::decode(std::span<const int> sessions, std::span<const std::uint32_t> tokens) {
    if (sessions.empty() || sessions.size() != tokens.size() ||
        sessions.size() > static_cast<std::size_t>(impl_->session_limit))
        throw std::invalid_argument("Invalid Gemma decode batch");
    return impl_->run(sessions, tokens, 1);
}
void GemmaEngine::release(int session) {
    impl_->batch_cache.reset();
    impl_->sessions.erase(session);
}
void GemmaEngine::save_logits(const std::filesystem::path& path) const {
    if (!impl_->last_logits)
        throw std::invalid_argument("No Gemma logits are available");
    mx::save(path.string(), mx::astype(*impl_->last_logits, mx::float32));
}
Json GemmaEngine::info() const {
    std::size_t cache_bytes = 0;
    for (const auto& [id, state] : impl_->sessions) {
        (void)id;
        for (const auto& kv : state.cache)
            if (kv)
                cache_bytes += kv->keys.nbytes() + kv->values.nbytes();
    }
    return Json::Object{{"model_type", "gemma4_text"},
                        {"protocol_version", 2},
                        {"backend", "MLX C++ / Metal"},
                        {"quantization", "affine 4-bit, group 64"},
                        {"token_id_bits", 32},
                        {"text_only", true},
                        {"context_limit", impl_->context_limit},
                        {"prefill_query_reduction", impl_->reduce_prefill_queries},
                        {"batched_kv", impl_->batch_kv},
                        {"batch_cache_reuses", static_cast<std::size_t>(impl_->batch_cache_reuses)},
                        {"session_limit", impl_->session_limit},
                        {"active_sessions", impl_->sessions.size()},
                        {"weight_bytes", impl_->weight_bytes},
                        {"kv_bytes", cache_bytes},
                        {"mlx_active_bytes", mx::get_active_memory()},
                        {"mlx_peak_bytes", mx::get_peak_memory()},
                        {"load_ms", impl_->load_ms}};
}
} // namespace mini_mla
