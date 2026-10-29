#include <metal_simdgroup_matrix>
#include <metal_stdlib>
using namespace metal;

struct AttentionDimensions {
    uint heads, time, key_width, value_width;
    float scale;
};

template <typename T>
void flash_forward(device const T* q, device const T* k, device const T* v, device float* out,
                   device float* lse, constant AttentionDimensions& d, threadgroup float* keys,
                   threadgroup float* values, uint group, uint thread_id, uint simd_id, uint lane) {
    uint blocks = (d.time + 3) / 4;
    uint head = group / blocks;
    uint first = (group % blocks) * 4;
    uint row = first + simd_id;
    uint key_base = head * d.time * d.key_width;
    uint value_base = head * d.time * d.value_width;
    float query[4] = {0, 0, 0, 0}, result[4] = {0, 0, 0, 0};
    for (uint f = lane; f < d.key_width; f += 32)
        if (row < d.time)
            query[f / 32] = float(q[key_base + row * d.key_width + f]);
    float maximum = -INFINITY, normalizer = 0;
    uint end = min(first + 4, d.time);
    for (uint start = 0; start < end; start += 32) {
        uint count = min(32u, end - start);
        for (uint i = thread_id; i < count * d.key_width; i += 128)
            keys[i] = float(k[key_base + start * d.key_width + i]);
        for (uint i = thread_id; i < count * d.value_width; i += 128)
            values[i] = float(v[value_base + start * d.value_width + i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < count; ++j) {
            if (row < d.time && start + j <= row) {
                float score = 0;
                for (uint f = lane; f < d.key_width; f += 32)
                    score += query[f / 32] * keys[j * d.key_width + f];
                score = simd_sum(score) * d.scale;
                float next_maximum = max(maximum, score);
                float rescale = exp(maximum - next_maximum);
                float probability = exp(score - next_maximum);
                normalizer = normalizer * rescale + probability;
                for (uint f = lane; f < d.value_width; f += 32)
                    result[f / 32] =
                        result[f / 32] * rescale + probability * values[j * d.value_width + f];
                maximum = next_maximum;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < d.time) {
        for (uint f = lane; f < d.value_width; f += 32)
            out[value_base + row * d.value_width + f] = result[f / 32] / normalizer;
        if (lane == 0)
            lse[head * d.time + row] = maximum + log(normalizer);
    }
}

template <typename T>
void flash_backward_q(device const T* q, device const T* k, device const T* v,
                      device const float* out, device const float* incoming,
                      device const float* lse, device float* delta, device float* dq,
                      constant AttentionDimensions& d, threadgroup float* keys,
                      threadgroup float* values, uint group, uint thread_id, uint simd_id,
                      uint lane) {
    uint blocks = (d.time + 3) / 4;
    uint head = group / blocks;
    uint first = (group % blocks) * 4;
    uint row = first + simd_id;
    uint key_base = head * d.time * d.key_width;
    uint value_base = head * d.time * d.value_width;
    float query[4] = {0, 0, 0, 0}, upstream[4] = {0, 0, 0, 0}, result[4] = {0, 0, 0, 0};
    float correction = 0, logsumexp = 0;
    if (row < d.time) {
        for (uint f = lane; f < d.key_width; f += 32)
            query[f / 32] = float(q[key_base + row * d.key_width + f]);
        for (uint f = lane; f < d.value_width; f += 32) {
            upstream[f / 32] = float(incoming[value_base + row * d.value_width + f]);
            correction += upstream[f / 32] * float(out[value_base + row * d.value_width + f]);
        }
        correction = simd_sum(correction);
        logsumexp = lse[head * d.time + row];
        if (lane == 0)
            delta[head * d.time + row] = correction;
    }
    uint end = min(first + 4, d.time);
    for (uint start = 0; start < end; start += 32) {
        uint count = min(32u, end - start);
        for (uint i = thread_id; i < count * d.key_width; i += 128)
            keys[i] = float(k[key_base + start * d.key_width + i]);
        for (uint i = thread_id; i < count * d.value_width; i += 128)
            values[i] = float(v[value_base + start * d.value_width + i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < count; ++j) {
            if (row < d.time && start + j <= row) {
                float score = 0, dp = 0;
                for (uint f = lane; f < d.key_width; f += 32)
                    score += query[f / 32] * keys[j * d.key_width + f];
                for (uint f = lane; f < d.value_width; f += 32)
                    dp += upstream[f / 32] * values[j * d.value_width + f];
                float probability = exp(simd_sum(score) * d.scale - logsumexp);
                float ds = probability * (simd_sum(dp) - correction) * d.scale;
                for (uint f = lane; f < d.key_width; f += 32)
                    result[f / 32] += ds * keys[j * d.key_width + f];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < d.time)
        for (uint f = lane; f < d.key_width; f += 32)
            dq[key_base + row * d.key_width + f] = result[f / 32];
}

template <typename T>
void flash_backward_kv(device const T* q, device const T* k, device const T* v,
                       device const float* incoming, device const float* lse,
                       device const float* delta, device float* dk, device float* dv,
                       constant AttentionDimensions& d, threadgroup float* queries,
                       threadgroup float* upstreams, uint group, uint thread_id, uint simd_id,
                       uint lane) {
    uint blocks = (d.time + 3) / 4;
    uint head = group / blocks;
    uint first = (group % blocks) * 4;
    uint row = first + simd_id;
    uint key_base = head * d.time * d.key_width;
    uint value_base = head * d.time * d.value_width;
    float key[4] = {0, 0, 0, 0}, value[4] = {0, 0, 0, 0};
    float key_gradient[4] = {0, 0, 0, 0}, value_gradient[4] = {0, 0, 0, 0};
    if (row < d.time) {
        for (uint f = lane; f < d.key_width; f += 32)
            key[f / 32] = float(k[key_base + row * d.key_width + f]);
        for (uint f = lane; f < d.value_width; f += 32)
            value[f / 32] = float(v[value_base + row * d.value_width + f]);
    }
    for (uint start = first; start < d.time; start += 32) {
        uint count = min(32u, d.time - start);
        for (uint i = thread_id; i < count * d.key_width; i += 128)
            queries[i] = float(q[key_base + start * d.key_width + i]);
        for (uint i = thread_id; i < count * d.value_width; i += 128)
            upstreams[i] = float(incoming[value_base + start * d.value_width + i]);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint j = 0; j < count; ++j) {
            if (row < d.time && start + j >= row) {
                float score = 0, dp = 0;
                for (uint f = lane; f < d.key_width; f += 32)
                    score += key[f / 32] * queries[j * d.key_width + f];
                for (uint f = lane; f < d.value_width; f += 32)
                    dp += value[f / 32] * upstreams[j * d.value_width + f];
                uint query_row = head * d.time + start + j;
                float probability = exp(simd_sum(score) * d.scale - lse[query_row]);
                float ds = probability * (simd_sum(dp) - delta[query_row]) * d.scale;
                for (uint f = lane; f < d.key_width; f += 32)
                    key_gradient[f / 32] += ds * queries[j * d.key_width + f];
                for (uint f = lane; f < d.value_width; f += 32)
                    value_gradient[f / 32] += probability * upstreams[j * d.value_width + f];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (row < d.time) {
        for (uint f = lane; f < d.key_width; f += 32)
            dk[key_base + row * d.key_width + f] = key_gradient[f / 32];
        for (uint f = lane; f < d.value_width; f += 32)
            dv[value_base + row * d.value_width + f] = value_gradient[f / 32];
    }
}

void tile_product(threadgroup float* result, threadgroup const float* a, threadgroup const float* b,
                  uint rows, uint columns, uint inner, uint a_stride, uint b_stride,
                  bool transpose_a, bool transpose_b, uint simd_id, bool accumulate) {
    uint column_tiles = columns / 8;
    for (uint tile = simd_id; tile < (rows / 8) * column_tiles; tile += 4) {
        uint row = (tile / column_tiles) * 8, column = (tile % column_tiles) * 8;
        simdgroup_float8x8 sum;
        if (accumulate)
            simdgroup_load(sum, result + row * columns + column, columns);
        else {
            sum.thread_elements()[0] = 0;
            sum.thread_elements()[1] = 0;
        }
        for (uint start = 0; start < inner; start += 8) {
            simdgroup_float8x8 left, right;
            auto a_tile = a + (transpose_a ? start * a_stride + row : row * a_stride + start);
            auto b_tile = b + (transpose_b ? column * b_stride + start : start * b_stride + column);
            simdgroup_load(left, a_tile, a_stride, ulong2(0), transpose_a);
            simdgroup_load(right, b_tile, b_stride, ulong2(0), transpose_b);
            simdgroup_multiply_accumulate(sum, left, right, sum);
        }
        simdgroup_store(sum, result + row * columns + column, columns);
    }
}

template <typename T>
void flash_forward_matrix(device const T* q, device const T* k, device const T* v,
                          device float* out, device float* lse, constant AttentionDimensions& d,
                          threadgroup float* scratch, threadgroup float*, uint group, uint tid,
                          uint simd_id, uint lane) {
    uint blocks = (d.time + 7) / 8, head = group / blocks, first = (group % blocks) * 8;
    uint kb = head * d.time * d.key_width, vb = head * d.time * d.value_width;
    auto queries = scratch;
    auto keys = queries + 8 * d.key_width;
    auto values = keys + 32 * d.key_width;
    auto result = values + 32 * d.value_width;
    auto probabilities = result + 8 * d.value_width;
    auto maximum = probabilities + 256;
    auto normalizer = maximum + 8;
    auto rescale = normalizer + 8;
    for (uint i = tid; i < 8 * d.key_width; i += 128)
        queries[i] = first + i / d.key_width < d.time ? float(q[kb + first * d.key_width + i]) : 0;
    for (uint i = tid; i < 8 * d.value_width; i += 128)
        result[i] = 0;
    if (tid < 8) {
        maximum[tid] = -INFINITY;
        normalizer[tid] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint start = 0; start < min(first + 8, d.time); start += 32) {
        for (uint i = tid; i < 32 * d.key_width; i += 128)
            keys[i] = start + i / d.key_width < d.time ? float(k[kb + start * d.key_width + i]) : 0;
        for (uint i = tid; i < 32 * d.value_width; i += 128)
            values[i] =
                start + i / d.value_width < d.time ? float(v[vb + start * d.value_width + i]) : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_product(probabilities, queries, keys, 8, 32, d.key_width, d.key_width, d.key_width,
                     false, true, simd_id, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint row = simd_id; row < 8; row += 4) {
            float probability = 0, factor = 0, next_maximum = 0, sum = 0;
            if (first + row < d.time) {
                float score = start + lane <= first + row ? probabilities[row * 32 + lane] * d.scale
                                                          : -INFINITY;
                next_maximum = max(maximum[row], simd_max(score));
                factor = exp(maximum[row] - next_maximum);
                probability = exp(score - next_maximum);
                sum = simd_sum(probability);
            }
            probabilities[row * 32 + lane] = probability;
            if (lane == 0) {
                maximum[row] = next_maximum;
                normalizer[row] = normalizer[row] * factor + sum;
                rescale[row] = factor;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 8 * d.value_width; i += 128)
            result[i] *= rescale[i / d.value_width];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_product(result, probabilities, values, 8, d.value_width, 32, 32, d.value_width, false,
                     false, simd_id, true);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = tid; i < 8 * d.value_width; i += 128)
        if (first + i / d.value_width < d.time)
            out[vb + first * d.value_width + i] = result[i] / normalizer[i / d.value_width];
    if (tid < 8 && first + tid < d.time)
        lse[head * d.time + first + tid] = maximum[tid] + log(normalizer[tid]);
}

template <typename T>
void flash_backward_q_matrix(device const T* q, device const T* k, device const T* v,
                             device const float* out, device const float* incoming,
                             device const float* lse, device float* delta, device float* dq,
                             constant AttentionDimensions& d, threadgroup float* scratch,
                             threadgroup float*, uint group, uint tid, uint simd_id, uint lane) {
    uint blocks = (d.time + 7) / 8, head = group / blocks, first = (group % blocks) * 8;
    uint kb = head * d.time * d.key_width, vb = head * d.time * d.value_width;
    auto queries = scratch;
    auto keys = queries + 8 * d.key_width;
    auto values = keys + 32 * d.key_width;
    auto upstreams = values + 32 * d.value_width;
    auto result = upstreams + 8 * d.value_width;
    auto scores = result + 8 * d.key_width;
    auto dp = scores + 256;
    auto ds = dp + 256;
    auto correction = ds + 256;
    auto logsumexp = correction + 8;
    for (uint i = tid; i < 8 * d.key_width; i += 128) {
        queries[i] = first + i / d.key_width < d.time ? float(q[kb + first * d.key_width + i]) : 0;
        result[i] = 0;
    }
    for (uint i = tid; i < 8 * d.value_width; i += 128)
        upstreams[i] =
            first + i / d.value_width < d.time ? incoming[vb + first * d.value_width + i] : 0;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint row = simd_id; row < 8; row += 4) {
        float sum = 0;
        if (first + row < d.time)
            for (uint f = lane; f < d.value_width; f += 32)
                sum += upstreams[row * d.value_width + f] *
                       out[vb + (first + row) * d.value_width + f];
        sum = simd_sum(sum);
        if (lane == 0) {
            correction[row] = sum;
            logsumexp[row] = first + row < d.time ? lse[head * d.time + first + row] : 0;
            if (first + row < d.time)
                delta[head * d.time + first + row] = sum;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint start = 0; start < min(first + 8, d.time); start += 32) {
        for (uint i = tid; i < 32 * d.key_width; i += 128)
            keys[i] = start + i / d.key_width < d.time ? float(k[kb + start * d.key_width + i]) : 0;
        for (uint i = tid; i < 32 * d.value_width; i += 128)
            values[i] =
                start + i / d.value_width < d.time ? float(v[vb + start * d.value_width + i]) : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_product(scores, queries, keys, 8, 32, d.key_width, d.key_width, d.key_width, false,
                     true, simd_id, false);
        tile_product(dp, upstreams, values, 8, 32, d.value_width, d.value_width, d.value_width,
                     false, true, simd_id, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 256; i += 128) {
            uint row = i / 32;
            float p = first + row < d.time && start + i % 32 <= first + row
                          ? exp(scores[i] * d.scale - logsumexp[row])
                          : 0;
            ds[i] = p * (dp[i] - correction[row]) * d.scale;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_product(result, ds, keys, 8, d.key_width, 32, 32, d.key_width, false, false, simd_id,
                     true);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = tid; i < 8 * d.key_width; i += 128)
        if (first + i / d.key_width < d.time)
            dq[kb + first * d.key_width + i] = result[i];
}

template <typename T>
void flash_backward_kv_matrix(device const T* q, device const T* k, device const T* v,
                              device const float* incoming, device const float* lse,
                              device const float* delta, device float* dk, device float* dv,
                              constant AttentionDimensions& d, threadgroup float* scratch,
                              threadgroup float*, uint group, uint tid, uint simd_id, uint lane) {
    uint blocks = (d.time + 7) / 8, head = group / blocks, first = (group % blocks) * 8;
    uint kb = head * d.time * d.key_width, vb = head * d.time * d.value_width;
    auto queries = scratch;
    auto upstreams = queries + 32 * d.key_width;
    auto keys = upstreams + 32 * d.value_width;
    auto values = keys + 8 * d.key_width;
    auto key_gradient = values + 8 * d.value_width;
    auto value_gradient = key_gradient + 8 * d.key_width;
    auto scores = value_gradient + 8 * d.value_width;
    auto dp = scores + 256;
    auto ds = dp + 256;
    auto probabilities = ds + 256;
    for (uint i = tid; i < 8 * d.key_width; i += 128) {
        keys[i] = first + i / d.key_width < d.time ? float(k[kb + first * d.key_width + i]) : 0;
        key_gradient[i] = 0;
    }
    for (uint i = tid; i < 8 * d.value_width; i += 128) {
        values[i] =
            first + i / d.value_width < d.time ? float(v[vb + first * d.value_width + i]) : 0;
        value_gradient[i] = 0;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint start = first; start < d.time; start += 32) {
        for (uint i = tid; i < 32 * d.key_width; i += 128)
            queries[i] =
                start + i / d.key_width < d.time ? float(q[kb + start * d.key_width + i]) : 0;
        for (uint i = tid; i < 32 * d.value_width; i += 128)
            upstreams[i] =
                start + i / d.value_width < d.time ? incoming[vb + start * d.value_width + i] : 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_product(scores, queries, keys, 32, 8, d.key_width, d.key_width, d.key_width, false,
                     true, simd_id, false);
        tile_product(dp, upstreams, values, 32, 8, d.value_width, d.value_width, d.value_width,
                     false, true, simd_id, false);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint i = tid; i < 256; i += 128) {
            uint query_row = start + i / 8, key_row = first + i % 8;
            float p = query_row < d.time && key_row < d.time && query_row >= key_row
                          ? exp(scores[i] * d.scale - lse[head * d.time + query_row])
                          : 0;
            probabilities[i] = p;
            ds[i] =
                query_row < d.time ? p * (dp[i] - delta[head * d.time + query_row]) * d.scale : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        tile_product(key_gradient, ds, queries, 8, d.key_width, 32, 8, d.key_width, true, false,
                     simd_id, true);
        tile_product(value_gradient, probabilities, upstreams, 8, d.value_width, 32, 8,
                     d.value_width, true, false, simd_id, true);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = tid; i < 8 * d.key_width; i += 128)
        if (first + i / d.key_width < d.time)
            dk[kb + first * d.key_width + i] = key_gradient[i];
    for (uint i = tid; i < 8 * d.value_width; i += 128)
        if (first + i / d.value_width < d.time)
            dv[vb + first * d.value_width + i] = value_gradient[i];
}

#define FLASH_FORWARD(NAME, TYPE, FUNCTION)                                                        \
    kernel void NAME(                                                                              \
        device const TYPE* q [[buffer(0)]], device const TYPE* k [[buffer(1)]],                    \
        device const TYPE* v [[buffer(2)]], device float* out [[buffer(3)]],                       \
        device float* lse [[buffer(4)]], constant AttentionDimensions& d [[buffer(5)]],            \
        threadgroup float* keys [[threadgroup(0)]], threadgroup float* values [[threadgroup(1)]],  \
        uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],     \
        uint simd_id [[simdgroup_index_in_threadgroup]],                                           \
        uint lane [[thread_index_in_simdgroup]]) {                                                 \
        FUNCTION(q, k, v, out, lse, d, keys, values, group, tid, simd_id, lane);                   \
    }
#define FLASH_BACKWARD_Q(NAME, TYPE, FUNCTION)                                                     \
    kernel void NAME(                                                                              \
        device const TYPE* q [[buffer(0)]], device const TYPE* k [[buffer(1)]],                    \
        device const TYPE* v [[buffer(2)]], device const float* out [[buffer(3)]],                 \
        device const float* incoming [[buffer(4)]], device const float* lse [[buffer(5)]],         \
        device float* delta [[buffer(6)]], device float* dq [[buffer(7)]],                         \
        constant AttentionDimensions& d [[buffer(8)]], threadgroup float* keys [[threadgroup(0)]], \
        threadgroup float* values [[threadgroup(1)]], uint group [[threadgroup_position_in_grid]], \
        uint tid [[thread_index_in_threadgroup]], uint simd_id [[simdgroup_index_in_threadgroup]], \
        uint lane [[thread_index_in_simdgroup]]) {                                                 \
        FUNCTION(q, k, v, out, incoming, lse, delta, dq, d, keys, values, group, tid, simd_id,     \
                 lane);                                                                            \
    }
#define FLASH_BACKWARD_KV(NAME, TYPE, FUNCTION)                                                    \
    kernel void NAME(                                                                              \
        device const TYPE* q [[buffer(0)]], device const TYPE* k [[buffer(1)]],                    \
        device const TYPE* v [[buffer(2)]], device const float* incoming [[buffer(3)]],            \
        device const float* lse [[buffer(4)]], device const float* delta [[buffer(5)]],            \
        device float* dk [[buffer(6)]], device float* dv [[buffer(7)]],                            \
        constant AttentionDimensions& d [[buffer(8)]],                                             \
        threadgroup float* queries [[threadgroup(0)]],                                             \
        threadgroup float* upstreams [[threadgroup(1)]],                                           \
        uint group [[threadgroup_position_in_grid]], uint tid [[thread_index_in_threadgroup]],     \
        uint simd_id [[simdgroup_index_in_threadgroup]],                                           \
        uint lane [[thread_index_in_simdgroup]]) {                                                 \
        FUNCTION(q, k, v, incoming, lse, delta, dk, dv, d, queries, upstreams, group, tid,         \
                 simd_id, lane);                                                                   \
    }
FLASH_FORWARD(flash_forward_f32, float, flash_forward)
FLASH_FORWARD(flash_forward_f16, half, flash_forward)
FLASH_BACKWARD_Q(flash_backward_q_f32, float, flash_backward_q)
FLASH_BACKWARD_Q(flash_backward_q_f16, half, flash_backward_q)
FLASH_BACKWARD_KV(flash_backward_kv_f32, float, flash_backward_kv)
FLASH_BACKWARD_KV(flash_backward_kv_f16, half, flash_backward_kv)

FLASH_FORWARD(flash_forward_matrix_f32, float, flash_forward_matrix)
FLASH_FORWARD(flash_forward_matrix_f16, half, flash_forward_matrix)
FLASH_BACKWARD_Q(flash_backward_q_matrix_f32, float, flash_backward_q_matrix)
FLASH_BACKWARD_Q(flash_backward_q_matrix_f16, half, flash_backward_q_matrix)
FLASH_BACKWARD_KV(flash_backward_kv_matrix_f32, float, flash_backward_kv_matrix)
FLASH_BACKWARD_KV(flash_backward_kv_matrix_f16, half, flash_backward_kv_matrix)
