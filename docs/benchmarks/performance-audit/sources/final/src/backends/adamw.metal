#include <metal_stdlib>
using namespace metal;

struct Parameter {
    device const float *value, *gradient, *first, *second;
    device float *next_value, *next_first, *next_second;
    uint count, first_block;
    float decay;
    uint padding;
};
struct Constants {
    uint count;
    float lr, beta1, beta2, epsilon, clip, bc1, bc2;
};
float group_sum(float value, threadgroup float* scratch, uint lane, uint group) {
    value = simd_sum(value);
    if (lane == 0) scratch[group] = value;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return simd_sum(lane < 8 ? scratch[lane] : 0.0f);
}
kernel void adamw_partial_norm(device const Parameter* parameters [[buffer(0)]],
                               device float* partials [[buffer(1)]],
                               constant Constants& c [[buffer(2)]],
                               device const uint2* tiles [[buffer(3)]],
                               uint block [[threadgroup_position_in_grid]],
                               uint tid [[thread_index_in_threadgroup]],
                               uint lane [[thread_index_in_simdgroup]],
                               uint group [[simdgroup_index_in_threadgroup]]) {
    const device Parameter& p = parameters[tiles[block].x];
    uint begin = tiles[block].y + tid;
    float value = 0;
    for (uint i = begin; i < min(begin + 1024, p.count); i += 256) {
        float g = p.gradient[i];
        value += g * g;
    }
    threadgroup float scratch[8];
    value = group_sum(value, scratch, lane, group);
    if (tid == 0) partials[block] = value;
}
kernel void adamw_global_norm(device const float* partials [[buffer(0)]],
                              device float* norm [[buffer(1)]],
                              constant uint& count [[buffer(2)]],
                              uint tid [[thread_index_in_threadgroup]],
                              uint lane [[thread_index_in_simdgroup]],
                              uint group [[simdgroup_index_in_threadgroup]]) {
    float value = 0;
    for (uint i = tid; i < count; i += 256) value += partials[i];
    threadgroup float scratch[8];
    value = group_sum(value, scratch, lane, group);
    if (tid == 0) norm[0] = sqrt(value);
}
kernel void adamw_update(device const Parameter* parameters [[buffer(0)]],
                         device const float* norm [[buffer(1)]],
                         constant Constants& c [[buffer(2)]],
                         device const uint2* tiles [[buffer(3)]],
                         uint block [[threadgroup_position_in_grid]],
                         uint tid [[thread_index_in_threadgroup]]) {
    const device Parameter& p = parameters[tiles[block].x];
    uint begin = tiles[block].y + tid;
    float scale = c.clip > 0 ? min(1.0f, c.clip / max(norm[0], 1e-30f)) : 1.0f;
    for (uint i = begin; i < min(begin + 1024, p.count); i += 256) {
        float g = p.gradient[i] * scale;
        float m = c.beta1 * p.first[i] + (1.0f - c.beta1) * g;
        float v = c.beta2 * p.second[i] + (1.0f - c.beta2) * (g * g);
        float update = (m / c.bc1) / (sqrt(v / c.bc2) + c.epsilon);
        p.next_value[i] = p.value[i] * (1.0f - c.lr * p.decay) - c.lr * update;
        p.next_first[i] = m;
        p.next_second[i] = v;
    }
}
