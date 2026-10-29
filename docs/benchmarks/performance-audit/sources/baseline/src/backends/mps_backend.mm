#include "adamw_source.hpp"
#include "mini_mla/backend.hpp"
#include "mini_mla/execution.hpp"
#include "mini_mla/io.hpp"
#include "mps_runtime.hpp"

#import <Accelerate/Accelerate.h>
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <stdexcept>
#include <tuple>

namespace mini_mla {
#if MINI_MLA_ENABLE_MPS
namespace {
using namespace mps;
struct ParameterBuffers {
    DeviceArray value, gradient, first, second, next_value, next_gradient, next_first, next_second;
};
class MetalAdamW {
  public:
    struct Entry {
        std::uint64_t value, gradient, first, second, next_value, next_first, next_second;
        std::uint32_t count, first_block;
        float decay;
        std::uint32_t padding = 0;
    };
    static_assert(sizeof(Entry) == 72);
    struct Constants {
        std::uint32_t count;
        float lr, beta1, beta2, epsilon, clip, bc1, bc2;
    };
    MetalAdamW(id<MTLDevice> device, const Model& model) {
        if (model.parameters().size() > UINT32_MAX)
            throw std::invalid_argument("Too many optimizer parameters");
        for (const auto& p : model.parameters()) {
            if (p.value.size() > UINT32_MAX - 1023)
                throw std::invalid_argument("Optimizer parameter exceeds Metal indexing limit");
            auto blocks = (p.value.size() + 1023) / 1024;
            if (blocks > UINT32_MAX - blocks_)
                throw std::invalid_argument("Optimizer workspace exceeds Metal indexing limit");
            blocks_ += static_cast<std::uint32_t>(blocks);
        }
        if (blocks_ > INT_MAX)
            throw std::invalid_argument("AdamW tile map exceeds tensor indexing limit");
        table_ = [device newBufferWithLength:model.parameters().size() * sizeof(Entry)
                                     options:MTLResourceStorageModeShared];
        tiles_ = [device newBufferWithLength:static_cast<std::size_t>(blocks_) * 8
                                     options:MTLResourceStorageModeShared];
        if (!tiles_)
            throw std::runtime_error("Cannot allocate AdamW tile map");
        auto* tiles = static_cast<std::uint32_t*>(tiles_.contents);
        std::size_t cursor = 0;
        for (std::size_t i = 0; i < model.parameters().size(); ++i)
            for (std::size_t offset = 0; offset < model.parameters()[i].value.size();
                 offset += 1024) {
                tiles[cursor++] = static_cast<std::uint32_t>(i);
                tiles[cursor++] = static_cast<std::uint32_t>(offset);
            }
        partials_ = allocate(device, shape({static_cast<int>(blocks_)}), MPSDataTypeFloat32);
        norm_ = allocate(device, shape({1}), MPSDataTypeFloat32);
        NSError* error = nil;
        auto options = [MTLCompileOptions new];
        if (@available(macOS 15.0, *))
            options.mathMode = MTLMathModeSafe;
        auto library = [device newLibraryWithSource:ns(metal::adamw_source)
                                            options:options
                                              error:&error];
        if (!table_ || !library)
            throw std::runtime_error("Cannot build Metal AdamW: " +
                                     cpp(error.localizedDescription));
        auto pipeline = [&](NSString* name) {
            auto function = [library newFunctionWithName:name];
            auto result = [device newComputePipelineStateWithFunction:function error:&error];
            if (!result || result.threadExecutionWidth != 32 ||
                result.maxTotalThreadsPerThreadgroup < 256)
                throw std::runtime_error("Unsupported AdamW pipeline: " +
                                         cpp(error.localizedDescription));
            return result;
        };
        partial_norm_ = pipeline(@"adamw_partial_norm");
        global_norm_ = pipeline(@"adamw_global_norm");
        update_ = pipeline(@"adamw_update");
    }
    DeviceArray encode(const Model& model, const Optimizer& optimizer,
                       const std::vector<ParameterBuffers>& buffers, float lr, float bc1, float bc2,
                       CommandBatch& commands, ExecutionStats& stats) {
        const auto& config = optimizer.config();
        Constants c{static_cast<std::uint32_t>(buffers.size()),
                    lr,
                    config.beta1,
                    config.beta2,
                    config.epsilon,
                    config.clip_norm,
                    bc1,
                    bc2};
        auto entries = static_cast<Entry*>(table_.contents);
        auto encoder = [commands.command computeCommandEncoder];
        if (!encoder)
            throw std::runtime_error("Cannot create AdamW compute encoder");
        std::uint32_t block = 0;
        for (std::size_t i = 0; i < buffers.size(); ++i) {
            const auto& b = buffers[i];
            const auto& p = model.parameters()[i];
            auto count = static_cast<std::uint32_t>(p.value.size());
            entries[i] = {b.value.buffer.gpuAddress,
                          b.gradient.buffer.gpuAddress,
                          b.first.buffer.gpuAddress,
                          b.second.buffer.gpuAddress,
                          b.next_value.buffer.gpuAddress,
                          b.next_first.buffer.gpuAddress,
                          b.next_second.buffer.gpuAddress,
                          count,
                          block,
                          p.role == ParameterRole::norm_scale ? 0 : config.weight_decay,
                          0};
            block += (count + 1023) / 1024;
            for (auto a : {b.value, b.gradient, b.first, b.second}) {
                [encoder useResource:a.buffer usage:MTLResourceUsageRead];
                commands.keep(a.buffer);
            }
            for (auto a : {b.next_value, b.next_first, b.next_second}) {
                [encoder useResource:a.buffer usage:MTLResourceUsageWrite];
                commands.keep(a.buffer);
            }
        }
        for (id object : std::array<id, 7>{table_, tiles_, partials_.buffer, norm_.buffer,
                                           partial_norm_, global_norm_, update_})
            commands.keep(object);
        [encoder setComputePipelineState:partial_norm_];
        [encoder setBuffer:table_ offset:0 atIndex:0];
        [encoder setBuffer:partials_.buffer offset:0 atIndex:1];
        [encoder setBytes:&c length:sizeof(c) atIndex:2];
        [encoder setBuffer:tiles_ offset:0 atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(blocks_, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [encoder setComputePipelineState:global_norm_];
        [encoder setBuffer:partials_.buffer offset:0 atIndex:0];
        [encoder setBuffer:norm_.buffer offset:0 atIndex:1];
        [encoder setBytes:&blocks_ length:sizeof(blocks_) atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(1, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder memoryBarrierWithScope:MTLBarrierScopeBuffers];
        [encoder setComputePipelineState:update_];
        [encoder setBuffer:table_ offset:0 atIndex:0];
        [encoder setBuffer:norm_.buffer offset:0 atIndex:1];
        [encoder setBytes:&c length:sizeof(c) atIndex:2];
        [encoder dispatchThreadgroups:MTLSizeMake(blocks_, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        stats.metal_kernel_submissions += 3;
        return norm_;
    }

  private:
    std::uint32_t blocks_ = 0;
    id<MTLBuffer> table_;
    id<MTLBuffer> tiles_;
    DeviceArray partials_, norm_;
    id<MTLComputePipelineState> partial_norm_, global_norm_, update_;
};
struct Staging {
    DeviceArray ids, targets, coefficients, loss;
};
enum class FlashStage {
    embedding,
    projections,
    post_attention,
    head,
    log_probs,
    head_backward,
    post_backward,
    projections_backward,
    embedding_backward,
    prefill_projections,
    prefill_head
};
struct FlashLayer {
    DeviceArray q, k, v, out, lse;
};
struct FlashWorkspace {
    int batch = 0, time = 0;
    std::vector<DeviceArray> hidden;
    std::vector<FlashLayer> layers;
    DeviceArray dx, residual_dx, attention_dx, dq, dk, dv, delta;
};
struct Program {
    std::uint64_t last_used = 0;
    MPSGraph* graph = [MPSGraph new];
    MPSGraphExecutable* executable = nil;
    std::vector<Tensor> weights, gradient_inputs, gradients, first_inputs, second_inputs,
        updated_values, updated_first, updated_second;
    std::vector<Tensor> cache_inputs, rope_inputs, cache_outputs, rope_outputs, projection_inputs,
        projection_outputs;
    Tensor ids = nil, targets = nil, coefficients = nil, position = nil, logits = nil,
           selected = nil, loss = nil, lr = nil, bc1 = nil, bc2 = nil, norm = nil;
    std::map<Tensor, DeviceArray> inputs, outputs;
    std::vector<Tensor> activation_inputs, activation_outputs;
    std::vector<std::size_t> parameter_indices;
    std::vector<Staging> staging;
};
struct ProgramPool {
    using Key = std::tuple<int, int, int>;
    struct Entry {
        std::shared_ptr<const Program> program;
        std::uint64_t last_used;
    };
    std::mutex mutex;
    std::uint64_t clock = 0;
    std::map<Key, Entry> programs;
};
class GraphBuilder {
  public:
    GraphBuilder(Program& p, const ModelConfig& c, bool half)
        : p_(p), c_(c), g_(p.graph), dtype_(half ? MPSDataTypeFloat16 : MPSDataTypeFloat32) {}
    Tensor constant(double v) { return [g_ constantWithScalar:v dataType:MPSDataTypeFloat32]; }
    Tensor add(Tensor a, Tensor b) {
        return [g_ additionWithPrimaryTensor:a secondaryTensor:b name:nil];
    }
    Tensor sub(Tensor a, Tensor b) {
        return [g_ subtractionWithPrimaryTensor:a secondaryTensor:b name:nil];
    }
    Tensor mul(Tensor a, Tensor b) {
        return [g_ multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
    }
    Tensor div(Tensor a, Tensor b) {
        return [g_ divisionWithPrimaryTensor:a secondaryTensor:b name:nil];
    }
    Tensor mm(Tensor a, Tensor b) {
        return [g_ matrixMultiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
    }
    Tensor cast(Tensor x, MPSDataType dtype) {
        return x.dataType == dtype ? x : [g_ castTensor:x toType:dtype name:nil];
    }
    Tensor reshape(Tensor x, MPSShape* s) { return [g_ reshapeTensor:x withShape:s name:nil]; }
    Tensor transpose(Tensor x, NSUInteger a, NSUInteger b) {
        return [g_ transposeTensor:x dimension:a withDimension:b name:nil];
    }
    Tensor slice(Tensor x, NSUInteger axis, int start, int count) {
        return [g_ sliceTensor:x dimension:axis start:start length:count name:nil];
    }
    Tensor sum(Tensor x) {
        return reshape([g_ reductionSumWithTensor:x axes:nil name:nil], shape({1}));
    }
    Tensor concat(Tensor a, Tensor b, NSUInteger axis) {
        return [g_ concatTensors:@[ a, b ] dimension:static_cast<NSInteger>(axis) name:nil];
    }
    Tensor softmax(Tensor x, NSInteger axis) { return [g_ softMaxWithTensor:x axis:axis name:nil]; }
    Tensor norm(Tensor x, Tensor weight, int width) {
        auto f = cast(x, MPSDataTypeFloat32);
        auto sq = [g_ reductionSumWithTensor:mul(f, f)
                                        axis:static_cast<NSInteger>(f.shape.count - 1)
                                        name:nil];
        auto inv = [g_
            reciprocalSquareRootWithTensor:add(div(sq, constant(width)), constant(c_.norm_epsilon))
                                      name:nil];
        return cast(mul(mul(f, inv), cast(weight, MPSDataTypeFloat32)), dtype_);
    }
    Tensor rope(Tensor x, int batch, int time, int heads, Tensor position = nil) {
        int width = c_.head_dim_rope, table_time = position ? c_.max_sequence_length : time;
        std::vector<float> cosine(static_cast<std::size_t>(table_time * width / 2)),
            sine(cosine.size());
        for (int t = 0; t < table_time; ++t)
            for (int j = 0; j < width; j += 2) {
                float angle =
                    static_cast<float>(t) *
                    std::pow(c_.rope_theta, -static_cast<float>(j) / static_cast<float>(width));
                auto index = static_cast<std::size_t>((t * width + j) / 2);
                cosine[index] = std::cos(angle);
                sine[index] = std::sin(angle);
            }
        auto table = [&](const std::vector<float>& values) {
            auto tensor = [g_ constantWithData:[NSData dataWithBytes:values.data()
                                                              length:values.size() * 4]
                                         shape:shape({table_time, width / 2})
                                      dataType:MPSDataTypeFloat32];
            if (position)
                tensor = [g_ gatherWithUpdatesTensor:tensor
                                       indicesTensor:position
                                                axis:0
                                     batchDimensions:0
                                                name:nil];
            return reshape(cast(tensor, dtype_), shape({1, time, 1, width / 2, 1}));
        };
        auto pairs = reshape(x, shape({batch, time, heads, width / 2, 2}));
        auto a = slice(pairs, 4, 0, 1), b = slice(pairs, 4, 1, 1);
        auto cs = table(cosine), sn = table(sine);
        return reshape(concat(sub(mul(a, cs), mul(b, sn)), add(mul(a, sn), mul(b, cs)), 4),
                       shape({batch, time, heads, width}));
    }
    void weights(const Model& m, bool half_storage = false) {
        for (const auto& p : m.parameters())
            p_.weights.push_back([g_
                placeholderWithShape:shape({p.value.rows, p.value.cols})
                            dataType:half_storage ? MPSDataTypeFloat16 : MPSDataTypeFloat32
                                name:ns(p.name)]);
    }
    Tensor weight(int layer, LayerParameter index) {
        return cast(p_.weights[2 + static_cast<std::size_t>(layer) * layer_parameter_count + index],
                    dtype_);
    }
    std::array<Tensor, 5> projections(Tensor x, int layer, int batch, int time) {
        auto h = norm(x, weight(layer, attn_norm), c_.d_model);
        auto cq = norm(mm(h, weight(layer, w_dq)), weight(layer, q_norm), c_.q_lora_rank);
        auto q =
            reshape(mm(cq, weight(layer, w_uq)), shape({batch, time, c_.n_heads, c_.qk_dim()}));
        q = concat(slice(q, 3, 0, c_.head_dim_nope),
                   rope(slice(q, 3, c_.head_dim_nope, c_.head_dim_rope), batch, time, c_.n_heads),
                   3);
        auto kv = mm(h, weight(layer, w_dkv));
        auto latent =
            norm(slice(kv, 2, 0, c_.kv_lora_rank), weight(layer, kv_norm), c_.kv_lora_rank);
        auto rotary = rope(reshape(slice(kv, 2, c_.kv_lora_rank, c_.head_dim_rope),
                                   shape({batch, time, 1, c_.head_dim_rope})),
                           batch, time, 1);
        auto keys = reshape(mm(latent, weight(layer, w_uk)),
                            shape({batch, time, c_.n_heads, c_.head_dim_nope}));
        auto shared = mul(rotary, [g_ constantWithScalar:1
                                                   shape:shape({1, 1, c_.n_heads, 1})
                                                dataType:dtype_]);
        keys = concat(keys, shared, 3);
        auto values = reshape(mm(latent, weight(layer, w_uv)),
                              shape({batch, time, c_.n_heads, c_.head_dim_value}));
        return {transpose(q, 1, 2), transpose(keys, 1, 2), transpose(values, 1, 2), latent, rotary};
    }
    Tensor projected_output(Tensor x, Tensor attention, int layer, int batch, int time) {
        auto out = reshape(transpose(cast(attention, dtype_), 1, 2),
                           shape({batch, time, c_.attn_width()}));
        auto residual = add(x, mm(out, weight(layer, w_o)));
        auto h = norm(residual, weight(layer, ffn_norm), c_.d_model);
        auto gate = mm(h, weight(layer, w_gate));
        auto gated =
            mul(mul(gate, [g_ sigmoidWithTensor:gate name:nil]), mm(h, weight(layer, w_up)));
        return add(residual, mm(gated, weight(layer, w_down)));
    }
    Tensor embedding(int batch, int time) {
        p_.ids = [g_ placeholderWithShape:shape({batch, time})
                                 dataType:MPSDataTypeInt32
                                     name:@"ids"];
        return [g_ gatherWithUpdatesTensor:cast(p_.weights[0], dtype_)
                             indicesTensor:p_.ids
                                      axis:0
                           batchDimensions:0
                                      name:nil];
    }
    Tensor head(Tensor x) {
        auto normalized = norm(x, cast(p_.weights[1], dtype_), c_.d_model);
        return cast(mm(normalized, transpose(cast(p_.weights[0], dtype_), 0, 1)),
                    MPSDataTypeFloat32);
    }
    Tensor combined_projection(int layer) {
        auto uv =
            [g_ transposeTensor:reshape(weight(layer, w_uv),
                                        shape({c_.kv_lora_rank, c_.n_heads, c_.head_dim_value}))
                    permutation:@[ @1, @0, @2 ]
                           name:nil];
        return mm(uv,
                  reshape(weight(layer, w_o), shape({c_.n_heads, c_.head_dim_value, c_.d_model})));
    }
    Tensor forward(int batch, int time, bool prefill) {
        p_.ids = [g_ placeholderWithShape:shape({batch, time})
                                 dataType:MPSDataTypeInt32
                                     name:@"ids"];
        auto embedding = cast(p_.weights[0], dtype_);
        Tensor x = [g_ gatherWithUpdatesTensor:embedding
                                 indicesTensor:p_.ids
                                          axis:0
                               batchDimensions:0
                                          name:nil];
        std::vector<float> mask(static_cast<std::size_t>(time) * static_cast<std::size_t>(time));
        for (int i = 0; i < time; ++i)
            for (int j = 0; j < time; ++j)
                mask[static_cast<std::size_t>(i * time + j)] = j <= i ? 0 : -1e30F;
        auto causal = [g_ constantWithData:[NSData dataWithBytes:mask.data() length:mask.size() * 4]
                                     shape:shape({1, 1, time, time})
                                  dataType:MPSDataTypeFloat32];
        for (int l = 0; l < c_.n_layers; ++l) {
            auto [q, k, v, ckv, kr] = projections(x, l, batch, time);
            auto scores = add(
                mul(mm(cast(q, MPSDataTypeFloat32), transpose(cast(k, MPSDataTypeFloat32), 2, 3)),
                    constant(1 / std::sqrt(static_cast<double>(c_.qk_dim())))),
                causal);
            auto probabilities = softmax(scores, 3);
            auto av = reshape(
                mm(reshape(cast(probabilities, dtype_), shape({batch * c_.n_heads, time, time})),
                   reshape(v, shape({batch * c_.n_heads, time, c_.head_dim_value}))),
                shape({batch, c_.n_heads, time, c_.head_dim_value}));
            x = projected_output(x, av, l, batch, time);
            if (prefill) {
                p_.cache_outputs.push_back(reshape(ckv, shape({time, c_.kv_lora_rank})));
                p_.rope_outputs.push_back(reshape(kr, shape({time, c_.head_dim_rope})));
                p_.projection_outputs.push_back(combined_projection(l));
            }
        }
        auto normalized = norm(x, cast(p_.weights[1], dtype_), c_.d_model);
        auto logits = cast(mm(normalized, transpose(embedding, 0, 1)), MPSDataTypeFloat32);
        if (prefill) {
            p_.position = [g_ placeholderWithShape:shape({1})
                                          dataType:MPSDataTypeInt32
                                              name:@"last_prompt_position"];
            logits = [g_ gatherWithUpdatesTensor:logits
                                   indicesTensor:p_.position
                                            axis:1
                                 batchDimensions:0
                                            name:nil];
            return reshape(logits, shape({1, c_.vocab_size}));
        }
        return reshape(logits, shape({batch * time, c_.vocab_size}));
    }
    Tensor selected(Tensor logits, int tokens) {
        p_.targets = [g_ placeholderWithShape:shape({tokens, 1})
                                     dataType:MPSDataTypeInt32
                                         name:@"targets"];
        auto labels = [g_ oneHotWithIndicesTensor:reshape(p_.targets, shape({tokens}))
                                            depth:static_cast<NSUInteger>(c_.vocab_size)
                                             axis:1
                                         dataType:MPSDataTypeFloat32
                                          onValue:1
                                         offValue:0
                                             name:nil];
        auto nll = [g_ softMaxCrossEntropyWithSourceTensor:logits
                                              labelsTensor:labels
                                                      axis:1
                                             reductionType:MPSGraphLossReductionTypeNone
                                                      name:nil];
        return reshape([g_ negativeWithTensor:nll name:nil], shape({tokens, 1}));
    }
    Tensor decode(int capacity) {
        p_.ids = [g_ placeholderWithShape:shape({1, 1}) dataType:MPSDataTypeInt32 name:@"token"];
        p_.position = [g_ placeholderWithShape:shape({1})
                                      dataType:MPSDataTypeInt32
                                          name:@"position"];
        auto embedding = cast(p_.weights[0], dtype_);
        auto x = [g_ gatherWithUpdatesTensor:embedding
                               indicesTensor:p_.ids
                                        axis:0
                             batchDimensions:0
                                        name:nil];
        std::vector<std::int32_t> positions(static_cast<std::size_t>(capacity));
        for (int i = 0; i < capacity; ++i)
            positions[static_cast<std::size_t>(i)] = i;
        auto pos = [g_ constantWithData:[NSData dataWithBytes:positions.data()
                                                       length:positions.size() * 4]
                                  shape:shape({1, 1, capacity})
                               dataType:MPSDataTypeInt32];
        auto valid = [g_ lessThanOrEqualToWithPrimaryTensor:pos
                                            secondaryTensor:p_.position
                                                       name:nil];
        for (int l = 0; l < c_.n_layers; ++l) {
            auto h = norm(x, weight(l, attn_norm), c_.d_model);
            auto cq = norm(mm(h, weight(l, w_dq)), weight(l, q_norm), c_.q_lora_rank);
            auto q = reshape(mm(cq, weight(l, w_uq)), shape({1, 1, c_.n_heads, c_.qk_dim()}));
            auto qn =
                reshape(slice(q, 3, 0, c_.head_dim_nope), shape({c_.n_heads, 1, c_.head_dim_nope}));
            auto qr = reshape(rope(slice(q, 3, c_.head_dim_nope, c_.head_dim_rope), 1, 1,
                                   c_.n_heads, p_.position),
                              shape({c_.n_heads, 1, c_.head_dim_rope}));
            auto kv = mm(h, weight(l, w_dkv));
            auto ckv =
                reshape(norm(slice(kv, 2, 0, c_.kv_lora_rank), weight(l, kv_norm), c_.kv_lora_rank),
                        shape({1, c_.kv_lora_rank}));
            auto kr = reshape(rope(reshape(slice(kv, 2, c_.kv_lora_rank, c_.head_dim_rope),
                                           shape({1, 1, 1, c_.head_dim_rope})),
                                   1, 1, 1, p_.position),
                              shape({1, c_.head_dim_rope}));
            auto cache = [g_ placeholderWithShape:shape({capacity, c_.kv_lora_rank})
                                         dataType:dtype_
                                             name:nil];
            auto rcache = [g_ placeholderWithShape:shape({capacity, c_.head_dim_rope})
                                          dataType:dtype_
                                              name:nil];
            auto combined =
                [g_ placeholderWithShape:shape({c_.n_heads, c_.kv_lora_rank, c_.d_model})
                                dataType:dtype_
                                    name:nil];
            p_.cache_inputs.push_back(cache);
            p_.rope_inputs.push_back(rcache);
            p_.projection_inputs.push_back(combined);
            cache = [g_ scatterWithDataTensor:cache
                                updatesTensor:ckv
                                indicesTensor:p_.position
                                         axis:0
                                         mode:MPSGraphScatterModeSet
                                         name:nil];
            rcache = [g_ scatterWithDataTensor:rcache
                                 updatesTensor:kr
                                 indicesTensor:p_.position
                                          axis:0
                                          mode:MPSGraphScatterModeSet
                                          name:nil];
            p_.cache_outputs.push_back(cache);
            p_.rope_outputs.push_back(rcache);
            auto uk =
                [g_ transposeTensor:reshape(weight(l, w_uk),
                                            shape({c_.kv_lora_rank, c_.n_heads, c_.head_dim_nope}))
                        permutation:@[ @1, @2, @0 ]
                               name:nil];
            auto qlatent = mm(qn, uk);
            auto scores = add(mm(cast(qlatent, MPSDataTypeFloat32),
                                 transpose(cast(cache, MPSDataTypeFloat32), 0, 1)),
                              mm(cast(qr, MPSDataTypeFloat32),
                                 transpose(cast(rcache, MPSDataTypeFloat32), 0, 1)));
            scores = mul(scores, constant(1 / std::sqrt(static_cast<double>(c_.qk_dim()))));
            scores = [g_ selectWithPredicateTensor:valid
                               truePredicateTensor:scores
                              falsePredicateTensor:constant(-1e30)
                                              name:nil];
            auto prob = cast(softmax(scores, 2), dtype_);
            auto context = mm(prob, cache);
            auto head_output = mm(context, combined);
            auto out = [g_ reductionSumWithTensor:head_output axis:0 name:nil];
            x = add(x, reshape(out, shape({1, 1, c_.d_model})));
            auto h2 = norm(x, weight(l, ffn_norm), c_.d_model);
            auto gate = mm(h2, weight(l, w_gate));
            x = add(
                x, mm(mul(mul(gate, [g_ sigmoidWithTensor:gate name:nil]), mm(h2, weight(l, w_up))),
                      weight(l, w_down)));
        }
        return reshape(
            cast(mm(norm(x, cast(p_.weights[1], dtype_), c_.d_model), transpose(embedding, 0, 1)),
                 MPSDataTypeFloat32),
            shape({1, c_.vocab_size}));
    }

  private:
    Program& p_;
    const ModelConfig& c_;
    MPSGraph* g_;
    MPSDataType dtype_;
};
class MpsExecution final : public Execution {
  public:
    MpsExecution(Model& model, Optimizer* optimizer, bool half, const ExecutionOptions& options,
                 const MpsExecution* source = nullptr)
        : model_(model), optimizer_(optimizer), half_(half), options_(options),
          device_(source ? source->device_ : MTLCreateSystemDefaultDevice()),
          program_pool_(source ? source->program_pool_ : std::make_shared<ProgramPool>()) {
        if (!device_)
            throw std::runtime_error("No Metal device is available to this process");
        uploaded_revision_ = model.revision();
        queue_ = [device_ newCommandQueue];
        graph_device_ = [MPSGraphDevice deviceWithMTLDevice:device_];
        if (!queue_ || !graph_device_)
            throw std::runtime_error("Cannot create Metal command queue");
        flash_supported_ = [device_ supportsFamily:MTLGPUFamilyApple7] &&
                           model.config().qk_dim() <= 128 && model.config().head_dim_value <= 128 &&
                           device_.maxThreadgroupMemoryLength >=
                               static_cast<NSUInteger>(
                                   32 * (model.config().qk_dim() + model.config().head_dim_value)) *
                                   4;
        if (options_.attention == "flash" && !flash_supported_)
            throw std::invalid_argument(
                "Flash attention requires an Apple7+ GPU and head widths <= 128");
        buffers_.resize(model.parameters().size());
        for (std::size_t i = 0; i < buffers_.size(); ++i) {
            const auto& p = model.parameters()[i];
            auto& b = buffers_[i];
            if (source) {
                b.value = source->buffers_[i].value;
                continue;
            }
            bool half_storage = half_ && !optimizer_;
            b.value = allocate(device_, shape({p.value.rows, p.value.cols}),
                               half_storage ? MPSDataTypeFloat16 : MPSDataTypeFloat32);
            if (half_storage) {
                vImage_Buffer floats{const_cast<float*>(p.value.data()), 1, p.value.size(),
                                     p.value.size() * 4};
                vImage_Buffer target{b.value.buffer.contents, 1, p.value.size(),
                                     p.value.size() * 2};
                if (vImageConvert_PlanarFtoPlanar16F(&floats, &target, kvImageNoFlags) !=
                    kvImageNoError)
                    throw std::runtime_error("Cannot convert inference weights to FP16");
            } else
                std::memcpy(b.value.buffer.contents, p.value.data(), p.value.size() * 4);
            if (optimizer_) {
                auto dimensions = b.value.data.shape;
                b.gradient = allocate(device_, dimensions, MPSDataTypeFloat32);
                b.next_gradient = allocate(device_, dimensions, MPSDataTypeFloat32);
                b.first = allocate(device_, dimensions, MPSDataTypeFloat32);
                b.next_first = allocate(device_, dimensions, MPSDataTypeFloat32);
                b.next_value = allocate(device_, dimensions, MPSDataTypeFloat32);
                const auto& s = optimizer_->states().at(i);
                std::memcpy(b.first.buffer.contents, s.first.data(), s.first.size() * 4);
                if (s.second.size()) {
                    b.second = allocate(device_, dimensions, MPSDataTypeFloat32);
                    b.next_second = allocate(device_, dimensions, MPSDataTypeFloat32);
                    std::memcpy(b.second.buffer.contents, s.second.data(), s.second.size() * 4);
                }
            }
        }
        const auto& c = model.config();
        auto dtype = half_ ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
        for (int l = 0; l < c.n_layers; ++l) {
            cache_.push_back(
                allocate(device_, shape({c.max_sequence_length, c.kv_lora_rank}), dtype));
            next_cache_.push_back(allocate(device_, cache_.back().data.shape, dtype));
            rope_.push_back(
                allocate(device_, shape({c.max_sequence_length, c.head_dim_rope}), dtype));
            next_rope_.push_back(allocate(device_, rope_.back().data.shape, dtype));
            projection_.push_back(
                allocate(device_, shape({c.n_heads, c.kv_lora_rank, c.d_model}), dtype));
        }
    }
    std::string device_name() const override {
        return cpp(device_.name) + " / MPSGraph / " + (half_ ? "fp16" : "fp32");
    }
    std::string attention_name() const override { return last_attention_; }
    ExecutionStats stats() const override { return stats_; }
    Result<std::unique_ptr<Execution>> fork_inference() override {
        if (optimizer_ || uploaded_revision_ != model_.revision())
            return Result<std::unique_ptr<Execution>>::failure(
                ErrorCode::invalid_argument, "Only current immutable inference can fork");
        return guarded<std::unique_ptr<Execution>>(
            [&] { return std::make_unique<MpsExecution>(model_, nullptr, half_, options_, this); });
    }
    Result<const Matrix*> forward(std::span<const std::uint16_t> ids, int b, int t) override {
        return guarded<const Matrix*>([&] {
            validate(ids, b, t);
            if (use_flash(t)) {
                auto& workspace = flash_workspace(b, t);
                CommandBatch commands(queue_, stats_);
                encode_flash_forward(ids, b, t, 0, commands, false);
                auto& head = flash_program(FlashStage::head, b, t);
                encode(head, {{head.activation_inputs[0], workspace.hidden.back()}}, {}, commands);
                commands.finish();
                output_.resize(b * t, model_.config().vocab_size);
                read(head.outputs.at(head.logits), output_.values);
                return &output_;
            }
            last_attention_ = "reference";
            auto& p = program(0, b, t);
            feed_ids(p, ids);
            run(p, {});
            output_.resize(b * t, model_.config().vocab_size);
            read(p.outputs.at(p.logits), output_.values);
            return &output_;
        });
    }
    Result<std::vector<float>> log_probabilities(const Batch& b) override {
        return guarded<std::vector<float>>([&] {
            validate(b.input, b.batch, b.sequence);
            validate_targets(b);
            if (use_flash(b.sequence)) {
                auto& workspace = flash_workspace(b.batch, b.sequence);
                CommandBatch commands(queue_, stats_);
                encode_flash_forward(b.input, b.batch, b.sequence, 0, commands, false);
                auto& head = flash_program(FlashStage::log_probs, b.batch, b.sequence);
                feed_targets(head, b.target);
                encode(head, {{head.activation_inputs[0], workspace.hidden.back()}}, {}, commands);
                commands.finish();
                std::vector<float> result(b.target.size());
                read(head.outputs.at(head.selected), result);
                return result;
            }
            last_attention_ = "reference";
            auto& p = program(1, b.batch, b.sequence);
            feed_ids(p, b.input);
            feed_targets(p, b.target);
            run(p, {});
            std::vector<float> lp(b.target.size());
            read(p.outputs.at(p.selected), lp);
            return lp;
        });
    }
    Result<double> accumulate(const Batch& b, std::span<const float> coefficients) override {
        return guarded<double>([&] {
            GradientBatch job{&b, coefficients};
            validate_job(job);
            CommandBatch commands(queue_, stats_);
            auto loss = encode_accumulation(job, 0, commands);
            commands.finish();
            return scalar(loss, "loss");
        });
    }
    Status zero_grad() override {
        gradients_empty_ = true;
        return ok();
    }
    Result<double> update(float lr) override {
        return guarded<double>([&] {
            CommandBatch commands(queue_, stats_);
            auto norm = encode_update(lr, commands);
            commands.finish();
            double value = scalar(norm, "gradient norm");
            commit_update();
            return value;
        });
    }
    Result<StepResult> step(std::span<const GradientBatch> jobs, float lr) override {
        if (options_.submission_window == 1)
            return Execution::step(jobs, lr);
        return guarded<StepResult>([&] {
            if (jobs.empty() || !std::isfinite(lr) || lr < 0)
                throw std::invalid_argument("Invalid training step");
            for (const auto& job : jobs)
                validate_job(job);
            StepResult result;
            auto window = static_cast<std::size_t>(options_.submission_window);
            for (std::size_t start = 0; start < jobs.size(); start += window) {
                CommandBatch commands(queue_, stats_);
                std::vector<DeviceArray> losses;
                auto end = std::min(start + window, jobs.size());
                for (auto i = start; i < end; ++i)
                    losses.push_back(encode_accumulation(jobs[i], i - start, commands));
                DeviceArray norm;
                if (end == jobs.size())
                    norm = encode_update(lr, commands);
                commands.finish();
                for (const auto& loss : losses)
                    result.loss += scalar(loss, "loss");
                if (end == jobs.size())
                    result.gradient_norm = scalar(norm, "gradient norm");
            }
            commit_update();
            return result;
        });
    }
    Status synchronize_to_host() override {
        if (uploaded_revision_ != model_.revision())
            return Status::failure(
                ErrorCode::invalid_argument,
                "Host weights changed; recreate MPS execution before synchronizing");
        if (!optimizer_)
            return ok();
        if (gradients_empty_)
            clear_gradients();
        for (std::size_t i = 0; i < buffers_.size(); ++i) {
            auto& p = model_.parameters()[i];
            const auto& b = buffers_[i];
            read(b.value, p.value.values);
            if (!ops::finite(p.value.values))
                return Status::failure(ErrorCode::numerical_error,
                                       "Nonfinite GPU parameter: " + p.name);
            if (optimizer_) {
                p.gradient.resize(p.value.rows, p.value.cols);
                read(b.gradient, p.gradient.values);
                auto& s = optimizer_->states()[i];
                read(b.first, s.first.values);
                if (s.second.size())
                    read(b.second, s.second.values);
            }
        }
        return ok();
    }
    Result<std::span<const float>> prefill(std::span<const std::uint16_t> ids) override {
        return guarded<std::span<const float>>([&] {
            validate(ids, 1, static_cast<int>(ids.size()));
            int bucket = 1;
            while (bucket < static_cast<int>(ids.size()))
                bucket *= 2;
            bucket = std::min(bucket, model_.config().max_sequence_length);
            if (use_flash(bucket)) {
                auto& workspace = flash_workspace(1, bucket);
                std::vector<std::uint16_t> padded(static_cast<std::size_t>(bucket), 0);
                std::copy(ids.begin(), ids.end(), padded.begin());
                CommandBatch commands(queue_, stats_);
                encode_flash_forward(padded, 1, bucket, 0, commands, true);
                auto& head = flash_program(FlashStage::prefill_head, 1, bucket);
                std::int32_t last = static_cast<std::int32_t>(ids.size() - 1);
                std::memcpy(head.inputs.at(head.position).buffer.contents, &last, 4);
                encode(head, {{head.activation_inputs[0], workspace.hidden.back()}}, {}, commands);
                commands.finish();
                position_ = static_cast<int>(ids.size());
                output_.resize(1, model_.config().vocab_size);
                read(head.outputs.at(head.logits), output_.values);
                return std::span<const float>{output_.values};
            }
            last_attention_ = "reference";
            auto& p = program(3, 1, bucket);
            std::vector<std::uint16_t> padded(static_cast<std::size_t>(bucket), 0);
            std::copy(ids.begin(), ids.end(), padded.begin());
            feed_ids(p, padded);
            std::int32_t last = static_cast<std::int32_t>(ids.size() - 1);
            std::memcpy(p.inputs.at(p.position).buffer.contents, &last, 4);
            CommandBatch commands(queue_, stats_);
            encode(p, {}, {}, commands);
            auto command = commands.command;
            auto blit = [command blitCommandEncoder];
            auto item = half_ ? 2U : 4U;
            for (std::size_t l = 0; l < cache_.size(); ++l) {
                const auto& ck = p.outputs.at(p.cache_outputs[l]);
                const auto& kr = p.outputs.at(p.rope_outputs[l]);
                const auto& combined = p.outputs.at(p.projection_outputs[l]);
                [blit copyFromBuffer:ck.buffer
                         sourceOffset:0
                             toBuffer:cache_[l].buffer
                    destinationOffset:0
                                 size:ids.size() *
                                      static_cast<std::size_t>(model_.config().kv_lora_rank) *
                                      item];
                [blit copyFromBuffer:kr.buffer
                         sourceOffset:0
                             toBuffer:rope_[l].buffer
                    destinationOffset:0
                                 size:ids.size() *
                                      static_cast<std::size_t>(model_.config().head_dim_rope) *
                                      item];
                [blit copyFromBuffer:combined.buffer
                         sourceOffset:0
                             toBuffer:projection_[l].buffer
                    destinationOffset:0
                                 size:combined.buffer.length];
            }
            [blit endEncoding];
            commands.finish();
            position_ = static_cast<int>(ids.size());
            output_.resize(1, model_.config().vocab_size);
            read(p.outputs.at(p.logits), output_.values);
            return std::span<const float>{output_.values};
        });
    }
    Result<std::span<const float>> decode(std::uint16_t token) override {
        return guarded<std::span<const float>>([&] {
            if (uploaded_revision_ != model_.revision())
                throw std::invalid_argument("Host weights changed; recreate the MPS execution");
            if (position_ < 1 || position_ >= model_.config().max_sequence_length ||
                token >= model_.config().vocab_size)
                throw std::invalid_argument("Invalid decode position or token");
            int bucket = 1;
            while (bucket <= position_)
                bucket *= 2;
            bucket = std::min(bucket, model_.config().max_sequence_length);
            auto& p = program(4, 1, bucket);
            std::uint16_t ids[] = {token};
            feed_ids(p, ids);
            std::int32_t position = position_;
            std::memcpy(p.inputs.at(p.position).buffer.contents, &position, 4);
            std::map<Tensor, DeviceArray> outputs;
            for (std::size_t l = 0; l < cache_.size(); ++l) {
                auto latent_shape = shape({bucket, model_.config().kv_lora_rank}),
                     rope_shape = shape({bucket, model_.config().head_dim_rope});
                p.inputs[p.cache_inputs[l]] = alias(cache_[l], latent_shape);
                p.inputs[p.rope_inputs[l]] = alias(rope_[l], rope_shape);
                p.inputs[p.projection_inputs[l]] = projection_[l];
                outputs[p.cache_outputs[l]] = alias(next_cache_[l], latent_shape);
                outputs[p.rope_outputs[l]] = alias(next_rope_[l], rope_shape);
            }
            run(p, outputs);
            for (std::size_t l = 0; l < cache_.size(); ++l) {
                std::swap(cache_[l], next_cache_[l]);
                std::swap(rope_[l], next_rope_[l]);
            }
            ++position_;
            output_.resize(1, model_.config().vocab_size);
            read(p.outputs.at(p.logits), output_.values);
            return std::span<const float>{output_.values};
        });
    }

  private:
    Model& model_;
    Optimizer* optimizer_;
    bool half_;
    ExecutionOptions options_;
    ExecutionStats stats_;
    bool flash_supported_ = false;
    bool gradients_empty_ = true;
    std::string last_attention_ = "reference";
    id<MTLDevice> device_;
    id<MTLCommandQueue> queue_;
    MPSGraphDevice* graph_device_;
    std::vector<ParameterBuffers> buffers_;
    std::vector<DeviceArray> cache_, next_cache_, rope_, next_rope_, projection_;
    std::map<std::tuple<int, int, int>, std::unique_ptr<Program>> programs_;
    std::shared_ptr<ProgramPool> program_pool_;
    std::unique_ptr<Program> update_;
    std::unique_ptr<MetalAdamW> adamw_;
    std::map<std::tuple<FlashStage, int, int>, std::unique_ptr<Program>> flash_programs_;
    FlashWorkspace flash_workspace_;
    std::unique_ptr<FlashAttention> flash_kernel_;
    Matrix output_;
    int position_ = 0;
    std::uint64_t program_clock_ = 0;
    std::uint64_t uploaded_revision_ = 0;
    template <class T, class F> Result<T> guarded(F fn) {
        @autoreleasepool {
            @try {
                try {
                    return Result<T>::success(fn());
                } catch (const std::exception& e) {
                    return Result<T>::failure(ErrorCode::gpu_error, e.what());
                }
            } @catch (NSException* e) {
                return Result<T>::failure(ErrorCode::gpu_error, cpp(e.reason));
            }
        }
    }
    void validate(std::span<const std::uint16_t> ids, int b, int t) {
        if (uploaded_revision_ != model_.revision())
            throw std::invalid_argument("Host weights changed; recreate MPS execution");
        if (b < 1 || t < 1 || t > model_.config().max_sequence_length || b > INT_MAX / t ||
            ids.size() != static_cast<std::size_t>(b) * static_cast<std::size_t>(t))
            throw std::invalid_argument("Invalid graph input shape");
        for (auto id : ids)
            if (id >= model_.config().vocab_size)
                throw std::invalid_argument("Graph input exceeds vocabulary");
    }
    void validate_targets(const Batch& b) {
        if (b.target.size() != b.input.size())
            throw std::invalid_argument("Target shape mismatch");
        for (auto id : b.target)
            if (id >= model_.config().vocab_size)
                throw std::invalid_argument("Target exceeds vocabulary");
    }
    void feed_ids(Program& p, std::span<const std::uint16_t> ids) {
        auto* dst = static_cast<std::int32_t*>(p.inputs.at(p.ids).buffer.contents);
        for (std::size_t i = 0; i < ids.size(); ++i)
            dst[i] = ids[i];
    }
    void feed_targets(Program& p, std::span<const std::uint16_t> ids) {
        auto* dst = static_cast<std::int32_t*>(p.inputs.at(p.targets).buffer.contents);
        for (std::size_t i = 0; i < ids.size(); ++i)
            dst[i] = ids[i];
    }
    static void read(const DeviceArray& a, std::span<float> out) {
        if (a.data.dataType != MPSDataTypeFloat32 || out.size() * 4 > a.buffer.length)
            throw std::runtime_error("MPS readback shape mismatch");
        std::memcpy(out.data(), a.buffer.contents, out.size() * 4);
    }
    void compile(Program& p, NSArray<Tensor>* targets) {
        auto types = [NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*> new];
        for (const auto& [tensor, data] : p.inputs)
            types[tensor] = [[MPSGraphShapedType alloc] initWithShape:data.data.shape
                                                             dataType:data.data.dataType];
        for (Tensor tensor : p.activation_inputs)
            types[tensor] = [[MPSGraphShapedType alloc] initWithShape:tensor.shape
                                                             dataType:tensor.dataType];
        auto descriptor = [MPSGraphCompilationDescriptor new];
        descriptor.optimizationLevel = MPSGraphOptimizationLevel1;
        descriptor.waitForCompilationCompletion = YES;
        p.executable = [p.graph compileWithDevice:graph_device_
                                            feeds:types
                                    targetTensors:targets
                                 targetOperations:nil
                            compilationDescriptor:descriptor];
        if (!p.executable)
            throw std::runtime_error("MPSGraph compilation failed");
        p.executable.options = MPSGraphOptionsNone;
        for (Tensor t in targets)
            if (!p.outputs.contains(t) &&
                std::find(p.activation_outputs.begin(), p.activation_outputs.end(), t) ==
                    p.activation_outputs.end())
                p.outputs.emplace(t, allocate(device_, t.shape, t.dataType));
    }
    void run(Program& p, const std::map<Tensor, DeviceArray>& supplied_outputs) {
        CommandBatch commands(queue_, stats_);
        encode(p, {}, supplied_outputs, commands);
        commands.finish();
    }
    void encode(Program& p, const std::map<Tensor, DeviceArray>& supplied_inputs,
                const std::map<Tensor, DeviceArray>& supplied_outputs, CommandBatch& commands,
                int layer = -1) {
        for (std::size_t i = 0; i < p.weights.size(); ++i)
            p.inputs[p.weights[i]] = buffers_[parameter_index(i, layer)].value;
        auto inputs = [NSMutableArray<MPSGraphTensorData*> new],
             results = [NSMutableArray<MPSGraphTensorData*> new];
        for (Tensor t in p.executable.feedTensors) {
            auto it = supplied_inputs.find(t);
            [inputs addObject:it == supplied_inputs.end() ? p.inputs.at(t).data : it->second.data];
        }
        for (Tensor t in p.executable.targetTensors) {
            auto it = supplied_outputs.find(t);
            [results
                addObject:it == supplied_outputs.end() ? p.outputs.at(t).data : it->second.data];
        }
        commands.keep(p.executable);
        [p.executable encodeToCommandBuffer:commands.command
                                inputsArray:inputs
                               resultsArray:results
                        executionDescriptor:commands.descriptor(inputs, results)];
        ++stats_.graph_submissions;
    }
    static std::size_t parameter_index(std::size_t i, int layer) {
        return layer >= 0 && i >= 2 && i < 2 + layer_parameter_count
                   ? i + static_cast<std::size_t>(layer) * layer_parameter_count
                   : i;
    }
    bool use_flash(int) const { return flash_supported_ && options_.attention == "flash"; }
    void validate_job(const GradientBatch& job) {
        if (!optimizer_ || !job.batch)
            throw std::invalid_argument("Gradient batch requires an optimizer and data");
        const auto& b = *job.batch;
        validate(b.input, b.batch, b.sequence);
        validate_targets(b);
        if (job.coefficients.size() != b.target.size() || !ops::finite(job.coefficients))
            throw std::invalid_argument("Invalid loss coefficients");
    }
    static double scalar(const DeviceArray& a, std::string_view name) {
        float value;
        read(a, std::span<float>{&value, 1});
        if (!std::isfinite(value))
            throw std::runtime_error("Nonfinite MPS " + std::string(name));
        return value;
    }
    static void write_ids(const DeviceArray& destination, std::span<const std::uint16_t> ids) {
        auto* target = static_cast<std::int32_t*>(destination.buffer.contents);
        for (std::size_t i = 0; i < ids.size(); ++i)
            target[i] = ids[i];
    }
    Staging& staging(Program& p, std::size_t slot) {
        while (p.staging.size() <= slot) {
            Staging s;
            if (p.ids)
                s.ids = allocate(device_, p.ids.shape, p.ids.dataType);
            if (p.targets)
                s.targets = allocate(device_, p.targets.shape, p.targets.dataType);
            if (p.coefficients)
                s.coefficients = allocate(device_, p.coefficients.shape, p.coefficients.dataType);
            if (p.loss)
                s.loss = allocate(device_, shape({1}), MPSDataTypeFloat32);
            p.staging.push_back(std::move(s));
        }
        return p.staging[slot];
    }
    void encode_gradients(Program& p, std::map<Tensor, DeviceArray> inputs,
                          std::map<Tensor, DeviceArray> outputs, CommandBatch& commands,
                          int layer = -1) {
        for (std::size_t j = 0; j < p.parameter_indices.size(); ++j) {
            auto i = parameter_index(p.parameter_indices[j], layer);
            if (p.gradient_inputs[j])
                inputs[p.gradient_inputs[j]] = buffers_[i].gradient;
            outputs[p.gradients[j]] = buffers_[i].next_gradient;
        }
        encode(p, inputs, outputs, commands, layer);
        for (auto index : p.parameter_indices) {
            auto& buffer = buffers_[parameter_index(index, layer)];
            std::swap(buffer.gradient, buffer.next_gradient);
        }
    }
    DeviceArray encode_accumulation(const GradientBatch& job, std::size_t slot,
                                    CommandBatch& commands) {
        const auto& batch = *job.batch;
        if (use_flash(batch.sequence)) {
            if (gradients_empty_)
                clear_gradients();
            return encode_flash_backward(job, slot, commands);
        }
        last_attention_ = "reference";
        auto& p = program(gradients_empty_ ? 6 : 2, batch.batch, batch.sequence);
        auto& s = staging(p, slot);
        write_ids(s.ids, batch.input);
        write_ids(s.targets, batch.target);
        std::memcpy(s.coefficients.buffer.contents, job.coefficients.data(),
                    job.coefficients.size_bytes());
        encode_gradients(p,
                         {{p.ids, s.ids}, {p.targets, s.targets}, {p.coefficients, s.coefficients}},
                         {{p.loss, s.loss}}, commands);
        gradients_empty_ = false;
        return s.loss;
    }
    void clear_gradients() {
        for (auto& b : buffers_)
            if (b.gradient.buffer)
                std::memset(b.gradient.buffer.contents, 0, b.gradient.buffer.length);
        gradients_empty_ = false;
    }
    DeviceArray encode_update(float lr, CommandBatch& commands) {
        if (!optimizer_ || !std::isfinite(lr) || lr < 0)
            throw std::invalid_argument("Invalid optimizer update");
        if (gradients_empty_)
            clear_gradients();
        float bc1 = static_cast<float>(1 - std::pow(static_cast<double>(optimizer_->config().beta1),
                                                    static_cast<double>(optimizer_->steps() + 1)));
        float bc2 = static_cast<float>(1 - std::pow(static_cast<double>(optimizer_->config().beta2),
                                                    static_cast<double>(optimizer_->steps() + 1)));
        if (optimizer_->config().kind == "adamw" &&
            device_.argumentBuffersSupport == MTLArgumentBuffersTier2) {
            if (!adamw_)
                adamw_ = std::make_unique<MetalAdamW>(device_, model_);
            return adamw_->encode(model_, *optimizer_, buffers_, lr, bc1, bc2, commands, stats_);
        }
        auto& p = update_program();
        std::memcpy(p.inputs.at(p.lr).buffer.contents, &lr, 4);
        std::memcpy(p.inputs.at(p.bc1).buffer.contents, &bc1, 4);
        std::memcpy(p.inputs.at(p.bc2).buffer.contents, &bc2, 4);
        std::map<Tensor, DeviceArray> outputs;
        for (std::size_t i = 0; i < buffers_.size(); ++i) {
            auto& b = buffers_[i];
            p.inputs[p.gradient_inputs[i]] = b.gradient;
            p.inputs[p.first_inputs[i]] = b.first;
            outputs[p.updated_values[i]] = b.next_value;
            outputs[p.updated_first[i]] = b.next_first;
            if (p.second_inputs[i]) {
                p.inputs[p.second_inputs[i]] = b.second;
                outputs[p.updated_second[i]] = b.next_second;
            }
        }
        encode(p, {}, outputs, commands);
        return p.outputs.at(p.norm);
    }
    void commit_update() {
        for (auto& b : buffers_) {
            std::swap(b.value, b.next_value);
            std::swap(b.first, b.next_first);
            if (b.second.buffer)
                std::swap(b.second, b.next_second);
        }
        optimizer_->set_steps(optimizer_->steps() + 1);
        model_.mark_weights_changed();
        uploaded_revision_ = model_.revision();
        position_ = 0;
    }
    FlashWorkspace& flash_workspace(int batch, int time) {
        if (flash_workspace_.batch == batch && flash_workspace_.time == time)
            return flash_workspace_;
        if (!flash_kernel_)
            flash_kernel_ = std::make_unique<FlashAttention>(device_, half_, model_.config());
        const auto& c = model_.config();
        if (static_cast<std::uint64_t>(batch) * static_cast<unsigned>(c.n_heads) *
                static_cast<unsigned>(time) *
                static_cast<unsigned>(std::max(c.qk_dim(), c.head_dim_value)) >
            UINT32_MAX)
            throw std::invalid_argument("Flash attention tensor exceeds its indexing limit");
        FlashWorkspace w;
        w.batch = batch;
        w.time = time;
        auto dtype = half_ ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
        auto hidden_shape = shape({batch, time, c.d_model});
        auto q_shape = shape({batch, c.n_heads, time, c.qk_dim()});
        auto v_shape = shape({batch, c.n_heads, time, c.head_dim_value});
        for (int l = 0; l <= c.n_layers; ++l)
            w.hidden.push_back(allocate(device_, hidden_shape, dtype));
        for (int l = 0; l < c.n_layers; ++l)
            w.layers.push_back(
                {allocate(device_, q_shape, dtype), allocate(device_, q_shape, dtype),
                 allocate(device_, v_shape, dtype), allocate(device_, v_shape, MPSDataTypeFloat32),
                 allocate(device_, shape({batch, c.n_heads, time}), MPSDataTypeFloat32)});
        if (optimizer_) {
            w.dx = allocate(device_, hidden_shape, MPSDataTypeFloat32);
            w.residual_dx = allocate(device_, hidden_shape, MPSDataTypeFloat32);
            w.attention_dx = allocate(device_, v_shape, MPSDataTypeFloat32);
            w.dq = allocate(device_, q_shape, MPSDataTypeFloat32);
            w.dk = allocate(device_, q_shape, MPSDataTypeFloat32);
            w.dv = allocate(device_, v_shape, MPSDataTypeFloat32);
            w.delta = allocate(device_, shape({batch, c.n_heads, time}), MPSDataTypeFloat32);
        }
        flash_workspace_ = std::move(w);
        return flash_workspace_;
    }
    Program& flash_program(FlashStage stage, int batch, int time) {
        auto key = std::tuple{stage, batch, time};
        if (auto found = flash_programs_.find(key); found != flash_programs_.end()) {
            found->second->last_used = ++program_clock_;
            return *found->second;
        }
        if (flash_programs_.size() >= 24) {
            auto oldest = std::min_element(flash_programs_.begin(), flash_programs_.end(),
                                           [](const auto& a, const auto& b) {
                                               return a.second->last_used < b.second->last_used;
                                           });
            flash_programs_.erase(oldest);
        }
        auto created = std::make_unique<Program>();
        auto& p = *created;
        p.last_used = ++program_clock_;
        GraphBuilder b(p, model_.config(), half_);
        b.weights(model_, half_ && !optimizer_);
        const auto& c = model_.config();
        auto dtype = half_ ? MPSDataTypeFloat16 : MPSDataTypeFloat32;
        auto hidden_shape = shape({batch, time, c.d_model});
        auto q_shape = shape({batch, c.n_heads, time, c.qk_dim()});
        auto v_shape = shape({batch, c.n_heads, time, c.head_dim_value});
        auto input = [&](MPSShape* dimensions, MPSDataType type) {
            auto tensor = [p.graph placeholderWithShape:dimensions dataType:type name:nil];
            p.activation_inputs.push_back(tensor);
            return tensor;
        };
        auto differentiate = [&](Tensor loss, std::vector<std::size_t> indices,
                                 NSArray<Tensor>* activations) {
            auto variables = [NSMutableArray<Tensor> arrayWithArray:activations];
            for (auto i : indices)
                [variables addObject:p.weights[i]];
            auto grads = [p.graph gradientForPrimaryTensor:loss withTensors:variables name:nil];
            for (auto i : indices) {
                if (!grads[p.weights[i]])
                    throw std::runtime_error("Missing segmented graph gradient");
                auto prior = [p.graph placeholderWithShape:p.weights[i].shape
                                                  dataType:MPSDataTypeFloat32
                                                      name:nil];
                p.parameter_indices.push_back(i);
                p.gradient_inputs.push_back(prior);
                p.inputs[prior] = buffers_[i].gradient;
                auto gradient = b.add(prior, b.cast(grads[p.weights[i]], MPSDataTypeFloat32));
                p.gradients.push_back(gradient);
                p.outputs[gradient] = buffers_[i].next_gradient;
            }
            return grads;
        };
        if (stage == FlashStage::embedding || stage == FlashStage::embedding_backward) {
            auto embedded = b.embedding(batch, time);
            if (stage == FlashStage::embedding)
                p.activation_outputs = {embedded};
            else {
                auto dx = input(hidden_shape, MPSDataTypeFloat32);
                differentiate(b.sum(b.mul(b.cast(embedded, MPSDataTypeFloat32), dx)), {0}, @[]);
            }
        } else {
            auto x = input(hidden_shape, dtype);
            if (stage == FlashStage::projections || stage == FlashStage::projections_backward ||
                stage == FlashStage::prefill_projections) {
                auto [q, k, v, latent, rotary] = b.projections(x, 0, batch, time);
                if (stage == FlashStage::projections_backward) {
                    auto dq = input(q_shape, MPSDataTypeFloat32),
                         dk = input(q_shape, MPSDataTypeFloat32),
                         dv = input(v_shape, MPSDataTypeFloat32);
                    auto residual = input(hidden_shape, MPSDataTypeFloat32);
                    auto loss = b.add(b.add(b.sum(b.mul(b.cast(q, MPSDataTypeFloat32), dq)),
                                            b.sum(b.mul(b.cast(k, MPSDataTypeFloat32), dk))),
                                      b.sum(b.mul(b.cast(v, MPSDataTypeFloat32), dv)));
                    auto grads = differentiate(loss,
                                               {2 + attn_norm, 2 + w_dq, 2 + q_norm, 2 + w_uq,
                                                2 + w_dkv, 2 + kv_norm, 2 + w_uk, 2 + w_uv},
                                               @[ x ]);
                    p.activation_outputs = {b.add(b.cast(grads[x], MPSDataTypeFloat32), residual)};
                } else {
                    p.activation_outputs = {q, k, v};
                    if (stage == FlashStage::prefill_projections) {
                        p.activation_outputs.push_back(latent);
                        p.activation_outputs.push_back(rotary);
                        p.activation_outputs.push_back(b.combined_projection(0));
                    }
                }
            } else if (stage == FlashStage::post_attention || stage == FlashStage::post_backward) {
                auto attention = input(v_shape, MPSDataTypeFloat32);
                auto output = b.projected_output(x, attention, 0, batch, time);
                if (stage == FlashStage::post_attention)
                    p.activation_outputs = {output};
                else {
                    auto upstream = input(hidden_shape, MPSDataTypeFloat32);
                    auto grads =
                        differentiate(b.sum(b.mul(b.cast(output, MPSDataTypeFloat32), upstream)),
                                      {2 + w_o, 2 + ffn_norm, 2 + w_gate, 2 + w_up, 2 + w_down},
                                      @[ attention, x ]);
                    p.activation_outputs = {b.cast(grads[attention], MPSDataTypeFloat32),
                                            b.cast(grads[x], MPSDataTypeFloat32)};
                }
            } else {
                p.logits = b.reshape(b.head(x), shape({batch * time, c.vocab_size}));
                if (stage == FlashStage::prefill_head) {
                    p.position = [p.graph placeholderWithShape:shape({1})
                                                      dataType:MPSDataTypeInt32
                                                          name:nil];
                    p.logits = [p.graph gatherWithUpdatesTensor:p.logits
                                                  indicesTensor:p.position
                                                           axis:0
                                                batchDimensions:0
                                                           name:nil];
                } else if (stage == FlashStage::log_probs || stage == FlashStage::head_backward) {
                    p.selected = b.selected(p.logits, batch * time);
                    if (stage == FlashStage::head_backward) {
                        p.coefficients = [p.graph placeholderWithShape:shape({batch * time, 1})
                                                              dataType:MPSDataTypeFloat32
                                                                  name:nil];
                        p.loss = b.sum(b.mul([p.graph negativeWithTensor:p.selected name:nil],
                                             p.coefficients));
                        auto grads = differentiate(p.loss, {0, 1}, @[ x ]);
                        p.activation_outputs = {b.cast(grads[x], MPSDataTypeFloat32)};
                    }
                }
            }
        }
        for (std::size_t i = 0; i < p.weights.size(); ++i)
            p.inputs[p.weights[i]] = buffers_[i].value;
        for (auto tensor : {p.ids, p.targets, p.coefficients, p.position})
            if (tensor)
                p.inputs[tensor] = allocate(device_, tensor.shape, tensor.dataType);
        auto targets = [NSMutableArray<Tensor> new];
        for (auto tensor : p.activation_outputs)
            [targets addObject:tensor];
        for (auto tensor : p.gradients)
            [targets addObject:tensor];
        if (p.loss)
            [targets addObject:p.loss];
        if (stage == FlashStage::head || stage == FlashStage::prefill_head)
            [targets addObject:p.logits];
        if (stage == FlashStage::log_probs)
            [targets addObject:p.selected];
        compile(p, targets);
        auto& result = *created;
        flash_programs_.emplace(key, std::move(created));
        return result;
    }
    void encode_flash_forward(std::span<const std::uint16_t> ids, int batch, int time,
                              std::size_t slot, CommandBatch& commands, bool prefill) {
        last_attention_ = "flash";
        auto& w = flash_workspace(batch, time);
        auto& embedding = flash_program(FlashStage::embedding, batch, time);
        auto& s = staging(embedding, slot);
        write_ids(s.ids, ids);
        encode(embedding, {{embedding.ids, s.ids}},
               {{embedding.activation_outputs[0], w.hidden[0]}}, commands);
        for (int l = 0; l < model_.config().n_layers; ++l) {
            auto i = static_cast<std::size_t>(l);
            auto& a = w.layers[i];
            auto& pre = flash_program(
                prefill ? FlashStage::prefill_projections : FlashStage::projections, batch, time);
            std::map<Tensor, DeviceArray> outputs{{pre.activation_outputs[0], a.q},
                                                  {pre.activation_outputs[1], a.k},
                                                  {pre.activation_outputs[2], a.v}};
            if (prefill) {
                outputs[pre.activation_outputs[3]] =
                    alias(cache_[i], pre.activation_outputs[3].shape);
                outputs[pre.activation_outputs[4]] =
                    alias(rope_[i], pre.activation_outputs[4].shape);
                outputs[pre.activation_outputs[5]] = projection_[i];
            }
            encode(pre, {{pre.activation_inputs[0], w.hidden[i]}}, outputs, commands, l);
            flash_kernel_->forward(commands.command, model_.config(), batch, time, a.q, a.k, a.v,
                                   a.out, a.lse);
            auto& post = flash_program(FlashStage::post_attention, batch, time);
            encode(post,
                   {{post.activation_inputs[0], w.hidden[i]}, {post.activation_inputs[1], a.out}},
                   {{post.activation_outputs[0], w.hidden[i + 1]}}, commands, l);
        }
    }
    DeviceArray encode_flash_backward(const GradientBatch& job, std::size_t slot,
                                      CommandBatch& commands) {
        const auto& batch = *job.batch;
        auto& w = flash_workspace(batch.batch, batch.sequence);
        encode_flash_forward(batch.input, batch.batch, batch.sequence, slot, commands, false);
        auto& head = flash_program(FlashStage::head_backward, batch.batch, batch.sequence);
        auto& s = staging(head, slot);
        write_ids(s.targets, batch.target);
        std::memcpy(s.coefficients.buffer.contents, job.coefficients.data(),
                    job.coefficients.size_bytes());
        encode_gradients(head,
                         {{head.activation_inputs[0], w.hidden.back()},
                          {head.targets, s.targets},
                          {head.coefficients, s.coefficients}},
                         {{head.activation_outputs[0], w.dx}, {head.loss, s.loss}}, commands);
        for (int l = model_.config().n_layers - 1; l >= 0; --l) {
            auto i = static_cast<std::size_t>(l);
            auto& a = w.layers[i];
            auto& post = flash_program(FlashStage::post_backward, batch.batch, batch.sequence);
            encode_gradients(post,
                             {{post.activation_inputs[0], w.hidden[i]},
                              {post.activation_inputs[1], a.out},
                              {post.activation_inputs[2], w.dx}},
                             {{post.activation_outputs[0], w.attention_dx},
                              {post.activation_outputs[1], w.residual_dx}},
                             commands, l);
            flash_kernel_->backward(commands.command, model_.config(), batch.batch, batch.sequence,
                                    a.q, a.k, a.v, a.out, w.attention_dx, a.lse, w.delta, w.dq,
                                    w.dk, w.dv);
            auto& pre =
                flash_program(FlashStage::projections_backward, batch.batch, batch.sequence);
            encode_gradients(pre,
                             {{pre.activation_inputs[0], w.hidden[i]},
                              {pre.activation_inputs[1], w.dq},
                              {pre.activation_inputs[2], w.dk},
                              {pre.activation_inputs[3], w.dv},
                              {pre.activation_inputs[4], w.residual_dx}},
                             {{pre.activation_outputs[0], w.dx}}, commands, l);
        }
        auto& embedding =
            flash_program(FlashStage::embedding_backward, batch.batch, batch.sequence);
        auto& es = staging(embedding, slot);
        write_ids(es.ids, batch.input);
        encode_gradients(embedding,
                         {{embedding.ids, es.ids}, {embedding.activation_inputs[0], w.dx}}, {},
                         commands);
        return s.loss;
    }
    Program& program(int mode, int batch, int time) {
        auto key = std::tuple{mode, batch, time};
        auto found = programs_.find(key);
        if (found != programs_.end()) {
            found->second->last_used = ++program_clock_;
            return *found->second;
        }
        if (programs_.size() >= 12) {
            auto oldest = std::min_element(programs_.begin(), programs_.end(),
                                           [](const auto& a, const auto& b) {
                                               return a.second->last_used < b.second->last_used;
                                           });
            programs_.erase(oldest);
        }
        std::unique_lock<std::mutex> shared_lock;
        if (!optimizer_) {
            shared_lock = std::unique_lock(program_pool_->mutex);
            if (auto cached = program_pool_->programs.find(key);
                cached != program_pool_->programs.end()) {
                cached->second.last_used = ++program_pool_->clock;
                auto copy = std::make_unique<Program>(*cached->second.program);
                auto& p = *copy;
                p.staging.clear();
                p.last_used = ++program_clock_;
                auto index_of = [](const std::vector<Tensor>& tensors, Tensor value) {
                    return static_cast<std::size_t>(
                        std::find(tensors.begin(), tensors.end(), value) - tensors.begin());
                };
                for (auto& [tensor, data] : p.inputs) {
                    if (auto wi = index_of(p.weights, tensor); wi < p.weights.size())
                        data = buffers_[wi].value;
                    else if (auto ci = index_of(p.cache_inputs, tensor); ci < p.cache_inputs.size())
                        data = alias(cache_[ci], tensor.shape);
                    else if (auto ri = index_of(p.rope_inputs, tensor); ri < p.rope_inputs.size())
                        data = alias(rope_[ri], tensor.shape);
                    else if (auto pi = index_of(p.projection_inputs, tensor);
                             pi < p.projection_inputs.size())
                        data = projection_[pi];
                    else
                        data = allocate(device_, tensor.shape, tensor.dataType);
                }
                for (auto& [tensor, data] : p.outputs) {
                    if (auto co = index_of(p.cache_outputs, tensor); co < p.cache_outputs.size())
                        data = alias(mode == 4 ? next_cache_[co] : cache_[co], tensor.shape);
                    else if (auto ro = index_of(p.rope_outputs, tensor); ro < p.rope_outputs.size())
                        data = alias(mode == 4 ? next_rope_[ro] : rope_[ro], tensor.shape);
                    else if (auto po = index_of(p.projection_outputs, tensor);
                             po < p.projection_outputs.size())
                        data = projection_[po];
                    else
                        data = allocate(device_, tensor.shape, tensor.dataType);
                }
                auto& result = *copy;
                programs_.emplace(key, std::move(copy));
                return result;
            }
        }
        auto created = std::make_unique<Program>();
        auto& p = *created;
        GraphBuilder b(p, model_.config(), half_);
        b.weights(model_, half_ && !optimizer_);
        p.last_used = ++program_clock_;
        for (std::size_t i = 0; i < p.weights.size(); ++i)
            p.inputs[p.weights[i]] = buffers_[i].value;
        p.logits = mode == 4 ? b.decode(time) : b.forward(batch, time, mode == 3);
        auto targets = [NSMutableArray<Tensor> new];
        if (mode == 0 || mode == 3 || mode == 4)
            [targets addObject:p.logits];
        if (mode == 1 || mode == 2 || mode == 6) {
            p.selected = b.selected(p.logits, batch * time);
            p.inputs[p.targets] = allocate(device_, p.targets.shape, p.targets.dataType);
        }
        if (mode == 1)
            [targets addObject:p.selected];
        if (mode == 2 || mode == 6) {
            p.coefficients = [p.graph placeholderWithShape:shape({batch * time, 1})
                                                  dataType:MPSDataTypeFloat32
                                                      name:@"loss_coefficients"];
            p.inputs[p.coefficients] =
                allocate(device_, p.coefficients.shape, p.coefficients.dataType);
            p.loss = b.sum(b.mul([p.graph negativeWithTensor:p.selected name:nil], p.coefficients));
            [targets addObject:p.loss];
            auto weights = [NSMutableArray<Tensor> new];
            for (auto weight : p.weights)
                [weights addObject:weight];
            auto grads = [p.graph gradientForPrimaryTensor:p.loss withTensors:weights name:nil];
            for (std::size_t i = 0; i < p.weights.size(); ++i) {
                if (!grads[p.weights[i]])
                    throw std::runtime_error("Missing training graph gradient");
                Tensor prior = nil;
                auto gradient = b.cast(grads[p.weights[i]], MPSDataTypeFloat32);
                if (mode == 2) {
                    prior = [p.graph placeholderWithShape:p.weights[i].shape
                                                 dataType:MPSDataTypeFloat32
                                                     name:nil];
                    p.inputs[prior] = buffers_[i].gradient;
                    gradient = b.add(prior, gradient);
                }
                p.gradient_inputs.push_back(prior);
                p.parameter_indices.push_back(i);
                p.gradients.push_back(gradient);
                [targets addObject:gradient];
                p.outputs[gradient] = buffers_[i].next_gradient;
            }
        }
        p.inputs[p.ids] = allocate(device_, p.ids.shape, p.ids.dataType);
        if (p.position)
            p.inputs[p.position] = allocate(device_, p.position.shape, p.position.dataType);
        for (std::size_t i = 0; i < p.cache_inputs.size(); ++i) {
            p.inputs[p.cache_inputs[i]] = alias(cache_[i], p.cache_inputs[i].shape);
            p.inputs[p.rope_inputs[i]] = alias(rope_[i], p.rope_inputs[i].shape);
            p.inputs[p.projection_inputs[i]] = projection_[i];
        }
        for (auto t : p.cache_outputs)
            [targets addObject:t];
        for (auto t : p.rope_outputs)
            [targets addObject:t];
        for (auto t : p.projection_outputs)
            [targets addObject:t];
        if (mode == 4)
            for (std::size_t i = 0; i < cache_.size(); ++i) {
                p.outputs[p.cache_outputs[i]] = alias(next_cache_[i], p.cache_outputs[i].shape);
                p.outputs[p.rope_outputs[i]] = alias(next_rope_[i], p.rope_outputs[i].shape);
            }
        compile(p, targets);
        if (!optimizer_) {
            if (program_pool_->programs.size() >= 24) {
                auto oldest =
                    std::min_element(program_pool_->programs.begin(), program_pool_->programs.end(),
                                     [](const auto& a, const auto& b) {
                                         return a.second.last_used < b.second.last_used;
                                     });
                program_pool_->programs.erase(oldest);
            }
            program_pool_->programs.emplace(
                key,
                ProgramPool::Entry{std::make_shared<const Program>(p), ++program_pool_->clock});
        }
        auto& result = *created;
        programs_.emplace(key, std::move(created));
        return result;
    }
    std::pair<Tensor, Tensor> muon_direction(GraphBuilder& b, MPSGraph* graph, Tensor g,
                                             Tensor first, int rows, int cols) {
        const auto& c = optimizer_->config();
        auto m = b.add(b.mul(b.constant(c.momentum), first), b.mul(b.constant(1 - c.momentum), g));
        auto x = c.nesterov
                     ? b.add(b.mul(b.constant(1 - c.momentum), g), b.mul(b.constant(c.momentum), m))
                     : m;
        NSUInteger row_axis = x.shape.count - 2, col_axis = x.shape.count - 1;
        bool transposed = rows > cols;
        if (transposed)
            x = b.transpose(x, row_axis, col_axis);
        Tensor sq;
        if (row_axis == 0)
            sq = b.sum(b.mul(x, x));
        else
            sq = b.reshape([graph reductionSumWithTensor:b.mul(x, x) axes:@[ @1, @2 ] name:nil],
                           shape({x.shape[0].intValue, 1, 1}));
        auto fnorm = [graph squareRootWithTensor:sq name:nil];
        x = b.div(x, [graph maximumWithPrimaryTensor:fnorm
                                     secondaryTensor:b.constant(c.epsilon)
                                                name:nil]);
        for (int k = 0; k < c.ns_steps; ++k) {
            auto a = b.mm(x, b.transpose(x, row_axis, col_axis));
            auto ax = b.mm(a, x);
            x = b.add(b.sub(b.mul(b.constant(3.4445), x), b.mul(b.constant(4.7750), ax)),
                      b.mul(b.constant(2.0315), b.mm(a, ax)));
        }
        if (transposed)
            x = b.transpose(x, row_axis, col_axis);
        return {m,
                b.mul(b.constant(0.2 * std::sqrt(static_cast<double>(std::max(rows, cols)))), x)};
    }
    Program& update_program() {
        if (update_)
            return *update_;
        auto created = std::make_unique<Program>();
        auto& p = *created;
        GraphBuilder b(p, model_.config(), false);
        b.weights(model_);
        const auto& c = optimizer_->config();
        p.lr = [p.graph placeholderWithShape:shape({1})
                                    dataType:MPSDataTypeFloat32
                                        name:@"learning_rate"];
        p.bc1 = [p.graph placeholderWithShape:shape({1})
                                     dataType:MPSDataTypeFloat32
                                         name:@"bias_correction_1"];
        p.bc2 = [p.graph placeholderWithShape:shape({1})
                                     dataType:MPSDataTypeFloat32
                                         name:@"bias_correction_2"];
        for (auto t : {p.lr, p.bc1, p.bc2})
            p.inputs[t] = allocate(device_, t.shape, t.dataType);
        Tensor sq = b.constant(0);
        for (std::size_t i = 0; i < buffers_.size(); ++i) {
            auto g = [p.graph placeholderWithShape:p.weights[i].shape
                                          dataType:MPSDataTypeFloat32
                                              name:nil];
            p.gradient_inputs.push_back(g);
            p.inputs[g] = buffers_[i].gradient;
            sq = b.add(sq, b.sum(b.mul(g, g)));
        }
        p.norm = [p.graph squareRootWithTensor:sq name:nil];
        auto gs =
            c.clip_norm > 0
                ? [p.graph
                      minimumWithPrimaryTensor:b.constant(1)
                               secondaryTensor:b.div(b.constant(c.clip_norm),
                                                     [p.graph
                                                         maximumWithPrimaryTensor:p.norm
                                                                  secondaryTensor:b.constant(1e-30)
                                                                             name:nil])
                                          name:nil]
                : b.constant(1);
        auto targets = [NSMutableArray<Tensor> arrayWithObject:p.norm];
        for (std::size_t i = 0; i < buffers_.size(); ++i) {
            auto first = [p.graph placeholderWithShape:p.weights[i].shape
                                              dataType:MPSDataTypeFloat32
                                                  name:nil];
            p.first_inputs.push_back(first);
            p.inputs[first] = buffers_[i].first;
        }
        std::vector<Tensor> muon_first(buffers_.size(), nil), muon_updates(buffers_.size(), nil);
        if (c.kind == "hybrid_muon_adamw" && options_.muon_layout == "batched") {
            std::map<std::pair<int, int>, std::vector<std::size_t>> groups;
            for (std::size_t i = 0; i < buffers_.size(); ++i) {
                const auto& param = model_.parameters()[i];
                if (param.role == ParameterRole::hidden_matrix)
                    groups[{param.value.rows, param.value.cols}].push_back(i);
            }
            for (const auto& [dimensions, indices] : groups) {
                for (std::size_t start = 0; start < indices.size(); start += 32) {
                    auto count = std::min(std::size_t{32}, indices.size() - start);
                    if (count == 1)
                        continue;
                    auto gradients = [NSMutableArray<Tensor> new];
                    auto moments = [NSMutableArray<Tensor> new];
                    for (std::size_t j = start; j < start + count; ++j) {
                        [gradients addObject:p.gradient_inputs[indices[j]]];
                        [moments addObject:p.first_inputs[indices[j]]];
                    }
                    auto g = b.mul([p.graph stackTensors:gradients axis:0 name:nil], gs);
                    auto first = [p.graph stackTensors:moments axis:0 name:nil];
                    auto [m, direction] =
                        muon_direction(b, p.graph, g, first, dimensions.first, dimensions.second);
                    for (std::size_t j = 0; j < count; ++j) {
                        auto i = indices[start + j];
                        muon_first[i] =
                            b.reshape(b.slice(m, 0, static_cast<int>(j), 1), p.weights[i].shape);
                        muon_updates[i] = b.reshape(b.slice(direction, 0, static_cast<int>(j), 1),
                                                    p.weights[i].shape);
                    }
                }
            }
        }
        for (std::size_t i = 0; i < buffers_.size(); ++i) {
            const auto& param = model_.parameters()[i];
            auto dimensions = p.weights[i].shape;
            auto first = p.first_inputs[i];
            auto g = b.mul(p.gradient_inputs[i], gs);
            Tensor m, v = nil, update;
            bool muon = c.kind == "hybrid_muon_adamw" && param.role == ParameterRole::hidden_matrix;
            if (muon) {
                if (muon_updates[i]) {
                    m = muon_first[i];
                    update = muon_updates[i];
                } else
                    std::tie(m, update) =
                        muon_direction(b, p.graph, g, first, param.value.rows, param.value.cols);
                p.second_inputs.push_back(nil);
            } else {
                auto second = [p.graph placeholderWithShape:dimensions
                                                   dataType:MPSDataTypeFloat32
                                                       name:nil];
                p.second_inputs.push_back(second);
                p.inputs[second] = buffers_[i].second;
                m = b.add(b.mul(b.constant(c.beta1), first), b.mul(b.constant(1 - c.beta1), g));
                v = b.add(b.mul(b.constant(c.beta2), second),
                          b.mul(b.constant(1 - c.beta2), b.mul(g, g)));
                update = b.div(b.div(m, p.bc1), b.add([p.graph squareRootWithTensor:b.div(v, p.bc2)
                                                                               name:nil],
                                                      b.constant(c.epsilon)));
            }
            float decay = param.role == ParameterRole::norm_scale ? 0 : c.weight_decay;
            auto value =
                b.sub(b.mul(p.weights[i], b.sub(b.constant(1), b.mul(p.lr, b.constant(decay)))),
                      b.mul(p.lr, update));
            p.updated_values.push_back(value);
            p.updated_first.push_back(m);
            p.updated_second.push_back(v);
            p.inputs[p.weights[i]] = buffers_[i].value;
            [targets addObject:value];
            [targets addObject:m];
            p.outputs[value] = buffers_[i].next_value;
            p.outputs[m] = buffers_[i].next_first;
            if (v) {
                [targets addObject:v];
                p.outputs[v] = buffers_[i].next_second;
            }
        }
        compile(p, targets);
        update_ = std::move(created);
        return *update_;
    }
};
} // namespace
#endif
Result<std::unique_ptr<Execution>> make_mps_execution(Model& model, Optimizer* optimizer,
                                                      std::string_view precision,
                                                      const ExecutionOptions& options) {
    auto valid = options.validate();
    if (!valid)
        return Result<std::unique_ptr<Execution>>::failure(valid.error());
#if MINI_MLA_ENABLE_MPS
    if (precision != "fp32" && precision != "fp16")
        return Result<std::unique_ptr<Execution>>::failure(ErrorCode::unsupported,
                                                           "MPS precision must be fp32 or fp16");
    @autoreleasepool {
        @try {
            try {
                if (@available(macOS 13.0, *))
                    return Result<std::unique_ptr<Execution>>::success(
                        std::make_unique<MpsExecution>(model, optimizer, precision == "fp16",
                                                       options));
                return Result<std::unique_ptr<Execution>>::failure(
                    ErrorCode::unsupported, "MPS execution requires macOS 13 or newer");
            } catch (const std::exception& e) {
                return Result<std::unique_ptr<Execution>>::failure(ErrorCode::gpu_error, e.what());
            }
        } @catch (NSException* e) {
            return Result<std::unique_ptr<Execution>>::failure(ErrorCode::gpu_error, cpp(e.reason));
        }
    }
#else
    (void)model;
    (void)optimizer;
    (void)precision;
    (void)options;
    return Result<std::unique_ptr<Execution>>::failure(ErrorCode::unsupported,
                                                       "MPS is disabled in this build");
#endif
}
Result<Json> metal_device_info() {
#if MINI_MLA_ENABLE_MPS
    @autoreleasepool {
        auto device = MTLCreateSystemDefaultDevice();
        if (!device)
            return Result<Json>::failure(ErrorCode::unsupported,
                                         "No Metal device is available to this process");
        return Result<Json>::success(Json::Object{
            {"device", cpp(device.name)},
            {"os", cpp(NSProcessInfo.processInfo.operatingSystemVersionString)},
            {"unified_memory", static_cast<bool>(device.hasUnifiedMemory)},
            {"working_set_bytes", static_cast<std::size_t>(device.recommendedMaxWorkingSetSize)},
            {"precision", Json::Array{"fp32", "fp16"}},
            {"attention", Json::Array{"reference", "flash"}},
            {"flash_head_width_limit", 128},
            {"submission_window_limit", 16}});
    }
#else
    return Result<Json>::failure(ErrorCode::unsupported, "MPS is disabled in this build");
#endif
}
} // namespace mini_mla
