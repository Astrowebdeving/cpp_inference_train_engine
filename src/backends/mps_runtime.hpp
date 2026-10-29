#pragma once
#include "flash_attention_source.hpp"
#include "mini_mla/execution.hpp"

#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>

namespace mini_mla::mps {
using Tensor = MPSGraphTensor*;
inline NSString* ns(std::string_view s) {
    return [[NSString alloc] initWithBytes:s.data() length:s.size() encoding:NSUTF8StringEncoding];
}
inline std::string cpp(NSString* s) {
    return s ? std::string(s.UTF8String) : std::string("Unknown Metal error");
}
inline MPSShape* shape(std::initializer_list<int> dimensions) {
    auto a = [NSMutableArray<NSNumber*> new];
    for (int n : dimensions)
        [a addObject:@(n)];
    return a;
}
struct DeviceArray {
    id<MTLBuffer> buffer = nil;
    MPSGraphTensorData* data = nil;
};
inline DeviceArray alias(const DeviceArray& source, MPSShape* dimensions) {
    return DeviceArray{source.buffer,
                       [[MPSGraphTensorData alloc] initWithMTLBuffer:source.buffer
                                                               shape:dimensions
                                                            dataType:source.data.dataType]};
}
inline DeviceArray allocate(id<MTLDevice> device, MPSShape* dimensions, MPSDataType dtype) {
    std::size_t count = 1;
    for (NSNumber* d in dimensions) {
        auto width = d.unsignedLongLongValue;
        if (!width || width > SIZE_MAX / count)
            throw std::invalid_argument("Invalid or overflowing Metal shape");
        count *= width;
    }
    std::size_t item = dtype == MPSDataTypeFloat16 ? 2 : 4;
    if (count > SIZE_MAX / item)
        throw std::invalid_argument("Metal allocation size overflows");
    auto buffer = [device newBufferWithLength:count * item options:MTLResourceStorageModeShared];
    if (!buffer)
        throw std::runtime_error("Metal allocation failed");
    std::memset(buffer.contents, 0, buffer.length);
    auto data =
        [[MPSGraphTensorData alloc] initWithMTLBuffer:buffer shape:dimensions dataType:dtype];
    if (!data)
        throw std::runtime_error("MPSGraph tensor allocation failed");
    return {buffer, data};
}
struct CommandError {
    std::mutex mutex;
    NSError* failure = nil;
};
class CommandBatch {
  public:
    explicit CommandBatch(id<MTLCommandQueue> queue, ExecutionStats& stats)
        : command([MPSCommandBuffer commandBufferFromCommandQueue:queue]), stats_(stats) {
        if (!command)
            throw std::runtime_error("Cannot create MPS command buffer");
    }
    CommandBatch(const CommandBatch&) = delete;
    CommandBatch& operator=(const CommandBatch&) = delete;
    ~CommandBatch() {
        if (!finished_) {
            @try {
                if (!committed_)
                    [command commit];
                [command waitUntilCompleted];
            } @catch (NSException*) {
            }
        }
    }
    MPSGraphExecutableExecutionDescriptor* descriptor(NSArray<MPSGraphTensorData*>* inputs,
                                                      NSArray<MPSGraphTensorData*>* results) {
        keep(inputs);
        keep(results);
        auto descriptor = [MPSGraphExecutableExecutionDescriptor new];
        auto errors = errors_;
        descriptor.waitUntilCompleted = NO;
        descriptor.completionHandler = ^(NSArray<MPSGraphTensorData*>*, NSError* error) {
          (void)inputs;
          (void)results;
          if (error) {
              std::lock_guard lock(errors->mutex);
              if (!errors->failure)
                  errors->failure = error;
          }
        };
        return descriptor;
    }
    void keep(id object) { [retained_ addObject:object]; }
    // Hands the encoded work to the GPU without blocking, so the host can encode the next batch
    // while this one executes. wait() is the matching completion fence.
    void commit() {
        if (committed_)
            return;
        [command commit];
        committed_ = true;
    }
    void wait() {
        commit();
        [command waitUntilCompleted];
        finished_ = true;
        ++stats_.host_waits;
        std::lock_guard lock(errors_->mutex);
        if (errors_->failure)
            throw std::runtime_error(cpp(errors_->failure.localizedDescription));
        if (command.error)
            throw std::runtime_error(cpp(command.error.localizedDescription));
    }
    void finish() { wait(); }
    MPSCommandBuffer* command;

  private:
    ExecutionStats& stats_;
    std::shared_ptr<CommandError> errors_ = std::make_shared<CommandError>();
    bool committed_ = false;
    bool finished_ = false;
    NSMutableArray* retained_ = [NSMutableArray new];
};
class FlashAttention {
  public:
    FlashAttention(id<MTLDevice> device, bool half, const ModelConfig& c) {
        auto key_width = static_cast<unsigned>(c.qk_dim());
        auto value_width = static_cast<unsigned>(c.head_dim_value);
        scratch_bytes_ = sizeof(float) * std::max({40U * (key_width + value_width) + 280U,
                                                   48U * key_width + 40U * value_width + 784U,
                                                   48U * (key_width + value_width) + 1024U});
        matrix_ = key_width % 8 == 0 && value_width % 8 == 0 && key_width <= 64 &&
                  value_width <= 64 && scratch_bytes_ <= device.maxThreadgroupMemoryLength;
        NSError* error = nil;
        auto options = [MTLCompileOptions new];
        if (@available(macOS 15.0, *))
            options.mathMode = MTLMathModeSafe;
        auto library = [device newLibraryWithSource:ns(metal::flash_attention_source)
                                            options:options
                                              error:&error];
        if (!library)
            throw std::runtime_error("Flash attention compilation failed: " +
                                     cpp(error.localizedDescription));
        auto pipeline = [&](std::string_view name) {
            auto function =
                [library newFunctionWithName:ns(std::string(name) + (matrix_ ? "_matrix" : "") +
                                                (half ? "_f16" : "_f32"))];
            auto result = [device newComputePipelineStateWithFunction:function error:&error];
            if (!result || result.threadExecutionWidth != 32 ||
                result.maxTotalThreadsPerThreadgroup < 128)
                throw std::runtime_error("Unsupported Flash attention pipeline: " +
                                         cpp(error.localizedDescription));
            return result;
        };
        forward_ = pipeline("flash_forward");
        backward_q_ = pipeline("flash_backward_q");
        backward_kv_ = pipeline("flash_backward_kv");
    }
    void forward(MPSCommandBuffer* command, const ModelConfig& c, int batch, int time,
                 const DeviceArray& q, const DeviceArray& k, const DeviceArray& v,
                 const DeviceArray& out, const DeviceArray& lse) {
        dispatch(command, forward_, {q.buffer, k.buffer, v.buffer, out.buffer, lse.buffer}, c,
                 batch, time);
    }
    void backward(MPSCommandBuffer* command, const ModelConfig& c, int batch, int time,
                  const DeviceArray& q, const DeviceArray& k, const DeviceArray& v,
                  const DeviceArray& out, const DeviceArray& incoming, const DeviceArray& lse,
                  const DeviceArray& delta, const DeviceArray& dq, const DeviceArray& dk,
                  const DeviceArray& dv) {
        dispatch(command, backward_q_,
                 {q.buffer, k.buffer, v.buffer, out.buffer, incoming.buffer, lse.buffer,
                  delta.buffer, dq.buffer},
                 c, batch, time);
        dispatch(command, backward_kv_,
                 {q.buffer, k.buffer, v.buffer, incoming.buffer, lse.buffer, delta.buffer,
                  dk.buffer, dv.buffer},
                 c, batch, time);
    }

  private:
    struct Dimensions {
        std::uint32_t heads, time, key_width, value_width;
        float scale;
    };
    id<MTLComputePipelineState> forward_, backward_q_, backward_kv_;
    bool matrix_ = false;
    NSUInteger scratch_bytes_ = 0;
    void dispatch(MPSCommandBuffer* command, id<MTLComputePipelineState> pipeline,
                  std::initializer_list<id<MTLBuffer>> buffers, const ModelConfig& c, int batch,
                  int time) {
        auto encoder = [command computeCommandEncoder];
        if (!encoder)
            throw std::runtime_error("Cannot encode Flash attention");
        [encoder setComputePipelineState:pipeline];
        NSUInteger index = 0;
        for (auto buffer : buffers)
            [encoder setBuffer:buffer offset:0 atIndex:index++];
        Dimensions d{static_cast<std::uint32_t>(c.n_heads), static_cast<std::uint32_t>(time),
                     static_cast<std::uint32_t>(c.qk_dim()),
                     static_cast<std::uint32_t>(c.head_dim_value),
                     1 / std::sqrt(static_cast<float>(c.qk_dim()))};
        [encoder setBytes:&d length:sizeof(d) atIndex:index];
        [encoder
            setThreadgroupMemoryLength:matrix_ ? scratch_bytes_ : 32U * d.key_width * sizeof(float)
                               atIndex:0];
        [encoder setThreadgroupMemoryLength:matrix_ ? 16U : 32U * d.value_width * sizeof(float)
                                    atIndex:1];
        unsigned rows = matrix_ ? 8U : 4U;
        auto groups = static_cast<NSUInteger>(batch) * d.heads * ((d.time + rows - 1) / rows);
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        [encoder endEncoding];
    }
};
} // namespace mini_mla::mps
