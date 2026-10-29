// Micro-benchmark: what does MPSGraph matmul achieve on this model's shapes, and does the
// tensor layout (batched 3-D versus flattened 2-D) change it?
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static MPSGraphDevice* graph_device;

static MPSShape* shp(std::initializer_list<int> d) {
    auto a = [NSMutableArray<NSNumber*> new];
    for (int n : d)
        [a addObject:@(n)];
    return a;
}
static MPSGraphTensorData* buffer(MPSShape* s, MPSDataType t) {
    std::size_t count = 1;
    for (NSNumber* d in s)
        count *= d.unsignedLongValue;
    std::size_t item = t == MPSDataTypeFloat16 ? 2 : 4;
    auto b = [device newBufferWithLength:count * item options:MTLResourceStorageModeShared];
    if (t == MPSDataTypeFloat16) {
        auto* p = static_cast<uint16_t*>(b.contents);
        for (std::size_t i = 0; i < count; ++i)
            p[i] = 0x2000 + (i * 2654435761u) % 0x1000; // small positive halves
    } else {
        auto* p = static_cast<float*>(b.contents);
        for (std::size_t i = 0; i < count; ++i)
            p[i] = 0.001f * static_cast<float>((i * 2654435761u) % 1000);
    }
    return [[MPSGraphTensorData alloc] initWithMTLBuffer:b shape:s dataType:t];
}

struct Case {
    std::string name;
    double macs; // multiply-accumulates per run
    MPSGraph* graph;
    MPSGraphExecutable* exe;
    NSArray<MPSGraphTensorData*>* inputs;
    NSArray<MPSGraphTensorData*>* outputs;
};

static Case build(const std::string& name, double macs, MPSDataType t,
                  std::vector<MPSShape*> in_shapes,
                  std::function<MPSGraphTensor*(MPSGraph*, NSArray<MPSGraphTensor*>*)> body) {
    Case c;
    c.name = name;
    c.macs = macs;
    c.graph = [MPSGraph new];
    auto placeholders = [NSMutableArray<MPSGraphTensor*> new];
    auto feeds = [NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*> new];
    auto inputs = [NSMutableArray<MPSGraphTensorData*> new];
    for (auto s : in_shapes) {
        auto p = [c.graph placeholderWithShape:s dataType:t name:nil];
        [placeholders addObject:p];
        feeds[p] = [[MPSGraphShapedType alloc] initWithShape:s dataType:t];
    }
    auto out = body(c.graph, placeholders);
    auto d = [MPSGraphCompilationDescriptor new];
    d.optimizationLevel = MPSGraphOptimizationLevel1;
    d.waitForCompilationCompletion = YES;
    c.exe = [c.graph compileWithDevice:graph_device
                                 feeds:feeds
                         targetTensors:@[ out ]
                      targetOperations:nil
                 compilationDescriptor:d];
    for (MPSGraphTensor* f in c.exe.feedTensors)
        [inputs addObject:buffer(f.shape, f.dataType)];
    c.inputs = inputs;
    c.outputs = @[ buffer(out.shape, out.dataType) ];
    return c;
}

static double run(Case& c, int repeat) {
    // `repeat` encodes of the same graph in one command buffer, so launch overhead amortises.
    std::vector<double> samples;
    for (int trial = 0; trial < 12; ++trial) {
        @autoreleasepool {
            auto cb = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
            auto start = std::chrono::steady_clock::now();
            for (int r = 0; r < repeat; ++r) {
                auto ed = [MPSGraphExecutableExecutionDescriptor new];
                ed.waitUntilCompleted = NO;
                [c.exe encodeToCommandBuffer:cb
                                 inputsArray:c.inputs
                                resultsArray:c.outputs
                         executionDescriptor:ed];
            }
            [cb commit];
            [cb waitUntilCompleted];
            double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start)
                            .count();
            if (trial >= 2)
                samples.push_back(ms / repeat);
        }
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

int main() {
    @autoreleasepool {
        device = MTLCreateSystemDefaultDevice();
        queue = [device newCommandQueue];
        graph_device = [MPSGraphDevice deviceWithMTLDevice:device];
        std::printf("device: %s\n", device.name.UTF8String);
        const int B = 8, T = 128, N = B * T, D = 512, F = 1360, V = 16384;
        auto mm = ^(MPSGraph* g, MPSGraphTensor* a, MPSGraphTensor* b) {
          return [g matrixMultiplicationWithPrimaryTensor:a secondaryTensor:b name:nil];
        };
        auto tr = ^(MPSGraph* g, MPSGraphTensor* a, NSUInteger i, NSUInteger j) {
          return [g transposeTensor:a dimension:i withDimension:j name:nil];
        };
        std::vector<Case> cases;
        for (auto t : {MPSDataTypeFloat16, MPSDataTypeFloat32}) {
            std::string p = t == MPSDataTypeFloat16 ? "f16 " : "f32 ";
            // ---- forward x @ W, feed-forward width ----
            cases.push_back(build(p + "fwd  3D [8,128,512]@[512,1360]", double(N) * D * F, t,
                                  {shp({B, T, D}), shp({D, F})},
                                  [&](MPSGraph* g, NSArray<MPSGraphTensor*>* in) {
                                      return mm(g, in[0], in[1]);
                                  }));
            cases.push_back(build(p + "fwd  2D [1024,512]@[512,1360]", double(N) * D * F, t,
                                  {shp({N, D}), shp({D, F})},
                                  [&](MPSGraph* g, NSArray<MPSGraphTensor*>* in) {
                                      return mm(g, in[0], in[1]);
                                  }));
            // ---- weight gradient x^T @ dy ----
            cases.push_back(build(
                p + "dW   3D sum_b x_b^T@dy_b", double(N) * D * F, t,
                {shp({B, T, D}), shp({B, T, F})}, [&](MPSGraph* g, NSArray<MPSGraphTensor*>* in) {
                    auto per = mm(g, tr(g, in[0], 1, 2), in[1]);
                    return [g reductionSumWithTensor:per axis:0 name:nil];
                }));
            cases.push_back(build(p + "dW   2D [512,1024]@[1024,1360]", double(N) * D * F, t,
                                  {shp({N, D}), shp({N, F})},
                                  [&](MPSGraph* g, NSArray<MPSGraphTensor*>* in) {
                                      return mm(g, tr(g, in[0], 0, 1), in[1]);
                                  }));
            // ---- input gradient dy @ W^T ----
            cases.push_back(build(p + "dx   3D [8,128,1360]@[1360,512]", double(N) * D * F, t,
                                  {shp({B, T, F}), shp({D, F})},
                                  [&](MPSGraph* g, NSArray<MPSGraphTensor*>* in) {
                                      return mm(g, in[0], tr(g, in[1], 0, 1));
                                  }));
            cases.push_back(build(p + "dx   2D [1024,1360]@[1360,512]", double(N) * D * F, t,
                                  {shp({N, F}), shp({D, F})},
                                  [&](MPSGraph* g, NSArray<MPSGraphTensor*>* in) {
                                      return mm(g, in[0], tr(g, in[1], 0, 1));
                                  }));
            // ---- vocabulary head ----
            cases.push_back(build(p + "head 3D [8,128,512]@[16384,512]^T", double(N) * D * V, t,
                                  {shp({B, T, D}), shp({V, D})},
                                  [&](MPSGraph* g, NSArray<MPSGraphTensor*>* in) {
                                      return mm(g, in[0], tr(g, in[1], 0, 1));
                                  }));
            cases.push_back(build(p + "head 2D [1024,512]@[16384,512]^T", double(N) * D * V, t,
                                  {shp({N, D}), shp({V, D})},
                                  [&](MPSGraph* g, NSArray<MPSGraphTensor*>* in) {
                                      return mm(g, in[0], tr(g, in[1], 0, 1));
                                  }));
            cases.push_back(build(p + "head 2D pre-transposed [1024,512]@[512,16384]",
                                  double(N) * D * V, t, {shp({N, D}), shp({D, V})},
                                  [&](MPSGraph* g, NSArray<MPSGraphTensor*>* in) {
                                      return mm(g, in[0], in[1]);
                                  }));
            cases.push_back(build(p + "head dE 2D [16384,1024]@[1024,512]", double(N) * D * V, t,
                                  {shp({N, V}), shp({N, D})},
                                  [&](MPSGraph* g, NSArray<MPSGraphTensor*>* in) {
                                      return mm(g, tr(g, in[0], 0, 1), in[1]);
                                  }));
            cases.push_back(build(
                p + "head dE 3D sum_b dl_b^T@h_b", double(N) * D * V, t,
                {shp({B, T, V}), shp({B, T, D})}, [&](MPSGraph* g, NSArray<MPSGraphTensor*>* in) {
                    auto per = mm(g, tr(g, in[0], 1, 2), in[1]);
                    return [g reductionSumWithTensor:per axis:0 name:nil];
                }));
            // ---- Muon-shaped products ----
            cases.push_back(build(p + "muon X@X^T [30,512,1360]", 30.0 * D * D * F, t,
                                  {shp({30, D, F})},
                                  [&](MPSGraph* g, NSArray<MPSGraphTensor*>* in) {
                                      return mm(g, in[0], tr(g, in[0], 1, 2));
                                  }));
            cases.push_back(build(p + "muon A@A   [30,512,512]", 30.0 * D * D * D, t,
                                  {shp({30, D, D})},
                                  [&](MPSGraph* g, NSArray<MPSGraphTensor*>* in) {
                                      return mm(g, in[0], in[0]);
                                  }));
            cases.push_back(build(p + "muon P@X   [30,512,512]@[30,512,1360]", 30.0 * D * D * F, t,
                                  {shp({30, D, D}), shp({30, D, F})},
                                  [&](MPSGraph* g, NSArray<MPSGraphTensor*>* in) {
                                      return mm(g, in[0], in[1]);
                                  }));
        }
        std::printf("%-52s %9s %10s\n", "case", "ms", "GFLOP/s");
        for (auto& c : cases) {
            double ms = run(c, 4);
            std::printf("%-52s %9.3f %10.0f\n", c.name.c_str(), ms, 2 * c.macs / ms / 1e6);
        }
    }
    return 0;
}
