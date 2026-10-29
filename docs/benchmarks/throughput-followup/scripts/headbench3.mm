// Isolate the head/loss/embedding backward cliff. Each case is one autodiff subgraph.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

static id<MTLDevice> device;
static id<MTLCommandQueue> queue;
static MPSGraphDevice* graph_device;
typedef MPSGraphTensor* T;

static MPSShape* shp(std::initializer_list<int> d) {
    auto a = [NSMutableArray<NSNumber*> new];
    for (int n : d)
        [a addObject:@(n)];
    return a;
}
static MPSGraphTensorData* buffer(MPSShape* s, MPSDataType t, int modulo) {
    std::size_t count = 1;
    for (NSNumber* d in s)
        count *= d.unsignedLongValue;
    std::size_t item = t == MPSDataTypeFloat16 ? 2 : 4;
    auto b = [device newBufferWithLength:count * item options:MTLResourceStorageModeShared];
    if (t == MPSDataTypeFloat16) {
        auto* p = static_cast<uint16_t*>(b.contents);
        for (std::size_t i = 0; i < count; ++i)
            p[i] = 0x2000 + (i * 2654435761u) % 0x1000;
    } else if (t == MPSDataTypeInt32) {
        auto* p = static_cast<int32_t*>(b.contents);
        for (std::size_t i = 0; i < count; ++i)
            p[i] = static_cast<int32_t>((i * 2654435761u) % static_cast<unsigned>(modulo));
    } else {
        auto* p = static_cast<float*>(b.contents);
        for (std::size_t i = 0; i < count; ++i)
            p[i] = 0.001f * static_cast<float>((i * 2654435761u) % 1000);
    }
    return [[MPSGraphTensorData alloc] initWithMTLBuffer:b shape:s dataType:t];
}

struct Graph {
    MPSGraph* g = [MPSGraph new];
    NSMutableArray<T>* feeds = [NSMutableArray new];
    T in(MPSShape* s, MPSDataType t) {
        auto p = [g placeholderWithShape:s dataType:t name:nil];
        [feeds addObject:p];
        return p;
    }
    T mm(T a, T b) { return [g matrixMultiplicationWithPrimaryTensor:a secondaryTensor:b name:nil]; }
    T tr(T a) { return [g transposeTensor:a dimension:0 withDimension:1 name:nil]; }
    T cast(T a, MPSDataType t) { return a.dataType == t ? a : [g castTensor:a toType:t name:nil]; }
    T mul(T a, T b) { return [g multiplicationWithPrimaryTensor:a secondaryTensor:b name:nil]; }
    T sum(T a) { return [g reductionSumWithTensor:a axes:nil name:nil]; }
    T reshape(T a, MPSShape* s) { return [g reshapeTensor:a withShape:s name:nil]; }
};

static double timeit(Graph& G, NSArray<T>* targets, int V) {
    auto feeds = [NSMutableDictionary<T, MPSGraphShapedType*> new];
    for (T f in G.feeds)
        feeds[f] = [[MPSGraphShapedType alloc] initWithShape:f.shape dataType:f.dataType];
    auto d = [MPSGraphCompilationDescriptor new];
    d.optimizationLevel = MPSGraphOptimizationLevel1;
    d.waitForCompilationCompletion = YES;
    auto exe = [G.g compileWithDevice:graph_device
                                feeds:feeds
                        targetTensors:targets
                     targetOperations:nil
                compilationDescriptor:d];
    auto inputs = [NSMutableArray<MPSGraphTensorData*> new];
    for (T f in exe.feedTensors)
        [inputs addObject:buffer(f.shape, f.dataType, V)];
    auto outputs = [NSMutableArray<MPSGraphTensorData*> new];
    for (T t in exe.targetTensors)
        [outputs addObject:buffer(t.shape, t.dataType, V)];
    std::vector<double> samples;
    for (int trial = 0; trial < 9; ++trial) {
        @autoreleasepool {
            auto cb = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
            auto start = std::chrono::steady_clock::now();
            auto ed = [MPSGraphExecutableExecutionDescriptor new];
            ed.waitUntilCompleted = NO;
            [exe encodeToCommandBuffer:cb inputsArray:inputs resultsArray:outputs executionDescriptor:ed];
            [cb commit];
            [cb waitUntilCompleted];
            double ms = std::chrono::duration<double, std::milli>(
                            std::chrono::steady_clock::now() - start)
                            .count();
            if (trial >= 2)
                samples.push_back(ms);
        }
    }
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

int main(int argc, char** argv) {
    @autoreleasepool {
        device = MTLCreateSystemDefaultDevice();
        queue = [device newCommandQueue];
        graph_device = [MPSGraphDevice deviceWithMTLDevice:device];
        const int D = 512, V = argc > 1 ? atoi(argv[1]) : 16384;
        std::printf("%-46s", "case \\ N");
        std::vector<int> Ns{512, 1024, 1152, 2048, 4096, 8192};
        for (int N : Ns)
            std::printf(" %8d", N);
        std::printf("\n");
        auto F16 = MPSDataTypeFloat16, F32 = MPSDataTypeFloat32;
        using Builder = std::function<NSArray<T>*(Graph&, int)>;
        auto ce = [&](Graph& G, T logits, T targets, T coeff, int n) {
            T labels = [G.g oneHotWithIndicesTensor:G.reshape(targets, shp({n}))
                                              depth:static_cast<NSUInteger>(V)
                                               axis:1
                                           dataType:F32
                                            onValue:1
                                           offValue:0
                                               name:nil];
            T nll = [G.g softMaxCrossEntropyWithSourceTensor:logits
                                                labelsTensor:labels
                                                        axis:1
                                               reductionType:MPSGraphLossReductionTypeNone
                                                        name:nil];
            return G.sum(G.mul(G.reshape(nll, shp({n, 1})), coeff));
        };
        auto slice = [&](Graph& G, T x, int begin, int count) {
            return [G.g sliceTensor:x dimension:0 start:begin length:count name:nil];
        };
        // whole head, autodiff, optionally split into row chunks before the vocabulary product
        auto head = [&](int chunk, bool per_chunk_loss, bool wide = false) {
            return [&, chunk, per_chunk_loss, wide](Graph& G, int N) -> NSArray<T>* {
                T h = G.in(shp({N, D}), F16);
                T E = G.in(shp({V, D}), F32);
                T targets = G.in(shp({N, 1}), MPSDataTypeInt32);
                T coeff = G.in(shp({N, 1}), F32);
                T Et = G.tr(wide ? E : G.cast(E, F16));
                if (wide)
                    h = G.cast(h, F32);
                T loss = nil;
                if (chunk <= 0 || N <= chunk) {
                    loss = ce(G, G.cast(G.mm(h, Et), F32), targets, coeff, N);
                } else if (per_chunk_loss) {
                    for (int begin = 0; begin < N; begin += chunk) {
                        int n = std::min(chunk, N - begin);
                        T part = ce(G, G.cast(G.mm(slice(G, h, begin, n), Et), F32),
                                    slice(G, targets, begin, n), slice(G, coeff, begin, n), n);
                        loss = loss ? [G.g additionWithPrimaryTensor:loss secondaryTensor:part name:nil] : part;
                    }
                } else {
                    auto parts = [NSMutableArray<T> new];
                    for (int begin = 0; begin < N; begin += chunk)
                        [parts addObject:G.mm(slice(G, h, begin, std::min(chunk, N - begin)), Et)];
                    loss = ce(G, G.cast([G.g concatTensors:parts dimension:0 name:nil], F32), targets, coeff, N);
                }
                auto gr = [G.g gradientForPrimaryTensor:loss withTensors:@[ h, E ] name:nil];
                return @[ gr[h], gr[E], loss ];
            };
        };
        std::vector<std::pair<std::string, Builder>> cases{
            {"G   flattened f16 head, unchunked", head(0, false)},
            {"G3  f16 head,  512-row chunks", head(512, true)},
            {"G6  f16 head,  256-row chunks", head(256, true)},
            {"G7  f16 head,  768-row chunks", head(768, true)},
            {"G4  f32 head, unchunked", head(0, false, true)},
            {"G5  f32 head, 1024-row chunks", head(1024, true, true)},
            {"G8  f32 head, 2048-row chunks", head(2048, true, true)},
        };
        for (auto& [name, build] : cases) {
            std::printf("%-46s", name.c_str());
            for (int N : Ns) {
                Graph G;
                auto targets = build(G, N);
                std::printf(" %8.2f", timeit(G, targets, V));
                std::fflush(stdout);
            }
            std::printf("\n");
        }
    }
    return 0;
}
