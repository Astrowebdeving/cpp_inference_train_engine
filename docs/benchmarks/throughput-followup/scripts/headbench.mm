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
        std::vector<int> Ns{1024, 1100, 1536, 2048};
        for (int N : Ns)
            std::printf(" %8d", N);
        std::printf("\n");
        auto F16 = MPSDataTypeFloat16, F32 = MPSDataTypeFloat32;
        using Builder = std::function<NSArray<T>*(Graph&, int)>;
        // selected log-prob through one-hot softmax cross entropy, exactly as the engine builds it
        auto onehot_loss = [&](Graph& G, T logits, int N) {
            T targets = G.in(shp({N, 1}), MPSDataTypeInt32);
            T coeff = G.in(shp({N, 1}), F32);
            T labels = [G.g oneHotWithIndicesTensor:G.reshape(targets, shp({N}))
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
            return G.sum(G.mul(G.reshape(nll, shp({N, 1})), coeff));
        };
        std::vector<std::pair<std::string, Builder>> cases{
            {"A  loss only: d loss / d logits(f32)",
             [&](Graph& G, int N) -> NSArray<T>* {
                 T logits = G.in(shp({N, V}), F32);
                 T loss = onehot_loss(G, logits, N);
                 auto gr = [G.g gradientForPrimaryTensor:loss withTensors:@[ logits ] name:nil];
                 return @[ gr[logits] ];
             }},
            {"B  cast f32->f16 of [N,V]",
             [&](Graph& G, int N) -> NSArray<T>* { return @[ G.cast(G.in(shp({N, V}), F32), F16) ]; }},
            {"C  dh = dl @ E          [N,V]@[V,512]",
             [&](Graph& G, int N) -> NSArray<T>* {
                 return @[ G.mm(G.in(shp({N, V}), F16), G.in(shp({V, D}), F16)) ];
             }},
            {"D  dEt = h^T @ dl       [512,N]@[N,V]",
             [&](Graph& G, int N) -> NSArray<T>* {
                 return @[ G.mm(G.tr(G.in(shp({N, D}), F16)), G.in(shp({N, V}), F16)) ];
             }},
            {"E  dE  = (h^T @ dl)^T   materialised",
             [&](Graph& G, int N) -> NSArray<T>* {
                 return @[ G.tr(G.mm(G.tr(G.in(shp({N, D}), F16)), G.in(shp({N, V}), F16))) ];
             }},
            {"F  dE  = dl^T @ h       [V,N]@[N,512]",
             [&](Graph& G, int N) -> NSArray<T>* {
                 return @[ G.mm(G.tr(G.in(shp({N, V}), F16)), G.in(shp({N, D}), F16)) ];
             }},
            {"G  head autodiff: grads wrt h and E (f16 mm)",
             [&](Graph& G, int N) -> NSArray<T>* {
                 T h = G.in(shp({N, D}), F16);
                 T E = G.in(shp({V, D}), F32);
                 T logits = G.cast(G.mm(h, G.tr(G.cast(E, F16))), F32);
                 T loss = onehot_loss(G, logits, N);
                 auto gr = [G.g gradientForPrimaryTensor:loss withTensors:@[ h, E ] name:nil];
                 return @[ gr[h], gr[E] ];
             }},
            {"H  head forward only (logits f32)",
             [&](Graph& G, int N) -> NSArray<T>* {
                 T h = G.in(shp({N, D}), F16);
                 T E = G.in(shp({V, D}), F32);
                 return @[ G.cast(G.mm(h, G.tr(G.cast(E, F16))), F32) ];
             }},
            {"I  embedding gather grad (scatter into [V,512])",
             [&](Graph& G, int N) -> NSArray<T>* {
                 T E = G.in(shp({V, D}), F32);
                 T ids = G.in(shp({N}), MPSDataTypeInt32);
                 T up = G.in(shp({N, D}), F32);
                 T x = [G.g gatherWithUpdatesTensor:G.cast(E, F16)
                                      indicesTensor:ids
                                               axis:0
                                    batchDimensions:0
                                               name:nil];
                 T loss = G.sum(G.mul(G.cast(x, F32), up));
                 auto gr = [G.g gradientForPrimaryTensor:loss withTensors:@[ E ] name:nil];
                 return @[ gr[E] ];
             }},
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
