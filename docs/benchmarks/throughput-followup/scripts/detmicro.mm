// Which MPSGraph operations give different bits in different processes? Prints one hash per case.
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>
typedef MPSGraphTensor* T;
static id<MTLDevice> device; static id<MTLCommandQueue> queue; static MPSGraphDevice* gd;
static MPSShape* shp(std::initializer_list<int> d) { auto a = [NSMutableArray<NSNumber*> new]; for (int n : d) [a addObject:@(n)]; return a; }
static MPSGraphTensorData* fill(MPSShape* s, MPSDataType t, int modulo, uint32_t seed) {
    std::size_t count = 1; for (NSNumber* d in s) count *= d.unsignedLongValue;
    std::size_t item = t == MPSDataTypeFloat16 ? 2 : 4;
    auto b = [device newBufferWithLength:count * item options:MTLResourceStorageModeShared];
    uint32_t x = seed;
    auto next = [&] { x = x * 1664525u + 1013904223u; return x >> 8; };
    if (t == MPSDataTypeInt32) { auto* p = (int32_t*)b.contents; for (std::size_t i = 0; i < count; ++i) p[i] = (int32_t)(next() % (uint32_t)modulo); }
    else if (t == MPSDataTypeFloat16) { auto* p = (__fp16*)b.contents; for (std::size_t i = 0; i < count; ++i) p[i] = (__fp16)(((float)(next() % 2001) - 1000.0f) * 0.0002f); }
    else { auto* p = (float*)b.contents; for (std::size_t i = 0; i < count; ++i) p[i] = ((float)(next() % 2001) - 1000.0f) * 0.0002f; }
    return [[MPSGraphTensorData alloc] initWithMTLBuffer:b shape:s dataType:t];
}
struct G { MPSGraph* g = [MPSGraph new]; NSMutableArray<T>* feeds = [NSMutableArray new];
    T in(MPSShape* s, MPSDataType t) { auto p = [g placeholderWithShape:s dataType:t name:nil]; [feeds addObject:p]; return p; }
    T mm(T a, T b) { return [g matrixMultiplicationWithPrimaryTensor:a secondaryTensor:b name:nil]; }
    T tr(T a) { return [g transposeTensor:a dimension:0 withDimension:1 name:nil]; }
    T cast(T a, MPSDataType t) { return a.dataType == t ? a : [g castTensor:a toType:t name:nil]; } };
static uint64_t run(G& G_, NSArray<T>* targets, int V, int level) {
    auto feeds = [NSMutableDictionary<T, MPSGraphShapedType*> new];
    for (T f in G_.feeds) feeds[f] = [[MPSGraphShapedType alloc] initWithShape:f.shape dataType:f.dataType];
    auto d = [MPSGraphCompilationDescriptor new];
    d.optimizationLevel = level ? MPSGraphOptimizationLevel1 : MPSGraphOptimizationLevel0; d.waitForCompilationCompletion = YES;
    auto exe = [G_.g compileWithDevice:gd feeds:feeds targetTensors:targets targetOperations:nil compilationDescriptor:d];
    auto in = [NSMutableArray<MPSGraphTensorData*> new]; uint32_t seed = 12345;
    for (T f in exe.feedTensors) [in addObject:fill(f.shape, f.dataType, V, seed += 77)];
    auto out = [NSMutableArray<MPSGraphTensorData*> new]; std::vector<id<MTLBuffer>> bufs;
    for (T t in exe.targetTensors) { std::size_t count = 1; for (NSNumber* dd in t.shape) count *= dd.unsignedLongValue;
        auto b = [device newBufferWithLength:count * (t.dataType == MPSDataTypeFloat16 ? 2 : 4) options:MTLResourceStorageModeShared]; bufs.push_back(b);
        [out addObject:[[MPSGraphTensorData alloc] initWithMTLBuffer:b shape:t.shape dataType:t.dataType]]; }
    auto cb = [MPSCommandBuffer commandBufferFromCommandQueue:queue];
    auto ed = [MPSGraphExecutableExecutionDescriptor new]; ed.waitUntilCompleted = NO;
    [exe encodeToCommandBuffer:cb inputsArray:in resultsArray:out executionDescriptor:ed];
    [cb commit]; [cb waitUntilCompleted];
    uint64_t h = 1469598103934665603ull;
    for (auto b : bufs) { auto* p = (const uint8_t*)b.contents; for (NSUInteger i = 0; i < b.length; ++i) { h ^= p[i]; h *= 1099511628211ull; } }
    return h;
}
int main(int argc, char** argv) {
    @autoreleasepool {
        device = MTLCreateSystemDefaultDevice(); queue = [device newCommandQueue]; gd = [MPSGraphDevice deviceWithMTLDevice:device];
        int level = argc > 1 ? atoi(argv[1]) : 1;
        const int N = 256, D = 512, V = 16384, F = 1360;
        auto F16 = MPSDataTypeFloat16, F32 = MPSDataTypeFloat32;
        using B = std::function<NSArray<T>*(G&)>;
        auto ce = [&](G& g, T logits) {
            T targets = g.in(shp({N, 1}), MPSDataTypeInt32); T coeff = g.in(shp({N, 1}), F32);
            T labels = [g.g oneHotWithIndicesTensor:[g.g reshapeTensor:targets withShape:shp({N}) name:nil] depth:(NSUInteger)V axis:1 dataType:F32 onValue:1 offValue:0 name:nil];
            T nll = [g.g softMaxCrossEntropyWithSourceTensor:logits labelsTensor:labels axis:1 reductionType:MPSGraphLossReductionTypeNone name:nil];
            return [g.g reductionSumWithTensor:[g.g multiplicationWithPrimaryTensor:[g.g reshapeTensor:nll withShape:shp({N, 1}) name:nil] secondaryTensor:coeff name:nil] axes:nil name:nil]; };
        std::vector<std::pair<std::string, B>> cases{
            {"f32 head forward  h@E^T", [&](G& g) -> NSArray<T>* { return @[ g.mm(g.in(shp({N, D}), F32), g.tr(g.in(shp({V, D}), F32))) ]; }},
            {"f16 head forward  h@E^T", [&](G& g) -> NSArray<T>* { return @[ g.mm(g.in(shp({N, D}), F16), g.tr(g.in(shp({V, D}), F16))) ]; }},
            {"f32 dh = dl@E", [&](G& g) -> NSArray<T>* { return @[ g.mm(g.in(shp({N, V}), F32), g.in(shp({V, D}), F32)) ]; }},
            {"f16 dh = dl@E", [&](G& g) -> NSArray<T>* { return @[ g.mm(g.in(shp({N, V}), F16), g.in(shp({V, D}), F16)) ]; }},
            {"f32 dE = h^T@dl", [&](G& g) -> NSArray<T>* { return @[ g.mm(g.tr(g.in(shp({N, D}), F32)), g.in(shp({N, V}), F32)) ]; }},
            {"f16 dE = h^T@dl", [&](G& g) -> NSArray<T>* { return @[ g.mm(g.tr(g.in(shp({N, D}), F16)), g.in(shp({N, V}), F16)) ]; }},
            {"f32 layer x@W [N,512]@[512,1360]", [&](G& g) -> NSArray<T>* { return @[ g.mm(g.in(shp({N, D}), F32), g.in(shp({D, F}), F32)) ]; }},
            {"f32 layer dW x^T@dy", [&](G& g) -> NSArray<T>* { return @[ g.mm(g.tr(g.in(shp({N, D}), F32)), g.in(shp({N, F}), F32)) ]; }},
            {"f32 loss + d loss/d logits", [&](G& g) -> NSArray<T>* { T l = g.in(shp({N, V}), F32); T loss = ce(g, l); auto gr = [g.g gradientForPrimaryTensor:loss withTensors:@[ l ] name:nil]; return @[ loss, gr[l] ]; }},
            {"f32 head autodiff (h,E grads)", [&](G& g) -> NSArray<T>* { T h = g.in(shp({N, D}), F16); T E = g.in(shp({V, D}), F32);
                T loss = ce(g, g.mm(g.cast(h, F32), g.tr(E))); auto gr = [g.g gradientForPrimaryTensor:loss withTensors:@[ h, E ] name:nil]; return @[ loss, gr[h], gr[E] ]; }},
            {"f16 head autodiff (h,E grads)", [&](G& g) -> NSArray<T>* { T h = g.in(shp({N, D}), F16); T E = g.in(shp({V, D}), F32);
                T loss = ce(g, g.cast(g.mm(h, g.tr(g.cast(E, F16))), F32)); auto gr = [g.g gradientForPrimaryTensor:loss withTensors:@[ h, E ] name:nil]; return @[ loss, gr[h], gr[E] ]; }},
        };
        for (auto& [name, build] : cases) { G g; auto t = build(g); std::printf("%-36s %016llx\n", name.c_str(), (unsigned long long)run(g, t, V, level)); }
    }
    return 0;
}
