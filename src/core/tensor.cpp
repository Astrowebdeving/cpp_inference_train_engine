#include "mini_mla/tensor.hpp"

#include <algorithm>

namespace mini_mla {
Result<TensorView> TensorView::create(const Buffer& b, TensorDesc d) {
    auto bytes = d.required_bytes();
    if (!bytes)
        return Result<TensorView>::failure(bytes.error());
    if (d.device != b.device() || bytes.value() > b.size_bytes())
        return Result<TensorView>::failure(ErrorCode::invalid_argument,
                                           "Tensor descriptor exceeds storage or has wrong device");
    return Result<TensorView>::success(TensorView(b, d));
}
Result<Tensor> Tensor::create(Buffer b, TensorDesc d) {
    auto view = TensorView::create(b, d);
    if (!view)
        return Result<Tensor>::failure(view.error());
    return Result<Tensor>::success(Tensor(std::move(b), d));
}
Result<TensorView> TensorView::reshape(std::span<const std::int64_t> dims) const {
    auto s = Shape::make(dims);
    if (!s)
        return Result<TensorView>::failure(s.error());
    auto before = desc_.shape.numel();
    auto after = s.value().numel();
    if (!before || !after || before.value() != after.value() || !desc_.is_contiguous())
        return Result<TensorView>::failure(
            ErrorCode::invalid_argument,
            "Reshape must preserve element count and contiguous layout");
    auto d = TensorDesc::make_contiguous(s.value(), desc_.dtype, desc_.device, desc_.byte_offset);
    if (!d)
        return Result<TensorView>::failure(d.error());
    return create(*storage_, d.value());
}
Result<TensorView> TensorView::narrow(std::size_t axis, std::int64_t start,
                                      std::int64_t count) const {
    if (axis >= desc_.shape.rank || start < 0 || count < 0 || start > desc_.shape.dims[axis] ||
        count > desc_.shape.dims[axis] - start)
        return Result<TensorView>::failure(ErrorCode::invalid_argument,
                                           "Slice exceeds tensor dimension");
    auto d = desc_;
    std::size_t offset;
    if (__builtin_mul_overflow(static_cast<std::size_t>(start),
                               static_cast<std::size_t>(d.strides[axis]), &offset) ||
        __builtin_mul_overflow(offset, dtype_size_bytes(d.dtype), &offset) ||
        __builtin_add_overflow(d.byte_offset, offset, &d.byte_offset))
        return Result<TensorView>::failure(ErrorCode::invalid_argument, "Slice offset overflows");
    d.shape.dims[axis] = count;
    return create(*storage_, d);
}
Result<TensorView> TensorView::transpose(std::size_t a, std::size_t b) const {
    if (a >= desc_.shape.rank || b >= desc_.shape.rank)
        return Result<TensorView>::failure(ErrorCode::invalid_argument,
                                           "Transpose axis exceeds rank");
    auto d = desc_;
    std::swap(d.shape.dims[a], d.shape.dims[b]);
    std::swap(d.strides[a], d.strides[b]);
    return create(*storage_, d);
}
Result<MutableTensorView> MutableTensorView::create(Buffer& b, TensorDesc d) {
    auto view = TensorView::create(b, d);
    if (!view)
        return Result<MutableTensorView>::failure(view.error());
    return Result<MutableTensorView>::success(MutableTensorView(b, view.value()));
}
Result<MutableTensorView> MutableTensorView::reshape(std::span<const std::int64_t> dims) const {
    auto view = view_.reshape(dims);
    if (!view)
        return Result<MutableTensorView>::failure(view.error());
    return create(*storage_, view.value().desc());
}
Result<MutableTensorView> MutableTensorView::narrow(std::size_t axis, std::int64_t start,
                                                    std::int64_t count) const {
    auto view = view_.narrow(axis, start, count);
    if (!view)
        return Result<MutableTensorView>::failure(view.error());
    return create(*storage_, view.value().desc());
}
Result<MutableTensorView> MutableTensorView::transpose(std::size_t a, std::size_t b) const {
    auto view = view_.transpose(a, b);
    if (!view)
        return Result<MutableTensorView>::failure(view.error());
    return create(*storage_, view.value().desc());
}
} // namespace mini_mla
