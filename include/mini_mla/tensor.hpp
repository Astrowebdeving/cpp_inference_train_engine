#pragma once
#include "mini_mla/buffer.hpp"
#include "mini_mla/shape.hpp"

namespace mini_mla {
class TensorView {
  public:
    [[nodiscard]] static Result<TensorView> create(const Buffer&, TensorDesc);
    [[nodiscard]] const TensorDesc& desc() const noexcept { return desc_; }
    [[nodiscard]] const Buffer& buffer() const noexcept { return *storage_; }
    [[nodiscard]] Result<TensorView> reshape(std::span<const std::int64_t>) const;
    [[nodiscard]] Result<TensorView> narrow(std::size_t axis, std::int64_t start,
                                            std::int64_t count) const;
    [[nodiscard]] Result<TensorView> transpose(std::size_t a, std::size_t b) const;
    template <class T> [[nodiscard]] Result<std::span<const T>> data() const {
        if (desc_.device != DeviceType::cpu || desc_.dtype != dtype_of<T> || !desc_.is_contiguous())
            return Result<std::span<const T>>::failure(
                ErrorCode::invalid_argument,
                "Typed access requires matching contiguous CPU storage");
        auto n = desc_.shape.numel();
        if (!n)
            return Result<std::span<const T>>::failure(n.error());
        if (n.value() == 0)
            return Result<std::span<const T>>::success({});
        return Result<std::span<const T>>::success(
            {reinterpret_cast<const T*>(storage_->bytes().data() + desc_.byte_offset), n.value()});
    }

  private:
    TensorView(const Buffer& b, TensorDesc d) : storage_(&b), desc_(d) {}
    const Buffer* storage_;
    TensorDesc desc_;
};

class MutableTensorView {
  public:
    [[nodiscard]] static Result<MutableTensorView> create(Buffer&, TensorDesc);
    [[nodiscard]] const TensorDesc& desc() const noexcept { return view_.desc(); }
    [[nodiscard]] const TensorView& read_only() const noexcept { return view_; }
    [[nodiscard]] Result<MutableTensorView> reshape(std::span<const std::int64_t>) const;
    [[nodiscard]] Result<MutableTensorView> narrow(std::size_t axis, std::int64_t start,
                                                   std::int64_t count) const;
    [[nodiscard]] Result<MutableTensorView> transpose(std::size_t a, std::size_t b) const;
    template <class T> [[nodiscard]] Result<std::span<T>> data() const {
        auto checked = view_.data<T>();
        if (!checked)
            return Result<std::span<T>>::failure(checked.error());
        if (checked.value().empty())
            return Result<std::span<T>>::success({});
        return Result<std::span<T>>::success(
            {reinterpret_cast<T*>(storage_->bytes().data() + desc().byte_offset),
             checked.value().size()});
    }

  private:
    MutableTensorView(Buffer& b, TensorView v) : storage_(&b), view_(std::move(v)) {}
    Buffer* storage_;
    TensorView view_;
};

class Tensor {
  public:
    [[nodiscard]] static Result<Tensor> create(Buffer, TensorDesc);
    Tensor(Tensor&&) noexcept = default;
    Tensor& operator=(Tensor&&) noexcept = default;
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;
    [[nodiscard]] const TensorDesc& desc() const noexcept { return desc_; }
    [[nodiscard]] Buffer& buffer() noexcept { return storage_; }
    [[nodiscard]] const Buffer& buffer() const noexcept { return storage_; }
    [[nodiscard]] Result<TensorView> view() const { return TensorView::create(storage_, desc_); }
    [[nodiscard]] Result<MutableTensorView> mutable_view() {
        return MutableTensorView::create(storage_, desc_);
    }

  private:
    Tensor(Buffer b, TensorDesc d) : storage_(std::move(b)), desc_(d) {}
    Buffer storage_;
    TensorDesc desc_;
};
} // namespace mini_mla
