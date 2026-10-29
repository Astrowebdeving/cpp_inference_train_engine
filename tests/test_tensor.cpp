#include "mini_mla/backend.hpp"
#include "mini_mla/buffer.hpp"
#include "mini_mla/shape.hpp"
#include "mini_mla/tensor.hpp"
#include "mini_mla/testing.hpp"

#include <cstdint>
#include <cstring>
#include <limits>

using namespace mini_mla;

namespace {

template <typename T> [[nodiscard]] bool succeeded(const Result<T>& result, const char* label) {
    if (result) {
        return true;
    }
    std::fprintf(stderr, "       %s failed: %s\n", label, result.error().message.c_str());
    return false;
}

[[nodiscard]] Shape make_shape_or_die(std::initializer_list<std::int64_t> dims) {
    auto shape = Shape::make(std::span<const std::int64_t>{dims.begin(), dims.size()});
    if (!shape) {
        detail::fatal("test setup: Shape::make rejected a shape it should accept");
    }
    return shape.value();
}

} // namespace

MINI_MLA_TEST(shape_make_accepts_valid_ranks) {
    const std::int64_t dims[] = {2, 3, 4};
    auto shape = Shape::make(dims);
    MINI_MLA_REQUIRE(succeeded(shape, "Shape::make"));
    MINI_MLA_CHECK(shape.value().rank == 3);
    MINI_MLA_CHECK(shape.value().dim(0) == 2);
    MINI_MLA_CHECK(shape.value().dim(2) == 4);

    MINI_MLA_CHECK(shape.value().dims[3] == 0);
}

MINI_MLA_TEST(shape_make_rejects_excess_rank) {
    const std::int64_t dims[] = {2, 3, 4, 5, 6};
    auto shape = Shape::make(dims);
    MINI_MLA_CHECK(!shape.has_value());
    if (!shape) {
        MINI_MLA_CHECK(shape.error().code == ErrorCode::invalid_argument);
    }
}

MINI_MLA_TEST(shape_make_rejects_negative_dims) {
    const std::int64_t dims[] = {2, -1};
    auto shape = Shape::make(dims);
    MINI_MLA_CHECK(!shape.has_value());
}

MINI_MLA_TEST(shape_numel_is_the_product_of_dims) {
    auto numel = make_shape_or_die({2, 3, 4}).numel();
    MINI_MLA_REQUIRE(succeeded(numel, "numel"));
    MINI_MLA_CHECK(numel.value() == 24);
}

MINI_MLA_TEST(shape_numel_of_scalar_is_one) {

    auto numel = make_shape_or_die({}).numel();
    MINI_MLA_REQUIRE(succeeded(numel, "numel"));
    MINI_MLA_CHECK(numel.value() == 1);
}

MINI_MLA_TEST(shape_numel_allows_zero_sized_dims) {

    auto numel = make_shape_or_die({4, 0, 8}).numel();
    MINI_MLA_REQUIRE(succeeded(numel, "numel"));
    MINI_MLA_CHECK(numel.value() == 0);
}

MINI_MLA_TEST(shape_numel_detects_overflow) {

    const std::int64_t big = std::int64_t{1} << 40;
    auto numel = make_shape_or_die({big, big}).numel();
    MINI_MLA_CHECK(!numel.has_value());
    if (!numel) {
        MINI_MLA_CHECK(numel.error().code == ErrorCode::invalid_argument);
    }
}

MINI_MLA_TEST(strides_are_row_major) {
    auto strides = contiguous_strides(make_shape_or_die({2, 3, 4}));
    MINI_MLA_REQUIRE(succeeded(strides, "contiguous_strides"));
    MINI_MLA_CHECK(strides.value()[0] == 12);
    MINI_MLA_CHECK(strides.value()[1] == 4);
    MINI_MLA_CHECK(strides.value()[2] == 1);
    MINI_MLA_CHECK(strides.value()[3] == 0);
}

MINI_MLA_TEST(strides_of_rank_one_is_unit) {
    auto strides = contiguous_strides(make_shape_or_die({5}));
    MINI_MLA_REQUIRE(succeeded(strides, "contiguous_strides"));
    MINI_MLA_CHECK(strides.value()[0] == 1);
}

MINI_MLA_TEST(strides_of_rank_zero_does_not_underflow) {

    auto strides = contiguous_strides(make_shape_or_die({}));
    MINI_MLA_REQUIRE(succeeded(strides, "contiguous_strides"));
    for (std::size_t i = 0; i < Shape::max_rank; ++i) {
        MINI_MLA_CHECK(strides.value()[i] == 0);
    }
}

MINI_MLA_TEST(desc_make_contiguous_fills_strides_and_metadata) {
    auto desc =
        TensorDesc::make_contiguous(make_shape_or_die({2, 3, 4}), DType::f32, DeviceType::cpu);
    MINI_MLA_REQUIRE(succeeded(desc, "make_contiguous"));
    MINI_MLA_CHECK(desc.value().strides[0] == 12);
    MINI_MLA_CHECK(desc.value().strides[2] == 1);
    MINI_MLA_CHECK(desc.value().dtype == DType::f32);
    MINI_MLA_CHECK(desc.value().device == DeviceType::cpu);
    MINI_MLA_CHECK(desc.value().byte_offset == 0);
}

MINI_MLA_TEST(desc_make_contiguous_propagates_byte_offset) {
    auto desc =
        TensorDesc::make_contiguous(make_shape_or_die({4}), DType::f32, DeviceType::cpu, 128);
    MINI_MLA_REQUIRE(succeeded(desc, "make_contiguous"));
    MINI_MLA_CHECK(desc.value().byte_offset == 128);
}

MINI_MLA_TEST(required_bytes_of_dense_tensor) {
    auto desc =
        TensorDesc::make_contiguous(make_shape_or_die({2, 3, 4}), DType::f32, DeviceType::cpu);
    MINI_MLA_REQUIRE(succeeded(desc, "make_contiguous"));
    auto bytes = desc.value().required_bytes();
    MINI_MLA_REQUIRE(succeeded(bytes, "required_bytes"));
    MINI_MLA_CHECK(bytes.value() == 24 * 4);
}

MINI_MLA_TEST(required_bytes_includes_byte_offset) {
    auto desc =
        TensorDesc::make_contiguous(make_shape_or_die({4}), DType::f32, DeviceType::cpu, 128);
    MINI_MLA_REQUIRE(succeeded(desc, "make_contiguous"));
    auto bytes = desc.value().required_bytes();
    MINI_MLA_REQUIRE(succeeded(bytes, "required_bytes"));
    MINI_MLA_CHECK(bytes.value() == 128 + 4 * 4);
}

MINI_MLA_TEST(required_bytes_of_strided_view_exceeds_numel_times_itemsize) {

    TensorDesc desc{};
    desc.shape = make_shape_or_die({2, 2});
    desc.strides = {3, 1, 0, 0};
    desc.dtype = DType::f32;
    desc.device = DeviceType::cpu;

    auto bytes = desc.required_bytes();
    MINI_MLA_REQUIRE(succeeded(bytes, "required_bytes"));
    MINI_MLA_CHECK_MSG(bytes.value() == 20,
                       "numel*itemsize would give 16 and overrun by one element");
}

MINI_MLA_TEST(required_bytes_of_empty_tensor_is_just_the_offset) {

    TensorDesc desc{};
    desc.shape = make_shape_or_die({0, 8});
    desc.strides = {8, 1, 0, 0};
    desc.dtype = DType::f32;
    desc.device = DeviceType::cpu;
    desc.byte_offset = 64;

    auto bytes = desc.required_bytes();
    MINI_MLA_REQUIRE(succeeded(bytes, "required_bytes"));
    MINI_MLA_CHECK(bytes.value() == 64);
}

MINI_MLA_TEST(buffer_allocate_returns_usable_memory) {
    auto buffer = Buffer::allocate_cpu(1024);
    MINI_MLA_REQUIRE(succeeded(buffer, "allocate_cpu"));
    MINI_MLA_CHECK(buffer.value().size_bytes() == 1024);
    MINI_MLA_CHECK(!buffer.value().empty());
    MINI_MLA_CHECK(buffer.value().device() == DeviceType::cpu);
    MINI_MLA_CHECK(buffer.value().bytes().size() == 1024);

    std::span<std::byte> span = buffer.value().bytes();
    std::memset(span.data(), 0xAB, span.size());
    MINI_MLA_CHECK(span[0] == std::byte{0xAB});
    MINI_MLA_CHECK(span[1023] == std::byte{0xAB});
}

MINI_MLA_TEST(buffer_honours_requested_alignment) {
    auto buffer = Buffer::allocate_cpu(100, 64);
    MINI_MLA_REQUIRE(succeeded(buffer, "allocate_cpu"));
    const auto address = reinterpret_cast<std::uintptr_t>(buffer.value().bytes().data());
    MINI_MLA_CHECK_MSG(address % 64 == 0, "note: 100 is not a multiple of 64 -- std::aligned_alloc "
                                          "would be undefined here, ::operator new is not");
}

MINI_MLA_TEST(buffer_rejects_bad_alignment) {
    MINI_MLA_CHECK(!Buffer::allocate_cpu(64, 0).has_value());
    MINI_MLA_CHECK(!Buffer::allocate_cpu(64, 3).has_value());
    MINI_MLA_CHECK(!Buffer::allocate_cpu(64, 100).has_value());
}

MINI_MLA_TEST(buffer_zero_bytes_is_a_valid_empty_buffer) {

    auto buffer = Buffer::allocate_cpu(0);
    MINI_MLA_REQUIRE(succeeded(buffer, "allocate_cpu(0)"));
    MINI_MLA_CHECK(buffer.value().empty());
    MINI_MLA_CHECK(buffer.value().size_bytes() == 0);
    MINI_MLA_CHECK(buffer.value().bytes().empty());
}

MINI_MLA_TEST(buffer_default_constructed_is_empty_and_safe) {
    Buffer buffer;
    MINI_MLA_CHECK(buffer.empty());
    MINI_MLA_CHECK(buffer.size_bytes() == 0);
    MINI_MLA_CHECK(buffer.bytes().empty());
    MINI_MLA_CHECK(buffer.bytes().data() == nullptr);
}

MINI_MLA_TEST(buffer_move_transfers_ownership) {
    auto allocated = Buffer::allocate_cpu(256);
    MINI_MLA_REQUIRE(succeeded(allocated, "allocate_cpu"));

    Buffer source = std::move(allocated).value();
    MINI_MLA_CHECK(source.size_bytes() == 256);

    Buffer target = std::move(source);
    MINI_MLA_CHECK(target.size_bytes() == 256);

    MINI_MLA_CHECK(source.empty());
    MINI_MLA_CHECK(source.size_bytes() == 0);
}

MINI_MLA_TEST(buffer_move_assignment_releases_the_old_allocation) {

    auto first = Buffer::allocate_cpu(128);
    auto second = Buffer::allocate_cpu(512);
    MINI_MLA_REQUIRE(succeeded(first, "allocate_cpu"));
    MINI_MLA_REQUIRE(succeeded(second, "allocate_cpu"));

    Buffer target = std::move(first).value();
    target = std::move(second).value();
    MINI_MLA_CHECK(target.size_bytes() == 512);
}

MINI_MLA_TEST(cpu_backend_reports_a_device) {
    auto backend = make_cpu_backend();
    MINI_MLA_REQUIRE(succeeded(backend, "make_cpu_backend"));
    const DeviceInfo info = backend.value()->device_info();
    MINI_MLA_CHECK(info.type == DeviceType::cpu);
    MINI_MLA_CHECK(!info.name.empty());
    MINI_MLA_CHECK(info.total_memory_bytes > 0);
}

MINI_MLA_TEST(cpu_backend_round_trips_data) {
    auto backend = make_cpu_backend();
    MINI_MLA_REQUIRE(succeeded(backend, "make_cpu_backend"));

    auto buffer = backend.value()->allocate(64, BufferUsage::storage);
    MINI_MLA_REQUIRE(succeeded(buffer, "allocate"));

    std::byte written[64];
    for (std::size_t i = 0; i < sizeof(written); ++i) {
        written[i] = static_cast<std::byte>(i);
    }
    MINI_MLA_REQUIRE(succeeded(backend.value()->upload(buffer.value(), written), "upload"));

    std::byte read_back[64] = {};
    MINI_MLA_REQUIRE(succeeded(backend.value()->download(buffer.value(), read_back), "download"));
    MINI_MLA_CHECK(std::memcmp(written, read_back, sizeof(written)) == 0);
}

MINI_MLA_TEST(cpu_backend_rejects_oversized_transfers) {
    auto backend = make_cpu_backend();
    MINI_MLA_REQUIRE(succeeded(backend, "make_cpu_backend"));

    auto buffer = backend.value()->allocate(16, BufferUsage::storage);
    MINI_MLA_REQUIRE(succeeded(buffer, "allocate"));

    std::byte too_much[32] = {};

    MINI_MLA_CHECK(!backend.value()->upload(buffer.value(), too_much).has_value());

    std::byte too_small[8] = {};
    MINI_MLA_CHECK(!backend.value()->download(buffer.value(), too_small).has_value());
}

MINI_MLA_TEST(contiguity_dense_tensor_is_contiguous) {
    auto desc =
        TensorDesc::make_contiguous(make_shape_or_die({2, 3, 4}), DType::f32, DeviceType::cpu);
    MINI_MLA_REQUIRE(succeeded(desc, "make_contiguous"));
    MINI_MLA_CHECK(desc.value().is_contiguous());
}

MINI_MLA_TEST(contiguity_strided_window_is_not_contiguous) {
    TensorDesc desc{};
    desc.shape = make_shape_or_die({2, 2});
    desc.strides = {3, 1, 0, 0};
    desc.dtype = DType::f32;
    MINI_MLA_CHECK(!desc.is_contiguous());
}

MINI_MLA_TEST(contiguity_ignores_extent_one_dimensions) {

    TensorDesc desc{};
    desc.shape = make_shape_or_die({4, 1, 8});
    desc.strides = {8, 999, 1, 0};
    desc.dtype = DType::f32;
    MINI_MLA_CHECK(desc.is_contiguous());
}

MINI_MLA_TEST(contiguity_empty_tensor_is_vacuously_contiguous) {
    TensorDesc desc{};
    desc.shape = make_shape_or_die({0, 8});
    desc.strides = {8, 1, 0, 0};
    desc.dtype = DType::f32;
    MINI_MLA_CHECK(desc.is_contiguous());
}

MINI_MLA_TEST(mutable_views_alias_owned_storage_and_check_layout) {
    auto buffer = Buffer::allocate_cpu(6 * sizeof(float));
    MINI_MLA_REQUIRE(buffer);
    auto desc = TensorDesc::make_contiguous(make_shape_or_die({2, 3}), DType::f32, DeviceType::cpu);
    MINI_MLA_REQUIRE(desc);
    auto tensor = Tensor::create(std::move(buffer).value(), desc.value());
    MINI_MLA_REQUIRE(tensor);
    auto view = tensor.value().mutable_view();
    MINI_MLA_REQUIRE(view);
    auto values = view.value().data<float>();
    MINI_MLA_REQUIRE(values);
    values.value()[4] = 7;
    auto row = view.value().narrow(0, 1, 1);
    MINI_MLA_REQUIRE(row);
    auto row_values = row.value().data<float>();
    MINI_MLA_REQUIRE(row_values);
    MINI_MLA_CHECK(row_values.value()[1] == 7);
    row_values.value()[1] = 9;
    auto read = tensor.value().view();
    MINI_MLA_REQUIRE(read);
    auto read_values = read.value().data<float>();
    MINI_MLA_REQUIRE(read_values);
    MINI_MLA_CHECK(read_values.value()[4] == 9);
    auto transposed = view.value().transpose(0, 1);
    MINI_MLA_REQUIRE(transposed);
    MINI_MLA_CHECK(!transposed.value().data<float>());
    MINI_MLA_CHECK(!view.value().narrow(0, 2, 1));
    std::int64_t dims[] = {3, 2};
    MINI_MLA_REQUIRE(view.value().reshape(dims));
}
MINI_MLA_TEST_MAIN()
