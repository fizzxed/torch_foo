#include <ATen/DeviceGuard.h>
#include <ATen/EmptyTensor.h>
#include <ATen/core/TensorBody.h>
#include <ATen/ops/_to_copy_native.h>
#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/DeviceType.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/Layout.h>
#include <c10/core/ScalarType.h>
#include <c10/core/TensorOptions.h>
#include <c10/util/ArrayRef.h>
#include <cstring>
#include <iostream>
#include <optional>
#include <torch/library.h>
#include <torch/torch.h>

#include "DummyDeviceGuard.h"

namespace foo_core {

inline bool is_foo(const at::Tensor& tensor)
{
    return tensor.is_privateuseone();
}

// =====================================
// ============= KERNELS ===============
// =====================================

// basic dummy empty function, so we can directly construct tensors on the custom device
// This dummy test device will just use the CPU allocator, and ignores pinned memory.
//
// Note: this kernel is very simple because our "custom device" just uses the normal TensorImpl object
// to store data under the hood.
// In PyTorch core today, both cpu and cuda are implemented with an ordinary TensorImpl class.
// Sometimes, backends prefer to subclass TensorImpl in order to store extra information.
// If this is the case, then this kernel is where you'll be responsible for creating and returning
// a fresh at::Tensor object, that properly stores a TensorImpl of your subclass.

// Empty Tensor Factories
at::Tensor custom_empty_memory_format(c10::IntArrayRef size, std::optional<c10::ScalarType> dtype_opt, std::optional<c10::Layout> layout_opt, std::optional<c10::Device> device_opt, std::optional<bool> pin_memory_opt, std::optional<c10::MemoryFormat> memory_format_opt)
{
    const c10::ScalarType dtype = c10::dtype_or_default(dtype_opt);
    const c10::Device device = c10::device_or_default(device_opt);
    TORCH_CHECK(device.is_privateuseone());
    TORCH_CHECK(
        c10::layout_or_default(layout_opt) == c10::Layout::Strided,
        "Non strided layout not supported"
    )
    TORCH_CHECK(
        !c10::pinned_memory_or_default(pin_memory_opt),
        "Pin memory can only be on CPU"
    )
    const DummyDeviceGuard guard(device); // Example of using our specialized device guard
    auto allocator = c10::GetAllocator(c10::DeviceType::PrivateUse1); // Will get the global_dummy_allocator we registered
    constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
    std::cout << "Custom aten::empty.memory_format() called!" << std::endl;
    return at::detail::empty_generic(size, allocator, private_use_ks, c10::dtype_or_default(dtype), memory_format_opt);
}

at::Tensor custom_empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride, std::optional<c10::ScalarType> dtype_opt, std::optional<c10::Layout> layout_opt, std::optional<c10::Device> device_opt, std::optional<bool> pin_memory_opt)
{
    const c10::ScalarType dtype = c10::dtype_or_default(dtype_opt);
    const c10::Device device = c10::device_or_default(device_opt);
    TORCH_CHECK(device.is_privateuseone());
    TORCH_CHECK(
        c10::layout_or_default(layout_opt) == c10::Layout::Strided,
        "Non strided layout not supported");
    TORCH_CHECK(
        !c10::pinned_memory_or_default(pin_memory_opt),
        "Pin memory can only be on CPU");
    const DummyDeviceGuard guard(device);
    constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
    auto allocator = c10::GetAllocator(c10::DeviceType::PrivateUse1);
    std::cout << "Custom aten::empty.strided() called!" << std::endl;
    return at::detail::empty_strided_generic(size, stride, allocator, private_use_ks, dtype);
}

at::Tensor& custom_copy_(at::Tensor& self, const at::Tensor& src, bool non_blocking)
{
    const DummyDeviceGuard guard(self.device());
    std::cout << "Custom aten::copy_() called!" << std::endl;
    TORCH_CHECK(is_foo(self), "self must be on a foo device to dispatch here");

    if (self.numel() == 0) {
        return self;
    }
    // Secretly Just perform the CPU copy
    return at::native::copy_(self, src, non_blocking);

}

at::Tensor custom__to_copy(const at::Tensor& self, std::optional<c10::ScalarType> dtype_opt, std::optional<c10::Layout> layout_opt, std::optional<c10::Device> device_opt, std::optional<bool> pin_memory_opt, bool non_blocking, std::optional<c10::MemoryFormat> memory_format_opt)
{
    const c10::ScalarType dtype = c10::dtype_or_default(dtype_opt);
    const c10::Device device = c10::device_or_default(device_opt);

    std::cout << "Custom aten::_to_copy() called!" << std::endl;
    if (is_foo(self) && device.type() == c10::DeviceType::CPU) {
        // Foo -> CPU
        auto cpu_tensor = at::empty_like(self, self.options().device(c10::DeviceType::CPU));
        cpu_tensor.copy_(self);
        return cpu_tensor;

    } else if (self.is_cpu() && device.is_privateuseone()) {
        // CPU -> Foo
        auto cpu_copy = self.to(self.options().device(at::kCPU), non_blocking, true, memory_format_opt);
        auto copy = custom_empty_strided(self.sizes(), self.strides(), dtype_opt, layout_opt, device_opt, pin_memory_opt);
        std::memcpy(copy.storage().data_ptr().get(), self.storage().data_ptr().get(), self.storage().nbytes());
        return copy;
    } else {
        // Unsupported
        TORCH_CHECK(false, "Unsupported");
        return self;
    }
}

at::Tensor custom__copy_from(const at::Tensor& self, const at::Tensor& dst, bool non_blocking)
{
    const at::OptionalDeviceGuard guard(at::device_of(self));
    std::cout << "Custom aten::_copy_from() called!" << std::endl;
    TORCH_CHECK(self.is_cpu() || self.device().type() == c10::DeviceType::PrivateUse1, "Dummy test only allows copy from cpu -> dummy device.");
    TORCH_CHECK(dst.is_cpu() || dst.device().type() == c10::DeviceType::PrivateUse1, "Dummy test only allows copy from cpu -> dummy device.");
    // Some dummy asserts for the basic use case: inputs are the same size / dtype, all contiguous.
    TORCH_CHECK(self.sizes() == dst.sizes());
    TORCH_CHECK(self.scalar_type() == dst.scalar_type());
    TORCH_CHECK(self.is_contiguous() && dst.is_contiguous());

    std::memcpy(dst.storage().data_ptr().get(), self.storage().data_ptr().get(), self.storage().nbytes());
    return dst;
}

at::Tensor & custom_fill__scalar(at::Tensor & self, const at::Scalar & value) {
  const at::OptionalDeviceGuard device_guard(at::device_of(self));
  std::cout << "Custom aten::fill_.Scalar() called!" << std::endl;
  // Not bothering to implement.
  // Should fill the tensor's data with "value".
  return self;
}

// Adds two tensors (A + (B * 2)), ignore alpha for now.
at::Tensor custom_add(const at::Tensor& self_, const at::Tensor& other_, const at::Scalar & alpha) {
    const at::OptionalDeviceGuard guard(at::device_of(self_));
    std::cout << "Custom aten::add.Tensor() called!" << std::endl;
    TORCH_CHECK(self_.sizes() == other_.sizes());
    TORCH_INTERNAL_ASSERT(self_.device().type() == c10::DeviceType::PrivateUse1); // These device guards will become different once we implement PrivateUse1 device guards
    TORCH_INTERNAL_ASSERT(other_.device().type() == c10::DeviceType::PrivateUse1);
    at::Tensor self = self_.contiguous();
    at::Tensor other = other_.contiguous();
    at::Tensor result = torch::empty(self.sizes(), self.options());
    const float* self_ptr = self.data_ptr<float>();
    const float* other_ptr = other.data_ptr<float>();
    float* result_ptr = result.data_ptr<float>();
    for (int64_t i = 0; i < result.numel(); i++) {
    result_ptr[i] = self_ptr[i] + other_ptr[i] * 2;
    }
    return result;
}

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    // m.impl("add.Tensor", TORCH_FN(custom_add));
    m.impl("empty.memory_format", TORCH_FN(custom_empty_memory_format));
    m.impl("empty_strided", TORCH_FN(custom_empty_strided));
    m.impl("copy_", TORCH_FN(custom_copy_));
    m.impl("_to_copy", TORCH_FN(custom__to_copy));

    m.impl("fill_.Scalar", TORCH_FN(custom_fill__scalar));
    // m.impl("_copy_from", TORCH_FN(custom__copy_from));
}

} // namespace foo_core
