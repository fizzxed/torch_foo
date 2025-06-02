#include <ATen/DeviceGuard.h>
#include <ATen/EmptyTensor.h>
#include <c10/core/Allocator.h>
#include <c10/core/Device.h>
#include <c10/core/DeviceGuard.h>
#include <c10/core/DispatchKey.h>
#include <c10/core/DispatchKeySet.h>
#include <c10/core/TensorOptions.h>
#include <iostream>
#include <torch/library.h>
#include <torch/torch.h>



namespace foo_core {

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

at::Tensor custom_empty_memory_format(at::IntArrayRef size, c10::optional<at::ScalarType> dtype, c10::optional<at::Layout> layout, c10::optional<at::Device> device, c10::optional<bool> pin_memory, c10::optional<at::MemoryFormat> memory_format)
{
    const c10::OptionalDeviceGuard guard(device);
    std::cout << "Custom aten::empty.memory_format() called!" << std::endl;
    auto allocator = c10::GetAllocator(c10::DeviceType::PrivateUse1); // Will get the global_dummy_allocator we registered
    constexpr c10::DispatchKeySet private_use_ks(c10::DispatchKey::PrivateUse1);
    return at::detail::empty_generic(size, allocator, private_use_ks, c10::dtype_or_default(dtype), memory_format);
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
    TORCH_INTERNAL_ASSERT(self_.device().type() == at::DeviceType::CPU); // These device guards will become different once we implement PrivateUse1 device guards
    TORCH_INTERNAL_ASSERT(other_.device().type() == at::DeviceType::CPU);
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
    m.impl("add.Tensor", TORCH_FN(custom_add));
    m.impl("empty.memory_format", TORCH_FN(custom_empty_memory_format));
    m.impl("fill_.Scalar", &custom_fill__scalar);
    m.impl("_copy_from", TORCH_FN(custom__copy_from));
}

} // namespace foo_core
