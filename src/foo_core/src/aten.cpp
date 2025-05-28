#include <torch/library.h>
#include <torch/torch.h>

namespace foo_core {

// Adds two tensors (A + (B * 2)), ignore alpha for now.
at::Tensor custom_add(const at::Tensor& self_, const at::Tensor& other_, const at::Scalar & alpha) {
    TORCH_WARN("Calling custom_add")
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
}

} // namespace foo_core
