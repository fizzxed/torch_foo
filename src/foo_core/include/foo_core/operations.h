#pragma once

#include <torch/torch.h>

namespace foo_core {

// Add two tensors element-wise
torch::Tensor add(const torch::Tensor& a, const torch::Tensor& b);

// Multiply two tensors element-wise
torch::Tensor multiply(const torch::Tensor& a, const torch::Tensor& b);

}  // namespace foo_core
