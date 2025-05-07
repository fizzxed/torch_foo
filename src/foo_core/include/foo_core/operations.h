#pragma once

#include <torch/torch.h>

namespace foo_core {

at::Tensor mymuladd_cpu(const at::Tensor& a, const at::Tensor& b, double c);

at::Tensor mymul_cpu(const at::Tensor& a, const at::Tensor& b);

void myadd_out_cpu(const at::Tensor& a, const at::Tensor& b, at::Tensor& out);

// Add two tensors element-wise
torch::Tensor add(const torch::Tensor& a, const torch::Tensor& b);

// Multiply two tensors element-wise
torch::Tensor multiply(const torch::Tensor& a, const torch::Tensor& b);

}  // namespace foo_core
