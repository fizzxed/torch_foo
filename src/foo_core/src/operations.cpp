#include "foo_core/operations.h"

namespace foo_core {

torch::Tensor add(const torch::Tensor& a, const torch::Tensor& b) {
    return a + b;
}

torch::Tensor multiply(const torch::Tensor& a, const torch::Tensor& b) {
    return a * b;
}

}  // namespace foo_core
