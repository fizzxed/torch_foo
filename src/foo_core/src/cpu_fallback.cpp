#include <ATen/native/CPUFallback.h>
#include <torch/library.h>

namespace foo_core {

void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack)
{
    // Add custom fallback logic here, or warnings that we are falling back to CPU
    at::native::cpu_fallback(op, stack);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&cpu_fallback>());
}

} // namespace foo_core
