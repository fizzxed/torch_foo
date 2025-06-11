#include <ATen/native/CPUFallback.h>
#include <iostream>
#include <string>
#include <torch/library.h>
#include <unordered_set>

namespace foo_core {

bool has_op_name_warned(const std::string& op_name)
{
    static std::unordered_set<std::string> _op_lists = {};
   if (_op_lists.find(op_name) != _op_lists.end()) {
       return true;
   }
   _op_lists.insert(op_name);
   return false;
}

void cpu_fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack)
{
    // Add custom fallback logic here, or warnings that we are falling back to CPU
    if (!has_op_name_warned(c10::toString(op.schema().operator_name()))) {
        std::cout << "Falling back to CPU for operator: " << op.operator_name().name << std::endl;
        // TORCH_WARN("CAUTION: The operator '",
        //             op.schema().operator_name(),
        //             "' is not currently supported ",
        //             "on the backend and will fall back to run on the CPU.",
        //             " This may have performance implications.");
    }

    at::native::cpu_fallback(op, stack);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFromBoxedFunction<&cpu_fallback>());
}

} // namespace foo_core
