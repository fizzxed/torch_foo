#include <c10/util/Exception.h>
#include <torch/library.h>

namespace foo_core {
int register_privateuse1_backend(const std::string &backend_name)
{
    c10::register_privateuse1_backend(backend_name);
    TORCH_WARN("Registering backend:", backend_name);
    return 0;
}

static const int _temp_ = register_privateuse1_backend("foo");
} // namespace foo_core
