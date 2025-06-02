/** \file bindings.cpp
* \brief Creates a python module that includes registration and implementation of operations and backends from C++
*
* The C++ core contains the implementation and registration of operations and backends that we
* want to extend PyTorch with. To use these operations and backends, we must load the C++ shared
* library (e.g. the `.so` files on Linux or `.dll` files on Windows) so that the static initializers'
* (e.g. `TORCH_LIBRARY` and `TORCH_LIBRARY_IMPL`) registration code runs.
*
* There are a few ways of accomplishing this.
*   1. We define a Python module that will load the shared library upon import. This can be done in at least two ways:
*       - We could use Pybind11 which is what we do here. Unfortunately we lose the ability to compile against the Python
*         limited API.
*       - We could use the stable C API from Python to create the module. This would be preferable but once we need
          to expose additional Python bindings or go between torch.Tensor in python and at::Tensor in C++, using pybind11
          along with torch_python's pybind utilities becomes nearly necessary.
* ```
* #include <Python.h>
* PyMODINIT_FUNC PyInit__C(void) {
*   static struct PyModuleDef module_def = {PyModuleDef_HEAD_INIT, "_C", NULL, -1, NULL};
*   return PyModule_Create(&module_def);
* }
* ```
*       - We could directly load the shared library using something like `torch.ops.load_library("/path/to/library.so")`
*         The downside is we can't expose any additional python bindings from C++ this way.
*   2. Instead of an import statement which would require us to compile the extension ahead of time, we can make use of
*      `torch.utils.cpp_extension.load` which will build and compile the .cpp files into a shared library and load it
*      into the current Python process.
*
*/

#include <torch/csrc/utils/pybind.h>
#include "foo_core/operations.h"

namespace torch_foo {
namespace {

// TODO: Remove
at::Tensor temp_func(const at::Tensor& a, const at::Tensor& b)
{
    return foo_core::mymuladd_cpu(a, b, 10.0);
}

void InitFooBindings(py::module m)
{
    // Extra initialization code here

    // Functions to be exposed for the backend module
    m.def("current_device", []() {
        return 0; // Hardcoded 0 device.
    }, "Returns the curent device index");
    m.def("device_count", []() {
        return 2; // Hardcode 2 total devices.
    }, "Returns the total number of devices available");

    // Functions over torch tensors
    m.def("add", &foo_core::add, "add two tensors element-wise", py::arg("a"), py::arg("b"))
    m.def("multiply", &foo_core::multiply, "Multiply two tensors element-wise", py::arg("a"), py::arg("b"));
}

} // namespace
} // namespace torch_foo

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    torch_foo::InitFooBindings(m);
}
