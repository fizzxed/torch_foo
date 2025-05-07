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
void InitFooBindings(py::module m)
{
    // Extra initialization code here
    // Example of adding a function to the module:
    m.def("foo", []() { return "foo"; }, "Foo function");
    // m.def("add", &foo_core::add, "add two tensors element-wise", py::arg("a"), py::arg("b"))
    // m.def("multiply", &foo_core::multiply, "Multiply two tensors element-wise", py::arg("a"), py::arg("b"));
}

// Register the operators
TORCH_LIBRARY(foo, m)
{
    m.def("mymuladd(Tensor a, Tensor b, float c) -> Tensor");
    m.def("mymul(Tensor a, Tensor b) -> Tensor");
    m.def("myadd_out(Tensor a, Tensor b, Tensor(a!) out) -> ()");
}
// Register the implementations for the operators. Will be updating this to privateuse1 soon.
TORCH_LIBRARY_IMPL(foo, CPU, m)
{
    m.impl("mymuladd", &foo_core::mymuladd_cpu);
    m.impl("mymul", &foo_core::mymul_cpu);
    m.impl("myadd_out", &foo_core::myadd_out_cpu);
}

} // namespace
} // namespace torch_foo

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { torch_foo::InitFooBindings(m); }
