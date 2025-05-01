#include <torch/csrc/utils/pybind.h>
#include "foo_core/operations.h"

namespace py = pybind11;

PYBIND11_MODULE(torch_foo, m) {
    m.doc() = "PyTorch C++ extension for tensor operations";
    m.def("add", &foo_core::add, "Add two tensors element-wise",
          py::arg("a"), py::arg("b"));
    m.def("multiply", &foo_core::multiply, "Multiply two tensors element-wise",
          py::arg("a"), py::arg("b"));
}
