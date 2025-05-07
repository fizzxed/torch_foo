
An example of a simple PyTorch extension that uses scikit-build-core (CMake) as the build system and `uv pip` as the frontend.

To see an example using libtorch_python (pytorch/python bindings + pybind11) in the C++ extension, checkout the `use_torch_python_API` branch.

Currently:
We make use of PyTorch's included PyBind11 library to create a Python extension that binds the C++ library.

Goal:
Register custom ATen C++ operations through the dispatcher which will let us access them from Python. Maybe this will let us use the python limited API. See Notes below.

```
├── CMakeLists.txt
├── pyproject.toml
├── python                       # the actual python module
│   └── torch_foo
│       └── __init__.py
├── requirements.txt
├── src
│   ├── foo_core                  # the C++ library that will be in charge of core operations
│   │   ├── CMakeLists.txt
│   │   ├── include
│   │   │   └── foo_core
│   │   │       └── operations.h
│   │   └── src
│   │       └── operations.cpp
│   └── torch_foo                 # the python extension that simply binds the C++ library
│       ├── CMakeLists.txt
│       └── src
│           └── bindings.cpp
└── tests
    └── test_torch_foo.py
```

Building:
```
uv venv
uv pip install -r requirements.txt
uv pip install -e . # we have set no-build-isolation in pyproject.toml
```

Tests:
```
python -m pytest tests/
```

To disable autoloading of the extension, set the environment variable `TORCH_DEVICE_BACKEND_AUTOLOAD'` to `0`.

To update requirements
```
uv pip freeze > requirements.txt --exclude-editable
```

TODO/Notes:
- See if we can support Python's Limited API. This would mean moving from pybind11 to nanobind. See [
pytorch-nanobind-cuda-example
](https://github.com/jannismoeller/pytorch-nanobind-cuda-example) for an example of an extension that does this with the same build system. Also look at the [nanobind scikit-build-core example](https://github.com/wjakob/nanobind_example). Currently `torch_python` uses more than the limited API so we would have to avoid using it in `torch_foo`.
  - Note that it seems that the canonical way for C++ extensions to include torch headers is to `#include <torch/extension.h>` which pulls in all the C++ headers, as defined in `torch/all.h` as well as the Python bindings in `torch/python.h` (which includes pybind11 and probably uses more than the limited API in other facets too, therefore no limited API support). Maybe could get around this by only using the C++ headers from `torch/all.h` and avoiding the Python bindings if we are just registering ops and backends in C++. We could then use the limited API to create a module to be imported in Python.
  - This is now being taken more seriously for pytorch extensions that use pytorch's `CppExtension` class for building. See [this PR](https://github.com/pytorch/pytorch/pull/145764)
- The `foo_core` library will likely need some code generation to register ops and fallbacks for kernels defined in C++. This can be achieved using [torchgen](https://github.com/pytorch/pytorch/tree/master/torchgen) which Ascend's NPU's pytorch extension also uses. See [here](https://github.com/Ascend/pytorch/blob/master/generate_code.sh) for the entrypoint in which NPU generates code.


Useful headers:
- `<torch/torch.h>`     - All pure C++ headers for the C++ frontend. Also synonymous with `<torch/all.h>` which has the include statement for all C++ headers.
- `<torch/python.h>`    - Python bindings for the C++ frontend. (includes `Python.h`)
- `<torch/extension.h>` - Includes the above two header files. For use in C++ extensions that want Pytorch's C++/Python bindings and also Pybind11.
- `<torch/library.h>`   - The API for extending PyTorch's core library of operators with user defined operators and data types. Provides `TORCH_LIBRARY` and `TORCH_LIBRARY_IMPL` macros, which are important for registering new operators and for providing implementations for those operators, respectively.

Useful links:
- [`torch._C` creation](https://github.com/pytorch/pytorch/blob/e889937850759fe69a8c7de6326984102ed9b088/torch/csrc/Module.cpp#L1833) as part of [`initModule`](https://github.com/pytorch/pytorch/blob/e889937850759fe69a8c7de6326984102ed9b088/torch/csrc/Module.cpp#L1794)
- [`bind_module`](https://github.com/pytorch/pytorch/blob/e889937850759fe69a8c7de6326984102ed9b088/torch/csrc/api/include/torch/python.h#L217) which creates a pybind11  class object for a `nn::Module` subclass type and adds default bindings. The meat of the work is done in [`add_module_bindings`](https://github.com/pytorch/pytorch/blob/e889937850759fe69a8c7de6326984102ed9b088/torch/csrc/api/include/torch/python.h#L106)
- [`InitNpuBindings.cpp`](https://github.com/Ascend/pytorch/blob/a72f1dde999b6d78839c9691d11256e1d821a891/torch_npu/csrc/InitNpuBindings.cpp) How the Ascend NPU initializes python/C++ bindings for the NPU torch extension. In particular their module is defined [here](https://github.com/Ascend/pytorch/blob/a72f1dde999b6d78839c9691d11256e1d821a891/torch_npu/csrc/InitNpuBindings.cpp#L170) without using pybind11.
It seems that in the end they still use pybind11 to create bindings if you follow their included headers, so might as well model after the [torch_xla bindings](https://github.com/pytorch/xla/blob/master/torch_xla/csrc/init_python_bindings.cpp)
- [`torch._C` stub](https://github.com/pytorch/pytorch/pull/39375/files) Here they make the `_C` extension a thin C wrapper (`torch/csrc/stub.c`) that links against `torch_python` which provides the `initModule` implementation.  Ascend NPU doesn't bother with the thin C wrapper and has the `PyInit__C` function in `InitNpuBindings.cpp`. `stub.c` was never intended to keep existing, according to it's [PR review](https://github.com/pytorch/pytorch/pull/12742#discussion_r229892371), so leaning more towards the Ascend NPU's approach is probably a good idea.
- [`torchgen` creation](https://github.com/pytorch/pytorch/issues/73212) Here PyTorch decided to make the pytorch codegen utilities publically available to external libraries that have C++ extensions (us). [`gen.py`](https://github.com/pytorch/pytorch/blob/main/torchgen/gen.py) is the entry point for generating code.
- [autoloading](https://pytorch.org/tutorials/prototype/python_extension_autoload.html) is a nice to have so users don't have to manually import the extension module.
- [extension-cpp](https://github.com/pytorch/extension-cpp) An example of writing a C++/CUDA extension for PyTorch that implements a custom op for both CPU and CUDA written by the pytorch team.
- [Custom Operators Manual](https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/) for information on how to extend the list of PyTorch operators.
