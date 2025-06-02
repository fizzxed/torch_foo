
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

# Building:
```bash
uv venv
uv pip install -r requirements.txt
uv pip install -e . # we have set no-build-isolation in pyproject.toml
```

# Building the library standalone

```bash
# Create build directory
mkdir build && cd build

# Configure with CMake (pointing to your libtorch installation)
cmake .. -DCMAKE_PREFIX_PATH=/path/to/libtorch

# Build
cmake --build .

# Install (optional)
cmake --install .
```

TODO: Add capability to use foo_core in a separate C++ project and document here.

# Tests
```bash
python -m pytest tests/
```
# Autoloading
This extension makes use of PyTorch's [autoloading](https://pytorch.org/tutorials/prototype/python_extension_autoload.html) feature.

To disable autoloading of the extension, set the environment variable `TORCH_DEVICE_BACKEND_AUTOLOAD'` to `0`.

# Developing
To update requirements
```bash
uv pip freeze > requirements.txt --exclude-editable
```

Sanity check to see that everything is building correctly. Should make this a test.
```bash
uv pip uninstall torch_foo && rm -rf build/ && uv pip install -e .
python
>>> import torch
>>> a = torch.tensor([1.0, 2.0, 3.0])
>>> b = torch.tensor([4.0, 5.0, 6.0])
>>> torch.ops.foo.mymuladd(a, b, 2)
tensor([ 6., 12., 20.])
```

# Building LibTorch
libtorch is typically distributed with shared libraries because the CUDA runtime can [no longer be linked statically](https://discuss.pytorch.org/t/libtorch-cxx11-abi-static-library/211309). However if we don't need CUDA then we can build pytorch from source and statically link any dependencies we want.
```bash
git clone -b main --recurse-submodule https://github.com/pytorch/pytorch.git
cd pytorch
uv venv
source .venv/bin/activate
uv pip install cmake ninja
uv pip install -r requirements.txt
mkdir build && cd build
cmake -GNinja -DBUILD_SHARED_LIBS:BOOL=OFF -DCMAKE_BUILD_TYPE:STRING=MinSizeRel -DPYTHON_EXECUTABLE:PATH=`which python3` -DCMAKE_INSTALL_PREFIX:PATH=../install -DUSE_CUDA:BOOL=OFF -DUSE_CUDNN:BOOL=OFF -DUSE_XNNPACK:BOOL=OFF -DBUILD_TEST:BOOL=OFF -DBUILD_PYTHON:BOOL=OFF ..
cmake --build . --target install
```
this build will take a while so it would be helpful to run on a machine with a lot of cores and memory.
Note that because libtorch uses static initializers to register things like custom operators and device guards, `libtorch_cpu will` need to be
linked with `--whole-archive` negating any binary size savings.


# Notes
- See if we can support Python's Limited API. This would mean moving from pybind11 to nanobind. See [
pytorch-nanobind-cuda-example
](https://github.com/jannismoeller/pytorch-nanobind-cuda-example) for an example of an extension that does this with the same build system. Also look at the [nanobind scikit-build-core example](https://github.com/wjakob/nanobind_example). Currently `torch_python` uses more than the limited API so we would have to avoid using it in `torch_foo`.
  - Note that it seems that the canonical way for C++ extensions to include torch headers is to `#include <torch/extension.h>` which pulls in all the C++ headers, as defined in `torch/all.h` as well as the Python bindings in `torch/python.h` (which includes pybind11 and probably uses more than the limited API in other facets too, therefore no limited API support). Maybe could get around this by only using the C++ headers from `torch/all.h` and avoiding the Python bindings if we are just registering ops and backends in C++. We could then use the limited API to create a module to be imported in Python.
  - This is now being taken more seriously for pytorch extensions that use pytorch's `CppExtension` class for building. See [this PR](https://github.com/pytorch/pytorch/pull/145764)
- The `foo_core` library will likely need some code generation to register ops for kernels defined in C++. This can be achieved using [torchgen](https://github.com/pytorch/pytorch/tree/master/torchgen) which Ascend's NPU's pytorch extension also uses. See [here](https://github.com/Ascend/pytorch/blob/master/generate_code.sh) for the entrypoint in which NPU generates code.
  - see also this [discussion](https://dev-discuss.pytorch.org/t/backend-fallbacks/195) on fallbacks. and the [pytorch wiki](https://github.com/pytorch/pytorch/wiki/Boxing-and-Unboxing-in-the-PyTorch-Operator-Library) for discussion on boxing and unboxing.
  - Once upon a time, the XLA torch extension ran some codegen to add CPU fallback kernels. This was taken out in this [PR](https://github.com/pytorch/pytorch/pull/58065) (see this [deleted file](https://github.com/pytorch/pytorch/pull/58065/files#diff-4cd06a72bfdef66dd62378d7f0adb439fd2dfac0190c65072c6e67def7ff7b04)) in favor of using a boxed kernel fallback to CPU.
  - It seems like all pytorch backend extensions now register boxed kernel fallbacks. See Intel's [extension as an example](https://github.com/HabanaAI/gaudi-pytorch-bridge/blob/1167bf4dbc9fc0d1a20102d4ac91b406d44a2d2e/hpu_ops/cpu_fallback.cpp#L116) (Note that they build off a fork of pytorch that adds their dispatch keys to the dispatcher, so they don't use PrivateUse1)
- At somepoint we will have to rename the backend. This can be done in C++ as well which could be useful. NPU also does this [here](https://github.com/Ascend/pytorch/blob/24a508cd1022f9a383dbc53f0a9ab6b526e4fcda/torch_npu/csrc/core/npu/impl/NPUGuardImpl.cpp#L188) though they do also do it in the [`__init__.py` file](https://github.com/Ascend/pytorch/blob/24a508cd1022f9a383dbc53f0a9ab6b526e4fcda/torch_npu/__init__.py#L185) but according to the [implementation](register_privateuse1_backend) this is fine since both set it to "npu"
- [`generate_methods_for_privateuse1_backend`](https://docs.pytorch.org/docs/stable/generated/torch.utils.generate_methods_for_privateuse1_backend.html) to monkeypatch onto the torch objects. In the [pr comments](https://github.com/pytorch/pytorch/pull/98066#issuecomment-1496128826) they mention that the backend's ought to do this themselves for proper typing and are just provided for convenience. User's don't absolutely need this, but is a nice to have.
- The Pytorch tutorials imply that we could get away with just writing and registering our custom aten kernels for privateuse1 in C++. However, if we did that we would only be able to call into Aten ops by bypassing Pytorch sugaring via:
```python
tensor =  torch.ops.aten.empty.memory_format(
    [2, 2],
    dtype=torch.float32,
    layout=torch.strided,
    device=torch.device("privateuseone:0"),
    pin_memory=False,
    memory_format=torch.contiguous_format
)
```
If we wanted to fully integrate with the python side of Pytorch, we also need to expose a backend module that implements the following functions [referenced here](https://docs.pytorch.org/docs/stable/generated/torch.utils.rename_privateuse1_backend.html):
  1. is_available() -> bool Returns a bool indicating if our backend is currently available.
  2. current_device() -> int Returns the index of a currently selected device.
And then register it via `torch._register_device_module("foo", BackendModule)`


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
- [Custom Operators Manual](https://docs.google.com/document/d/1_W62p8WJOQQUzPsJYa7s701JXt0qf2OfLub2sbkHOaU/) for information on how to extend the list of PyTorch operators. Covers things like supporting `torch.compile`, adding autograd support in either python or C++.
- [State of Cuda Extensions in Pytorch](https://github.com/pytorch/pytorch/issues/152032)
- [build system overview](https://stackoverflow.com/questions/25941536/what-is-a-cmake-generator/61651241#61651241)
- [spglib](https://github.com/spglib/spglib/tree/develop) an example of a project that uses CMake and scikit-build-core to create a python extension over their C library. Also builds both static and shared libraries, could be useful to support static libtorch.
- [pytorch internals blogpost](https://lernapparat.de/selective-excursion-into-pytorch-internals/) for a look into how some important parts of PyTorch worked circa 2018.
