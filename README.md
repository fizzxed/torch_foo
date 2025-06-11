
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
