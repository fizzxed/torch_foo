
An example of a simple PyTorch extension that uses scikit-build-core (CMake) as the build system and `uv pip` as the frontend.

We make use of PyTorch's included PyBind11 library to create a Python extension that binds the C++ library.

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


To update requirements
```
uv pip freeze > requirements.txt --exclude-editable
```
