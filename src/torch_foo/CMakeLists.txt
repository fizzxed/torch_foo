# Make sure the library name matches what we will import in Python
# WITH_SOABI adds a suffix to the library name detailing the Python version and ABI
Python3_add_library(_C MODULE WITH_SOABI
    src/bindings.cpp
)

# We have to find the torch_python library so we can make use of torch's pybind11 interface
# https://github.com/pytorch/pytorch/issues/108041
# Maybe just including pybind11 ourselves would be better?
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
message(STATUS "Torch Python Library: ${TORCH_PYTHON_LIBRARY}")

# Link against the core library and the torch python library.
# This pulls in any include directories and libraries listed as PUBLIC
target_link_libraries(_C PRIVATE
    foo_core
    ${TORCH_PYTHON_LIBRARY}
)

# Set C++ Standard. pybind11 uses C++14 or higher
target_compile_features(_C PRIVATE cxx_std_14)

# Install the library to the Python Module torch_foo
install(TARGETS _C DESTINATION torch_foo)
