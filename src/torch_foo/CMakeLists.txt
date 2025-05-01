# find_package(pybind11 CONFIG REQUIRED)

Python3_add_library(torch_foo MODULE
    src/bindings.cpp
)

# Set include directories
target_include_directories(torch_foo PRIVATE
    ${TORCH_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}/../foo_core/include
)

# We have to find the torch_python library so we can make use of torch's pybind11 interface
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
message(STATUS "Torch Python Library: ${TORCH_PYTHON_LIBRARY}")


# Link against the core library and libtorch.
target_link_libraries(torch_foo PRIVATE
    foo_core
    ${TORCH_LIBRARIES}
    ${TORCH_PYTHON_LIBRARY}
)

# Set C++ Standard. pybind11 uses C++14 or higher
target_compile_features(torch_foo PRIVATE cxx_std_14)


# Install the Python Module
install(TARGETS torch_foo DESTINATION torch_foo)
install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/__init__.py DESTINATION torch_foo)
