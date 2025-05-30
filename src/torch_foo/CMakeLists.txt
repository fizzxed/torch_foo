cmake_minimum_required(VERSION 3.26)
#[=============================================================================[
#                           Basic project definition                           #
]=============================================================================]


#[=============================================================================[
#                                   Options                                   #
]=============================================================================]

option(FOO_INSTALL "Foo: Install project" ${PROJECT_IS_TOP_LEVEL})

#[=============================================================================[
#                            Project configuration                            #
]=============================================================================]

include(GNUInstallDirs)


#[=============================================================================[
#                                Public targets                                #
]=============================================================================]

# Make sure the library name matches what we will import in Python
# WITH_SOABI adds a suffix to the library name detailing the Python version and ABI
Python3_add_library(_C MODULE WITH_SOABI
    src/bindings.cpp
)
add_library(foo::python ALIAS _C)

#[=============================================================================[
#                              External packages                              #
]=============================================================================]

# We have to find the torch_python library so we can make use of torch's pybind11 interface
# https://github.com/pytorch/pytorch/issues/108041
# Maybe just including pybind11 ourselves would be better?
find_library(TORCH_PYTHON_LIBRARY torch_python PATH "${TORCH_INSTALL_PREFIX}/lib")
message(STATUS "Torch Python Library: ${TORCH_PYTHON_LIBRARY}")

#[=============================================================================[
#                               Main definition                               #
]=============================================================================]

# Link against the core library and the torch python library.
# This pulls in any include directories and libraries listed as PUBLIC
target_link_libraries(_C PRIVATE
    $<LINK_LIBRARY:WHOLE_ARCHIVE,foo_core> # TODO: Look into using object libraries instead
    ${TORCH_PYTHON_LIBRARY}
)

# Set C++ Standard. pybind11 uses C++14 or higher
target_compile_features(_C PRIVATE cxx_std_14)

# Add the TORCH_EXTENSION_NAME definition (equivalent to -DTORCH_EXTENSION_NAME={name})
# So we don't have to keep the C++ code in sync with the build system.
# PyTorch's BuildExtension sets this for us normally.
target_compile_definitions(_C PRIVATE
    TORCH_EXTENSION_NAME=$<TARGET_NAME:_C>
)

#[=============================================================================[
#                              Install or Export                              #
]=============================================================================]

if (NOT SKBUILD AND FOO_INSTALL)
    message(WARNING "Installing the python bindings outside of scikit-build-core environment is not supported.")
endif()

# Install the library to the Python Module torch_foo
install(TARGETS _C DESTINATION torch_foo)
