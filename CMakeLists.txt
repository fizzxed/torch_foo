cmake_minimum_required(VERSION 3.26)

#[=============================================================================[
#                           Basic project definition                           #
]=============================================================================]

project(torch_foo LANGUAGES CXX)

# Have CMake output compile_commands.json for LSP servers.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


#[=============================================================================[
#                                   Options                                   #
]=============================================================================]

# Boilerplate default values
# if (CMAKE_BUILD_TYPE STREQUAL "Debug")
#     set(_Foo_default_debug ON)
# else()
#     set(_Foo_default_debug OFF)
# endif()

option(FOO_WITH_PYTHON "Foo: Build Python interface" OFF)
option(FOO_WITH_TESTS "Foo: Build unit tests" ${PROJECT_IS_TOP_LEVEL})
option(FOO_SHARED_LIBS "Foo: Build as a shared library" ${PROJECT_IS_TOP_LEVEL})
option(FOO_INSTALL "Foo: Install project" ${PROJECT_IS_TOP_LEVEL})
option(FOO_WARNINGS "Foo: Enable warning messages" ON)
option(FOO_INFO "Foo: Enable info messages" ON)
option(FOO_DEBUG "Foo: Build in debug mode" ${_Spglib_default_debug})
option(FOO_COMPILATION_WARNING "Foo: Enable compilation warnings" OFF)
mark_as_advanced(
        FOO_DEBUG
        FOO_WARNINGS
        FOO_COMPILATION_WARNING
)

#[=============================================================================[
#                            Project configuration                            #
]=============================================================================]
# Include basic tools
include(GNUInstallDirs)
# Define basic parameters
# set(BUILD_SHARED_LIBS ${FOO_SHARED_LIBS}) TODO: build foo_core as a shared library
# https://alexreinking.com/blog/building-a-dual-shared-and-static-library-with-cmake.html


#[=============================================================================[
#                              External packages                              #
]=============================================================================]

# TODO: Branch on SKBUILD and FOO_WITH_PYTHON to determine where torch is
# If with SKBUILD probably using the libtorch that comes with pytorch
# otherwise torch installation passed via  -DCMAKE_PREFIX_PATH=/path/to/libtorch
# Also should probably not live at the root CMakelists.txt

# Find Python
find_package(Python3 COMPONENTS Interpreter Development.Module NumPy REQUIRED)
# Find Torch from the local Python installation.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")


# Find torch
find_package(Torch REQUIRED)
message(STATUS "Torch found at ${TORCH_PREFIX_PATH}")
message(STATUS "Torch include directories: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")

#[=============================================================================[
#                               Main definition                               #
]=============================================================================]

# TODO: Do we need to add TORCH_CXX_FLAGS, is seven setting CMAKE_CXX_FLAGS a good practice anymore?
# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add the core library first. This is the library that does all the work of the extension.
add_subdirectory(src/foo_core)

# Add the Python C++ extension only when asked
if (FOO_WITH_PYTHON)
    add_subdirectory(src/torch_foo)
endif()
