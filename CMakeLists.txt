cmake_minimum_required(VERSION 3.26)
project(torch_foo LANGUAGES CXX)

# Have CMake output compile_commands.json for LSP servers.
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Find Python
find_package(Python3 COMPONENTS Interpreter Development.Module NumPy REQUIRED)

# Find Torch from the local Python installation. Put in top level because it's used by all subdirectories.
execute_process(
    COMMAND "${Python_EXECUTABLE}" -c "import torch; print(torch.utils.cmake_prefix_path)"
    OUTPUT_VARIABLE TORCH_CMAKE_PREFIX_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
list(APPEND CMAKE_PREFIX_PATH "${TORCH_CMAKE_PREFIX_PATH}")
find_package(Torch REQUIRED)
message(STATUS "Torch found at ${TORCH_PREFIX_PATH}")
message(STATUS "Torch include directories: ${TORCH_INCLUDE_DIRS}")
message(STATUS "Torch libraries: ${TORCH_LIBRARIES}")

# TODO: Do we need to add TORCH_CXX_FLAGS, is seven setting CMAKE_CXX_FLAGS a good practice anymore?
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# Add the core library first. This is the library that does all the work of the extension.
add_subdirectory(src/foo_core)
# Add the extension
add_subdirectory(src/torch_foo)
