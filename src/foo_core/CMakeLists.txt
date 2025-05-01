# The CMake file for the core library. This should be buildable without building any python parts.

add_library(foo_core STATIC
    src/operations.cpp
)

# Set position independent code.
set_target_properties(foo_core PROPERTIES
    POSITION_INDEPENDENT_CODE ON)

# Set include directories
target_include_directories(foo_core
    PUBLIC
        $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
        $<INSTALL_INTERFACE:include>
        ${TORCH_INCLUDE_DIRS}
)

# Link against LibTorch
target_link_libraries(foo_core
    PUBLIC
        "${TORCH_LIBRARIES}"
)

# Set the C++ Standard
target_compile_features(foo_core PUBLIC cxx_std_14)
