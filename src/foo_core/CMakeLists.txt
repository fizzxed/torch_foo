# The CMake file for the core library. This should be buildable without building any python parts.

# TODO: make foo_core be static or shared depending on the value of BUILD_SHARED_LIBS
add_library(foo_core STATIC
    src/operations.cpp
    src/cpu_fallback.cpp
    src/register_name.cpp
    src/aten.cpp
    src/DummyDeviceGuard.cpp
    src/DummyAllocator.cpp
)

# Set position independent code. This is defaulted to ON for shared libraries. Keeping for verbosity.
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

# Installation rules for the standalone mode (no scikit-build-core)
# if (NOT DEFINED SKBUILD)
#     include(GNUInstallDirs)
#     install(TARGETS foo_core
#         LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#     )
# endif()
