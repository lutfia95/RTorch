cmake_minimum_required(VERSION 3.18)
project(equiload_ffi LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

option(EQUILOAD_CPU_BRIDGE "Link CPU tensor operations without requiring a CUDA compiler" OFF)
if(EQUILOAD_CPU_BRIDGE)
  # A CUDA-enabled Python wheel has the CPU headers and shared libraries needed
  # by this bridge, even on machines without a CUDA development toolkit.
  set(TORCH_INCLUDE_DIRS "$ENV{LIBTORCH}/include" "$ENV{LIBTORCH}/include/torch/csrc/api/include")
  set(TORCH_LIBRARIES "")
  foreach(component torch torch_cpu c10)
    find_library(EQUILOAD_${component}_LIBRARY ${component}
      PATHS "$ENV{LIBTORCH}/lib" NO_DEFAULT_PATH REQUIRED)
    list(APPEND TORCH_LIBRARIES "${EQUILOAD_${component}_LIBRARY}")
  endforeach()
  file(STRINGS "$ENV{LIBTORCH}/share/cmake/Torch/TorchConfig.cmake" abi_line
    REGEX "set\\(TORCH_CXX_FLAGS.*_GLIBCXX_USE_CXX11_ABI")
  if(DEFINED ENV{EQUILOAD_CXX11_ABI} AND "$ENV{EQUILOAD_CXX11_ABI}" MATCHES "^[01]$")
    set(TORCH_CXX_FLAGS "-D_GLIBCXX_USE_CXX11_ABI=$ENV{EQUILOAD_CXX11_ABI}")
  elseif(abi_line MATCHES "-D_GLIBCXX_USE_CXX11_ABI=[01]")
    set(TORCH_CXX_FLAGS "${CMAKE_MATCH_0}")
  elseif(CMAKE_SYSTEM_NAME STREQUAL "Linux")
    message(FATAL_ERROR "Set EQUILOAD_CXX11_ABI to the active torch._C._GLIBCXX_USE_CXX11_ABI (0 or 1)")
  endif()
else()
  find_package(Torch REQUIRED)
endif()
option(EQUILOAD_ENABLE_HDF5 "Enable the HDF5 loader" OFF)
if(EQUILOAD_ENABLE_HDF5)
  find_package(HDF5 COMPONENTS C REQUIRED)
endif()

include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_INCLUDES ${TORCH_INCLUDE_DIRS})
set(CMAKE_REQUIRED_FLAGS ${TORCH_CXX_FLAGS})
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)
check_cxx_source_compiles(
  "#include <torch/torch.h>
   int main() {
     auto u16 = c10::kUInt16;
     auto u32 = c10::kUInt32;
     auto u64 = c10::kUInt64;
     return static_cast<int>(u16) + static_cast<int>(u32) + static_cast<int>(u64);
   }"
  EQUILOAD_HAS_EXTENDED_UNSIGNED_TYPES
)

add_library(equiload_ffi STATIC
  src/equiload.cpp
)

if(TORCH_CXX_FLAGS)
  separate_arguments(EQUILOAD_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(equiload_ffi PRIVATE ${EQUILOAD_TORCH_CXX_FLAGS})
endif()

if(EQUILOAD_HAS_EXTENDED_UNSIGNED_TYPES)
  target_compile_definitions(equiload_ffi PRIVATE EQUILOAD_HAS_EXTENDED_UNSIGNED_TYPES=1)
endif()

if(EQUILOAD_ENABLE_HDF5)
  target_compile_definitions(equiload_ffi PRIVATE EQUILOAD_WITH_HDF5=1)
  target_include_directories(equiload_ffi PRIVATE ${HDF5_INCLUDE_DIRS})
endif()

target_include_directories(equiload_ffi PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
  ${TORCH_INCLUDE_DIRS}
)

target_link_libraries(equiload_ffi PUBLIC
  ${TORCH_LIBRARIES}
)

if(EQUILOAD_ENABLE_HDF5)
  target_link_libraries(equiload_ffi PUBLIC ${HDF5_C_LIBRARIES})
  file(WRITE "${CMAKE_BINARY_DIR}/equiload-hdf5-link-libs.txt" "")
  foreach(HDF5_LIBRARY ${HDF5_C_LIBRARIES})
    file(APPEND "${CMAKE_BINARY_DIR}/equiload-hdf5-link-libs.txt" "${HDF5_LIBRARY}\n")
  endforeach()
  install(FILES "${CMAKE_BINARY_DIR}/equiload-hdf5-link-libs.txt" DESTINATION lib)
endif()

install(TARGETS equiload_ffi
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

option(EQUILOAD_BUILD_TESTS "Build EquiLoad C ABI tests" OFF)
if(EQUILOAD_BUILD_TESTS)
  enable_testing()
  add_executable(equiload_tensor_test tests/equiload_tensor_test.cpp)
  if(MSVC)
    target_compile_options(equiload_tensor_test PRIVATE /UNDEBUG)
  else()
    target_compile_options(equiload_tensor_test PRIVATE -UNDEBUG)
  endif()
  target_link_libraries(equiload_tensor_test PRIVATE equiload_ffi)
  add_test(NAME equiload_tensor_test COMMAND equiload_tensor_test)
endif()
