cmake_minimum_required(VERSION 3.18)
project(rtorch_ffi LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Torch REQUIRED)
option(RTORCH_ENABLE_HDF5 "Enable the HDF5 loader" OFF)
if(RTORCH_ENABLE_HDF5)
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
  RTORCH_HAS_EXTENDED_UNSIGNED_TYPES
)

add_library(rtorch_ffi STATIC
  src/rtorch.cpp
)

if(TORCH_CXX_FLAGS)
  separate_arguments(RTORCH_TORCH_CXX_FLAGS NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
  target_compile_options(rtorch_ffi PRIVATE ${RTORCH_TORCH_CXX_FLAGS})
endif()

if(RTORCH_HAS_EXTENDED_UNSIGNED_TYPES)
  target_compile_definitions(rtorch_ffi PRIVATE RTORCH_HAS_EXTENDED_UNSIGNED_TYPES=1)
endif()

if(RTORCH_ENABLE_HDF5)
  target_compile_definitions(rtorch_ffi PRIVATE RTORCH_WITH_HDF5=1)
  target_include_directories(rtorch_ffi PRIVATE ${HDF5_INCLUDE_DIRS})
endif()

target_include_directories(rtorch_ffi PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(rtorch_ffi PUBLIC
  ${TORCH_LIBRARIES}
)

if(RTORCH_ENABLE_HDF5)
  target_link_libraries(rtorch_ffi PUBLIC ${HDF5_C_LIBRARIES})
  file(WRITE "${CMAKE_BINARY_DIR}/rtorch-hdf5-link-libs.txt" "")
  foreach(HDF5_LIBRARY ${HDF5_C_LIBRARIES})
    file(APPEND "${CMAKE_BINARY_DIR}/rtorch-hdf5-link-libs.txt" "${HDF5_LIBRARY}\n")
  endforeach()
  install(FILES "${CMAKE_BINARY_DIR}/rtorch-hdf5-link-libs.txt" DESTINATION lib)
endif()

install(TARGETS rtorch_ffi
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

option(RTORCH_BUILD_TESTS "Build RTorch C ABI tests" OFF)
if(RTORCH_BUILD_TESTS)
  enable_testing()
  add_executable(rtorch_tensor_test tests/tensor_test.cpp)
  target_link_libraries(rtorch_tensor_test PRIVATE rtorch_ffi)
  add_test(NAME rtorch_tensor_test COMMAND rtorch_tensor_test)
endif()
