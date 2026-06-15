cmake_minimum_required(VERSION 3.18)
project(rtorch_ffi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Torch REQUIRED)

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

if(RTORCH_HAS_EXTENDED_UNSIGNED_TYPES)
  target_compile_definitions(rtorch_ffi PRIVATE RTORCH_HAS_EXTENDED_UNSIGNED_TYPES=1)
endif()

target_include_directories(rtorch_ffi PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(rtorch_ffi PUBLIC
  ${TORCH_LIBRARIES}
)

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
