cmake_minimum_required(VERSION 3.18)
project(rtorch_ffi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Torch REQUIRED)

add_library(rtorch_ffi STATIC
  src/rtorch.cpp
)

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
