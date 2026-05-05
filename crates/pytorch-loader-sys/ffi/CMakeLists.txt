cmake_minimum_required(VERSION 3.18)
project(pytorch_loader_ffi LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Torch REQUIRED)

add_library(pytorch_loader_ffi STATIC
  src/pytorch_loader.cpp
)

target_include_directories(pytorch_loader_ffi PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(pytorch_loader_ffi PUBLIC
  ${TORCH_LIBRARIES}
)

install(TARGETS pytorch_loader_ffi
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)
