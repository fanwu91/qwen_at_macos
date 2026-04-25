cmake_minimum_required(VERSION 3.20)
project(qwen_benchmark CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# llama.cpp is in third_party/ as a git submodule
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)

add_subdirectory(third_party/llama.cpp)

add_executable(qwen_benchmark benchmark.cpp metrics.cpp)

target_include_directories(qwen_benchmark PRIVATE
  third_party/llama.cpp/include
  third_party/llama.cpp/examples/llava
)

target_link_libraries(qwen_benchmark PRIVATE
  llama
  llava
)
