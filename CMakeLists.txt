cmake_minimum_required(VERSION 3.14)
project(qwen_benchmark CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(LLAMA_DIR ${CMAKE_SOURCE_DIR}/third_party/llama.cpp)

# 主执行程序
add_executable(benchmark src/benchmark.cpp)

# 头文件路径
target_include_directories(benchmark PRIVATE
    ${LLAMA_DIR}/include
    ${LLAMA_DIR}/common
    ${LLAMA_DIR}/tools/mtmd
    ${LLAMA_DIR}/ggml/include
)

# 库文件路径
target_link_directories(benchmark PRIVATE
    ${LLAMA_DIR}/build/bin
)

# 链接库文件
target_link_libraries(benchmark PRIVATE llama-common mtmd llama ggml)
