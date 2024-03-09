cmake_minimum_required(VERSION 3.22)  # 设CMake最小版本号
project(ctest)  # 设置工程名

set(INCLUDE_PATH ${PROJECT_SOURCE_DIR}/include)
include_directories(${INCLUDE_PATH})

set(LIB_PATH ${PROJECT_SOURCE_DIR}/libs)
link_directories(${LIB_PATH})

aux_source_directory(. DIR_SRCS)
add_executable(${PROJECT_NAME} ${DIR_SRCS})

target_link_libraries(${PROJECT_NAME}
    rust_wrapper onnxruntime
)
