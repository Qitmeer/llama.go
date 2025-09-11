cmake_minimum_required(VERSION 3.14)
project(llama_core)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)

# llama.cpp
set(GGML_USE_OPENMP OFF)
set(GGML_OPENMP OFF)
set(GGML_STATIC ON)
set(BUILD_SHARED_LIBS OFF)
set(LLAMA_BUILD_COMMON ON)
set(LLAMA_CURL OFF)
add_subdirectory(llama.cpp ${CMAKE_BINARY_DIR}/llama)
add_subdirectory(./llama.cpp/tools/mtmd)

# whisper.cpp
set(WHISPER_COMMON_TARGET "whisper-common")
set(WHISPER_BUILD_EXAMPLES ON)
add_subdirectory(whisper.cpp ${CMAKE_BINARY_DIR}/whisper)

# core
set(SRCS src/interactive.cpp src/process.cpp src/runner.cpp src/event_processor.cpp src/embedding.cpp src/whisper_service.cpp src/scheduler.cpp src/server_context.h)
set(TARGET llama_core)

include_directories(./include)
include_directories(./llama.cpp/include)
include_directories(./llama.cpp/common)
include_directories(./llama.cpp/tools/mtmd)
include_directories(./whisper.cpp/include)
include_directories(./whisper.cpp)

link_directories(${CMAKE_BINARY_DIR}/lib)

add_library(${TARGET} STATIC ${SRCS})
target_link_libraries(${TARGET} PRIVATE common llama whisper-common whisper mtmd ${CMAKE_THREAD_LIBS_INIT} ${FFMPEG_LIBRARIES})

# test
option(BUILD_TEST "Build the testing tree." OFF)

if(BUILD_TEST)
    include(CTest)
    add_subdirectory(tests)
endif()