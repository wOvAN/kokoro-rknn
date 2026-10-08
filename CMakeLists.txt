cmake_minimum_required(VERSION 3.15)
project(kokoro-infer LANGUAGES CXX)

option(USE_RKNN "Enable RKNN decoder backend (RK3588 NPU)" OFF)
option(USE_ORT_CUDA "Enable ONNX Runtime CUDA/TensorRT EPs" OFF)
option(BUILD_SERVER "Build kokoro-server (needs drogon + opus)" ON)
option(BUILD_CLI "Build kokoro-cli" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake/")

# Vendored server deps (see README: deps/drogon-install, deps/opusenc-install).
set(DEPS_PREFIX "${CMAKE_SOURCE_DIR}/deps" CACHE PATH "Local dependency install roots")
if(EXISTS "${DEPS_PREFIX}/drogon-install/lib/cmake/Drogon")
  list(APPEND CMAKE_PREFIX_PATH "${DEPS_PREFIX}/drogon-install")
endif()

# ---- ONNX Runtime --------------------------------------------------------
# Two layouts supported:
#   1. ORT_ROOT=/path/to/onnxruntime-linux-*-1.14.x  (paroli's flat layout)
#   2. Unset → use system install. Arch's onnxruntime-cpu puts headers under
#      /usr/include/onnxruntime/, which is what the cxx_api expects.
set(ORT_ROOT "" CACHE STRING "Path to ONNX Runtime root (optional, else system)")
if(ORT_ROOT STREQUAL "" AND EXISTS "${CMAKE_SOURCE_DIR}/third_party/onnxruntime/include/onnxruntime_cxx_api.h")
  set(ORT_ROOT "${CMAKE_SOURCE_DIR}/third_party/onnxruntime")
endif()
if(ORT_ROOT STREQUAL "")
  find_path(ORT_INCLUDE_DIR onnxruntime_cxx_api.h
    PATHS /usr/include /usr/local/include
    PATH_SUFFIXES onnxruntime)
  if(NOT ORT_INCLUDE_DIR)
    message(FATAL_ERROR "onnxruntime_cxx_api.h not found; set -DORT_ROOT=...")
  endif()
  include_directories(${ORT_INCLUDE_DIR})
else()
  include_directories(${ORT_ROOT}/include)
  link_directories(${ORT_ROOT}/lib)
endif()

# ---- espeak-ng ----------------------------------------------------------
# Either ESPEAK_NG_ROOT=/install/prefix (with include/espeak-ng/ and lib/) or
# system install. Used by src/phonemizer.cpp directly — we don't wrap
# piper-phonemize since the code only touches the espeak-ng C API.
set(ESPEAK_NG_ROOT "" CACHE STRING "Path to espeak-ng install prefix (optional)")
if(NOT ESPEAK_NG_ROOT STREQUAL "")
  include_directories(${ESPEAK_NG_ROOT}/include)
  link_directories(${ESPEAK_NG_ROOT}/lib)
endif()

# ---- BLAS for iSTFT sgemm ------------------------------------------------
# Try cblas (openblas/atlas) via the standard finder.
find_package(BLAS REQUIRED)
# cblas.h location varies; try common spots.
find_path(CBLAS_INCLUDE_DIR cblas.h
  PATH_SUFFIXES openblas blis)
if(NOT CBLAS_INCLUDE_DIR)
  message(FATAL_ERROR "cblas.h not found (install libopenblas-dev or similar)")
endif()
include_directories(${CBLAS_INCLUDE_DIR})

# ---- Other deps ----------------------------------------------------------
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(nlohmann_json REQUIRED)

# ---- misaki-cpp (English G2P front end) ---------------------------------
# The faithful C++ port of misaki/en.py — replaces the old in-tree English
# pipeline (src/lexicon.cpp + src/num2words_en.cpp are now unused). It ships its
# own data (dictionaries, perceptron tagger, spaCy contraction exceptions) under
# misaki-cpp/data, which is staged next to the binaries as "misaki-data".
# Keep its espeak in sync with ours so a single libespeak-ng is linked.
if(NOT ESPEAK_NG_ROOT STREQUAL "")
  set(ESPEAK_ROOT "${ESPEAK_NG_ROOT}" CACHE STRING "espeak-ng install prefix" FORCE)
endif()
set(MISAKI_BUILD_TOOLS OFF CACHE BOOL "" FORCE)  # no misaki-cli/test in this build
add_subdirectory(misaki-cpp)
set(MISAKI_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/misaki-cpp/data")
set(RU_ESPEAK_DATA_DIR "${CMAKE_CURRENT_SOURCE_DIR}/data/espeak-data"
    CACHE PATH "Russian espeak-ng-data tree (staged into build dir for tests)")
if(NOT EXISTS "${RU_ESPEAK_DATA_DIR}/ru_dict"
   AND EXISTS "/opt/espeak-ng-data/ru_dict")
  set(RU_ESPEAK_DATA_DIR "/opt/espeak-ng-data" CACHE PATH "" FORCE)
endif()

# ============== libkokoro ================================================
add_library(kokoro
  src/istft.cpp
  src/phonemizer.cpp
  src/g2p.cpp
  src/ru_g2p.cpp
  src/paths.cpp
  src/kokoro.cpp
  src/onnx-decoder.cpp)

target_include_directories(kokoro PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(kokoro
  PUBLIC
    spdlog::spdlog
    fmt::fmt
    nlohmann_json::nlohmann_json
  PRIVATE
    misaki
    onnxruntime
    espeak-ng
    openblas
    ${BLAS_LIBRARIES}
    pthread
    dl)

if(USE_RKNN)
  if(EXISTS "${CMAKE_SOURCE_DIR}/third_party/rknn/include/rknn_api.h")
    include_directories(${CMAKE_SOURCE_DIR}/third_party/rknn/include)
  endif()
  target_sources(kokoro PRIVATE src/rknn-decoder.cpp src/rknn-matmul-op.cpp)
  target_compile_definitions(kokoro PRIVATE USE_RKNN)
  target_link_libraries(kokoro PRIVATE rknnrt)
endif()

if(USE_ORT_CUDA)
  target_compile_definitions(kokoro PRIVATE USE_ORT_CUDA)
endif()

# Copy runtime data into the build directory (once per build).
if(BUILD_TESTS AND NOT EXISTS "${RU_ESPEAK_DATA_DIR}/ru_dict")
  message(FATAL_ERROR
    "BUILD_TESTS requires Russian espeak-ng-data at ${RU_ESPEAK_DATA_DIR} "
    "(e.g. -DRU_ESPEAK_DATA_DIR=/opt/espeak-ng-data in the dev container)")
endif()
if(EXISTS "${RU_ESPEAK_DATA_DIR}/ru_dict")
  set(_espeak_stage
    COMMAND cp -aL ${RU_ESPEAK_DATA_DIR} ${CMAKE_BINARY_DIR}/espeak-ng-data)
else()
  set(_espeak_stage
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/espeak-ng-data)
endif()
add_custom_target(stage_runtime_data_target ALL
  COMMAND ${CMAKE_COMMAND} -E rm -rf
          ${CMAKE_BINARY_DIR}/misaki-data
          ${CMAKE_BINARY_DIR}/espeak-ng-data
  COMMAND cp -aL ${MISAKI_DATA_DIR} ${CMAKE_BINARY_DIR}/misaki-data
  ${_espeak_stage}
  COMMENT "Staging misaki + Russian espeak data into build dir")
function(stage_runtime_data tgt)
  add_dependencies(${tgt} stage_runtime_data_target)
endfunction()

# ============== kokoro-cli ===============================================
if(BUILD_CLI)
  add_executable(kokoro-cli cli/main.cpp)
  target_link_libraries(kokoro-cli PRIVATE kokoro)
  stage_runtime_data(kokoro-cli)
endif()

# ============== kokoro-server ============================================
if(BUILD_SERVER)
  find_package(Drogon REQUIRED)
  find_package(Opus REQUIRED)

  set(OPUSENC_PREFIX "${DEPS_PREFIX}/opusenc-install")
  if(EXISTS "${OPUSENC_PREFIX}/include/opus/opusenc.h")
    set(OPUSENC_INCLUDE_DIR "${OPUSENC_PREFIX}/include")
  endif()
  find_library(OPUSENC_LIBRARY NAMES opusenc
               PATHS "${OPUSENC_PREFIX}/lib" NO_DEFAULT_PATH)
  if(NOT OPUSENC_LIBRARY)
    find_library(OPUSENC_LIBRARY NAMES opusenc)
  endif()
  if(NOT OPUSENC_INCLUDE_DIR OR NOT OPUSENC_LIBRARY)
    message(FATAL_ERROR
      "libopusenc not found. Build into deps/opusenc-install (see README).")
  endif()

  find_path(LAME_INCLUDE_DIR lame/lame.h)
  find_library(LAME_LIBRARY NAMES mp3lame)
  if(NOT LAME_INCLUDE_DIR OR NOT LAME_LIBRARY)
    message(FATAL_ERROR "libmp3lame not found (install libmp3lame-dev)")
  endif()

  add_executable(kokoro-server
    server/main.cpp
    server/api.cpp
    server/engine_pool.cpp
    server/mp3_encoder.cpp
    server/OggOpusEncoder.cpp)

  target_include_directories(kokoro-server PRIVATE
    ${OPUS_INCLUDE_DIRS} ${OPUSENC_INCLUDE_DIR} ${LAME_INCLUDE_DIR})
  target_link_libraries(kokoro-server PRIVATE
    kokoro
    Drogon::Drogon
    soxr
    ${OPUSENC_LIBRARY}
    ${OPUS_LIBRARIES}
    ${LAME_LIBRARY}
    ogg)
  target_precompile_headers(kokoro-server PRIVATE server/pch.hpp)
  stage_runtime_data(kokoro-server)
endif()

# ============== tests ====================================================
option(BUILD_TESTS "Build unit/integration tests" ON)
if(BUILD_TESTS)
  enable_testing()
  add_executable(test-ru-g2p tests/test_ru_g2p.cpp)
  target_link_libraries(test-ru-g2p PRIVATE kokoro)
  stage_runtime_data(test-ru-g2p)
  add_test(NAME ru_g2p_unit COMMAND test-ru-g2p WORKING_DIRECTORY ${CMAKE_BINARY_DIR})

  if(BUILD_CLI AND USE_RKNN)
    add_test(
      NAME cli_russian_npu
      COMMAND kokoro-cli --text "Привет" --voice sveta
        --out ${CMAKE_BINARY_DIR}/test-out.wav
      WORKING_DIRECTORY ${CMAKE_BINARY_DIR})
    set_tests_properties(cli_russian_npu PROPERTIES
      PASS_REGULAR_EXPRESSION "wrote .*test-out.wav")
  endif()
endif()
