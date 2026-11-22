# Adapted from fngine (github.com/satellitedown/fngine, Apache-2.0) for ingine.
cmake_minimum_required(VERSION 3.28)

# Select a single target before CUDA compiler detection. This fork defaults to the RTX 4090.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 89 CACHE STRING "CUDA architectures to build")
endif()
if(NOT CMAKE_CUDA_ARCHITECTURES MATCHES "^(120a|89|86)$")
  message(FATAL_ERROR
    "ingine supports only CMAKE_CUDA_ARCHITECTURES=120a (RTX 5090), 89 (RTX 4090), or 86 (RTX 3090); got "
    "'${CMAKE_CUDA_ARCHITECTURES}'")
endif()

# Prefer the installer's private toolkit, then /opt/cuda, when nvcc is not chosen explicitly.
if(NOT DEFINED CMAKE_CUDA_COMPILER AND NOT DEFINED ENV{CUDACXX})
  foreach(_nvcc ${CMAKE_CURRENT_SOURCE_DIR}/../.cuda-toolkit/nvidia/cu13/bin/nvcc
                ${CMAKE_CURRENT_SOURCE_DIR}/../.cuda-toolkit/bin/nvcc /opt/cuda/bin/nvcc)
    if(EXISTS ${_nvcc})
      set(CMAKE_CUDA_COMPILER ${_nvcc} CACHE FILEPATH "CUDA compiler")
      break()
    endif()
  endforeach()
endif()

project(ingine LANGUAGES C CXX CUDA)

option(INGINE_BUILD_TESTS "Build ingine tests" OFF)
option(INGINE_BUILD_APPS "Build the ingine executables (serve, bench, validate)" ON)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CXX_SCAN_FOR_MODULES OFF) # no C++20 modules; skip the dependency scan
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_GENERATOR MATCHES "Ninja")
  set_property(GLOBAL PROPERTY JOB_POOLS ingine_link=1)
  set(CMAKE_JOB_POOL_LINK ingine_link)
endif()

include(cmake/Dependencies.cmake)

# Both SM8x targets use stream ordering and the cluster-free kernels. Ada has native FP8 conversion.
if(CMAKE_CUDA_ARCHITECTURES MATCHES "^(86|89)$")
  add_compile_definitions(INGINE_AMPERE=1)
endif()

add_subdirectory(third_party)
add_subdirectory(src/core)

# Every other component directory owns its CMakeLists.txt and defines the target named here; a
# directory is added only once it has one, so the tree builds while components are in progress.
#   src/text              -> ingine_text
#   src/kernels/<area>    -> ingine_kernels_<area>   (aggregated by ingine_kernels)
#   src/model             -> ingine_model
#   src/engine            -> ingine_engine
#   src/serve             -> ingine_serve
#   apps/serve|bench|validate -> ingine-serve | ingine-bench | ingine-validate
function(ingine_add_component dir)
  if(EXISTS ${PROJECT_SOURCE_DIR}/${dir}/CMakeLists.txt)
    add_subdirectory(${dir})
  endif()
endfunction()

ingine_add_component(src/text)

set(INGINE_KERNEL_AREAS common dense moe attn head prefill draft)
foreach(area ${INGINE_KERNEL_AREAS})
  ingine_add_component(src/kernels/${area})
endforeach()
add_library(ingine_kernels INTERFACE)
foreach(area ${INGINE_KERNEL_AREAS})
  if(TARGET ingine_kernels_${area})
    target_link_libraries(ingine_kernels INTERFACE ingine_kernels_${area})
  endif()
endforeach()

ingine_add_component(src/offload)
ingine_add_component(src/vision)
ingine_add_component(src/model)
ingine_add_component(src/engine)
ingine_add_component(src/serve)

if(INGINE_BUILD_APPS)
  # The shell launcher can use a source build as well as the private-toolkit installation.
  find_package(Python3 QUIET COMPONENTS Interpreter)
  if(Python3_Interpreter_FOUND)
    set(_manifest "${PROJECT_SOURCE_DIR}/../runtime-manifest.json")
    set(_profile_writer "${PROJECT_SOURCE_DIR}/../scripts/write_profile.py")
    set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS "${_manifest}" "${_profile_writer}")
    execute_process(COMMAND "${Python3_EXECUTABLE}" "${_profile_writer}" "${_manifest}"
                    OUTPUT_FILE "${PROJECT_BINARY_DIR}/profile.env.tmp" COMMAND_ERROR_IS_FATAL ANY)
    file(RENAME "${PROJECT_BINARY_DIR}/profile.env.tmp" "${PROJECT_BINARY_DIR}/profile.env")
  endif()
  foreach(app serve bench validate)
    ingine_add_component(apps/${app})
  endforeach()
endif()

if(INGINE_BUILD_TESTS)
  enable_testing()
  add_subdirectory(tests)
endif()
