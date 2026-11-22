add_library(ingine_kernels_prefill STATIC
  attention.cu
  moe.cu
  rows.cu
  gemm.cpp
)
if(CMAKE_CUDA_ARCHITECTURES STREQUAL "89")
  target_sources(ingine_kernels_prefill PRIVATE small_moe.cu)
  target_compile_definitions(ingine_kernels_prefill PRIVATE INGINE_ADA_SMALL_MOE=1)
endif()
ingine_cuda_target(ingine_kernels_prefill)
ingine_internal_includes(ingine_kernels_prefill)
target_include_directories(ingine_kernels_prefill PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(ingine_kernels_prefill PUBLIC ingine_core CUDA::cudart CUDA::cublasLt)
