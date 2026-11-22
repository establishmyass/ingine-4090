add_library(ingine_kernels_attn STATIC
  attn.cu
)
ingine_cuda_target(ingine_kernels_attn)
ingine_internal_includes(ingine_kernels_attn)
target_include_directories(ingine_kernels_attn PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(ingine_kernels_attn PUBLIC ingine_core CUDA::cudart)

if(CMAKE_CUDA_ARCHITECTURES STREQUAL "89")
  # Eight KV heads across 128 SMs: two blocks per SM at one row, one at verify widths.
  target_compile_definitions(ingine_kernels_attn PRIVATE INGINE_ATTN_KS1=32 INGINE_ATTN_KSM=16)
endif()
