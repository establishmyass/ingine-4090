# Expert offloading (pinned host experts, VRAM slot cache, CPU expert pool). The CPU kernels need AVX-512 at
# run time (checked before use); only cpu_experts.cpp is compiled for it.
add_library(ingine_offload STATIC
  expert_store.cpp
  cpu_pool.cpp
  cpu_experts.cpp
)
ingine_internal_includes(ingine_offload)
target_include_directories(ingine_offload PUBLIC ${PROJECT_SOURCE_DIR}/src)
target_link_libraries(ingine_offload PUBLIC ingine_core ingine_kernels_moe CUDA::cudart Threads::Threads)
set_source_files_properties(cpu_experts.cpp PROPERTIES COMPILE_OPTIONS
  "-O3;-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512bf16;-mf16c;-mfma;-fno-fast-math")
set_source_files_properties(cpu_pool.cpp PROPERTIES COMPILE_OPTIONS "-O2;-msse2")
