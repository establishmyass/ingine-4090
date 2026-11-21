# Expert offloading: CPU expert kernels vs the GPU kernels on real checkpoint experts.
ingine_add_test(NAME offload_cpu_experts SOURCES test_cpu_experts.cpp LIBS ingine_offload ingine_kernels_moe ingine_core
                GPU MODEL TIMEOUT 600)
