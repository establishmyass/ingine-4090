# Decode kernel tests against FP64 host oracles (synthetic weights at the model's shapes, < 1 GB of GPU memory).
ingine_add_test(NAME kernels_dense SOURCES test_dense.cpp LIBS ingine_kernels_dense GPU ORACLE TIMEOUT 600)
ingine_add_test(NAME kernels_moe SOURCES test_moe.cpp LIBS ingine_kernels_moe GPU ORACLE TIMEOUT 600)
ingine_add_test(NAME kernels_attn SOURCES test_attn.cpp LIBS ingine_kernels_attn GPU ORACLE TIMEOUT 900)
ingine_add_test(NAME kernels_head SOURCES test_head.cpp LIBS ingine_kernels_head GPU ORACLE TIMEOUT 600)

# Decode kernel timings at production sizes (graph-replayed like the decode step); exits 77 unless INGINE_BENCH=1.
ingine_add_test(NAME kernels_bench SOURCES bench_decode.cu LIBS ingine_kernels_moe ingine_kernels_dense ingine_kernels_attn GPU
                LABELS bench TIMEOUT 900)
