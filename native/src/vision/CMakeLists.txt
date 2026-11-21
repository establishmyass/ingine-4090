# ingine_vision: image decoding (stb_image) and the Inkling 2 vision tower on the CPU. The tower's kernels need
# AVX-512 BF16 at run time (vision_supported()); only vit.cpp is compiled for it.
add_library(ingine_vision STATIC
  image.cpp
  vit.cpp
  encoder.cpp
)
ingine_internal_includes(ingine_vision)
target_include_directories(ingine_vision PUBLIC ${PROJECT_SOURCE_DIR}/src ${PROJECT_SOURCE_DIR}/include)
target_link_libraries(ingine_vision PUBLIC ingine_core ingine_offload spdlog::spdlog Threads::Threads)
set_source_files_properties(vit.cpp PROPERTIES COMPILE_OPTIONS
  "-O3;-mavx512f;-mavx512bw;-mavx512vl;-mavx512dq;-mavx512bf16;-mfma;-fno-fast-math")
# stb_image: third-party code, quiet
set_source_files_properties(image.cpp PROPERTIES COMPILE_OPTIONS "-O2;-w")
