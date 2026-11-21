// The Inkling 2 vision tower on the CPU (transformers Inkling2VisionModel + Inkling2VisionMerger):
//   patches [4T, 1200] -> patch embedding (Linear, no bias) + bilinearly resampled 32 x 32 position table,
//   class token in front -> pre-LayerNorm -> 23 x [x += ls_1 * attn(LN1(x)); x += ls_2 * mlp(LN2(x))]
//   (16 heads of 64, full non-causal attention, axial 2D RoPE on q / k, exact-erf GELU MLP 1024 -> 4096)
//   -> drop the class token, 2 x 2 patches per image token -> proj_in 4096 -> GELU -> proj_out 2304.
//
// Weights stay BF16 in host memory (0.59 GiB), packed for the GEMM microkernel; activations are FP32 between
// operations and BF16 at GEMM inputs (VDPBF16PS: BF16 products, FP32 accumulation). Requires AVX-512 BF16
// (vision_supported()).
#pragma once

#include "vision/image.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

namespace ingine {
class SafetensorsModel;
}
namespace ingine::offload {
class CpuPool;
}

namespace ingine::vision {

inline constexpr int kVitHidden = 1024, kVitLayers = 23, kVitHeads = 16, kVitHeadDim = 64, kVitMlp = 4096;
inline constexpr int kPosGrid = 32, kTextHidden = 2304, kProjHidden = 4096;

class VisionTower {
 public:
  // Reads model.visual.* from the checkpoint (BF16) into host memory, packing on `pool`.
  VisionTower(const SafetensorsModel& ckpt, offload::CpuPool& pool);
  ~VisionTower();
  VisionTower(const VisionTower&) = delete;
  VisionTower& operator=(const VisionTower&) = delete;

  // One image -> BF16 bits [rows * cols, 2304]. Not reentrant (scratch buffers are members).
  std::vector<std::uint16_t> forward(const Patches& p);
  std::size_t weight_bytes() const;
  double load_seconds() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// Kernels, exposed for tests.
namespace cpu {
// W [N][K] BF16 (PyTorch Linear layout) -> GEMM panels [N/32][K/2][32][2]. N % 32 == 0, K even.
void pack_weight(const std::uint16_t* w, int N, int K, std::uint16_t* out);
// C [M][N] (row stride ldc) = A [M][K] (BF16, row stride lda) . W^T, W packed by pack_weight.
void gemm_f32(offload::CpuPool& pool, const std::uint16_t* A, std::size_t lda, int M, const std::uint16_t* packed,
              int N, int K, float* C, std::size_t ldc);
// The vectorized erf / exp, one lane at a time.
float erf_lane(float x);
float exp_lane(float x);
}  // namespace cpu

}  // namespace ingine::vision
