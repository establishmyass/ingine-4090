// CPU expert kernels for offloaded experts (AVX-512): one routed expert's W13 (+ softcapped SwiGLU) and W2
// rows for 1..8 tokens, reading the checkpoint encoding in pinned host memory.
//
// Numerics follow the GPU kernels (kernels/moe/moe.cu), not a quantized approximation: every weight is decoded
// to the BF16 value model.decode_experts produces (NVFP4: bf16(fp32(e2m1 * e4m3 scale) * factor); MXFP8:
// e4m3 * 2^(e8m0 - 127), exact), activations stay BF16 values, products are exact in FP32 and accumulate in
// FP32 (AVX512_BF16 VDPBF16PS: two products per lane per instruction; subnormal BF16 inputs count as zero).
// gate / up sums and the activation are rounded to BF16 where the GPU rounds them, and the W2 outputs are
// BF16-rounded. Only the FP32 summation order differs from the GPU's tensor-core order.
//
// Structure adapted from Strata's NVFP4 kernels (github.com/Niko1221/Strata, src/kernels/cpu/nvfp4_avx512.cpp,
// MIT): register-blocked rows for several tokens per weight load and software prefetch ahead of the stream.
// Strata quantizes activations to Q8_0 and uses VNNI; here the activations are exact, which costs nothing
// measurable because both are bound by DRAM bandwidth.
#pragma once

#include <cstddef>
#include <cstdint>

namespace ingine::offload {

inline constexpr int kH = 2304;  // hidden
inline constexpr int kI = 1152;  // expert intermediate

enum class ExpertFmt : int { NVFP4 = 0, MXFP8 = 1 };

// One expert in checkpoint encoding (host pointers). Row strides: NVFP4 codes K/2, scales K/16 (e4m3);
// MXFP8 codes K, scales K/32 (E8M0). W13 rows interleave gate (2p) and up (2p + 1).
struct ExpertView {
  ExpertFmt fmt = ExpertFmt::NVFP4;
  const std::uint8_t* w13 = nullptr;
  const std::uint8_t* w13s = nullptr;
  const std::uint8_t* w2 = nullptr;
  const std::uint8_t* w2s = nullptr;
  float f13 = 1.f, f2 = 1.f;  // NVFP4 per-expert factors
};

// Activations of one token as BF16 bits for both formats: `nat` in element order (MXFP8), `perm` with each
// 32-value chunk in the NVFP4 kernel's order (its even elements, then its odd ones, as VDPBF16PS pairs them).
struct PreparedRow {
  const std::uint16_t* nat = nullptr;
  const std::uint16_t* perm = nullptr;
};
void prepare_row(const std::uint16_t* nat, int K, std::uint16_t* perm);

// act[t][p] (BF16 bits) for gate/up pairs p in [p0, p1), nt tokens (<= 8).
void expert_up(const ExpertView& e, const PreparedRow* x, int nt, std::uint16_t* const* act, int p0, int p1);
// y[t][r] = bf16(W2 row r . act[t]) for rows r in [r0, r1), written as BF16 bits; act prepared (K = I).
void expert_down(const ExpertView& e, const PreparedRow* act, int nt, std::uint16_t* const* y, int r0, int r1);

// Whether this CPU runs the kernels (AVX-512 F/BW/VL/DQ, AVX512_BF16, F16C).
bool cpu_experts_supported();

// FP32 -> BF16 bits, round to nearest even (keeps subnormals, like __float2bfloat16_rn).
inline std::uint16_t bf16_bits(float f) {
  std::uint32_t u;
  __builtin_memcpy(&u, &f, 4);
  if ((u & 0x7f800000u) == 0x7f800000u) return std::uint16_t((u >> 16) | ((u & 0xffffu) ? 0x40u : 0u));
  u += 0x7fffu + ((u >> 16) & 1u);
  return std::uint16_t(u >> 16);
}
inline float bf16_float(std::uint16_t b) {
  const std::uint32_t u = std::uint32_t(b) << 16;
  float f;
  __builtin_memcpy(&f, &u, 4);
  return f;
}

}  // namespace ingine::offload
