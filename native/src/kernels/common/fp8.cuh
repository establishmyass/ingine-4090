// Exact E4M3 (FP8) and E2M1 (FP4) -> FP16 decoding for every build target.
//
// RTX 5090 builds (sm_120a) use the hardware conversions (cvt.rn.f16x2.e4m3x2 / .e2m1x2), unchanged.
// The RTX 3090 port (sm_86, INGINE_AMPERE) has neither instruction, and cuda_fp8.h's fallback below sm_89
// decodes one byte at a time with a normalisation loop for subnormals. Here instead each code is moved
// into the top of an FP16 field with a byte permute and two masks:
// * E4M3 s.eeee.mmm -> FP16 s.0eeee.mmm0000000 is the value times 2^-8 (exponent bias 15 vs 7), also for
//   E4M3 subnormals, which land on FP16 subnormals with the same significand;
// * E2M1 s.ee.m -> FP16 s.000ee.m000000000 is the value times 2^-14 (exponent bias 15 vs 1);
// and one __hmul2 by 2^8 / 2^14 restores the value. The product is exact (a power of two times an exact
// FP16 value, result in the normal range), so every finite code decodes to the same FP16 bits as the
// hardware instruction. E4M3 NaN (0x7f / 0xff) decodes to +-480 instead of NaN; ingine never holds one:
// the KV caches and FP8 dense weights are written saturated (SATFINITE) and checkpoint codes are finite.
#pragma once

#include <cuda_fp16.h>

#include <cstdint>

namespace ingine::kernels::fp {

// Two E4M3 codes (low 16 bits of v, element 0 in the low byte) -> f16x2 (element 0 in the low half).
__device__ __forceinline__ uint32_t e4m3x2_to_f16x2(uint32_t v) {
#ifndef INGINE_AMPERE
  uint32_t r;
  asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"(static_cast<uint16_t>(v)));
  return r;
#else
  const uint32_t t = __byte_perm(v, 0u, 0x1404);  // bytes [0, e0, 0, e1]: each code in its lane's high byte
  const uint32_t h = (t & 0x80008000u) | ((t >> 1) & 0x3f803f80u);
  const __half2 r = __hmul2(*reinterpret_cast<const __half2*>(&h), __half2(__half_raw{0x5c00}, __half_raw{0x5c00}));
  return *reinterpret_cast<const uint32_t*>(&r);
#endif
}

// Eight E2M1 codes (byte i = codes 2i (low nibble) and 2i+1) -> four f16x2: h[i] = (code 2i, code 2i+1).
__device__ __forceinline__ void e2m1x8_to_f16x2x4(uint32_t w, uint32_t (&h)[4]) {
#ifndef INGINE_AMPERE
  asm("{\n"
      ".reg .b8 b0, b1, b2, b3;\n"
      "mov.b32 {b0, b1, b2, b3}, %4;\n"
      "cvt.rn.f16x2.e2m1x2 %0, b0;\n"
      "cvt.rn.f16x2.e2m1x2 %1, b1;\n"
      "cvt.rn.f16x2.e2m1x2 %2, b2;\n"
      "cvt.rn.f16x2.e2m1x2 %3, b3;\n"
      "}"
      : "=r"(h[0]), "=r"(h[1]), "=r"(h[2]), "=r"(h[3])
      : "r"(w));
#else
  const uint32_t w4 = w << 4;  // byte i of w4 has code 2i in its high nibble; byte i of w has code 2i+1
  const __half2 k = __half2(__half_raw{0x7400}, __half_raw{0x7400});  // 2^14
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    // lane 0 high byte = byte i of w4, lane 1 high byte = byte i of w: codes in bits 15..12 / 31..28
    const uint32_t t = __byte_perm(w4, w, ((4 + i) << 12) | (i << 4));
    const uint32_t u = (t & 0x80008000u) | ((t >> 3) & 0x0e000e00u);
    const __half2 r = __hmul2(*reinterpret_cast<const __half2*>(&u), k);
    h[i] = *reinterpret_cast<const uint32_t*>(&r);
  }
#endif
}

// Two E4M3 codes -> two floats (exact).
__device__ __forceinline__ float2 e4m3x2_to_float2(uint32_t v) {
  const uint32_t h = e4m3x2_to_f16x2(v);
  return __half22float2(*reinterpret_cast<const __half2*>(&h));
}

}  // namespace ingine::kernels::fp
