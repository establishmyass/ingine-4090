// See cpu_experts.h. Compiled with -mavx512f -mavx512bw -mavx512vl -mavx512dq -mavx512bf16 -mf16c -mfma and
// without fast-math: the MXFP8 decode relies on FP32 subnormals not being flushed (MXCSR default).
#include "offload/cpu_experts.h"

#include <immintrin.h>

#include <cmath>

namespace ingine::offload {
namespace {

// E4M3 (e4m3fn) block scales as FP32, exact; 0x7f / 0xff (NaN) never occur in a checkpoint.
struct E4M3Table {
  float v[256];
  constexpr E4M3Table() : v() {
    for (int x = 0; x < 256; ++x) {
      const int e = (x >> 3) & 0xf, m = x & 7;
      float r = 1.f;
      if (e == 0) {
        r = float(m) / 8.f;
        for (int i = 0; i < 6; ++i) r *= 0.5f;  // 2^-6
      } else {
        r = 1.f + float(m) / 8.f;
        for (int i = 7; i < e; ++i) r *= 2.f;
        for (int i = e; i < 7; ++i) r *= 0.5f;
      }
      v[x] = (x & 0x80) ? -r : r;
    }
  }
};
constexpr E4M3Table kE4M3;

constexpr int kPrefetch = 2048;  // bytes ahead in the code stream (Strata's NVFP4 default)

// The two decoded halves of a 32-value chunk as one BF16 vector (round to nearest even; VCVTNE2PS2BF16 treats
// subnormal inputs as zero, and decoded weights never are: |code * scale * factor| >= 2^-10 * factor).
inline __m512bh pack(__m512 lo, __m512 hi) { return _mm512_cvtne2ps_pbh(hi, lo); }  // lanes 0-15 lo, 16-31 hi

template <int NT>
inline void dot_nvfp4(const std::uint8_t* codes, const std::uint8_t* scales, float factor, int K,
                      const PreparedRow* x, float* out) {
  const __m512 lut = _mm512_setr_ps(0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f,
                                    -6.f);
  const __m512 f = _mm512_set1_ps(factor);
  const __m512i m4 = _mm512_set1_epi32(15);
  __m512 acc[NT];
  for (int t = 0; t < NT; ++t) acc[t] = _mm512_setzero_ps();
  const int C = K / 32;
  for (int c = 0; c < C; ++c) {
    _mm_prefetch(reinterpret_cast<const char*>(codes) + 16 * c + kPrefetch, _MM_HINT_T0);
    // byte i = elements 2i (low nibble) and 2i + 1; bytes 0-7 use block scale 2c, bytes 8-15 scale 2c + 1
    const __m512i v = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(codes + 16 * c)));
    const __m512 sv = _mm512_mask_blend_ps(__mmask16(0xff00), _mm512_set1_ps(kE4M3.v[scales[2 * c]]),
                                           _mm512_set1_ps(kE4M3.v[scales[2 * c + 1]]));
    // decode_experts: code * scale exact, one FP32 rounding for the factor, one BF16 rounding (in pack)
    const __m512 wl = _mm512_mul_ps(_mm512_mul_ps(_mm512_permutexvar_ps(_mm512_and_si512(v, m4), lut), sv), f);
    const __m512 wh = _mm512_mul_ps(_mm512_mul_ps(_mm512_permutexvar_ps(_mm512_srli_epi32(v, 4), lut), sv), f);
    const __m512bh w = pack(wl, wh);  // lanes: even elements, then odd ones (prepare_row's `perm` order)
    for (int t = 0; t < NT; ++t)
      acc[t] = _mm512_dpbf16_ps(acc[t], w, (__m512bh)_mm512_loadu_si512(x[t].perm + 32 * c));
  }
  for (int t = 0; t < NT; ++t) out[t] = _mm512_reduce_add_ps(acc[t]);
}

template <int NT>
inline void dot_mxfp8(const std::uint8_t* codes, const std::uint8_t* scales, int K, const PreparedRow* x,
                      float* out) {
  const __m512i sign = _mm512_set1_epi32(0x80), mag = _mm512_set1_epi32(0x7f);
  const __m512 two120 = _mm512_set1_ps(0x1p120f);  // e4m3 bits in an FP32 field are the value times 2^-120
  __m512 acc[NT];
  for (int t = 0; t < NT; ++t) acc[t] = _mm512_setzero_ps();
  const int C = K / 32;
  auto decode16 = [&](const std::uint8_t* p, __m512 scv) {
    const __m512i v = _mm512_cvtepu8_epi32(_mm_loadu_si128(reinterpret_cast<const __m128i*>(p)));
    const __m512i bits = _mm512_or_si512(_mm512_slli_epi32(_mm512_and_si512(v, sign), 24),
                                         _mm512_slli_epi32(_mm512_and_si512(v, mag), 20));
    // exact: the e4m3 value (subnormals included, no DAZ in FP32) times two powers of two
    return _mm512_mul_ps(_mm512_mul_ps(_mm512_castsi512_ps(bits), two120), scv);
  };
  for (int c = 0; c < C; ++c) {
    _mm_prefetch(reinterpret_cast<const char*>(codes) + 32 * c + kPrefetch, _MM_HINT_T0);
    const std::uint8_t s = scales[c];
    const std::uint32_t sb = s ? std::uint32_t(s) << 23 : 0x00400000u;  // 2^(s - 127); s = 0 -> 2^-127
    float sc;
    __builtin_memcpy(&sc, &sb, 4);
    const __m512 scv = _mm512_set1_ps(sc);
    const __m512bh w = pack(decode16(codes + 32 * c, scv), decode16(codes + 32 * c + 16, scv));  // exact in BF16
    for (int t = 0; t < NT; ++t)
      acc[t] = _mm512_dpbf16_ps(acc[t], w, (__m512bh)_mm512_loadu_si512(x[t].nat + 32 * c));
  }
  for (int t = 0; t < NT; ++t) out[t] = _mm512_reduce_add_ps(acc[t]);
}

template <int NT>
inline void dot_row(const ExpertView& e, bool w13, int row, const PreparedRow* x, float* out) {
  const int K = w13 ? kH : kI;
  const std::uint8_t* codes = w13 ? e.w13 : e.w2;
  const std::uint8_t* scales = w13 ? e.w13s : e.w2s;
  if (e.fmt == ExpertFmt::NVFP4)
    dot_nvfp4<NT>(codes + std::size_t(row) * (K / 2), scales + std::size_t(row) * (K / 16), w13 ? e.f13 : e.f2, K, x,
                  out);
  else
    dot_mxfp8<NT>(codes + std::size_t(row) * K, scales + std::size_t(row) * (K / 32), K, x, out);
}

inline float round_bf16(float v) { return bf16_float(bf16_bits(v)); }

template <int NT>
void up_rows(const ExpertView& e, const PreparedRow* x, std::uint16_t* const* act, int p0, int p1) {
  float g[NT], u[NT];
  for (int p = p0; p < p1; ++p) {
    dot_row<NT>(e, true, 2 * p, x, g);
    dot_row<NT>(e, true, 2 * p + 1, x, u);
    for (int t = 0; t < NT; ++t) {
      // moe_up_kernel: gate / up rounded to BF16, then softcap(g, 4) * sigmoid(g) * softcap(u, 25), BF16
      const float gate = round_bf16(g[t]), up = round_bf16(u[t]);
      const float a = std::tanh(gate / 4.f) * 4.f * (1.f / (1.f + std::exp(-gate))) * (std::tanh(up / 25.f) * 25.f);
      act[t][p] = bf16_bits(a);
    }
  }
}

template <int NT>
void down_rows(const ExpertView& e, const PreparedRow* a, std::uint16_t* const* y, int r0, int r1) {
  float s[NT];
  for (int r = r0; r < r1; ++r) {
    dot_row<NT>(e, false, r, a, s);
    for (int t = 0; t < NT; ++t) y[t][r] = bf16_bits(s[t]);
  }
}

}  // namespace

void prepare_row(const std::uint16_t* nat, int K, std::uint16_t* perm) {
  for (int c = 0; c < K / 32; ++c)
    for (int j = 0; j < 16; ++j) {
      perm[32 * c + j] = nat[32 * c + 2 * j];
      perm[32 * c + 16 + j] = nat[32 * c + 2 * j + 1];
    }
}

void expert_up(const ExpertView& e, const PreparedRow* x, int nt, std::uint16_t* const* act, int p0, int p1) {
  switch (nt) {
    case 1: up_rows<1>(e, x, act, p0, p1); break;
    case 2: up_rows<2>(e, x, act, p0, p1); break;
    case 3: up_rows<3>(e, x, act, p0, p1); break;
    case 4: up_rows<4>(e, x, act, p0, p1); break;
    case 5: up_rows<5>(e, x, act, p0, p1); break;
    case 6: up_rows<6>(e, x, act, p0, p1); break;
    case 7: up_rows<7>(e, x, act, p0, p1); break;
    default: up_rows<8>(e, x, act, p0, p1); break;
  }
}

void expert_down(const ExpertView& e, const PreparedRow* act, int nt, std::uint16_t* const* y, int r0, int r1) {
  switch (nt) {
    case 1: down_rows<1>(e, act, y, r0, r1); break;
    case 2: down_rows<2>(e, act, y, r0, r1); break;
    case 3: down_rows<3>(e, act, y, r0, r1); break;
    case 4: down_rows<4>(e, act, y, r0, r1); break;
    case 5: down_rows<5>(e, act, y, r0, r1); break;
    case 6: down_rows<6>(e, act, y, r0, r1); break;
    case 7: down_rows<7>(e, act, y, r0, r1); break;
    default: down_rows<8>(e, act, y, r0, r1); break;
  }
}

bool cpu_experts_supported() {
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
         __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq") &&
         __builtin_cpu_supports("avx512bf16") && __builtin_cpu_supports("f16c") && __builtin_cpu_supports("fma");
}

}  // namespace ingine::offload
