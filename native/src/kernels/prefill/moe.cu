// Prefill MoE (model._moe / _moe_grouped for T tokens): router selection, expert-sorted pair lists, grouped
// expert GEMMs that decode the NVFP4 / MXFP8 weights on the fly, and the deterministic combine.
//
// Grouped GEMM design (structure from fngine src/kernels/moe/moe_grouped_gemm.cu). A CTA computes 128
// (token, expert) pairs of one expert x 128 weight rows, K in steps of 64, 8 warps as 2 (rows) x 4 (columns),
// mma.m16n8k16 BF16 with FP32 accumulation. Per step the next activation tile is gathered with cp.async and
// the next weight tile's raw codes/scales are loaded into registers while the current step's MMAs run; the
// codes are then decoded exactly as model.decode_experts / prefill dequant (NVFP4: bf16((e2m1 * e4m3 scale) *
// factor), MXFP8: bf16(e4m3 * 2^(e8m0 - 127))) into the other BF16 shared-memory buffer, so the expert matrices
// are never materialised. One barrier per step. m16 row blocks past the tile's pair count skip their MMAs.
// gate/up epilogue: w13 rows are interleaved (2i gate, 2i + 1 up), so each thread holds gate and up of the same
// i: act = bf16(softcap(bf16 g, 4) * sigmoid(bf16 g) * softcap(bf16 u, 25)). down epilogue: bf16 per pair,
// placed at t * 8 + slot; moe_combine sums each token's slots in slot order (no atomics: deterministic).
#include "kernels/prefill/common.cuh"
#include "kernels/prefill/prefill.h"

#include <cmath>
#include <stdexcept>

namespace ingine::kernels::prefill {
namespace {

constexpr int TM = kMoeTileRows, TN = 128, BK = 64;
constexpr int NT = 256;
constexpr int KCH = BK / 8;  // 16-byte chunks per tile row
constexpr int NTILES_N = kHidden / TN;  // 18 (gate/up: 2304 interleaved rows; down: 2304 outputs)

__constant__ float kE2M1[16] = {0.f, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0.f, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f};

struct GemmSmem {
  uint16_t a[2][TM * BK];
  uint16_t w[2][TN * BK];
  float lut[16];
};

template <int FMT>
struct WRaw;
template <>
struct WRaw<0> {  // NVFP4: 32 codes (16 bytes) + 2 block scales
  uint4 codes;
  uint16_t scales;
};
template <>
struct WRaw<1> {  // MXFP8: 32 codes + 1 E8M0 scale
  uint4 codes[2];
  uint8_t scale;
};

// Thread -> weight row tid / 2, elements [32 (tid & 1), +32) of the step.
template <int FMT, int K>
__device__ __forceinline__ void load_w(const uint8_t* codes_row, const uint8_t* scales_row, int ks, WRaw<FMT>& r) {
  const int half = threadIdx.x & 1;
  if constexpr (FMT == 0) {
    r.codes = __ldg(reinterpret_cast<const uint4*>(codes_row + ks * (BK / 2) + half * 16));
    r.scales = __ldg(reinterpret_cast<const uint16_t*>(scales_row + ks * (BK / 16) + half * 2));
  } else {
    const uint4* c = reinterpret_cast<const uint4*>(codes_row + ks * BK + half * 32);
    r.codes[0] = __ldg(c);
    r.codes[1] = __ldg(c + 1);
    r.scale = __ldg(scales_row + ks * (BK / 32) + half);
  }
}

template <int FMT>
__device__ __forceinline__ void store_w(uint16_t* w, const float* lut, float factor, const WRaw<FMT>& r) {
  const int row = threadIdx.x >> 1, chunk0 = (threadIdx.x & 1) * 4;
  if constexpr (FMT == 0) {
    const uint32_t cw[4] = {r.codes.x, r.codes.y, r.codes.z, r.codes.w};
    const float2 sc = e4m3x2_to_float2(r.scales);
#pragma unroll
    for (int q = 0; q < 4; ++q) {  // 8 elements (4 bytes) per 16-byte chunk; block scale per 16 elements
      const float scale = q < 2 ? sc.x : sc.y;
      uint32_t out[4];
#pragma unroll
      for (int b = 0; b < 4; ++b) {
        const uint32_t byte = (cw[q] >> (8 * b)) & 0xffu;
        out[b] = pack_bf16x2((lut[byte & 0xf] * scale) * factor, (lut[byte >> 4] * scale) * factor);
      }
      *reinterpret_cast<uint4*>(w + swz<KCH>(row, chunk0 + q)) = make_uint4(out[0], out[1], out[2], out[3]);
    }
  } else {
    const float s = exp2f(float(r.scale) - 127.f);
#pragma unroll
    for (int q = 0; q < 4; ++q) {
      const uint4 c = r.codes[q >> 1];
      const uint32_t w0 = (q & 1) ? c.z : c.x, w1 = (q & 1) ? c.w : c.y;
      const float2 f0 = e4m3x2_to_float2(w0 & 0xffff), f1 = e4m3x2_to_float2(w0 >> 16);
      const float2 f2 = e4m3x2_to_float2(w1 & 0xffff), f3 = e4m3x2_to_float2(w1 >> 16);
      *reinterpret_cast<uint4*>(w + swz<KCH>(row, chunk0 + q)) =
          make_uint4(pack_bf16x2(f0.x * s, f0.y * s), pack_bf16x2(f1.x * s, f1.y * s), pack_bf16x2(f2.x * s, f2.y * s),
                     pack_bf16x2(f3.x * s, f3.y * s));
    }
  }
}

__device__ __forceinline__ float softcap(float x, float cap) { return tanhf(x * (1.f / cap)) * cap; }

// UP: A = x rows of the pairs' tokens (K = H), W = w13 (N = 2I), out act [pairs, I].
// !UP: A = act rows (K = I), W = w2 (N = H), out down [T * 8, H] at pair_dst.
template <int FMT, bool UP>
__global__ void __launch_bounds__(NT, 1)
    moe_gemm_kernel(const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales,
                    const float* __restrict__ factors, const __nv_bfloat16* __restrict__ a_src, MoeWork w,
                    __nv_bfloat16* __restrict__ out, const kernels::moe::ExpertTable* __restrict__ table) {
  constexpr int K = UP ? kHidden : kInter;
  constexpr int N = UP ? 2 * kInter : kHidden;
  constexpr int KS = K / BK;
  constexpr int64_t CODE_ROW = FMT == 0 ? K / 2 : K;
  constexpr int64_t SCALE_ROW = FMT == 0 ? K / 16 : K / 32;
  extern __shared__ __align__(128) unsigned char smem_raw[];
  GemmSmem& s = *reinterpret_cast<GemmSmem*>(smem_raw);

  const int tile_idx = blockIdx.x / NTILES_N, nt_idx = blockIdx.x % NTILES_N;
  if (tile_idx >= *w.tiles_n) return;
  const int4 tile = w.tiles[tile_idx];
  const int e = tile.x, pb = tile.y, n = tile.z;
  const int n0 = nt_idx * TN;
  const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
  if (tid < 16) s.lut[tid] = kE2M1[tid];

  // activation rows of this thread's cp.async chunks: rows tid / 8 + 32 i, chunk tid % 8
  const uint16_t* arow[TM / 32];
#pragma unroll
  for (int i = 0; i < TM / 32; ++i) {
    const int r = (tid >> 3) + 32 * i;
    const int rr = r < n ? r : 0;
    arow[i] = UP ? reinterpret_cast<const uint16_t*>(a_src) + size_t(w.pair_token[pb + rr]) * K
                 : reinterpret_cast<const uint16_t*>(a_src) + size_t(pb + rr) * K;
  }
  auto load_a = [&](int ks, int buf) {
#pragma unroll
    for (int i = 0; i < TM / 32; ++i) {
      const int r = (tid >> 3) + 32 * i, c = tid & 7;
      cp_async16(&s.a[buf][swz<KCH>(r, c)], arow[i] + ks * BK + c * 8, r < n ? 16 : 0);
    }
    cp_commit();
  };
  const uint8_t *codes_row, *scales_row;
  if (table) {  // offloading: the expert's own rows, in VRAM or (cold) read from host memory over PCIe
    const int64_t r = n0 + (tid >> 1);
    codes_row = (UP ? table->w13[e] : table->w2[e]) + r * CODE_ROW;
    scales_row = (UP ? table->w13s[e] : table->w2s[e]) + r * SCALE_ROW;
  } else {
    const int64_t wrow = int64_t(e) * N + n0 + (tid >> 1);
    codes_row = codes + wrow * CODE_ROW;
    scales_row = scales + wrow * SCALE_ROW;
  }
  const float factor = FMT == 0 ? __ldg(factors + e) : 1.f;

  const int wm = warp & 1, wn = warp >> 1;
  const int mact = min(max((n - wm * 64 + 15) / 16, 0), 4);  // active m16 blocks of this warp
  float acc[4][4][4];
#pragma unroll
  for (int a = 0; a < 4; ++a)
#pragma unroll
    for (int b = 0; b < 4; ++b) acc[a][b][0] = acc[a][b][1] = acc[a][b][2] = acc[a][b][3] = 0.f;

  WRaw<FMT> raw;
  load_a(0, 0);
  load_w<FMT, K>(codes_row, scales_row, 0, raw);
  __syncthreads();  // lut
  store_w<FMT>(s.w[0], s.lut, factor, raw);
  cp_wait<0>();
  __syncthreads();

#pragma unroll 1
  for (int ks = 0; ks < KS; ++ks) {
    const int buf = ks & 1;
    if (ks + 1 < KS) {
      load_a(ks + 1, buf ^ 1);
      load_w<FMT, K>(codes_row, scales_row, ks + 1, raw);
    }
    if (mact > 0) {
      const uint16_t* A = s.a[buf];
      const uint16_t* W = s.w[buf];
#pragma unroll
      for (int kk = 0; kk < BK / 16; ++kk) {
        unsigned b[4][2];
#pragma unroll
        for (int np = 0; np < 2; ++np)
          ldmatrix_x4(b[2 * np][0], b[2 * np][1], b[2 * np + 1][0], b[2 * np + 1][1],
                      smem_addr(W + swz<KCH>(wn * 32 + np * 16 + (lane & 7) + ((lane >> 4) << 3),
                                             kk * 2 + ((lane >> 3) & 1))));
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
          if (mt < mact) {
            unsigned a[4];
            ldmatrix_x4(a[0], a[1], a[2], a[3],
                        smem_addr(A + swz<KCH>(wm * 64 + mt * 16 + (lane & 15), kk * 2 + (lane >> 4))));
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) mma_bf16(acc[mt][nt], a, b[nt][0], b[nt][1]);
          }
        }
      }
    }
    if (ks + 1 < KS) {
      store_w<FMT>(s.w[buf ^ 1], s.lut, factor, raw);
      cp_wait<0>();
    }
    __syncthreads();
  }

#pragma unroll
  for (int mt = 0; mt < 4; ++mt) {
    if (mt >= mact) continue;
#pragma unroll
    for (int half = 0; half < 2; ++half) {
      const int row = wm * 64 + mt * 16 + (lane >> 2) + half * 8;
      if (row >= n) continue;
#pragma unroll
      for (int nt = 0; nt < 4; ++nt) {
        const int col = n0 + wn * 32 + nt * 8 + (lane & 3) * 2;
        const float c0 = acc[mt][nt][2 * half], c1 = acc[mt][nt][2 * half + 1];
        if constexpr (UP) {
          const float g = round_bf16(c0), u = round_bf16(c1);
          const float sig = 1.f / (1.f + expf(-g));
          const float v = __fmul_rn(__fmul_rn(softcap(g, 4.f), sig), softcap(u, 25.f));
          reinterpret_cast<uint16_t*>(out)[size_t(pb + row) * kInter + col / 2] = f2bf_bits(v);
        } else {
          const int dst = w.pair_dst[pb + row];
          *reinterpret_cast<uint32_t*>(reinterpret_cast<uint16_t*>(out) + size_t(dst) * kHidden + col) =
              pack_bf16x2(c0, c1);
        }
      }
    }
  }
}

template <int FMT, bool UP>
void launch_gemm(const void* codes, const void* scales, const float* factors, const __nv_bfloat16* a, int T,
                 const MoeWork& w, __nv_bfloat16* out, cudaStream_t stream, const kernels::moe::ExpertTable* table) {
  static const cudaError_t attr = cudaFuncSetAttribute(
      moe_gemm_kernel<FMT, UP>, cudaFuncAttributeMaxDynamicSharedMemorySize, int(sizeof(GemmSmem)));
  if (attr != cudaSuccess) throw std::runtime_error("prefill moe: shared memory attribute failed");
  moe_gemm_kernel<FMT, UP><<<moe_max_tiles(T) * NTILES_N, NT, sizeof(GemmSmem), stream>>>(
      static_cast<const uint8_t*>(codes), static_cast<const uint8_t*>(scales), factors, a, w, out, table);
}

// Warp per token: sigmoid scores, 8 rounds of arg-max over s + bias (lowest expert on ties).
__global__ void topk_kernel(const float* __restrict__ logits, const float* __restrict__ bias, int T,
                            int* __restrict__ idx, float* __restrict__ wts, int* __restrict__ counts) {
  const int t = blockIdx.x * (blockDim.x / 32) + (threadIdx.x >> 5), lane = threadIdx.x & 31;
  if (t >= T) return;
  float sc[4], key[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const int e = lane + 32 * i;
    sc[i] = 1.f / (1.f + expf(-logits[size_t(t) * kExperts + e]));
    key[i] = sc[i] + bias[e];
  }
  int sel_e[kTopK];
  float sel_s[kTopK];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    float best = -INFINITY;
    int be = kExperts;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      if (key[i] > best) best = key[i], be = lane + 32 * i;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
      const float ob = __shfl_xor_sync(0xffffffffu, best, o);
      const int oe = __shfl_xor_sync(0xffffffffu, be, o);
      if (ob > best || (ob == best && oe < be)) best = ob, be = oe;
    }
    sel_e[k] = be;
    float s_be = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i)
      if (lane + 32 * i == be) s_be = sc[i], key[i] = -INFINITY;
    sel_s[k] = __shfl_sync(0xffffffffu, s_be, be & 31);
  }
  if (lane == 0) {
    float sum = sel_s[0];
#pragma unroll
    for (int k = 1; k < kTopK; ++k) sum += sel_s[k];
    const float den = sum + 1e-20f;
#pragma unroll
    for (int k = 0; k < kTopK; ++k) {
      idx[size_t(t) * kTopK + k] = sel_e[k];
      wts[size_t(t) * kTopK + k] = sel_s[k] / den * 8.f;
      atomicAdd(counts + sel_e[k], 1);
    }
  }
}

// One CTA: expert offsets, tiles, scatter of the pairs into expert order (order within an expert does not
// affect any result: every pair row is computed independently).
__global__ void __launch_bounds__(1024) route_kernel(const int* __restrict__ idx, int T, MoeWork w) {
  __shared__ int offs[kExperts], tile0[kExperts + 1], cursor[kExperts];
  const int tid = threadIdx.x;
  if (tid == 0) {
    int o = 0, tl = 0;
    for (int e = 0; e < kExperts; ++e) {
      const int c = w.counts[e];
      offs[e] = o;
      tile0[e] = tl;
      o += c;
      tl += (c + TM - 1) / TM;
    }
    tile0[kExperts] = tl;
    *w.tiles_n = tl;
  }
  if (tid < kExperts) cursor[tid] = 0;
  __syncthreads();
  if (tid < kExperts) {
    const int c = w.counts[tid];
    for (int i = 0; i * TM < c; ++i) w.tiles[tile0[tid] + i] = make_int4(tid, offs[tid] + i * TM, min(TM, c - i * TM), 0);
  }
  for (int p = tid; p < T * kTopK; p += blockDim.x) {
    const int e = idx[p];
    const int pos = offs[e] + atomicAdd(&cursor[e], 1);
    w.pair_token[pos] = p / kTopK;
    w.pair_dst[pos] = p;
  }
}

// out[t] = bf16(sum_slot down[t, slot] * wts[t, slot]), 4 channels per thread.
__global__ void __launch_bounds__(kHidden / 4) combine_kernel(const uint2* __restrict__ down, const float* __restrict__ wts,
                                                            uint2* __restrict__ out) {
  const int t = blockIdx.x, c = threadIdx.x;
  float acc[4];
#pragma unroll
  for (int k = 0; k < kTopK; ++k) {
    const uint2 d = down[(size_t(t) * kTopK + k) * (kHidden / 4) + c];
    const float wk = wts[size_t(t) * kTopK + k];
    const float p[4] = {__fmul_rn(bf_lo(d.x), wk), __fmul_rn(bf_hi(d.x), wk), __fmul_rn(bf_lo(d.y), wk),
                        __fmul_rn(bf_hi(d.y), wk)};
#pragma unroll
    for (int i = 0; i < 4; ++i) acc[i] = k == 0 ? p[i] : __fadd_rn(acc[i], p[i]);
  }
  out[size_t(t) * (kHidden / 4) + c] = make_uint2(pack_bf16x2(acc[0], acc[1]), pack_bf16x2(acc[2], acc[3]));
}

}  // namespace

void moe_topk(const float* logits, const float* bias, int T, int* idx, float* wts, int* counts, cudaStream_t stream) {
  topk_kernel<<<(T + 7) / 8, 256, 0, stream>>>(logits, bias, T, idx, wts, counts);
}

void moe_route(const int* idx, int T, const MoeWork& w, cudaStream_t stream) {
  route_kernel<<<1, 1024, 0, stream>>>(idx, T, w);
}

void moe_gate_up(const MoeExperts& e, const __nv_bfloat16* x, int T, const MoeWork& w, __nv_bfloat16* act,
                 cudaStream_t stream) {
  if (e.fmt == 0) launch_gemm<0, true>(e.w13, e.w13_scale, e.w13_factor, x, T, w, act, stream, e.table);
  else launch_gemm<1, true>(e.w13, e.w13_scale, nullptr, x, T, w, act, stream, e.table);
}

void moe_down(const MoeExperts& e, const __nv_bfloat16* act, int T, const MoeWork& w, __nv_bfloat16* down,
              cudaStream_t stream) {
  if (e.fmt == 0) launch_gemm<0, false>(e.w2, e.w2_scale, e.w2_factor, act, T, w, down, stream, e.table);
  else launch_gemm<1, false>(e.w2, e.w2_scale, nullptr, act, T, w, down, stream, e.table);
}

void moe_combine(const __nv_bfloat16* down, const float* wts, int T, __nv_bfloat16* out, cudaStream_t stream) {
  combine_kernel<<<T, kHidden / 4, 0, stream>>>(reinterpret_cast<const uint2*>(down), wts,
                                                reinterpret_cast<uint2*>(out));
}

}  // namespace ingine::kernels::prefill
