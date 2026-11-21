// CPU vision tower (see vit.h). Compiled with AVX-512 F/BW/VL/DQ + AVX512_BF16; callers check
// vision_supported() first.
//
// GEMM: C = A . W^T with W packed in 32-column panels of BF16 pairs ([N/32][K/2][32][2]), so one 64-byte load
// is 16 columns x one k-pair, which VDPBF16PS multiplies with a broadcast pair of A. A 12 x 32 register tile
// (24 accumulators) per microkernel call, K in blocks of 128 pairs, jobs of 96 x 128 output elements on the
// CPU pool, each finished by an epilogue (bias, GELU, layer-scale residual, BF16 store, RoPE + head packing)
// while its tile is still in cache. The blocking follows the usual BLIS / llamafile tinyBLAS structure.
#include "vision/vit.h"

#include "core/file_read.h"
#include "core/safetensors.h"
#include "offload/cpu_pool.h"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ingine::vision {
namespace {

using u16 = std::uint16_t;

constexpr int MR = 12, NR = 32, KC = 128;  // microkernel rows / columns, K block (pairs)
constexpr int MB = 96, NB = 128;           // job tile
constexpr int QB = 48;                     // attention query rows per job

// ---- aligned buffers ------------------------------------------------------------------------------------------

template <class T>
struct Buf {
  T* p = nullptr;
  std::size_t n = 0;
  Buf() = default;
  explicit Buf(std::size_t count) { resize(count); }
  ~Buf() { std::free(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
  Buf(Buf&& o) noexcept : p(o.p), n(o.n) { o.p = nullptr, o.n = 0; }
  Buf& operator=(Buf&& o) noexcept {
    std::swap(p, o.p), std::swap(n, o.n);
    return *this;
  }
  void resize(std::size_t count) {  // contents are not kept
    if (count <= n) return;
    std::free(p);
    const std::size_t bytes = (count * sizeof(T) + 63) / 64 * 64;
    p = static_cast<T*>(std::aligned_alloc(64, bytes));
    if (!p) throw std::bad_alloc();
    n = count;
  }
  T* data() const { return p; }
  T& operator[](std::size_t i) const { return p[i]; }
};

// ---- scalar / vector helpers ----------------------------------------------------------------------------------

inline float bf16f(u16 b) {
  const std::uint32_t u = std::uint32_t(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
inline u16 f2bf(float f) {  // round to nearest even (finite)
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
  return u16(u >> 16);
}

inline void store_bf16(u16* dst, __m512 v) {
  _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst), (__m256i)_mm512_cvtneps_pbh(v));
}

// exp: 2^n * e^r, n = round(x / ln 2), |r| <= ln 2 / 2, degree-6 Taylor polynomial (< 2 ulp); x <= 88.
inline __m512 exp512(__m512 x) {
  x = _mm512_max_ps(x, _mm512_set1_ps(-87.f));
  const __m512 n = _mm512_roundscale_ps(_mm512_mul_ps(x, _mm512_set1_ps(1.44269504088896341f)),
                                        _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
  __m512 r = _mm512_fnmadd_ps(n, _mm512_set1_ps(0.693359375f), x);
  r = _mm512_fnmadd_ps(n, _mm512_set1_ps(-2.12194440e-4f), r);
  __m512 p = _mm512_set1_ps(1.f / 720);
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.f / 120));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.f / 24));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.f / 6));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(0.5f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.f));
  p = _mm512_fmadd_ps(p, r, _mm512_set1_ps(1.f));
  return _mm512_scalef_ps(p, n);
}

// erf: rational minimax approximation on [-4, 4] (Eigen / XLA generic_fast_erf_float), |error| ~1e-7.
inline __m512 erf512(__m512 x) {
  x = _mm512_min_ps(_mm512_max_ps(x, _mm512_set1_ps(-4.f)), _mm512_set1_ps(4.f));
  const __m512 x2 = _mm512_mul_ps(x, x);
  __m512 p = _mm512_fmadd_ps(x2, _mm512_set1_ps(-2.72614225801306e-10f), _mm512_set1_ps(2.77068142495902e-08f));
  p = _mm512_fmadd_ps(x2, p, _mm512_set1_ps(-2.10102402082508e-06f));
  p = _mm512_fmadd_ps(x2, p, _mm512_set1_ps(-5.69250639462346e-05f));
  p = _mm512_fmadd_ps(x2, p, _mm512_set1_ps(-7.34990630326855e-04f));
  p = _mm512_fmadd_ps(x2, p, _mm512_set1_ps(-2.95459980854025e-03f));
  p = _mm512_fmadd_ps(x2, p, _mm512_set1_ps(-1.60960333262415e-02f));
  p = _mm512_mul_ps(x, p);
  __m512 q = _mm512_fmadd_ps(x2, _mm512_set1_ps(-1.45660718464996e-05f), _mm512_set1_ps(-2.13374055278905e-04f));
  q = _mm512_fmadd_ps(x2, q, _mm512_set1_ps(-1.68282697438203e-03f));
  q = _mm512_fmadd_ps(x2, q, _mm512_set1_ps(-7.37332916720468e-03f));
  q = _mm512_fmadd_ps(x2, q, _mm512_set1_ps(-1.42647390514189e-02f));
  return _mm512_div_ps(p, q);
}

// exact (erf) GELU: x / 2 * (1 + erf(x / sqrt 2))
inline __m512 gelu512(__m512 x) {
  const __m512 e = erf512(_mm512_mul_ps(x, _mm512_set1_ps(0.70710678118654752f)));
  return _mm512_mul_ps(_mm512_mul_ps(x, _mm512_set1_ps(0.5f)), _mm512_add_ps(e, _mm512_set1_ps(1.f)));
}

// ---- GEMM -----------------------------------------------------------------------------------------------------

// C[r][0..32) (+)= sum over kp pairs of A[r][2k..2k+1] . B panel; B advances 64 elements per pair.
template <int R>
inline void ukernel(const u16* __restrict A, std::size_t lda, const u16* __restrict B, int kp, float* __restrict C,
                    int ldc, bool load) {
  __m512 c0[R], c1[R];
#pragma GCC unroll 12
  for (int r = 0; r < R; ++r) {
    c0[r] = load ? _mm512_loadu_ps(C + r * ldc) : _mm512_setzero_ps();
    c1[r] = load ? _mm512_loadu_ps(C + r * ldc + 16) : _mm512_setzero_ps();
  }
  for (int k = 0; k < kp; ++k) {
    const __m512bh b0 = (__m512bh)_mm512_loadu_si512(B + std::size_t(k) * 64);
    const __m512bh b1 = (__m512bh)_mm512_loadu_si512(B + std::size_t(k) * 64 + 32);
#pragma GCC unroll 12
    for (int r = 0; r < R; ++r) {
      std::int32_t a;
      std::memcpy(&a, A + r * lda + 2 * k, 4);
      const __m512bh av = (__m512bh)_mm512_set1_epi32(a);
      c0[r] = _mm512_dpbf16_ps(c0[r], av, b0);
      c1[r] = _mm512_dpbf16_ps(c1[r], av, b1);
    }
  }
#pragma GCC unroll 12
  for (int r = 0; r < R; ++r) {
    _mm512_storeu_ps(C + r * ldc, c0[r]);
    _mm512_storeu_ps(C + r * ldc + 16, c1[r]);
  }
}

void ukernel_tail(int rows, const u16* A, std::size_t lda, const u16* B, int kp, float* C, int ldc, bool load) {
  switch (rows) {
#define CASE(n) \
  case n: ukernel<n>(A, lda, B, kp, C, ldc, load); break;
    CASE(1) CASE(2) CASE(3) CASE(4) CASE(5) CASE(6) CASE(7) CASE(8) CASE(9) CASE(10) CASE(11)
#undef CASE
    default: break;
  }
}

// C tile [rows][cols] (row stride ldc) = A rows . W columns [n0, n0 + cols) over the full K (KP pairs).
void tile_gemm(const u16* A, std::size_t lda, int rows, const u16* Bp, int KP, int n0, int cols, float* C, int ldc) {
  for (int kb = 0; kb < KP; kb += KC) {
    const int kc = std::min(KC, KP - kb);
    const bool load = kb > 0;
    for (int nb = n0 / NR; nb < (n0 + cols) / NR; ++nb) {
      const u16* B = Bp + (std::size_t(nb) * KP + kb) * 64;
      float* Cn = C + (nb * NR - n0);
      int m = 0;
      for (; m + MR <= rows; m += MR) ukernel<MR>(A + m * lda + 2 * kb, lda, B, kc, Cn + std::size_t(m) * ldc, ldc, load);
      if (m < rows) ukernel_tail(rows - m, A + m * lda + 2 * kb, lda, B, kc, Cn + std::size_t(m) * ldc, ldc, load);
    }
  }
}

float* job_tile() {
  alignas(64) static thread_local float tile[MB * NB];
  return tile;
}

void run_jobs(offload::CpuPool& pool, int njobs, const auto& f) {
  constexpr int kMax = 60000;  // the pool's claim word holds 16-bit job counts
  for (int j0 = 0; j0 < njobs; j0 += kMax) {
    const int n = std::min(kMax, njobs - j0);
    pool.run(n, [&](int j) { f(j0 + j); });
  }
}

// epi(m0, rows, n0, cols, tile, NB) for every MB x NB tile of A [M][K] . W^T (W: N x K packed).
template <class Epi>
void gemm(offload::CpuPool& pool, const u16* A, std::size_t lda, int M, const u16* Bp, int N, int K, const Epi& epi) {
  const int mt = (M + MB - 1) / MB, nt = (N + NB - 1) / NB;
  run_jobs(pool, mt * nt, [&](int job) {
    const int i = job / nt, j = job % nt;
    const int m0 = i * MB, rows = std::min(MB, M - m0), n0 = j * NB, cols = std::min(NB, N - n0);
    float* tile = job_tile();
    tile_gemm(A + std::size_t(m0) * lda, lda, rows, Bp, K / 2, n0, cols, tile, NB);
    epi(m0, rows, n0, cols, static_cast<const float*>(tile), NB);
  });
}

// ---- weights --------------------------------------------------------------------------------------------------

struct Linear {
  Buf<u16> w;  // packed
  int N = 0, K = 0;
  std::vector<float> bias;  // empty: none
};

struct LayerW {
  std::vector<float> ln1_w, ln1_b, ln2_w, ln2_b, ls1, ls2;
  Linear qkv, o, fc1, fc2;
};

void layer_norm_row(const float* x, const float* w, const float* b, float* out_f32, u16* out_bf16) {
  constexpr int D = kVitHidden;
  __m512 s = _mm512_setzero_ps();
  for (int i = 0; i < D; i += 16) s = _mm512_add_ps(s, _mm512_loadu_ps(x + i));
  const float mean = _mm512_reduce_add_ps(s) / D;
  const __m512 mv = _mm512_set1_ps(mean);
  __m512 v = _mm512_setzero_ps();
  for (int i = 0; i < D; i += 16) {
    const __m512 d = _mm512_sub_ps(_mm512_loadu_ps(x + i), mv);
    v = _mm512_fmadd_ps(d, d, v);
  }
  const float rstd = 1.f / std::sqrt(_mm512_reduce_add_ps(v) / D + 1e-5f);
  const __m512 rv = _mm512_set1_ps(rstd);
  for (int i = 0; i < D; i += 16) {
    const __m512 d = _mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(x + i), mv), rv);
    const __m512 y = _mm512_fmadd_ps(d, _mm512_loadu_ps(w + i), _mm512_loadu_ps(b + i));
    if (out_f32) _mm512_storeu_ps(out_f32 + i, y);
    if (out_bf16) store_bf16(out_bf16 + i, y);
  }
}

}  // namespace

// ---- exposed kernels ------------------------------------------------------------------------------------------

namespace cpu {

void pack_weight(const u16* w, int N, int K, u16* out) {
  const int KP = K / 2;
  for (int nb = 0; nb < N / NR; ++nb)
    for (int kp = 0; kp < KP; ++kp)
      for (int j = 0; j < NR; ++j) {
        const u16* src = w + std::size_t(nb * NR + j) * K + 2 * kp;
        u16* dst = out + ((std::size_t(nb) * KP + kp) * NR + j) * 2;
        dst[0] = src[0], dst[1] = src[1];
      }
}

void gemm_f32(offload::CpuPool& pool, const u16* A, std::size_t lda, int M, const u16* packed, int N, int K, float* C,
              std::size_t ldc) {
  gemm(pool, A, lda, M, packed, N, K, [&](int m0, int rows, int n0, int cols, const float* t, int ldt) {
    for (int r = 0; r < rows; ++r) std::memcpy(C + (m0 + r) * ldc + n0, t + r * ldt, std::size_t(cols) * 4);
  });
}

float erf_lane(float x) {
  alignas(64) float v[16];
  _mm512_store_ps(v, erf512(_mm512_set1_ps(x)));
  return v[0];
}
float exp_lane(float x) {
  alignas(64) float v[16];
  _mm512_store_ps(v, exp512(_mm512_set1_ps(x)));
  return v[0];
}

}  // namespace cpu

// ---- the tower ------------------------------------------------------------------------------------------------

struct VisionTower::Impl {
  explicit Impl(offload::CpuPool& p) : pool(p) {}

  offload::CpuPool& pool;
  Linear patch, proj_in, proj_out;
  std::vector<float> cls;        // class_embedding + cls_position_embedding
  std::vector<float> pos_table;  // [32 * 32][1024]
  std::vector<float> ln_pre_w, ln_pre_b;
  LayerW layers[kVitLayers];
  std::size_t bytes = 0;
  double load_seconds = 0;

  // per-image scratch (grow-only)
  Buf<float> x, rope_cos, rope_sin;
  Buf<u16> xn, h, qh, kp, vp;
  std::vector<int> pos_idx;
  std::vector<float> pos_w;

  void load(const SafetensorsModel& ckpt);
  std::vector<u16> forward(const Patches& p);
};

void VisionTower::Impl::load(const SafetensorsModel& ckpt) {
  const auto t0 = std::chrono::steady_clock::now();
  const std::string V = "model.visual.", B = V + "backbone.";
  struct Want {
    std::string name;
    std::vector<std::int64_t> shape;
    std::byte* dst = nullptr;
  };
  std::vector<Want> want;
  auto add = [&](std::string name, std::vector<std::int64_t> shape) { want.push_back({std::move(name), std::move(shape)}); };
  add(B + "embeddings.class_embedding", {kVitHidden});
  add(B + "embeddings.cls_position_embedding", {kVitHidden});
  add(B + "embeddings.patch_embedding.weight", {kVitHidden, kPatchDim});
  add(B + "embeddings.position_embedding", {kPosGrid * kPosGrid, kVitHidden});
  add(B + "ln_pre.weight", {kVitHidden});
  add(B + "ln_pre.bias", {kVitHidden});
  for (int l = 0; l < kVitLayers; ++l) {
    const std::string L = B + "layers." + std::to_string(l) + ".";
    for (const char* n : {"ln_1.weight", "ln_1.bias", "ln_2.weight", "ln_2.bias", "ls_1.gamma", "ls_2.gamma",
                          "self_attn.q_proj.bias", "self_attn.k_proj.bias", "self_attn.v_proj.bias",
                          "self_attn.out_proj.bias", "mlp.c_proj.bias"})
      add(L + n, {kVitHidden});
    for (const char* n : {"self_attn.q_proj.weight", "self_attn.k_proj.weight", "self_attn.v_proj.weight",
                          "self_attn.out_proj.weight"})
      add(L + n, {kVitHidden, kVitHidden});
    add(L + "mlp.c_fc.weight", {kVitMlp, kVitHidden});
    add(L + "mlp.c_fc.bias", {kVitMlp});
    add(L + "mlp.c_proj.weight", {kVitHidden, kVitMlp});
  }
  add(V + "proj_in.weight", {kProjHidden, 4 * kVitHidden});
  add(V + "proj_out.weight", {kTextHidden, kProjHidden});

  // one staging buffer, every range at its file offset modulo 4 KiB (O_DIRECT-capable reads)
  std::vector<ReadRange> ranges;
  std::size_t cursor = 0;
  std::vector<std::size_t> at;
  for (const Want& w : want) {
    const SafetensorsRef r = ckpt.at(w.name);
    if (r->dtype != "BF16" || r->shape != w.shape)
      throw std::runtime_error("vision tower: unexpected dtype or shape of " + w.name);
    const std::size_t o = aligned_layout(cursor, r->offset);
    at.push_back(o);
    ranges.push_back({r.file, r->offset, r->bytes, nullptr});
    cursor = o + r->bytes;
  }
  Buf<std::byte> staging((cursor + 2 * kDirectAlign + 4095) / 4096 * 4096);
  // aligned_alloc(64) is not 4 KiB aligned: shift the whole layout so offsets keep their 4 KiB residues
  std::byte* base = reinterpret_cast<std::byte*>(
      (reinterpret_cast<std::uintptr_t>(staging.data()) + kDirectAlign - 1) & ~std::uintptr_t(kDirectAlign - 1));
  for (std::size_t i = 0; i < want.size(); ++i) want[i].dst = ranges[i].dst = base + at[i];
  read_ranges(ranges, 8);

  auto vec = [&](std::size_t i) {
    const u16* s = reinterpret_cast<const u16*>(want[i].dst);
    std::size_t n = 1;
    for (auto d : want[i].shape) n *= std::size_t(d);
    std::vector<float> v(n);
    for (std::size_t k = 0; k < n; ++k) v[k] = bf16f(s[k]);
    return v;
  };
  auto bf = [&](std::size_t i) { return reinterpret_cast<const u16*>(want[i].dst); };

  // linear weights to pack: (target, source rows in order). QKV is one N = 3072 matrix.
  struct PackJob {
    Linear* lin;
    std::vector<const u16*> parts;  // row blocks concatenated along N
    int part_rows;
  };
  std::vector<PackJob> jobs;
  std::size_t i = 0;
  const std::vector<float> cls_e = vec(i++), cls_p = vec(i++);
  cls.resize(kVitHidden);
  for (int k = 0; k < kVitHidden; ++k) cls[k] = cls_e[k] + cls_p[k];
  patch.N = kVitHidden, patch.K = kPatchDim;
  jobs.push_back({&patch, {bf(i++)}, kVitHidden});
  pos_table = vec(i++);
  ln_pre_w = vec(i++), ln_pre_b = vec(i++);
  for (int l = 0; l < kVitLayers; ++l) {
    LayerW& L = layers[l];
    L.ln1_w = vec(i++), L.ln1_b = vec(i++), L.ln2_w = vec(i++), L.ln2_b = vec(i++);
    L.ls1 = vec(i++), L.ls2 = vec(i++);
    const std::vector<float> bq = vec(i++), bk = vec(i++), bv = vec(i++);
    L.qkv.bias = bq;
    L.qkv.bias.insert(L.qkv.bias.end(), bk.begin(), bk.end());
    L.qkv.bias.insert(L.qkv.bias.end(), bv.begin(), bv.end());
    L.o.bias = vec(i++);
    L.fc2.bias = vec(i++);
    const u16 *wq = bf(i++), *wk = bf(i++), *wv = bf(i++), *wo = bf(i++);
    L.qkv.N = 3 * kVitHidden, L.qkv.K = kVitHidden;
    jobs.push_back({&L.qkv, {wq, wk, wv}, kVitHidden});
    L.o.N = kVitHidden, L.o.K = kVitHidden;
    jobs.push_back({&L.o, {wo}, kVitHidden});
    L.fc1.N = kVitMlp, L.fc1.K = kVitHidden;
    jobs.push_back({&L.fc1, {bf(i++)}, kVitMlp});
    L.fc1.bias = vec(i++);
    L.fc2.N = kVitHidden, L.fc2.K = kVitMlp;
    jobs.push_back({&L.fc2, {bf(i++)}, kVitHidden});
  }
  proj_in.N = kProjHidden, proj_in.K = 4 * kVitHidden;
  jobs.push_back({&proj_in, {bf(i++)}, kProjHidden});
  proj_out.N = kTextHidden, proj_out.K = kProjHidden;
  jobs.push_back({&proj_out, {bf(i++)}, kTextHidden});

  // pack: one pool job per (matrix, source block)
  struct Piece {
    const PackJob* job;
    int part;
  };
  std::vector<Piece> pieces;
  for (PackJob& j : jobs) {
    j.lin->w.resize(std::size_t(j.lin->N) * j.lin->K);
    bytes += std::size_t(j.lin->N) * j.lin->K * 2;
    for (int p = 0; p < int(j.parts.size()); ++p) pieces.push_back({&j, p});
  }
  run_jobs(pool, int(pieces.size()), [&](int k) {
    const Piece& pc = pieces[std::size_t(k)];
    Linear& lin = *pc.job->lin;
    const int rows = pc.job->part_rows;
    // rows [part * rows, (part + 1) * rows) of the packed matrix: whole panels (rows % 32 == 0)
    cpu::pack_weight(pc.job->parts[std::size_t(pc.part)], rows, lin.K,
                     lin.w.data() + std::size_t(pc.part) * rows * lin.K);
  });
  bytes += pos_table.size() * 4;
  load_seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
}

std::vector<u16> VisionTower::Impl::forward(const Patches& pt) {
  constexpr int D = kVitHidden;
  const int T = pt.rows * pt.cols, P = 4 * T, N = P + 1;
  const int Npad = (N + NR - 1) / NR * NR;
  if (T <= 0) throw std::invalid_argument("vision tower: empty image");

  x.resize(std::size_t(N) * D);
  xn.resize(std::size_t(N) * D);
  h.resize(std::size_t(N) * kVitMlp);
  qh.resize(std::size_t(kVitHeads) * Npad * kVitHeadDim);
  kp.resize(std::size_t(kVitHeads) * Npad * kVitHeadDim);
  vp.resize(std::size_t(kVitHeads) * Npad * kVitHeadDim);
  rope_cos.resize(std::size_t(N) * 32);
  rope_sin.resize(std::size_t(N) * 32);
  const std::size_t head_stride = std::size_t(Npad) * kVitHeadDim;

  // ---- per-image tables: position-embedding taps and RoPE angles, patches in merge-block order
  const int Hp = pt.rows * 2, Wp = pt.cols * 2;
  pos_idx.assign(std::size_t(P) * 4, 0);
  pos_w.assign(std::size_t(P) * 4, 0.f);
  auto taps = [](int index, int size, int t[2], float w[2]) {
    const float src = (float(index) + 0.5f) * float(kPosGrid) / float(size) - 0.5f;
    const float fl = std::floor(src);
    for (int o = 0; o < 2; ++o) {
      t[o] = std::clamp(int(fl) + o, 0, kPosGrid - 1);
      w[o] = std::max(0.f, 1.f - std::abs(src - fl - float(o)));
    }
  };
  float inv_freq[16];
  for (int j = 0; j < 16; ++j) inv_freq[j] = float(1.0 / std::pow(10000.0, double(2 * j) / 32.0));
  for (int j = 0; j < 32; ++j) rope_cos[j] = 1.f, rope_sin[j] = 0.f;  // class token at (0, 0)
  for (int p = 0; p < P; ++p) {
    const int tok = p / 4, mr = (p % 4) / 2, mc = p % 2;
    const int row = (tok / pt.cols) * 2 + mr, col = (tok % pt.cols) * 2 + mc;
    int th[2], tw[2];
    float wh[2], ww[2];
    taps(row, Hp, th, wh);
    taps(col, Wp, tw, ww);
    for (int a = 0; a < 2; ++a)
      for (int b = 0; b < 2; ++b) {
        pos_idx[std::size_t(p) * 4 + a * 2 + b] = th[a] * kPosGrid + tw[b];
        pos_w[std::size_t(p) * 4 + a * 2 + b] = wh[a] * ww[b];
      }
    float* c = rope_cos.data() + std::size_t(p + 1) * 32;
    float* s = rope_sin.data() + std::size_t(p + 1) * 32;
    for (int j = 0; j < 32; ++j) {
      const float ang = j < 16 ? float(col + 1) * inv_freq[j] : float(row + 1) * inv_freq[j - 16];
      c[j] = std::cos(ang), s[j] = std::sin(ang);
    }
  }
  // padded keys must be finite zeros (masked in the softmax, multiplied by zero probabilities)
  for (int hd = 0; hd < kVitHeads; ++hd) {
    for (int n = N; n < Npad; ++n)
      for (int d = 0; d < kVitHeadDim; ++d) {
        kp[hd * head_stride + ((std::size_t(n / NR) * 32 + d / 2) * NR + n % NR) * 2 + (d & 1)] = 0;
        vp[hd * head_stride + ((std::size_t(d / NR) * (Npad / 2) + n / 2) * NR + d % NR) * 2 + (n & 1)] = 0;
      }
  }

  // ---- embeddings: patches . Wpatch^T + interpolated position embedding; class token first
  std::memcpy(x.data(), cls.data(), D * 4);
  gemm(pool, pt.data.data(), kPatchDim, P, patch.w.data(), D, kPatchDim,
       [&](int m0, int rows, int n0, int cols, const float* t, int ldt) {
         for (int r = 0; r < rows; ++r) {
           const int p = m0 + r;
           float* dst = x.data() + std::size_t(p + 1) * D + n0;
           const int* ix = pos_idx.data() + std::size_t(p) * 4;
           const float* wt = pos_w.data() + std::size_t(p) * 4;
           for (int c = 0; c < cols; c += 16) {
             __m512 v = _mm512_loadu_ps(t + r * ldt + c);
             for (int k = 0; k < 4; ++k)
               v = _mm512_fmadd_ps(_mm512_set1_ps(wt[k]),
                                   _mm512_loadu_ps(pos_table.data() + std::size_t(ix[k]) * D + n0 + c), v);
             _mm512_storeu_ps(dst + c, v);
           }
         }
       });
  const int row_jobs = (N + 15) / 16;
  run_jobs(pool, row_jobs, [&](int j) {
    for (int r = j * 16; r < std::min(N, j * 16 + 16); ++r)
      layer_norm_row(x.data() + std::size_t(r) * D, ln_pre_w.data(), ln_pre_b.data(), x.data() + std::size_t(r) * D,
                     nullptr);
  });

  for (int l = 0; l < kVitLayers; ++l) {
    const LayerW& L = layers[l];
    // ---- attention
    run_jobs(pool, row_jobs, [&](int j) {
      for (int r = j * 16; r < std::min(N, j * 16 + 16); ++r)
        layer_norm_row(x.data() + std::size_t(r) * D, L.ln1_w.data(), L.ln1_b.data(), nullptr,
                       xn.data() + std::size_t(r) * D);
    });
    // q, k, v (+ bias); RoPE on q and k in FP32, then BF16 into the per-head layouts
    gemm(pool, xn.data(), D, N, L.qkv.w.data(), 3 * D, D, [&](int m0, int rows, int n0, int cols, const float* t, int ldt) {
      const int which = n0 / D;  // 0 q, 1 k, 2 v (tiles never straddle)
      for (int r = 0; r < rows; ++r) {
        const int n = m0 + r;
        const float* src = t + r * ldt;
        const float* cs = rope_cos.data() + std::size_t(n) * 32;
        const float* sn = rope_sin.data() + std::size_t(n) * 32;
        for (int c = 0; c < cols; c += 2) {
          const int col = n0 + c - which * D, hd = col / kVitHeadDim, d = col % kVitHeadDim;
          float a = src[c] + L.qkv.bias[std::size_t(n0 + c)], b = src[c + 1] + L.qkv.bias[std::size_t(n0 + c + 1)];
          if (which < 2) {
            const float co = cs[d / 2], si = sn[d / 2];
            const float ra = a * co - b * si, rb = b * co + a * si;
            a = ra, b = rb;
          }
          if (which == 0) {
            u16* q = qh.data() + hd * head_stride + std::size_t(n) * kVitHeadDim + d;
            q[0] = f2bf(a), q[1] = f2bf(b);
          } else if (which == 1) {
            u16* k = kp.data() + hd * head_stride + ((std::size_t(n / NR) * 32 + d / 2) * NR + n % NR) * 2;
            k[0] = f2bf(a), k[1] = f2bf(b);
          } else {
            const std::size_t base = hd * head_stride + (std::size_t(n / 2)) * NR * 2 + (n & 1);
            vp[base + (std::size_t(d / NR) * (Npad / 2)) * NR * 2 + (d % NR) * 2] = f2bf(a);
            vp[base + (std::size_t((d + 1) / NR) * (Npad / 2)) * NR * 2 + ((d + 1) % NR) * 2] = f2bf(b);
          }
        }
      }
    });
    // softmax(q k^T / 8) v per head, QB query rows per job; output BF16 [N][16 * 64] into xn
    const int qblocks = (N + QB - 1) / QB;
    run_jobs(pool, kVitHeads * qblocks, [&](int job) {
      const int hd = job / qblocks, r0 = (job % qblocks) * QB, rows = std::min(QB, N - r0);
      static thread_local Buf<float> S;
      static thread_local Buf<u16> Pm;
      alignas(64) static thread_local float O[QB * kVitHeadDim];
      S.resize(std::size_t(QB) * Npad);
      Pm.resize(std::size_t(QB) * Npad);
      tile_gemm(qh.data() + hd * head_stride + std::size_t(r0) * kVitHeadDim, kVitHeadDim, rows,
                kp.data() + hd * head_stride, kVitHeadDim / 2, 0, Npad, S.data(), Npad);
      const __m512 scale = _mm512_set1_ps(0.125f);
      for (int r = 0; r < rows; ++r) {
        float* s = S.data() + std::size_t(r) * Npad;
        u16* pr = Pm.data() + std::size_t(r) * Npad;
        __m512 mx = _mm512_set1_ps(-INFINITY);
        for (int c = 0; c < N; c += 16) {
          const __mmask16 m = c + 16 <= N ? __mmask16(0xffff) : __mmask16((1u << (N - c)) - 1);
          mx = _mm512_mask_max_ps(mx, m, mx, _mm512_maskz_loadu_ps(m, s + c));
        }
        const __m512 mv = _mm512_set1_ps(_mm512_reduce_max_ps(mx));
        __m512 sum = _mm512_setzero_ps();
        for (int c = 0; c < Npad; c += 16) {
          const __mmask16 m = c + 16 <= N ? __mmask16(0xffff) : c < N ? __mmask16((1u << (N - c)) - 1) : __mmask16(0);
          const __m512 e = _mm512_maskz_mov_ps(m, exp512(_mm512_mul_ps(_mm512_sub_ps(_mm512_loadu_ps(s + c), mv), scale)));
          _mm512_storeu_ps(s + c, e);
          sum = _mm512_add_ps(sum, e);
        }
        const __m512 inv = _mm512_set1_ps(1.f / _mm512_reduce_add_ps(sum));
        for (int c = 0; c < Npad; c += 16) store_bf16(pr + c, _mm512_mul_ps(_mm512_loadu_ps(s + c), inv));
      }
      tile_gemm(Pm.data(), Npad, rows, vp.data() + hd * head_stride, Npad / 2, 0, kVitHeadDim, O, kVitHeadDim);
      for (int r = 0; r < rows; ++r)
        for (int d = 0; d < kVitHeadDim; d += 16)
          store_bf16(xn.data() + std::size_t(r0 + r) * D + hd * kVitHeadDim + d, _mm512_loadu_ps(O + r * kVitHeadDim + d));
    });
    // x += ls_1 * (attn . Wo^T + bo)
    auto residual = [&](const Linear& lin, const std::vector<float>& ls) {
      return [&](int m0, int rows, int n0, int cols, const float* t, int ldt) {
        for (int r = 0; r < rows; ++r) {
          float* dst = x.data() + std::size_t(m0 + r) * D + n0;
          for (int c = 0; c < cols; c += 16) {
            const __m512 v = _mm512_add_ps(_mm512_loadu_ps(t + r * ldt + c), _mm512_loadu_ps(lin.bias.data() + n0 + c));
            _mm512_storeu_ps(dst + c, _mm512_fmadd_ps(_mm512_loadu_ps(ls.data() + n0 + c), v, _mm512_loadu_ps(dst + c)));
          }
        }
      };
    };
    gemm(pool, xn.data(), D, N, L.o.w.data(), D, D, residual(L.o, L.ls1));

    // ---- MLP
    run_jobs(pool, row_jobs, [&](int j) {
      for (int r = j * 16; r < std::min(N, j * 16 + 16); ++r)
        layer_norm_row(x.data() + std::size_t(r) * D, L.ln2_w.data(), L.ln2_b.data(), nullptr,
                       xn.data() + std::size_t(r) * D);
    });
    gemm(pool, xn.data(), D, N, L.fc1.w.data(), kVitMlp, D, [&](int m0, int rows, int n0, int cols, const float* t, int ldt) {
      for (int r = 0; r < rows; ++r)
        for (int c = 0; c < cols; c += 16) {
          const __m512 v = _mm512_add_ps(_mm512_loadu_ps(t + r * ldt + c), _mm512_loadu_ps(L.fc1.bias.data() + n0 + c));
          store_bf16(h.data() + std::size_t(m0 + r) * kVitMlp + n0 + c, gelu512(v));
        }
    });
    gemm(pool, h.data(), kVitMlp, N, L.fc2.w.data(), D, kVitMlp, residual(L.fc2, L.ls2));
  }

  // ---- merger: the 4 patches of each image token side by side (rows 1.. are already in that order)
  run_jobs(pool, row_jobs, [&](int j) {
    for (int r = std::max(1, j * 16); r < std::min(N, j * 16 + 16); ++r)
      for (int c = 0; c < D; c += 16)
        store_bf16(h.data() + std::size_t(r - 1) * D + c, _mm512_loadu_ps(x.data() + std::size_t(r) * D + c));
  });
  gemm(pool, h.data(), 4 * D, T, proj_in.w.data(), kProjHidden, 4 * D,
       [&](int m0, int rows, int n0, int cols, const float* t, int ldt) {
         for (int r = 0; r < rows; ++r)
           for (int c = 0; c < cols; c += 16)
             store_bf16(xn.data() + std::size_t(m0 + r) * kProjHidden + n0 + c, gelu512(_mm512_loadu_ps(t + r * ldt + c)));
       });
  std::vector<u16> out(std::size_t(T) * kTextHidden);
  gemm(pool, xn.data(), kProjHidden, T, proj_out.w.data(), kTextHidden, kProjHidden,
       [&](int m0, int rows, int n0, int cols, const float* t, int ldt) {
         for (int r = 0; r < rows; ++r)
           for (int c = 0; c < cols; c += 16)
             store_bf16(out.data() + std::size_t(m0 + r) * kTextHidden + n0 + c, _mm512_loadu_ps(t + r * ldt + c));
       });
  return out;
}

VisionTower::VisionTower(const SafetensorsModel& ckpt, offload::CpuPool& pool) : impl_(std::make_unique<Impl>(pool)) {
  impl_->load(ckpt);
}
VisionTower::~VisionTower() = default;
std::vector<std::uint16_t> VisionTower::forward(const Patches& p) { return impl_->forward(p); }
std::size_t VisionTower::weight_bytes() const { return impl_->bytes; }
double VisionTower::load_seconds() const { return impl_->load_seconds; }

}  // namespace ingine::vision
