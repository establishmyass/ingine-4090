// Decode kernel timings at production sizes: moe_up / moe_down (NVFP4 and MXFP8, all 128 experts allocated,
// random routing per call), the attention projection GEMVs (proj, proj_conv) and the router, for M = 1, 2, 4, 8
// tokens. Each measurement replays a CUDA graph of 48 launches (one per decode visit, every launch with its own
// routing), as the decode graph does; reports microseconds per launch and the weight bytes streamed per second.
// Random data; ~1.3 GB of GPU memory. Runs only with INGINE_BENCH=1.
#include "kernels/attn/attn.h"
#include "kernels/dense/dense.h"
#include "kernels/moe/moe.h"
#include "common/test_util.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cstdint>
#include <string>
#include <functional>
#include <numeric>
#include <vector>

using namespace ingine::test;
namespace moe = ingine::kernels::moe;
namespace dense = ingine::kernels::dense;
namespace attn = ingine::kernels::attn;

namespace {

constexpr int H = 2304, I = 1152, E = 128, TOPK = 8, VISITS = 48;

__global__ void fill_bytes(uint8_t* p, size_t n, uint32_t seed, uint8_t mask, uint8_t base) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    uint32_t h = uint32_t(i) * 2654435761u ^ seed;
    h ^= h >> 15, h *= 2246822519u, h ^= h >> 13;
    p[i] = uint8_t((h & mask) | base);
  }
}
__global__ void fill_bf16(uint16_t* p, size_t n, uint32_t seed, float scale) {
  for (size_t i = blockIdx.x * size_t(blockDim.x) + threadIdx.x; i < n; i += size_t(gridDim.x) * blockDim.x) {
    uint32_t h = uint32_t(i) * 2654435761u ^ seed;
    h ^= h >> 15, h *= 2246822519u, h ^= h >> 13;
    const float u = (h & 0xffffff) / 16777216.f - 0.5f;
    p[i] = __bfloat16_as_ushort(__float2bfloat16(u * scale));
  }
}

template <class T>
T* dalloc(size_t n) {
  T* p = nullptr;
  INGINE_TEST_CUDA(cudaMalloc(&p, n * sizeof(T) + 64));
  INGINE_TEST_CUDA(cudaMemset(p, 0, n * sizeof(T) + 64));
  return p;
}
void bytes(void* p, size_t n, uint32_t seed, uint8_t mask = 0xff, uint8_t base = 0) {
  fill_bytes<<<1024, 256>>>(static_cast<uint8_t*>(p), n, seed, mask, base);
}

// Microseconds per launch of `launch(i)` for i = 0..VISITS-1, replayed as one CUDA graph.
float graph_us(const std::function<void(int, cudaStream_t)>& launch, cudaStream_t s) {
  cudaGraph_t g;
  cudaGraphExec_t ge;
  INGINE_TEST_CUDA(cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal));
  for (int i = 0; i < VISITS; ++i) launch(i, s);
  INGINE_TEST_CUDA(cudaStreamEndCapture(s, &g));
  INGINE_TEST_CUDA(cudaGraphInstantiate(&ge, g, 0));
  cudaEvent_t a, b;
  cudaEventCreate(&a), cudaEventCreate(&b);
  std::vector<float> t;
  for (int r = 0; r < 12; ++r) {
    cudaEventRecord(a, s);
    INGINE_TEST_CUDA(cudaGraphLaunch(ge, s));
    cudaEventRecord(b, s);
    INGINE_TEST_CUDA(cudaEventSynchronize(b));
    float ms;
    cudaEventElapsedTime(&ms, a, b);
    if (r >= 2) t.push_back(ms);
  }
  cudaGraphExecDestroy(ge);
  cudaGraphDestroy(g);
  std::sort(t.begin(), t.end());
  return t[t.size() / 2] * 1000.f / VISITS;
}

struct Experts {
  bool nv;
  uint8_t *w13, *w13s, *w2, *w2s;
  float *f13, *f2;
  size_t up_bytes, down_bytes;  // per expert
};

Experts make_experts(bool nv) {
  Experts x{};
  x.nv = nv;
  const size_t r13 = 2 * I, r2 = H;
  const size_t c13 = nv ? H / 2 : H, s13 = nv ? H / 16 : H / 32;
  const size_t c2 = nv ? I / 2 : I, s2 = nv ? I / 16 : I / 32;
  x.w13 = dalloc<uint8_t>(E * r13 * c13), x.w13s = dalloc<uint8_t>(E * r13 * s13);
  x.w2 = dalloc<uint8_t>(E * r2 * c2), x.w2s = dalloc<uint8_t>(E * r2 * s2);
  if (nv) {
    bytes(x.w13, E * r13 * c13, 1), bytes(x.w2, E * r2 * c2, 2);
    bytes(x.w13s, E * r13 * s13, 3, 0x0f, 0x30), bytes(x.w2s, E * r2 * s2, 4, 0x0f, 0x30);  // e4m3 in [0.5, 1)
  } else {
    bytes(x.w13, E * r13 * c13, 1, 0x77), bytes(x.w2, E * r2 * c2, 2, 0x77);  // no NaN codes
    bytes(x.w13s, E * r13 * s13, 3, 0x03, 0x70), bytes(x.w2s, E * r2 * s2, 4, 0x03, 0x70);
  }
  std::vector<float> f(E, 1e-3f);
  x.f13 = dalloc<float>(E), x.f2 = dalloc<float>(E);
  INGINE_TEST_CUDA(cudaMemcpy(x.f13, f.data(), E * 4, cudaMemcpyHostToDevice));
  INGINE_TEST_CUDA(cudaMemcpy(x.f2, f.data(), E * 4, cudaMemcpyHostToDevice));
  x.up_bytes = r13 * (c13 + s13);
  x.down_bytes = r2 * (c2 + s2);
  return x;
}

}  // namespace

int main() {
  if (skip_without_gpu()) return kSkip;
  const char* run = std::getenv("INGINE_BENCH");
  if (!run || std::string(run) != "1") {
    std::printf("skip: set INGINE_BENCH=1 to run the decode kernel benchmark\n");
    return kSkip;
  }
  const char* only = std::getenv("INGINE_BENCH_ONLY");  // substring filter: "moe", "proj", "router"
  auto want = [&](const char* what) { return !only || std::strstr(what, only); };
  cudaStream_t s;
  INGINE_TEST_CUDA(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking));
  Rng rng(7);

  // routing: per visit and token, 8 distinct experts; tokens share experts with probability ~1/3 (as a draft
  // block of consecutive tokens does)
  std::vector<int> idx_h(size_t(VISITS) * 8 * TOPK);
  std::vector<int> uniq(9 * VISITS);  // unique experts per (M, visit)
  for (int v = 0; v < VISITS; ++v) {
    std::vector<int> prev;
    for (int m = 0; m < 8; ++m) {
      std::vector<int> sel;
      while (int(sel.size()) < TOPK) {
        int e = (!prev.empty() && rng.uniform() < 0.33) ? prev[rng.next() % prev.size()] : int(rng.next() % E);
        if (std::find(sel.begin(), sel.end(), e) == sel.end()) sel.push_back(e);
      }
      for (int k = 0; k < TOPK; ++k) idx_h[(size_t(v) * 8 + m) * TOPK + k] = sel[k];
      prev = sel;
    }
  }
  int* idx = dalloc<int>(idx_h.size());
  INGINE_TEST_CUDA(cudaMemcpy(idx, idx_h.data(), idx_h.size() * 4, cudaMemcpyHostToDevice));
  auto unique_bytes = [&](int M, size_t per_expert) {
    size_t sum = 0;
    for (int v = 0; v < VISITS; ++v) {
      std::vector<int> u(idx_h.begin() + size_t(v) * 64, idx_h.begin() + size_t(v) * 64 + M * TOPK);
      std::sort(u.begin(), u.end());
      sum += size_t(std::unique(u.begin(), u.end()) - u.begin()) * per_expert;
    }
    return double(sum) / VISITS;
  };
  float* wts = dalloc<float>(size_t(VISITS) * 64);
  {
    std::vector<float> w(size_t(VISITS) * 64);
    for (auto& x : w) x = float(rng.uniform(0.1, 1.5));
    INGINE_TEST_CUDA(cudaMemcpy(wts, w.data(), w.size() * 4, cudaMemcpyHostToDevice));
  }
  uint16_t* x = dalloc<uint16_t>(size_t(8) * H);
  uint16_t* act = dalloc<uint16_t>(size_t(64) * I);
  fill_bf16<<<64, 256>>>(x, size_t(8) * H, 11, 4.f);
  fill_bf16<<<64, 256>>>(act, size_t(64) * I, 12, 2.f);
  float* conv_w = dalloc<float>(size_t(H) * 4);
  float* conv_state = dalloc<float>(size_t(VISITS) * 3 * H);
  float* out = dalloc<float>(size_t(8) * H);

  if (want("moe")) {
    for (bool nv : {true, false}) {
      Experts ex = make_experts(nv);
      for (int M : {1, 2, 4, 8}) {
        const auto fmt = nv ? moe::Format::NVFP4 : moe::Format::MXFP8;
        const float up = graph_us([&](int v, cudaStream_t st) {
          moe::moe_up(reinterpret_cast<const __nv_bfloat16*>(x), M, idx + v * 64, ex.w13, ex.w13s, ex.f13, fmt,
                      reinterpret_cast<__nv_bfloat16*>(act), false, st);
        }, s);
        const float down = graph_us([&](int v, cudaStream_t st) {
          moe::moe_down(reinterpret_cast<const __nv_bfloat16*>(act), M, idx + v * 64, wts + v * 64, ex.w2, ex.w2s,
                        ex.f2, fmt, conv_w, conv_state + size_t(v) * 3 * H, out, nullptr, false, st);
        }, s);
        const double ub = unique_bytes(M, ex.up_bytes), db = unique_bytes(M, ex.down_bytes);
        std::printf("moe %s M=%d: up %7.2f us (%5.1f GB/s, %5.2f MB)  down %7.2f us (%5.1f GB/s, %5.2f MB)\n",
                    nv ? "nvfp4" : "mxfp8", M, up, ub / up / 1e3, ub / 1e6, down, db / down / 1e3, db / 1e6);
      }
      for (void* p : {(void*)ex.w13, (void*)ex.w13s, (void*)ex.w2, (void*)ex.w2s, (void*)ex.f13, (void*)ex.f2})
        cudaFree(p);
    }
  }

  if (want("proj") || want("router")) {
    // FP8 attention projections of 48 visits (sliding layout: 8600 rows) and output projections [H, 3072]
    constexpr int R = 8600;
    std::vector<uint8_t*> wp(VISITS), wo(VISITS);
    std::vector<float*> sp(VISITS), so(VISITS);
    for (int v = 0; v < VISITS; ++v) {
      wp[v] = dalloc<uint8_t>(size_t(R) * H), wo[v] = dalloc<uint8_t>(size_t(H) * 3072);
      bytes(wp[v], size_t(R) * H, 100 + v, 0x77), bytes(wo[v], size_t(H) * 3072, 200 + v, 0x77);
      sp[v] = dalloc<float>(R), so[v] = dalloc<float>(H);
      std::vector<float> sc(R, 1e-2f);
      INGINE_TEST_CUDA(cudaMemcpy(sp[v], sc.data(), R * 4, cudaMemcpyHostToDevice));
      INGINE_TEST_CUDA(cudaMemcpy(so[v], sc.data(), H * 4, cudaMemcpyHostToDevice));
    }
    const std::vector<int> segs_h = {0, 3072, 0, 3072, 6144, 0, 6144, 6528, 0, 6528, 6552, 0,
                                     6552, 7576, 1, 7576, 8600, 2};
    int* segs = dalloc<int>(segs_h.size());
    INGINE_TEST_CUDA(cudaMemcpy(segs, segs_h.data(), segs_h.size() * 4, cudaMemcpyHostToDevice));
    uint16_t* xin = dalloc<uint16_t>(size_t(3) * 8 * H);
    fill_bf16<<<64, 256>>>(xin, size_t(3) * 8 * H, 13, 2.f);
    uint16_t* xo = dalloc<uint16_t>(size_t(8) * 3072);
    fill_bf16<<<64, 256>>>(xo, size_t(8) * 3072, 14, 2.f);
    float* pout = dalloc<float>(size_t(8) * R);
    uint16_t* rw = dalloc<uint16_t>(size_t(VISITS) * E * H);
    fill_bf16<<<256, 256>>>(rw, size_t(VISITS) * E * H, 15, 0.1f);
    float* rbias = dalloc<float>(E);
    int* ridx = dalloc<int>(64);
    float* rwts = dalloc<float>(64);
    for (int M : {1, 2, 4, 8}) {
      if (want("proj")) {
        const float pu = graph_us([&](int v, cudaStream_t st) {
          dense::proj(reinterpret_cast<const __nv_bfloat16*>(xin), M, dense::Matrix{wp[v], sp[v], R}, segs, 6, pout,
                      false, st);
        }, s);
        const float ou = graph_us([&](int v, cudaStream_t st) {
          dense::proj_conv(reinterpret_cast<const __nv_bfloat16*>(xo), M, dense::Matrix{wo[v], so[v], H}, conv_w,
                           conv_state + size_t(v) * 3 * H, out, nullptr, false, st);
        }, s);
        std::printf("proj M=%d: %7.2f us (%5.1f GB/s)  proj_conv %7.2f us (%5.1f GB/s)\n", M, pu,
                    double(R) * H / pu / 1e3, ou, double(H) * 3072 / ou / 1e3);
      }
      if (want("router")) {
        const float ru = graph_us([&](int v, cudaStream_t st) {
          dense::router(reinterpret_cast<const __nv_bfloat16*>(x), M, rw + size_t(v) * E * H, false, rbias, ridx,
                        rwts, false, st);
        }, s);
        std::printf("router M=%d: %7.2f us\n", M, ru);
      }
    }
  }
  if (want("attn")) {
    // one full-attention visit (FP8 cache) at several context lengths, and one sliding visit (BF16 ring)
    const size_t max_ctx = size_t(1) << 17;
    uint8_t* kc = dalloc<uint8_t>(max_ctx * 1024);
    uint8_t* vc = dalloc<uint8_t>(max_ctx * 1024);
    bytes(kc, max_ctx * 1024, 31, 0x37), bytes(vc, max_ctx * 1024, 32, 0x37);
    uint16_t* kr = dalloc<uint16_t>(1024 * 1024);
    uint16_t* vr = dalloc<uint16_t>(1024 * 1024);
    fill_bf16<<<256, 256>>>(kr, 1024 * 1024, 33, 1.f);
    fill_bf16<<<256, 256>>>(vr, 1024 * 1024, 34, 1.f);
    float* proj = dalloc<float>(size_t(8) * 8664);
    {
      std::vector<float> h(size_t(8) * 8664);
      for (auto& v : h) v = float(rng.uniform(-1, 1));
      INGINE_TEST_CUDA(cudaMemcpy(proj, h.data(), h.size() * 4, cudaMemcpyHostToDevice));
    }
    std::vector<float> ones(24 * 128, 1.f);
    float* qs = dalloc<float>(24 * 128);
    float* rs = dalloc<float>(24 * 16);
    float* relp = dalloc<float>(16 * 1024);
    float* kconv = dalloc<float>(4 * 1024);
    float* vconv = dalloc<float>(4 * 1024);
    INGINE_TEST_CUDA(cudaMemcpy(qs, ones.data(), 24 * 128 * 4, cudaMemcpyHostToDevice));
    INGINE_TEST_CUDA(cudaMemcpy(rs, ones.data(), 24 * 16 * 4, cudaMemcpyHostToDevice));
    float* cks = dalloc<float>(size_t(VISITS) * 3 * 1024);
    float* cvs = dalloc<float>(size_t(VISITS) * 3 * 1024);
    int64_t* kpos = dalloc<int64_t>(1024);
    int64_t* pos = dalloc<int64_t>(1);
    const size_t wsb = attn::workspace_bytes(8);
    void* ws = dalloc<uint8_t>(wsb);
    uint16_t* aout = dalloc<uint16_t>(size_t(8) * 3072);
    for (int ctx : {4096, 32768, 131064}) {
      const int64_t p0 = ctx - 8;
      INGINE_TEST_CUDA(cudaMemcpy(pos, &p0, 8, cudaMemcpyHostToDevice));
      std::vector<int64_t> kp(1024);
      for (int i = 0; i < 1024; ++i) kp[i] = p0 - 1024 + ((i - (p0 - 1024)) % 1024 + 1024) % 1024;
      for (int i = 0; i < 1024; ++i) kp[(p0 - 1024 + i) % 1024] = p0 - 1024 + i;
      INGINE_TEST_CUDA(cudaMemcpy(kpos, kp.data(), 1024 * 8, cudaMemcpyHostToDevice));
      for (int M : {1, 2, 4, 8}) {
        auto run = [&](bool full) {
          return graph_us([&](int v, cudaStream_t st) {
            attn::DecodeArgs a;
            a.proj = proj, a.M = M, a.q_scale = qs, a.r_scale = rs, a.rel_proj = relp, a.k_conv = kconv,
            a.v_conv = vconv, a.conv_k_state = cks + size_t(v) * 3 * 1024, a.conv_v_state = cvs + size_t(v) * 3 * 1024;
            a.pos = pos, a.workspace = ws, a.workspace_size = wsb, a.out = reinterpret_cast<__nv_bfloat16*>(aout);
            if (full) {
              a.k_cache = kc, a.v_cache = vc, a.cache_fp8 = true, a.cache_rows = int64_t(max_ctx), a.window = 0,
              a.log_scaling = true;
            } else {
              a.k_cache = kr, a.v_cache = vr, a.cache_fp8 = false, a.cache_rows = 1024, a.kpos = kpos, a.window = 1024;
            }
            attn::attn_decode(a, st);
          }, s);
        };
        const float fu = run(true);
        const float su = ctx == 4096 ? run(false) : 0.f;
        std::printf("attn ctx=%6d M=%d: full fp8 %7.2f us (%5.1f GB/s)", ctx, M, fu, double(p0) * 2048 / fu / 1e3);
        if (su > 0) std::printf("  sliding %6.2f us (%5.1f GB/s)", su, 1024.0 * 4096 / su / 1e3);
        std::printf("\n");
      }
    }
  }
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  return 0;
}
