// CPU expert kernels (offload/cpu_experts.h) against the GPU kernels (kernels/moe) on real checkpoint experts:
// layer 0 (NVFP4) and layer 20 (MXFP8), M = 1, 3, 8 tokens with random routing and activations.
// * moe_up's act rows vs expert_up, bit for bit (BF16): the only allowed difference is the FP32 summation
//   order before the gate / up BF16 rounding, so mismatches must be rare (< 0.2%) and small against the row.
// * moe_down (conv weights 0: out = bf16(sum_k w_k y_k) / sqrt 5) vs expert_down on the GPU's act + the same sum.
// Model directory: $INGINE_MODEL_DIR, else the configured default; skipped (77) when absent or without a GPU.
#include "common/test_util.h"

#include "core/safetensors.h"
#include "kernels/moe/moe.h"
#include "offload/cpu_experts.h"
#include "offload/cpu_pool.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <filesystem>
#include <thread>
#include <string>
#include <vector>

using namespace ingine;
namespace moe = ingine::kernels::moe;
namespace off = ingine::offload;
using ingine::test::DeviceArray;

namespace {

constexpr int H = 2304, I = 1152, E = 128;

std::vector<std::uint8_t> read_all(const SafetensorsModel& m, const std::string& name) {
  const SafetensorsRef r = m.at(name);
  std::vector<std::uint8_t> v(r->bytes);
  r.file->read_range(r->offset, r->bytes, v.data());
  return v;
}

std::vector<float> read_f32(const SafetensorsModel& m, const std::string& name) {
  const SafetensorsRef r = m.at(name);
  const std::vector<std::uint8_t> raw = read_all(m, name);
  std::vector<float> out;
  if (r->dtype == "F32") {
    out.resize(raw.size() / 4);
    std::memcpy(out.data(), raw.data(), raw.size());
  } else if (r->dtype == "BF16") {
    out.resize(raw.size() / 2);
    for (std::size_t i = 0; i < out.size(); ++i) {
      const std::uint32_t u = std::uint32_t(raw[2 * i] | (raw[2 * i + 1] << 8)) << 16;
      std::memcpy(&out[i], &u, 4);
    }
  } else {
    throw std::runtime_error("unexpected dtype " + r->dtype + " of " + name);
  }
  return out;
}

int ulp_diff(std::uint16_t a, std::uint16_t b) {
  auto key = [](std::uint16_t x) { return (x & 0x8000) ? -int(x & 0x7fff) : int(x & 0x7fff); };
  return std::abs(key(a) - key(b));
}

bool run_layer(const SafetensorsModel& m, int l, int M, std::uint64_t seed) {
  const std::string e = "model.llm.layers." + std::to_string(l) + ".mlp.experts.";
  const bool nv = m.at(e + "w13_weight")->dtype == "U8";
  const std::vector<std::uint8_t> w13 = read_all(m, e + "w13_weight"), w13s = read_all(m, e + "w13_weight_scale");
  const std::vector<std::uint8_t> w2 = read_all(m, e + "w2_weight"), w2s = read_all(m, e + "w2_weight_scale");
  std::vector<float> f13(E, 1.f), f2(E, 1.f);
  if (nv) {
    for (int w = 0; w < 2; ++w) {
      const std::string n = w ? "w2" : "w13";
      const std::vector<float> amax = read_f32(m, e + n + "_weight_amax");
      const std::vector<float> den = read_f32(m, e + n + "_weight_outer_scale_denominator");
      for (int i = 0; i < E; ++i) (w ? f2 : f13)[i] = amax[i] / den[den.size() == 1 ? 0 : i];
    }
  }
  DeviceArray<std::uint8_t> dw13(w13), dw13s(w13s), dw2(w2), dw2s(w2s);
  DeviceArray<float> df13(f13), df2(f2);

  ingine::test::Rng rng{seed};
  std::vector<std::uint16_t> xb(std::size_t(M) * H);
  for (auto& v : xb) {
    const __nv_bfloat16 b = ingine::test::float_to_bf16(float(rng.normal()) * 2.f);
    std::memcpy(&v, &b, 2);
  }
  std::vector<int> idx(std::size_t(M) * 8);
  for (int n = 0; n < M; ++n)
    for (int k = 0; k < 8; ++k) {
      int ex;
      bool dup;
      do {
        ex = int(rng.next() % E);
        dup = false;
        for (int j = 0; j < k; ++j) dup |= idx[n * 8 + j] == ex;
      } while (dup);
      idx[n * 8 + k] = ex;
    }
  std::vector<float> wts(std::size_t(M) * 8);
  for (auto& w : wts) w = float(rng.uniform()) * 2.f;

  DeviceArray<std::uint16_t> dx(xb);
  DeviceArray<int> didx(idx);
  DeviceArray<float> dwts(wts), conv_w(std::size_t(H) * 4), conv_state(std::size_t(3) * H), out(std::size_t(M) * H);
  DeviceArray<std::uint16_t> dact(std::size_t(M) * 8 * I);
  const moe::Format fmt = nv ? moe::Format::NVFP4 : moe::Format::MXFP8;
  moe::moe_up(reinterpret_cast<const __nv_bfloat16*>(dx.get()), M, didx.get(), dw13.get(), dw13s.get(), df13.get(), fmt,
              reinterpret_cast<__nv_bfloat16*>(dact.get()), false, nullptr);
  moe::moe_down(reinterpret_cast<const __nv_bfloat16*>(dact.get()), M, didx.get(), dwts.get(), dw2.get(), dw2s.get(),
                df2.get(), fmt, conv_w.get(), conv_state.get(), out.get(), nullptr, false, nullptr);
  INGINE_TEST_CUDA(cudaDeviceSynchronize());
  const std::vector<std::uint16_t> act_gpu = dact.download();
  const std::vector<float> out_gpu = out.download();

  const std::size_t s13 = nv ? std::size_t(2 * I) * H / 2 : std::size_t(2 * I) * H;
  const std::size_t s13s = nv ? std::size_t(2 * I) * H / 16 : std::size_t(2 * I) * H / 32;
  const std::size_t s2 = nv ? std::size_t(H) * I / 2 : std::size_t(H) * I;
  const std::size_t s2s = nv ? std::size_t(H) * I / 16 : std::size_t(H) * I / 32;
  auto view = [&](int ex) {
    off::ExpertView v;
    v.fmt = nv ? off::ExpertFmt::NVFP4 : off::ExpertFmt::MXFP8;
    v.w13 = w13.data() + s13 * ex, v.w13s = w13s.data() + s13s * ex;
    v.w2 = w2.data() + s2 * ex, v.w2s = w2s.data() + s2s * ex;
    v.f13 = f13[ex], v.f2 = f2[ex];
    return v;
  };

  long act_mismatch = 0, act_total = 0, act_maxulp = 0, act_large = 0;
  long y_mismatch = 0, y_total = 0, y_maxulp = 0;
  std::vector<std::uint16_t> xperm(H), act(I), aperm(I), y(H);
  std::vector<float> sum(std::size_t(M) * H, 0.f);
  for (int n = 0; n < M; ++n) {
    off::prepare_row(&xb[std::size_t(n) * H], H, xperm.data());
    const off::PreparedRow xr{&xb[std::size_t(n) * H], xperm.data()};
    for (int k = 0; k < 8; ++k) {
      const int ex = idx[n * 8 + k];
      std::uint16_t* ap = act.data();
      off::expert_up(view(ex), &xr, 1, &ap, 0, I);
      const std::uint16_t* ag = &act_gpu[(std::size_t(n) * 8 + k) * I];
      double rms = 0;
      for (int p = 0; p < I; ++p) rms += double(off::bf16_float(ag[p])) * off::bf16_float(ag[p]);
      rms = std::sqrt(rms / I);
      for (int p = 0; p < I; ++p) {
        const std::uint16_t c = act[p];
        ++act_total;
        if (c != ag[p]) {
          ++act_mismatch, act_maxulp = std::max<long>(act_maxulp, ulp_diff(c, ag[p]));
          // a summation-order difference moves a gate / up sum by ~1e-6 of its terms: large only relative
          // to values near zero, never relative to the row
          if (std::fabs(double(off::bf16_float(c)) - double(off::bf16_float(ag[p]))) > 0.02 * rms) ++act_large;
        }
      }
      // down on the GPU's act, so the comparison isolates the W2 kernel
      off::prepare_row(ag, I, aperm.data());
      const off::PreparedRow ar{ag, aperm.data()};
      std::uint16_t* yp = y.data();
      off::expert_down(view(ex), &ar, 1, &yp, 0, H);
      for (int r = 0; r < H; ++r) sum[std::size_t(n) * H + r] = sum[std::size_t(n) * H + r] + wts[n * 8 + k] * off::bf16_float(y[r]);
    }
  }
  const float inv_sqrt5 = 1.f / 2.2360679774997896f;
  long y_large = 0;
  for (int n = 0; n < M; ++n) {
    double rms = 0;
    for (int r = 0; r < H; ++r) rms += double(out_gpu[std::size_t(n) * H + r]) * out_gpu[std::size_t(n) * H + r];
    rms = std::sqrt(rms / H);
    for (int r = 0; r < H; ++r) {
      const std::size_t i = std::size_t(n) * H + r;
      const float o = (0.f + off::bf16_float(off::bf16_bits(sum[i]))) * inv_sqrt5;
      ++y_total;
      if (o != out_gpu[i]) {
        ++y_mismatch;
        y_maxulp = std::max<long>(y_maxulp, ulp_diff(off::bf16_bits(o / inv_sqrt5), off::bf16_bits(out_gpu[i] / inv_sqrt5)));
        if (std::fabs(double(o) - double(out_gpu[i])) > 0.02 * rms) ++y_large;
      }
    }
  }
  const double ar = double(act_mismatch) / double(act_total), yr = double(y_mismatch) / double(y_total);
  // allowed: rare BF16 rounding-edge flips (0.2%), never a difference that matters at the row's scale
  const bool ok = ar < 0.002 && act_large == 0 && yr < 0.002 && y_large == 0;
  std::printf("%s layer %d (%s) M=%d: act %ld/%ld differ (%.3f%%, max %ld ulp, %ld > 2%% of row rms); "
              "down+sum %ld/%ld differ (%.3f%%, max %ld ulp, %ld > 2%% of row rms)\n",
              ok ? "PASS" : "FAIL", l, nv ? "nvfp4" : "mxfp8", M, act_mismatch, act_total, 100 * ar, act_maxulp,
              act_large, y_mismatch, y_total, 100 * yr, y_maxulp, y_large);
  return ok;
}

}  // namespace

// INGINE_CPU_EXPERT_BENCH=1: time expert_up + expert_down (one token) per expert, on 1 thread and on N threads
// working on different experts, over the layer's host copy (cold in cache: experts are streamed from DRAM).
void bench(const SafetensorsModel& m, int l) {
  const std::string e = "model.llm.layers." + std::to_string(l) + ".mlp.experts.";
  const bool nv = m.at(e + "w13_weight")->dtype == "U8";
  const std::vector<std::uint8_t> w13 = read_all(m, e + "w13_weight"), w13s = read_all(m, e + "w13_weight_scale");
  const std::vector<std::uint8_t> w2 = read_all(m, e + "w2_weight"), w2s = read_all(m, e + "w2_weight_scale");
  const std::size_t s13 = w13.size() / E, s13s = w13s.size() / E, s2 = w2.size() / E, s2s = w2s.size() / E;
  const double expert_bytes = double(s13 + s13s + s2 + s2s);
  std::vector<std::uint16_t> xb(H, 0x3f80);
  for (int threads : {1, 4, 8, 16}) {
    const int per = 16;  // experts per thread
    auto t0 = std::chrono::steady_clock::now();
    std::vector<std::thread> th;
    for (int t = 0; t < threads; ++t)
      th.emplace_back([&, t] {
        std::vector<std::uint16_t> xperm(H), act(I), aperm(I), y(H);
        off::prepare_row(xb.data(), H, xperm.data());
        const off::PreparedRow xr{xb.data(), xperm.data()};
        for (int i = 0; i < per; ++i) {
          const int ex = (t * per + i) % E;
          off::ExpertView v;
          v.fmt = nv ? off::ExpertFmt::NVFP4 : off::ExpertFmt::MXFP8;
          v.w13 = w13.data() + s13 * ex, v.w13s = w13s.data() + s13s * ex, v.w2 = w2.data() + s2 * ex;
          v.w2s = w2s.data() + s2s * ex;
          std::uint16_t* ap = act.data();
          off::expert_up(v, &xr, 1, &ap, 0, I);
          off::prepare_row(act.data(), I, aperm.data());
          const off::PreparedRow ar{act.data(), aperm.data()};
          std::uint16_t* yp = y.data();
          off::expert_down(v, &ar, 1, &yp, 0, H);
        }
      });
    for (auto& x : th) x.join();
    const double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    const double n = double(threads) * per;
    std::printf("bench layer %d (%s), %2d threads: %6.1f us per expert per thread, %5.1f GB/s total\n", l,
                nv ? "nvfp4" : "mxfp8", threads, sec / per * 1e6, n * expert_bytes / sec / 1e9);
  }
}

// INGINE_CPU_EXPERT_BENCH=2: the engine's hand-off shape outside the engine - one expert at a time, its rows
// sliced over a CpuPool of T threads (up phase, then down phase), random experts (streamed from DRAM).
void bench_pool(const SafetensorsModel& m, int l) {
  const std::string e = "model.llm.layers." + std::to_string(l) + ".mlp.experts.";
  const bool nv = m.at(e + "w13_weight")->dtype == "U8";
  const std::vector<std::uint8_t> w13 = read_all(m, e + "w13_weight"), w13s = read_all(m, e + "w13_weight_scale");
  const std::vector<std::uint8_t> w2 = read_all(m, e + "w2_weight"), w2s = read_all(m, e + "w2_weight_scale");
  const std::size_t s13 = w13.size() / E, s13s = w13s.size() / E, s2 = w2.size() / E, s2s = w2s.size() / E;
  std::vector<std::uint16_t> xb(H, 0x3f80), xperm(H), act(I), aperm(I), y(H);
  off::prepare_row(xb.data(), H, xperm.data());
  for (int T : {4, 8, 16}) {
    off::CpuPool pool(T, 3000);
    ingine::test::Rng rng{7};
    double up_s = 0, down_s = 0;
    const int iters = 200;
    for (int it = 0; it < iters; ++it) {
      const int ex = int(rng.next() % E);
      off::ExpertView v;
      v.fmt = nv ? off::ExpertFmt::NVFP4 : off::ExpertFmt::MXFP8;
      v.w13 = w13.data() + s13 * ex, v.w13s = w13s.data() + s13s * ex, v.w2 = w2.data() + s2 * ex;
      v.w2s = w2s.data() + s2s * ex;
      const off::PreparedRow xr{xb.data(), xperm.data()};
      auto t0 = std::chrono::steady_clock::now();
      pool.run(T, [&](int j) {
        std::uint16_t* ap = act.data();
        off::expert_up(v, &xr, 1, &ap, I * j / T, I * (j + 1) / T);
      });
      auto t1 = std::chrono::steady_clock::now();
      off::prepare_row(act.data(), I, aperm.data());
      const off::PreparedRow ar{act.data(), aperm.data()};
      auto t2 = std::chrono::steady_clock::now();
      pool.run(T, [&](int j) {
        std::uint16_t* yp = y.data();
        off::expert_down(v, &ar, 1, &yp, H * j / T, H * (j + 1) / T);
      });
      auto t3 = std::chrono::steady_clock::now();
      up_s += std::chrono::duration<double>(t1 - t0).count();
      down_s += std::chrono::duration<double>(t3 - t2).count();
    }
    const double bytes = double(s13 + s13s + s2 + s2s);
    std::printf("pool layer %d (%s), %2d threads: up %.1f us, down %.1f us per expert = %.1f GB/s\n", l,
                nv ? "nvfp4" : "mxfp8", T, up_s / iters * 1e6, down_s / iters * 1e6, bytes * iters / (up_s + down_s) / 1e9);
  }
}

int main() {
  if (ingine::test::skip_without_gpu()) return ingine::test::kSkip;
  if (!off::cpu_experts_supported()) {
    std::printf("SKIP: no AVX-512 BF16 CPU\n");
    return ingine::test::kSkip;
  }
  const char* env = std::getenv("INGINE_MODEL_DIR");
  const std::filesystem::path dir = env && *env ? env : INGINE_MODEL_DIR;
  if (!std::filesystem::exists(dir / "config.json")) {
    std::printf("SKIP: no model at %s\n", dir.c_str());
    return ingine::test::kSkip;
  }
  const SafetensorsModel m(dir);
  if (const char* b = std::getenv("INGINE_CPU_EXPERT_BENCH"); b && *b == '2') {
    bench_pool(m, 0);
    return 0;
  }
  if (const char* b = std::getenv("INGINE_CPU_EXPERT_BENCH"); b && *b == '1') {
    bench(m, 0);
    bench(m, 20);
    return 0;
  }
  bool ok = true;
  for (int l : {0, 20})
    for (int M : {1, 3, 8}) ok = run_layer(m, l, M, 0x0ff10ad0ull + std::uint64_t(l * 10 + M)) && ok;
  std::printf(ok ? "ALL PASS\n" : "SOME FAILED\n");
  return ok ? 0 : 1;
}
