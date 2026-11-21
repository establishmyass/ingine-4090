// Image input on the CPU.
//   kernels: the vector erf / exp and the packed BF16 GEMM against FP64 oracles; decoding + preprocessing of
//            the golden images bit-exact against tools/vision_reference.py.
//   tower:   the vision tower's embeddings of the golden images against the FP64 reference, and the encode
//            time of a large image ($INGINE_VISION_BENCH_TOKENS, default 1024 tokens).
#include "core/safetensors.h"
#include "ingine/vision.h"
#include "offload/cpu_pool.h"
#include "vision/image.h"
#include "vision/vit.h"

#include <nlohmann/json.hpp>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>

namespace {

using namespace ingine::vision;

int failures = 0;
#define CHECK(cond, ...)                                       \
  do {                                                         \
    if (!(cond)) {                                             \
      std::printf("FAIL %s:%d: %s: ", __FILE__, __LINE__, #cond); \
      std::printf(__VA_ARGS__);                                \
      std::printf("\n");                                       \
      ++failures;                                              \
    }                                                          \
  } while (0)

std::vector<std::uint8_t> read_file(const std::filesystem::path& p) {
  std::ifstream in(p, std::ios::binary);
  return {std::istreambuf_iterator<char>(in), {}};
}

float bf16f(std::uint16_t b) {
  const std::uint32_t u = std::uint32_t(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}
std::uint16_t f2bf(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1u);
  return std::uint16_t(u >> 16);
}

struct Rng {
  std::uint64_t s;
  double uniform() {  // [-1, 1)
    s += 0x9e3779b97f4a7c15ull;
    std::uint64_t z = s;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    z ^= z >> 31;
    return double(z >> 11) / double(1ull << 52) - 1.0;
  }
};

const std::filesystem::path kGolden = std::filesystem::path(INGINE_GOLDEN_DIR) / "vision";

void test_math() {
  double erf_err = 0, exp_err = 0;
  for (int i = -60000; i <= 60000; ++i) {
    const float x = float(i) * 1e-4f;
    erf_err = std::max(erf_err, std::abs(double(cpu::erf_lane(x)) - std::erf(double(x))));
  }
  for (int i = -870000; i <= 0; ++i) {
    const float x = float(i) * 1e-4f;
    const double ref = std::exp(double(x));
    exp_err = std::max(exp_err, std::abs(double(cpu::exp_lane(x)) - ref) / ref);
  }
  std::printf("erf max abs error %.3g, exp max rel error %.3g\n", erf_err, exp_err);
  CHECK(erf_err < 1e-6, "%g", erf_err);
  CHECK(exp_err < 5e-7, "%g", exp_err);
}

void test_gemm(ingine::offload::CpuPool& pool) {
  Rng rng{7};
  const int Ms[] = {1, 5, 12, 13, 96, 97, 250};
  const int Ns[] = {32, 128, 160, 288};
  const int Ks[] = {2, 64, 256, 258, 1200};
  double worst = 0;
  for (int M : Ms)
    for (int N : Ns)
      for (int K : Ks) {
        std::vector<std::uint16_t> A(std::size_t(M) * (K + 6)), W(std::size_t(N) * K), Wp(std::size_t(N) * K);
        for (auto& a : A) a = f2bf(float(rng.uniform()));
        for (auto& w : W) w = f2bf(float(rng.uniform()));
        cpu::pack_weight(W.data(), N, K, Wp.data());
        std::vector<float> C(std::size_t(M) * (N + 3), -1.f);
        cpu::gemm_f32(pool, A.data(), K + 6, M, Wp.data(), N, K, C.data(), N + 3);
        for (int m = 0; m < M; ++m)
          for (int n = 0; n < N; ++n) {
            double ref = 0, mag = 0;
            for (int k = 0; k < K; ++k) {
              const double p = double(bf16f(A[std::size_t(m) * (K + 6) + k])) * bf16f(W[std::size_t(n) * K + k]);
              ref += p, mag += std::abs(p);
            }
            worst = std::max(worst, std::abs(C[std::size_t(m) * (N + 3) + n] - ref) / std::max(mag, 1e-30));
          }
      }
  std::printf("gemm: worst error / sum|products| %.3g\n", worst);
  CHECK(worst < 1e-6, "%g", worst);
}

void test_preprocess() {
  const auto manifest = nlohmann::json::parse(std::ifstream(kGolden / "manifest.json"));
  for (const auto& m : manifest) {
    const RgbImage img = decode_image(read_file(kGolden / m["image"].get<std::string>()));
    const Patches p = patchify(img);
    const std::vector<std::uint8_t> want = read_file(kGolden / m["pixels"].get<std::string>());
    CHECK(p.rows == m["rows"].get<int>() && p.cols == m["cols"].get<int>(), "%s grid", m["image"].get<std::string>().c_str());
    CHECK(want.size() == p.data.size() * 2, "%s size %zu vs %zu", m["image"].get<std::string>().c_str(), want.size(),
          p.data.size() * 2);
    std::size_t diff = 0;
    for (std::size_t i = 0; i < std::min(want.size() / 2, p.data.size()); ++i) {
      std::uint16_t w;
      std::memcpy(&w, want.data() + 2 * i, 2);
      diff += w != p.data[i];
    }
    std::printf("preprocess %s: %dx%d tokens, %zu of %zu values differ\n", m["image"].get<std::string>().c_str(),
                p.rows, p.cols, diff, p.data.size());
    CHECK(diff == 0, "%zu", diff);
  }
  // the token bound keeps the aspect ratio and fits
  const auto [w, h] = bounded_size(4032, 3024, 1024);
  std::printf("bounded 4032x3024 -> %dx%d (%d tokens)\n", w, h, token_grid(w) * token_grid(h));
  CHECK(token_grid(w) * token_grid(h) <= 1024 && token_grid(w) * token_grid(h) > 950, "%d", token_grid(w) * token_grid(h));
  CHECK(std::abs(double(w) / h - 4032.0 / 3024.0) < 0.01, "%d %d", w, h);
}

void test_tower(ingine::offload::CpuPool& pool, const std::string& model_dir) {
  ingine::SafetensorsModel ckpt(model_dir);
  VisionTower tower(ckpt, pool);
  std::printf("tower: %.0f MiB, loaded in %.2f s\n", double(tower.weight_bytes()) / (1 << 20), tower.load_seconds());
  const auto manifest = nlohmann::json::parse(std::ifstream(kGolden / "manifest.json"));
  for (const auto& m : manifest) {
    const Patches p = patchify(decode_image(read_file(kGolden / m["image"].get<std::string>())));
    const std::vector<std::uint16_t> out = tower.forward(p);
    const std::vector<std::uint8_t> raw = read_file(kGolden / m["embeds"].get<std::string>());
    std::vector<float> want(raw.size() / 4);
    std::memcpy(want.data(), raw.data(), raw.size());
    CHECK(want.size() == out.size(), "size");
    if (want.size() != out.size()) continue;
    double num = 0, den = 0, worst_cos = 1;
    const int T = p.rows * p.cols;
    for (int t = 0; t < T; ++t) {
      double dot = 0, a2 = 0, b2 = 0;
      for (int c = 0; c < kTextHidden; ++c) {
        const double a = bf16f(out[std::size_t(t) * kTextHidden + c]), b = want[std::size_t(t) * kTextHidden + c];
        num += (a - b) * (a - b), den += b * b;
        dot += a * b, a2 += a * a, b2 += b * b;
      }
      worst_cos = std::min(worst_cos, dot / std::sqrt(a2 * b2));
    }
    const double rel = std::sqrt(num / den);
    std::printf("tower %s: relative RMS error %.4f, worst token cosine %.6f\n", m["image"].get<std::string>().c_str(),
                rel, worst_cos);
    CHECK(rel < 0.03, "%g", rel);
    CHECK(worst_cos > 0.999, "%g", worst_cos);
  }
  // speed: a synthetic image of about $INGINE_VISION_BENCH_TOKENS tokens
  int tokens = 1024;
  if (const char* v = std::getenv("INGINE_VISION_BENCH_TOKENS")) tokens = std::atoi(v);
  const int side = int(std::sqrt(double(tokens))) * 40;
  RgbImage img;
  img.width = side, img.height = int(double(tokens) / (side / 40)) * 40;
  img.rgb.resize(std::size_t(img.width) * img.height * 3);
  for (std::size_t i = 0; i < img.rgb.size(); ++i) img.rgb[i] = std::uint8_t((i * 2654435761u) >> 24);
  const Patches p = patchify(img);
  for (int rep = 0; rep < 2; ++rep) {
    const auto t0 = std::chrono::steady_clock::now();
    tower.forward(p);
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    std::printf("encode %dx%d px (%d tokens, %d threads): %.0f ms\n", img.width, img.height, p.rows * p.cols,
                pool.threads(), s * 1e3);
  }
}

}  // namespace

int main(int argc, char** argv) {
  const std::string mode = argc > 1 ? argv[1] : "kernels";
  if (!vision_supported()) {
    std::printf("skip: no AVX-512 BF16\n");
    return 77;
  }
  int threads = int(std::thread::hardware_concurrency() / 2);
  if (const char* v = std::getenv("INGINE_VISION_THREADS")) threads = std::atoi(v);
  ingine::offload::CpuPool pool(std::max(1, threads), 1000);
  if (mode == "kernels") {
    test_math();
    test_gemm(pool);
    test_preprocess();
  } else {
    const std::string dir = std::getenv("INGINE_MODEL_DIR") ? std::getenv("INGINE_MODEL_DIR") : INGINE_MODEL_DIR;
    if (!has_vision_tower(dir)) {
      std::printf("skip: no vision tower in %s\n", dir.c_str());
      return 77;
    }
    test_tower(pool, dir);
  }
  std::printf(failures ? "FAILED (%d)\n" : "ok\n", failures);
  return failures ? 1 : 0;
}
