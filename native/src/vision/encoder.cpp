// CPU ImageEncoder: the vision tower on its own CPU pool (unpinned, so it shares cores with whatever else runs),
// weights read on a background thread at startup, one encode at a time, results cached by content.
#include "ingine/vision.h"

#include "core/safetensors.h"
#include "offload/cpu_pool.h"
#include "vision/image.h"
#include "vision/vit.h"

#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <list>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <thread>
#include <unordered_map>
#include <utility>

namespace ingine::vision {
namespace {

std::uint64_t mix64(std::uint64_t x) {
  x ^= x >> 30, x *= 0xbf58476d1ce4e5b9ull;
  x ^= x >> 27, x *= 0x94d049bb133111ebull;
  return x ^ (x >> 31);
}

// 64-bit content hash (two independent lanes over 8-byte words, then the tail and the length)
std::uint64_t content_hash(std::span<const std::uint8_t> b) {
  std::uint64_t h1 = 0x9e3779b97f4a7c15ull, h2 = 0xc2b2ae3d27d4eb4full;
  std::size_t i = 0;
  for (; i + 16 <= b.size(); i += 16) {
    std::uint64_t a, c;
    std::memcpy(&a, b.data() + i, 8);
    std::memcpy(&c, b.data() + i + 8, 8);
    h1 = (h1 ^ mix64(a)) * 0x100000001b3ull + 0x632be59bd9b4e019ull;
    h2 = (h2 ^ mix64(c + 0x165667b19e3779f9ull)) * 0x9e3779b97f4a7c15ull;
    h1 = (h1 << 31) | (h1 >> 33);
  }
  std::uint64_t tail = 0;
  for (std::size_t k = 0; i < b.size(); ++i, ++k) tail ^= std::uint64_t(b[i]) << (8 * (k % 8));
  return mix64(h1 ^ mix64(h2 ^ mix64(tail ^ b.size())));
}

int physical_cores() {
  std::set<std::pair<int, int>> cores;
  const unsigned n = std::thread::hardware_concurrency();
  for (unsigned c = 0; c < n; ++c) {
    const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(c) + "/topology/";
    std::ifstream pkg(base + "physical_package_id"), core(base + "core_id");
    int p = 0, k = 0;
    if (pkg >> p && core >> k) cores.insert({p, k});
  }
  if (!cores.empty()) return int(cores.size());
  return std::max(1, int(n) / 2);
}

class CpuImageEncoder final : public ImageEncoder {
 public:
  explicit CpuImageEncoder(const VisionOptions& o) : opt_(o) {
    if (!vision_supported()) throw std::runtime_error("vision: this CPU lacks AVX-512 BF16");
    if (const char* v = std::getenv("INGINE_VISION_THREADS"); v && *v) opt_.threads = std::atoi(v);
    if (opt_.threads <= 0) opt_.threads = physical_cores();
    opt_.max_tokens = std::clamp(opt_.max_tokens, 1, 4096);
    if (opt_.load_in_background)
      loader_ = std::thread([this] { load(); });
    else
      load();
  }
  ~CpuImageEncoder() override {
    if (loader_.joinable()) loader_.join();
  }

  int tokens_for(int width, int height) const override {
    const auto [w, h] = bounded_size(width, height, opt_.max_tokens);
    return token_grid(w) * token_grid(h);
  }

  EncodedImage encode(std::span<const std::uint8_t> file) override {
    const std::uint64_t key = mix64(content_hash(file) ^ (std::uint64_t(opt_.max_tokens) << 40));
    if (auto hit = lookup(key)) return *hit;
    const auto t0 = std::chrono::steady_clock::now();
    RgbImage img = decode_image(file);
    const int w0 = img.width, h0 = img.height;
    const auto [w, h] = bounded_size(img.width, img.height, opt_.max_tokens);
    if (w != img.width || h != img.height) img = resize(img, w, h);
    const Patches patches = patchify(img);
    VisionTower& vt = tower();
    std::unique_lock lock(encode_mu_);
    if (auto hit = lookup(key)) return *hit;  // encoded by another request meanwhile
    const auto t1 = std::chrono::steady_clock::now();
    auto embeds = std::make_shared<const std::vector<std::uint16_t>>(vt.forward(patches));
    lock.unlock();
    const auto t2 = std::chrono::steady_clock::now();
    EncodedImage e;
    e.rows = patches.rows, e.cols = patches.cols, e.width = w, e.height = h, e.hash = key;
    e.embeds = std::move(embeds);
    e.seconds = std::chrono::duration<double>(t2 - t0).count();
    spdlog::info("vision: {}x{}{} -> {} tokens ({}x{}), decode {:.0f} ms, encode {:.0f} ms ({} threads)", w0, h0,
                 w != w0 ? fmt::format(" resized to {}x{}", w, h) : std::string(), e.tokens(), e.rows, e.cols,
                 std::chrono::duration<double>(t1 - t0).count() * 1e3,
                 std::chrono::duration<double>(t2 - t1).count() * 1e3, opt_.threads);
    insert(e);
    return e;
  }

 private:
  void load() {
    try {
      const auto t0 = std::chrono::steady_clock::now();
      SafetensorsModel ckpt(opt_.model_dir);
      auto pool = std::make_unique<offload::CpuPool>(opt_.threads, 1000);
      auto tower = std::make_unique<VisionTower>(ckpt, *pool);
      spdlog::info("vision: tower loaded ({:.0f} MiB host memory, {} threads, max {} tokens per image) in {:.2f} s",
                   double(tower->weight_bytes()) / (1 << 20), opt_.threads, opt_.max_tokens,
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
      std::lock_guard lk(load_mu_);
      pool_ = std::move(pool);
      tower_ = std::move(tower);
      loaded_ = true;
    } catch (...) {
      std::lock_guard lk(load_mu_);
      load_err_ = std::current_exception();
      loaded_ = true;
    }
    load_cv_.notify_all();
  }

  VisionTower& tower() {
    std::unique_lock lk(load_mu_);
    load_cv_.wait(lk, [&] { return loaded_; });
    if (load_err_) std::rethrow_exception(load_err_);
    return *tower_;
  }

  std::optional<EncodedImage> lookup(std::uint64_t key) {
    std::lock_guard lk(cache_mu_);
    auto it = index_.find(key);
    if (it == index_.end()) return std::nullopt;
    lru_.splice(lru_.begin(), lru_, it->second);
    EncodedImage e = *it->second;
    e.cached = true, e.seconds = 0;
    return e;
  }

  void insert(const EncodedImage& e) {
    const std::size_t bytes = e.embeds->size() * 2;
    if (bytes > opt_.cache_bytes) return;
    std::lock_guard lk(cache_mu_);
    if (index_.count(e.hash)) return;
    lru_.push_front(e);
    index_[e.hash] = lru_.begin();
    cached_bytes_ += bytes;
    while (cached_bytes_ > opt_.cache_bytes && !lru_.empty()) {
      cached_bytes_ -= lru_.back().embeds->size() * 2;
      index_.erase(lru_.back().hash);
      lru_.pop_back();
    }
  }

  VisionOptions opt_;
  std::unique_ptr<offload::CpuPool> pool_;
  std::unique_ptr<VisionTower> tower_;
  std::thread loader_;
  std::mutex load_mu_;
  std::condition_variable load_cv_;
  bool loaded_ = false;
  std::exception_ptr load_err_;
  std::mutex encode_mu_;
  std::mutex cache_mu_;
  std::list<EncodedImage> lru_;
  std::unordered_map<std::uint64_t, std::list<EncodedImage>::iterator> index_;
  std::size_t cached_bytes_ = 0;
};

}  // namespace

bool vision_supported() {
  __builtin_cpu_init();
  return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") &&
         __builtin_cpu_supports("avx512vl") && __builtin_cpu_supports("avx512dq") &&
         __builtin_cpu_supports("avx512bf16") && __builtin_cpu_supports("fma");
}

bool has_vision_tower(const std::filesystem::path& model_dir) {
  try {
    return SafetensorsModel(model_dir).contains("model.visual.proj_out.weight");
  } catch (const std::exception&) {
    return false;
  }
}

std::unique_ptr<ImageEncoder> make_image_encoder(const VisionOptions& options) {
  return std::make_unique<CpuImageEncoder>(options);
}

}  // namespace ingine::vision
