#include "model/drafter.h"

#include "core/boot_trace.h"

#include "core/device.h"
#include "core/safetensors.h"
#include "kernels/draft/shortlist.h"

#include <unistd.h>

#include <algorithm>
#include <filesystem>
#include <system_error>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <fstream>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>

namespace ingine::model {

namespace kd = kernels::draft;
namespace kdense = kernels::dense;

namespace {

constexpr int H = kd::H;
constexpr int LOCAL_SLOTS = kd::WINDOW + kd::BLOCK;  // local ring + the drafted block's rows
constexpr int COMMIT_PIECE = 1024;                   // eager commit rows per piece

bool is_local(int l) { return l != 1; }  // dspark.LOCAL = (0, 2)

std::vector<float> bf16_to_float(const std::vector<std::byte>& raw) {
  std::vector<float> out(raw.size() / 2);
  for (std::size_t i = 0; i < out.size(); ++i) {
    std::uint16_t h;
    std::memcpy(&h, raw.data() + 2 * i, 2);
    const std::uint32_t u = std::uint32_t(h) << 16;
    std::memcpy(&out[i], &u, 4);
  }
  return out;
}

void capture_begin(cudaStream_t s) { CUDA_CHECK(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal)); }
cudaGraphExec_t capture_end(cudaStream_t s) {
  cudaGraph_t g = nullptr;
  CUDA_CHECK(cudaStreamEndCapture(s, &g));
  cudaGraphExec_t e = nullptr;
  const cudaError_t err = cudaGraphInstantiate(&e, g, 0);
  cudaGraphDestroy(g);
  CUDA_CHECK(err);
  return e;
}

}  // namespace

void* Drafter::alloc(std::size_t bytes) {
  storage_.emplace_back(bytes);
  return storage_.back().p;
}

Drafter::Drafter(const std::filesystem::path& model_dir, const Weights& w, int max_ctx, cudaStream_t stream,
                 int shortlist_rows)
    : w_(w), stream_(stream), ctx_(std::min(max_ctx, DRAFT_CTX)) {
  if (max_ctx < 2 * BLOCK) throw std::invalid_argument("Drafter: max_ctx must hold two blocks");
  head_ = {w.unembed.w.data, w.unembed.fp8() ? static_cast<const float*>(w.unembed.scale.data) : nullptr,
           w.unembed.rows()};
  V_ = w.unembed.rows();
  BootPhase p_load("  drafter: load()");
  load(model_dir);
  p_load.done();
  BootPhase p_rest("  drafter: buffers + tables");

  auto bf16s = [&](std::size_t n) { return static_cast<__nv_bfloat16*>(alloc(n * 2)); };
  auto f32s = [&](std::size_t n) { return static_cast<float*>(alloc(n * 4)); };
  auto i32s = [&](std::size_t n) { return static_cast<std::int32_t*>(alloc(n * 4)); };

  params_ = static_cast<kd::DraftParams*>(alloc(sizeof(kd::DraftParams)));
  CUDA_CHECK(cudaMemsetAsync(params_, 0, sizeof(kd::DraftParams), stream_));
  res_ = bf16s(BLOCK * H);
  xn_ = bf16s(BLOCK * H);
  q_ = bf16s(BLOCK * kd::Q_DIM);
  attn_o_ = bf16s(BLOCK * kd::Q_DIM);
  delta_ = f32s(BLOCK * H);
  act_ = bf16s(std::size_t(BLOCK) * inter_);
  f_rnn_ = bf16s(H);
  state_ = bf16s((GAMMA + 1) * kd::RANK);
  latent_ = bf16s(kd::RANK);
  qkv_out_ = f32s(BLOCK * kd::QKV_ROWS);
  gu_out_ = f32s(std::size_t(BLOCK) * 2 * inter_);
  // the shortlist head's base logits (rows past the ones in use stay -inf)
  base_ = f32s(std::size_t(GAMMA) * std::max(V_, shortlist_rows + 256 + CONTEXT_ROWS));
  attn_ws_ = f32s(std::max(kd::attention_workspace_floats(LOCAL_SLOTS), kd::attention_workspace_floats(ctx_)));
  topk_ws_ = alloc(kd::topk_workspace_bytes(GAMMA, V_));
  CUDA_CHECK(cudaMemsetAsync(topk_ws_, 0, kd::topk_workspace_bytes(GAMMA, V_), stream_));  // kept zeroed
  cand_val_ = f32s(GAMMA * kd::CANDIDATES);
  cand_logits_ = f32s(GAMMA * kd::CANDIDATES);
  part_ = f32s(GAMMA * kd::RNN_OUT);
  cand_idx_ = i32s(GAMMA * kd::CANDIDATES);
  prev_cand_ = i32s(GAMMA * kd::CANDIDATES);
  picks_ = static_cast<unsigned long long*>(alloc(GAMMA * sizeof(unsigned long long)));
  draft_logits_ = f32s(std::size_t(GAMMA) * V_);
  conf_ = f32s(GAMMA);
  qstats_ = f32s(2 * GAMMA);
  tokens_ = i32s(GAMMA);
  warm_feats_ = bf16s(std::size_t(BLOCK) * 4 * H);
  ctx_c_ = bf16s(std::size_t(COMMIT_PIECE) * H);
  ctx_kv_ = bf16s(std::size_t(COMMIT_PIECE) * 2 * kd::KV_DIM);
  commit_xin_ = bf16s(std::size_t(3) * BLOCK * H);
  commit_kv_ = f32s(std::size_t(BLOCK) * 3 * 2 * kd::KV_DIM);
  CUDA_CHECK(cudaMemsetAsync(warm_feats_, 0, std::size_t(BLOCK) * 4 * H * 2, stream_));
  CUDA_CHECK(cudaMemsetAsync(state_, 0, (GAMMA + 1) * kd::RANK * 2, stream_));  // state row 0 stays zero
  CUDA_CHECK(cudaMemsetAsync(prev_cand_, 0, GAMMA * kd::CANDIDATES * 4, stream_));
  CUDA_CHECK(cudaMemsetAsync(f_rnn_, 0, H * 2, stream_));
  kd::fill(draft_logits_, std::size_t(GAMMA) * V_, -INFINITY, stream_);

  // segment tables (row_begin, row_end, input) for the dense GEMV
  auto segments = [&](std::initializer_list<int> rows) {
    std::vector<int> host;
    int b = 0, input = 0;
    for (int r : rows) {
      host.insert(host.end(), {b, b + r, input++});
      b += r;
    }
    int* d = i32s(host.size());
    CUDA_CHECK(cudaMemcpyAsync(d, host.data(), host.size() * 4, cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    return d;
  };
  seg_qkv_ = segments({kd::QKV_ROWS});
  seg_gate_up_ = segments({2 * inter_});
  // the shortlist head: the shortlist_rows most frequent tokens and the specials, by ascending id (so equal
  // base logits break ties towards the lower id, as over the full head), then room for context tokens
  {
    short_in_.assign(V_, 0);
    int taken = 0;
    for (std::uint32_t t : kd::draft_shortlist_ranks())
      if (taken < shortlist_rows && t < std::uint32_t(V_) && !short_in_[t]) short_in_[t] = 1, ++taken;
    for (int t = 199990; t < V_; ++t) short_in_[t] = 1;  // <|unused|>, <|endoftext|> and the added tokens
    std::vector<std::int32_t> list;
    for (int t = 0; t < V_; ++t)
      if (short_in_[t]) list.push_back(t);
    short_K_ = int(list.size());
    const int rows = short_K_ + CONTEXT_ROWS;
    const std::size_t row_bytes = head_.scale ? std::size_t(H) : std::size_t(H) * 2;
    void* q = alloc(std::size_t(rows) * row_bytes);
    auto* qs = head_.scale ? f32s(rows) : nullptr;
    short_map_ = i32s(rows);
    short_ids_ = i32s(rows);
    CUDA_CHECK(cudaMemsetAsync(short_map_, 0, std::size_t(rows) * 4, stream_));
    CUDA_CHECK(cudaMemcpyAsync(short_ids_, list.data(), list.size() * 4, cudaMemcpyHostToDevice, stream_));
    kd::gather_head_rows(head_.w, head_.scale, row_bytes, short_ids_, short_K_, q, qs, short_map_, stream_);
    short_head_ = kdense::Matrix{q, qs, rows};
    seg_short_ = segments({short_K_});
    kd::fill(base_, std::size_t(GAMMA) * rows, -INFINITY, stream_);
  }
  seg_kv_ = segments({2 * kd::KV_DIM, 2 * kd::KV_DIM, 2 * kd::KV_DIM});  // layer l reads xin[l]

  for (int l = 0; l < 3; ++l) {
    DraftLayer& L = layers_[l];
    const int slots = is_local(l) ? LOCAL_SLOTS : ctx_;
    L.k_ctx = bf16s(std::size_t(slots) * kd::KV_DIM);
    L.v_ctx = bf16s(std::size_t(slots) * kd::KV_DIM);
    CUDA_CHECK(cudaMemsetAsync(L.k_ctx, 0, std::size_t(slots) * kd::KV_DIM * 2, stream_));
    CUDA_CHECK(cudaMemsetAsync(L.v_ctx, 0, std::size_t(slots) * kd::KV_DIM * 2, stream_));
    L.ctx_pos = is_local(l) ? i32s(kd::WINDOW) : nullptr;
    ctx_layers_.ctx_norm[l] = L.ctx_norm;
    ctx_layers_.k_norm_ctx[l] = L.k_norm_ctx;
    ctx_layers_.k_cache[l] = L.k_ctx;
    ctx_layers_.v_cache[l] = L.v_ctx;
    ctx_layers_.ctx_pos[l] = L.ctx_pos;
    ctx_layers_.size[l] = is_local(l) ? kd::WINDOW : ctx_;
  }

  accept_.target = i32s(BLOCK);
  accept_.probs = f32s(std::size_t(BLOCK) * V_);
  accept_.resid = f32s(V_);
  accept_.uniforms = f32s(GAMMA);
  accept_.ws = f32s(kd::sampled_accept_workspace_floats(V_));
  accept_.result = static_cast<kd::AcceptResult*>(alloc(sizeof(kd::AcceptResult)));

  reset();
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

Drafter::~Drafter() {
  for (auto& [b, g] : draft_graphs_) cudaGraphExecDestroy(g);
  if (commit_graph_) cudaGraphExecDestroy(commit_graph_);
  for (auto& [b, g] : advance_graphs_) cudaGraphExecDestroy(g);
}

// $INGINE_CACHE_DIR, else $XDG_CACHE_HOME/ingine, else ~/.cache/ingine; the fold file is named by a hash of every
// .safetensors file of the model directory (name, size, mtime), the vocabulary, the rank and the head format.
// Empty path: caching off ($INGINE_CACHE_DIR set to "-") or no home.
std::filesystem::path Drafter::fold_cache_path(const std::filesystem::path& model_dir, int V, bool fp8_head) {
  namespace fs = std::filesystem;
  fs::path dir;
  if (const char* c = std::getenv("INGINE_CACHE_DIR"); c && *c) {
    if (std::string(c) == "-") return {};
    dir = c;
  } else if (const char* x = std::getenv("XDG_CACHE_HOME"); x && *x) {
    dir = fs::path(x) / "ingine";
  } else if (const char* h = std::getenv("HOME"); h && *h) {
    dir = fs::path(h) / ".cache" / "ingine";
  } else {
    return {};
  }
  std::vector<std::string> parts;
  std::error_code ec;
  for (const auto& e : fs::directory_iterator(model_dir, ec)) {
    if (e.path().extension() != ".safetensors") continue;
    const auto size = fs::file_size(e.path(), ec);
    const auto mtime = fs::last_write_time(e.path(), ec).time_since_epoch().count();
    parts.push_back(e.path().filename().string() + ":" + std::to_string(size) + ":" + std::to_string(mtime));
  }
  std::sort(parts.begin(), parts.end());
  std::string key = "fold-v1|" + std::to_string(V) + "|" + std::to_string(kd::RANK) + "|" + (fp8_head ? "fp8" : "bf16");
  for (const auto& p : parts) key += "|" + p;
  std::uint64_t h = 1469598103934665603ull;  // FNV-1a
  for (unsigned char c : key) h = (h ^ c) * 1099511628211ull;
  char name[64];
  std::snprintf(name, sizeof name, "drafter-fold-%016llx.bin", static_cast<unsigned long long>(h));
  return dir / name;
}

void Drafter::load(const std::filesystem::path& model_dir) {
  SafetensorsFile f(model_dir / "mtp.safetensors");
  // Whole data region on the device, then the persistent (aligned, derived) tensors from it.
  const std::uint64_t base = f.data_offset();
  DeviceBuffer raw(f.file_size() - base);
  {
    std::vector<UploadRequest> reqs;
    for (const SafetensorsTensor& t : f.tensors())
      reqs.push_back(upload_request(f, t, static_cast<std::byte*>(raw.p) + (t.offset - base)));
    BootPhase p0("  drafter: uploader ctor");
    DeviceUploader up(UploadOptions{std::size_t(16) << 20, 4, 4});
    p0.done();
    BootPhase p("  drafter: upload mtp");
    up.upload(reqs, stream_);
  }
  auto src = [&](const std::string& name, std::initializer_list<std::int64_t> shape) {
    const SafetensorsTensor& t = f.at(name);
    if (t.dtype != "BF16") throw std::runtime_error("mtp.safetensors: " + name + " is not BF16");
    if (!std::equal(t.shape.begin(), t.shape.end(), shape.begin(), shape.end()))
      throw std::runtime_error("mtp.safetensors: unexpected shape of " + name);
    return static_cast<const std::byte*>(raw.p) + (t.offset - base);
  };
  // Copies tensors (concatenated along rows) into one new aligned device buffer.
  auto persist = [&](std::initializer_list<std::pair<std::string, std::vector<std::int64_t>>> parts) {
    std::size_t bytes = 0;
    for (const auto& [name, shape] : parts) bytes += f.at(name).bytes;
    auto* dst = static_cast<std::byte*>(alloc(bytes));
    std::size_t off = 0;
    for (const auto& [name, shape] : parts) {
      const SafetensorsTensor& t = f.at(name);
      if (t.dtype != "BF16" || t.shape != shape) throw std::runtime_error("mtp.safetensors: unexpected " + name);
      CUDA_CHECK(cudaMemcpyAsync(dst + off, static_cast<const std::byte*>(raw.p) + (t.offset - base), t.bytes,
                                 cudaMemcpyDeviceToDevice, stream_));
      off += t.bytes;
    }
    return reinterpret_cast<__nv_bfloat16*>(dst);
  };
  const std::int64_t h = H;
  inter_ = int(f.at("layers.0.mlp.gate_proj.weight").shape.at(0));
  const std::int64_t I = inter_;

  // alpha = bf16(softmax(fuse_weight.float(), -1))
  {
    const SafetensorsTensor& t = f.at("fuse_weight");
    if (t.dtype != "BF16" || t.shape != std::vector<std::int64_t>{4, 4})
      throw std::runtime_error("mtp.safetensors: unexpected fuse_weight");
    const std::vector<float> fw = bf16_to_float(f.read("fuse_weight"));
    __nv_bfloat16 host[16];
    for (int r = 0; r < 4; ++r) {
      float m = -INFINITY;
      for (int c = 0; c < 4; ++c) m = std::max(m, fw[r * 4 + c]);
      float e[4], s = 0.f;
      for (int c = 0; c < 4; ++c) s += (e[c] = std::exp(fw[r * 4 + c] - m));
      for (int c = 0; c < 4; ++c) host[r * 4 + c] = __float2bfloat16_rn(e[c] / s);
    }
    auto* d = static_cast<__nv_bfloat16*>(alloc(sizeof(host)));
    CUDA_CHECK(cudaMemcpyAsync(d, host, sizeof(host), cudaMemcpyHostToDevice, stream_));
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    alpha_ = d;
  }

  // FP8 rows of a row-concatenation (kernels::dense::fp8_rows, as dense.fp8_rows in dspark)
  DeviceBuffer cat(std::size_t(2) * I * h * 2);
  auto fp8 = [&](std::initializer_list<std::pair<std::string, std::int64_t>> parts) {
    std::int64_t rows = 0;
    for (const auto& [name, r] : parts) {
      const std::byte* s = src(name, {r, h});
      CUDA_CHECK(cudaMemcpyAsync(static_cast<std::byte*>(cat.p) + rows * h * 2, s, std::size_t(r * h * 2),
                                 cudaMemcpyDeviceToDevice, stream_));
      rows += r;
    }
    void* q = alloc(std::size_t(rows * h));
    auto* scale = static_cast<float*>(alloc(std::size_t(rows) * 4));
    kdense::fp8_rows(static_cast<const __nv_bfloat16*>(cat.p), int(rows), H, q, scale, stream_);
    return kdense::Matrix{q, scale, int(rows)};
  };

  // the three layers' context k/v projections in one matrix (one dense GEMV for a small commit)
  {
    auto kv = [](int l, const char* x) { return "layers." + std::to_string(l) + ".self_attn." + x + "_proj_ctx.weight"; };
    const std::vector<std::int64_t> shape{kd::KV_DIM, h};
    const __nv_bfloat16* all = persist({{kv(0, "k"), shape}, {kv(0, "v"), shape}, {kv(1, "k"), shape},
                                        {kv(1, "v"), shape}, {kv(2, "k"), shape}, {kv(2, "v"), shape}});
    wkv_all_ = kdense::Matrix{all, nullptr, 3 * 2 * kd::KV_DIM};
    for (int l = 0; l < 3; ++l) layers_[l].wkv_ctx = all + std::size_t(l) * 2 * kd::KV_DIM * H;
  }
  for (int l = 0; l < 3; ++l) {
    const std::string p = "layers." + std::to_string(l) + ".";
    const std::string a = p + "self_attn.";
    DraftLayer& L = layers_[l];
    L.in_norm = persist({{p + "input_layernorm.weight", {h}}});
    L.post_norm = persist({{p + "post_attention_layernorm.weight", {h}}});
    L.ctx_norm = persist({{p + "dflare_norm.weight", {h}}});
    L.q_norm = persist({{a + "q_norm.weight", {kd::HEAD_DIM}}});
    L.k_norm = persist({{a + "k_norm.weight", {kd::HEAD_DIM}}});
    L.k_norm_ctx = persist({{a + "k_norm_ctx.weight", {kd::HEAD_DIM}}});
    L.qkv = fp8({{a + "q_proj.weight", kd::Q_DIM}, {a + "k_proj.weight", kd::KV_DIM}, {a + "v_proj.weight", kd::KV_DIM}});
    L.gate_up = fp8({{p + "mlp.gate_proj.weight", I}, {p + "mlp.up_proj.weight", I}});
    L.wo = persist({{a + "o_proj.weight", {h, kd::Q_DIM}}});
    L.w_down = persist({{p + "mlp.down_proj.weight", {h, I}}});
  }
  final_norm_ = persist({{"norm.weight", {h}}});
  mask_embed_ = persist({{"mask_embed", {h}}});
  const std::string mh = "markov_head.";
  w1_ = persist({{mh + "markov_w1.weight", {f.at(mh + "markov_w1.weight").shape.at(0), kd::RANK}}});
  if (f.at(mh + "markov_w1.weight").shape.at(0) < V_)
    throw std::runtime_error("mtp.safetensors: markov_w1 has fewer rows than the vocabulary");
  rnn_w_ = persist({{mh + "rnn_proj.weight", {kd::RNN_OUT, kd::RNN_IN}}});
  rnn_b_ = persist({{mh + "rnn_proj.bias", {kd::RNN_OUT}}});
  f_norm_ = persist({{mh + "dflare_norm.weight", {h}}});

  // fold_w / fold_b (FP64, from the unembedding as the decoder stores it)
  fold_w_ = static_cast<__nv_bfloat16*>(alloc(std::size_t(V_) * kd::RANK * 2));
  fold_b_ = static_cast<__nv_bfloat16*>(alloc(std::size_t(V_) * 2));
  const __nv_bfloat16* hw = persist({{mh + "hidden_out.weight", {h, kd::RANK}}});
  const __nv_bfloat16* hb = persist({{mh + "hidden_out.bias", {h}}});
  // The fold is an FP64 product with the whole unembedding (~0.4 s on GPUs with 1/64-rate FP64, at every start)
  // of fixed inputs: cached on disk, keyed by the checkpoint files and the head's storage format.
  const std::filesystem::path cache = fold_cache_path(model_dir, V_, head_.scale != nullptr);
  const std::size_t wbytes = std::size_t(V_) * kd::RANK * 2, bbytes = std::size_t(V_) * 2;
  bool cached = false;
  if (!cache.empty()) {
    std::ifstream in(cache, std::ios::binary);
    std::vector<char> buf(wbytes + bbytes);
    if (in && in.read(buf.data(), std::streamsize(buf.size())) && in.peek() == std::ifstream::traits_type::eof()) {
      CUDA_CHECK(cudaMemcpyAsync(fold_w_, buf.data(), wbytes, cudaMemcpyHostToDevice, stream_));
      CUDA_CHECK(cudaMemcpyAsync(fold_b_, buf.data() + wbytes, bbytes, cudaMemcpyHostToDevice, stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));  // pageable source
      cached = true;
    }
  }
  if (!cached) {
    kd::fold_head(w_.unembed.w.data, head_.scale, V_, hw, hb, fold_w_, fold_b_, stream_);
    if (!cache.empty()) {
      std::vector<char> buf(wbytes + bbytes);
      CUDA_CHECK(cudaMemcpyAsync(buf.data(), fold_w_, wbytes, cudaMemcpyDeviceToHost, stream_));
      CUDA_CHECK(cudaMemcpyAsync(buf.data() + wbytes, fold_b_, bbytes, cudaMemcpyDeviceToHost, stream_));
      CUDA_CHECK(cudaStreamSynchronize(stream_));
      std::error_code ec;
      std::filesystem::create_directories(cache.parent_path(), ec);
      const std::filesystem::path tmp = cache.string() + ".tmp" + std::to_string(::getpid());
      {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out.write(buf.data(), std::streamsize(buf.size()));
        if (!out) ec = std::make_error_code(std::errc::io_error);
      }
      if (!ec) std::filesystem::rename(tmp, cache, ec);
      if (ec) std::filesystem::remove(tmp, ec);  // best effort: no cache next time
    }
  }
  CUDA_CHECK(cudaStreamSynchronize(stream_));  // raw and cat are freed on return
}

void Drafter::reset() {
  for (const DraftLayer& L : layers_)
    if (L.ctx_pos != nullptr) CUDA_CHECK(cudaMemsetAsync(L.ctx_pos, 0xff, kd::WINDOW * 4, stream_));
  length_ = 0;
  set_context_rows(0);
}

void Drafter::set_context_rows(int rows) {
  if (rows >= short_E_) return;
  for (std::size_t i = std::size_t(rows); i < short_extra_.size(); ++i) short_in_[short_extra_[i]] = 0;
  short_extra_.resize(std::size_t(rows));
  short_E_ = rows;
  const int seg[3] = {0, short_K_ + rows, 0};  // pageable source: copied before cudaMemcpyAsync returns
  CUDA_CHECK(cudaMemcpyAsync(seg_short_, seg, sizeof seg, cudaMemcpyHostToDevice, stream_));
  // the dropped rows' base logits back to -inf (the GEMV only writes the rows in use)
  for (int i = 0; i < GAMMA; ++i)
    kd::fill(base_ + std::size_t(i) * short_head_.rows + short_K_ + rows, std::size_t(CONTEXT_ROWS - rows), -INFINITY,
             stream_);
}

void Drafter::note_tokens(std::span<const std::int32_t> ids) {
  std::vector<std::int32_t> add;
  for (std::int32_t t : ids)
    if (t >= 0 && t < V_ && !short_in_[t] && short_E_ + int(add.size()) < CONTEXT_ROWS) {
      short_in_[t] = 1;
      add.push_back(t);
    }
  if (add.empty()) return;
  const std::size_t row = std::size_t(short_K_ + short_E_);
  const std::size_t row_bytes = head_.scale ? std::size_t(H) : std::size_t(H) * 2;
  // pageable sources: copied before cudaMemcpyAsync returns; short_ids_ is free once earlier gathers ran
  CUDA_CHECK(cudaMemcpyAsync(short_ids_, add.data(), add.size() * 4, cudaMemcpyHostToDevice, stream_));
  kd::gather_head_rows(head_.w, head_.scale, row_bytes, short_ids_, int(add.size()),
                       static_cast<std::uint8_t*>(const_cast<void*>(short_head_.w)) + row * row_bytes,
                       head_.scale ? const_cast<float*>(short_head_.scale) + row : nullptr, short_map_ + row, stream_);
  short_extra_.insert(short_extra_.end(), add.begin(), add.end());
  short_E_ += int(add.size());
  const int seg[3] = {0, short_K_ + short_E_, 0};
  CUDA_CHECK(cudaMemcpyAsync(seg_short_, seg, sizeof seg, cudaMemcpyHostToDevice, stream_));
}

void Drafter::set_seed(std::uint64_t seed) { kd::set_seed(params_, seed, stream_); }

void Drafter::global_written(int begin, int end) {
  if (checkpoint_ == nullptr || checkpoint_clobbered_ || end <= begin) return;
  // global slot = position % ctx; the checkpoint holds slots [0, min(L, ctx))
  const int live = std::min(checkpoint_length_, ctx_);
  checkpoint_clobbered_ = end - begin >= ctx_ || begin % ctx_ < live || begin / ctx_ != (end - 1) / ctx_;
}

// layout: local k/v of layers 0 and 2 (ring slots only), their ctx_pos, f_rnn, global k and v [ctx slots]
namespace {
constexpr std::size_t LOCAL_BYTES = std::size_t(kd::WINDOW) * kd::KV_DIM * 2;
}

std::size_t Drafter::checkpoint_bytes() const {
  return 4 * LOCAL_BYTES + 2 * kd::WINDOW * sizeof(std::int32_t) + std::size_t(H) * 2 +
         2 * std::size_t(ctx_) * kd::KV_DIM * 2;
}

void Drafter::save_checkpoint(void* dst) {
  auto* d = static_cast<std::byte*>(dst);
  auto copy = [&](std::size_t off, const void* src, std::size_t bytes) {
    CUDA_CHECK(cudaMemcpyAsync(d + off, src, bytes, cudaMemcpyDeviceToDevice, stream_));
  };
  std::size_t off = 0;
  for (int l : {0, 2}) {
    copy(off, layers_[l].k_ctx, LOCAL_BYTES);
    copy(off + LOCAL_BYTES, layers_[l].v_ctx, LOCAL_BYTES);
    off += 2 * LOCAL_BYTES;
  }
  for (int l : {0, 2}) {
    copy(off, layers_[l].ctx_pos, kd::WINDOW * sizeof(std::int32_t));
    off += kd::WINDOW * sizeof(std::int32_t);
  }
  copy(off, f_rnn_, std::size_t(H) * 2);
  off += std::size_t(H) * 2;
  const std::size_t live = std::size_t(std::min(length_, ctx_)) * kd::KV_DIM * 2;
  copy(off, layers_[1].k_ctx, live);
  copy(off + std::size_t(ctx_) * kd::KV_DIM * 2, layers_[1].v_ctx, live);
  checkpoint_ = dst;
  checkpoint_length_ = length_;
  checkpoint_clobbered_ = false;
  checkpoint_extra_ = short_extra_;
}

void Drafter::restore_checkpoint(const void* src) {
  if (src != checkpoint_) throw std::invalid_argument("Drafter::restore_checkpoint: not the last checkpoint saved");
  const auto* s = static_cast<const std::byte*>(src);
  auto copy = [&](void* dst, std::size_t off, std::size_t bytes) {
    CUDA_CHECK(cudaMemcpyAsync(dst, s + off, bytes, cudaMemcpyDeviceToDevice, stream_));
  };
  std::size_t off = 0;
  for (int l : {0, 2}) {
    copy(layers_[l].k_ctx, off, LOCAL_BYTES);
    copy(layers_[l].v_ctx, off + LOCAL_BYTES, LOCAL_BYTES);
    off += 2 * LOCAL_BYTES;
  }
  for (int l : {0, 2}) {
    copy(layers_[l].ctx_pos, off, kd::WINDOW * sizeof(std::int32_t));
    off += kd::WINDOW * sizeof(std::int32_t);
  }
  copy(f_rnn_, off, std::size_t(H) * 2);
  off += std::size_t(H) * 2;
  if (checkpoint_clobbered_) {
    const std::size_t live = std::size_t(std::min(checkpoint_length_, ctx_)) * kd::KV_DIM * 2;
    copy(layers_[1].k_ctx, off, live);
    copy(layers_[1].v_ctx, off + std::size_t(ctx_) * kd::KV_DIM * 2, live);
  }
  length_ = checkpoint_length_;
  checkpoint_clobbered_ = false;
  // the context tokens of the checkpoint: drop the ones added since (rebuild if they were reset meanwhile)
  const bool prefix = short_extra_.size() >= checkpoint_extra_.size() &&
                      std::equal(checkpoint_extra_.begin(), checkpoint_extra_.end(), short_extra_.begin());
  set_context_rows(prefix ? int(checkpoint_extra_.size()) : 0);
  if (!prefix) note_tokens(checkpoint_extra_);
}

void Drafter::run_commit(const __nv_bfloat16* feats, int n, int row0, bool last_piece) {
  for (int l = 0; l < 3; ++l) {
    const DraftLayer& L = layers_[l];
    kd::mix_rmsnorm(feats, n, alpha_ + 4 * l, L.ctx_norm, ctx_c_, stream_);
    gemm_.run(ctx_c_, n, L.wkv_ctx, 2 * kd::KV_DIM, H, ctx_kv_, stream_);
    const int size = is_local(l) ? kd::WINDOW : ctx_;
    kd::ctx_kv_store(ctx_kv_, n, params_, row0, n - std::min(n, size), L.k_norm_ctx, L.k_ctx, L.v_ctx, size,
                     L.ctx_pos, stream_);
  }
  if (last_piece) kd::mix_rmsnorm(feats + std::size_t(n - 1) * 4 * H, 1, alpha_ + 12, f_norm_, f_rnn_, stream_);
}

void Drafter::commit(const Tensor& feats, int start) {
  if (feats.dtype != DType::BF16 || feats.ne[0] != H || feats.ne[1] != 4 || !feats.is_contiguous())
    throw std::invalid_argument("Drafter::commit: feats must be contiguous bf16 [n, 4, 2304]");
  commit(static_cast<const __nv_bfloat16*>(feats.data), feats.ne[2] * feats.ne[3], start);
}

void Drafter::commit(const __nv_bfloat16* feats, int n, int start) {
  if (n <= 0) return;
  if (n <= BLOCK) {
    kd::set_commit(params_, feats, n, start, stream_);
    if (commit_graph_) CUDA_CHECK(cudaGraphLaunch(commit_graph_, stream_));
    else run_small_commit();
  } else {
    kd::set_commit_start(params_, start, stream_);
    for (int r0 = 0; r0 < n; r0 += COMMIT_PIECE) {
      const int k = std::min(COMMIT_PIECE, n - r0);
      run_commit(feats + std::size_t(r0) * 4 * H, k, r0, r0 + k == n);
    }
  }
  global_written(start, start + n);
  length_ = start + n;
}

// the k/v context rows of the n = params_->ncommit rows of params_->commit_feats in all three layers
void Drafter::run_small_commit() {
  kd::commit_mix(params_, alpha_, ctx_layers_, f_norm_, commit_xin_, f_rnn_, stream_);
  kdense::proj(commit_xin_, BLOCK, wkv_all_, seg_kv_, 3, commit_kv_, true, stream_);
  kd::commit_store(commit_kv_, params_, ctx_layers_, stream_);
}

void Drafter::run_draft(int bucket) {
  kd::embed_block(static_cast<const __nv_bfloat16*>(w_.embed.data), static_cast<const float*>(w_.embed_norm.data),
                  mask_embed_, layers_[0].in_norm, params_, res_, xn_, picks_, stream_);
  for (int l = 0; l < 3; ++l) {
    const DraftLayer& L = layers_[l];
    const bool local = is_local(l);
    kdense::proj(xn_, BLOCK, L.qkv, seg_qkv_, 1, qkv_out_, true, stream_);
    kd::qkv_rope(qkv_out_, L.q_norm, L.k_norm, params_, q_, L.k_ctx, L.v_ctx, local, ctx_, stream_);
    kd::block_attention(q_, L.k_ctx, L.v_ctx, local ? LOCAL_SLOTS : bucket, L.ctx_pos, params_, attn_ws_, attn_o_,
                        stream_);
    kdense::gemv(attn_o_, BLOCK, kd::Q_DIM, kdense::Matrix{L.wo, nullptr, H}, delta_, true, stream_);
    kd::add_rmsnorm(res_, delta_, L.post_norm, xn_, BLOCK, stream_);
    kdense::proj(xn_, BLOCK, L.gate_up, seg_gate_up_, 1, gu_out_, true, stream_);
    kd::silu_mul(gu_out_, BLOCK, inter_, act_, stream_);
    kdense::gemv(act_, BLOCK, inter_, kdense::Matrix{L.w_down, nullptr, H}, delta_, true, stream_);
    kd::add_rmsnorm(res_, delta_, l < 2 ? layers_[l + 1].in_norm : final_norm_, xn_, BLOCK, stream_);
  }
  const __nv_bfloat16* h = xn_ + H;  // rows 1..7
  kdense::proj(h, GAMMA, short_head_, seg_short_, 1, base_, true, stream_);
  kd::topk_candidates(base_, GAMMA, short_head_.rows, cand_val_, cand_idx_, topk_ws_, stream_);
  kd::map_candidates(cand_idx_, GAMMA * kd::CANDIDATES, short_map_, stream_);
  kd::markov_pre(rnn_w_, h, f_rnn_, part_, stream_);
  for (int i = 0; i < GAMMA; ++i) {
    kd::markov_step(i, rnn_w_, rnn_b_, w1_, part_, picks_, cand_idx_, params_, state_, latent_, stream_);
    kd::markov_logits(i, fold_w_, fold_b_, cand_val_, cand_idx_, latent_, params_, cand_logits_, picks_, stream_);
  }
  kd::markov_finish(cand_idx_, cand_logits_, picks_, prev_cand_, draft_logits_, V_, tokens_, conf_, qstats_, params_,
                    stream_);
}

// the smallest bucket holding the block; the whole ring once the context reaches it (and without graphs)
int Drafter::bucket_for(int P) const {
  for (const auto& [b, g] : draft_graphs_)
    if (b >= P + BLOCK) return b;
  return ctx_;
}

Drafter::Draft Drafter::draft(int anchor, int P, float temperature) {
  if (anchor < 0 || anchor >= V_) throw std::invalid_argument("Drafter::draft: anchor out of range");
  kd::set_draft_inputs(params_, anchor, P, temperature, stream_);
  global_written(P, P + BLOCK);  // the block's own rows
  const int bucket = bucket_for(P);
  if (auto it = draft_graphs_.find(bucket); it != draft_graphs_.end())
    CUDA_CHECK(cudaGraphLaunch(it->second, stream_));
  else
    run_draft(bucket);
  return {tokens_, draft_logits_, conf_, qstats_};
}

// set_round + small commit + draft, all reading device values
void Drafter::run_advance(int bucket) {
  kd::set_round(params_, round_, stream_);
  run_small_commit();
  run_draft(bucket);
}

void Drafter::bind_round(const kd::RoundInputs& in) {
  for (auto& [b, g] : advance_graphs_) cudaGraphExecDestroy(g);
  advance_graphs_.clear();
  round_ = in;
  round_bound_ = true;
  for (const auto& [b, g] : draft_graphs_) {  // the same buckets as the draft graphs (none before capture())
    capture_begin(stream_);
    run_advance(b);
    advance_graphs_[b] = capture_end(stream_);
  }
}

Drafter::Draft Drafter::advance(int start) {
  if (!round_bound_) throw std::logic_error("Drafter::advance: no bind_round()");
  pending_start_ = start;
  global_written(start, start + 2 * BLOCK);  // the commit (<= 8 rows) and the next block (P <= start + 8)
  const int bucket = bucket_for(start + BLOCK);  // P = start + n <= start + 8
  if (auto it = advance_graphs_.find(bucket); it != advance_graphs_.end())
    CUDA_CHECK(cudaGraphLaunch(it->second, stream_));
  else
    run_advance(bucket);
  return {tokens_, draft_logits_, conf_, qstats_};
}

void Drafter::capture() {
  std::vector<int> buckets;
  for (int b = 1024; b < ctx_; b *= 2) buckets.push_back(b);
  buckets.push_back(ctx_);
  // warm-up outside capture (cuBLASLt plans), then capture; the scratch context rows are reset afterwards
  kd::set_draft_inputs(params_, 0, 0, 0.f, stream_);
  kd::set_commit(params_, warm_feats_, BLOCK, 0, stream_);
  for (int b : buckets) run_draft(b);
  run_small_commit();
  CUDA_CHECK(cudaStreamSynchronize(stream_));
  for (int b : buckets) {
    capture_begin(stream_);
    run_draft(b);
    draft_graphs_[b] = capture_end(stream_);
  }
  capture_begin(stream_);
  run_small_commit();
  commit_graph_ = capture_end(stream_);
  if (round_bound_) bind_round(kd::RoundInputs(round_));
  reset();
  CUDA_CHECK(cudaStreamSynchronize(stream_));
}

}  // namespace ingine::model
