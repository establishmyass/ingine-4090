// The Generator (engine.Engine): one resident model, one request at a time. Prefill runs in chunks through
// the Prefiller; plain decoding replays the width-1 graph per token, pipelined (each step's input token
// stays on the device, the host launches the next step before it reads the current token); with
// options.spec the speculative loop (SpecLoop) drafts with DSpark and verifies with the width-2/4/8 graphs.
// A prompt that extends the tokens already in the state only prefills the new suffix; otherwise the longest
// prompt checkpoint that is a prefix of it is restored (the last prompt's in VRAM, a few earlier ones in pinned
// host memory), else it starts from an empty state.
#include "core/boot_trace.h"

#include <spdlog/spdlog.h>
#include "ingine/generator.h"

#include "core/device.h"
#include "kernels/head/head.h"
#include "model/drafter.h"
#include "model/prefill.h"
#include "model/runtime.h"
#include "model/spec.h"

#include <algorithm>
#include <chrono>
#include <list>
#include <memory>
#include <mutex>
#include <random>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace ingine::model {
namespace {

namespace head = kernels::head;

double seconds_since(std::chrono::steady_clock::time_point t) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - t).count();
}

const RuntimeOptions& without_spec(const RuntimeOptions& options) {
  if (options.spec) throw std::invalid_argument("make_generator over given weights: spec needs the model directory");
  return options;
}

// An image of a prompt held in the state (positions, size, content hash).
struct ImageKey {
  int pos = 0, tokens = 0;
  std::uint64_t hash = 0;
  bool operator==(const ImageKey&) const = default;
};

// An earlier prompt's checkpoint in pinned host memory (EngineGenerator::checkpoint_'s layout) and what it
// holds: the prompt, its images, the drafter's length and context tokens. Empty ids: unused or stale.
struct OlderCheckpoint {
  std::unique_ptr<PinnedHostBuffer> buf;
  std::vector<std::int32_t> ids;
  std::vector<ImageKey> images;
  int drafter_length = 0;
  std::vector<std::int32_t> drafter_extra;
  std::uint64_t seq = 0;
};

class EngineGenerator final : public Generator {
 public:
  explicit EngineGenerator(const RuntimeOptions& options)
      : EngineGenerator(options, std::make_unique<Runtime>(options)) {}
  EngineGenerator(const RuntimeOptions& options, Weights weights, Ablation ablation)
      : EngineGenerator(without_spec(options),
                        std::make_unique<Runtime>(options, std::move(weights), std::move(ablation))) {}

  std::string model_id() const override {
    return options_.ablation.empty() ? "inkling-2-mini" : "inkling-2-mini-abliterated";
  }

  int max_model_len() const override { return options_.max_ctx; }

 private:
  EngineGenerator(const RuntimeOptions& options, std::unique_ptr<Runtime> runtime)
      : options_(options),
        runtime_(std::move(runtime)),
        rt_(*runtime_),
        prefiller_(rt_.weights(), rt_.ablation(), options.prefill_chunk, rt_.stream()),
        chunk_tokens_(std::size_t(options.prefill_chunk) * sizeof(std::int32_t)),
        logits_(std::size_t(UNPADDED_VOCAB) * sizeof(float)),
        token_(sizeof(std::int32_t)),
        host_token_(sizeof(std::int32_t)) {
    if (options.park_host_mib < 0) throw std::invalid_argument("park_host_mib must be nonnegative");
    if (options.prefill_chunk < 1) throw std::invalid_argument("prefill_chunk must be positive");
    BootPhase spec_phase("generator: drafter + spec");
    if (options.spec) {
      {
        BootPhase p("  drafter: load");
        drafter_ = std::make_unique<Drafter>(options.model_dir, rt_.weights(), options.max_ctx, rt_.stream());
      }
      {
        BootPhase p("  drafter: capture");
        drafter_->capture();
      }
      taps_ = DeviceBuffer(std::size_t(options.prefill_chunk) * NUM_TAPS * HIDDEN * 2);
      BootPhase p("  spec loop");
      spec_ = std::make_unique<SpecLoop>(rt_, *drafter_);
    }
    // checkpoint layout: target state | drafter context | last prompt logits
    checkpoint_drafter_ = carved(rt_.checkpoint_bytes());
    checkpoint_logits_ = checkpoint_drafter_ + (drafter_ ? carved(drafter_->checkpoint_bytes()) : 0);
    checkpoint_ = DeviceBuffer(checkpoint_logits_ + logits_.bytes);
    // older prompt checkpoints in pinned host memory ($INGINE_PROMPT_CHECKPOINTS counts the VRAM one too)
    int keep = 4;
    if (const char* v = std::getenv("INGINE_PROMPT_CHECKPOINTS"); v && *v) keep = std::max(1, std::atoi(v));
    older_.resize(std::size_t(keep - 1));
    spdlog::info("prompt checkpoints: 1 in VRAM + {} in host memory (allocated when first used), {:.0f} MiB each",
                 older_.size(), double(checkpoint_.bytes) / (1 << 20));
    if (!older_.empty()) {
      CUDA_CHECK(cudaStreamCreateWithFlags(&copy_stream_, cudaStreamNonBlocking));
      CUDA_CHECK(cudaEventCreateWithFlags(&ev_ready_, cudaEventDisableTiming));
      CUDA_CHECK(cudaEventCreateWithFlags(&ev_moved_, cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaStreamSynchronize(rt_.stream()));
    spec_phase.done();
    warm_up();
  }

  // One tiny generation at startup ($INGINE_WARMUP=0 skips it): the first prefill pays one-time setup
  // (cuBLASLt, first-use allocations, ~60 ms) that would otherwise land on the first request's time to first
  // token. The state is then forgotten, so no request can reuse the dummy prompt.
  void warm_up() {
    if (const char* v = std::getenv("INGINE_WARMUP"); v && *v == '0') return;
    BootPhase p("generator: warm-up");
    const std::int32_t prompt[16] = {1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008, 1009, 1010, 1011, 1012, 1013, 1014, 1015};
    Sampling greedy;
    greedy.temperature = 0.f;
    generate(prompt, 2, greedy, {}, [](std::span<const std::int32_t>) { return true; });
    forget_state();
  }

 public:
  GenerateStats generate(std::span<const std::int32_t> prompt, int max_new_tokens, const Sampling& sampling,
                         std::span<const std::int32_t> stop_ids,
                         const std::function<bool(std::span<const std::int32_t>)>& on_tokens,
                         std::span<const PromptImage> images = {}) override {
    if (prompt.empty()) throw std::invalid_argument("prompt is empty");
    const int n = int(prompt.size());
    for (std::size_t i = 0; i < images.size(); ++i) {
      const PromptImage& im = images[i];
      const bool overlaps = i > 0 && im.pos < images[i - 1].pos + images[i - 1].tokens;
      if (im.tokens < 1 || im.pos < 0 || im.pos + im.tokens > n || overlaps)
        throw std::invalid_argument("image spans must be ordered, disjoint and inside the prompt");
      if (!im.embeds || im.embeds->size() != std::size_t(im.tokens) * HIDDEN)
        throw std::invalid_argument("image embeddings must be [tokens, 2304]");
    }
    if (!sampling.forced.empty()) max_new_tokens = std::min(max_new_tokens, int(sampling.forced.size()));
    // a verify call writes up to max(verify widths) positions
    const int reserve = drafter_ ? *std::max_element(rt_.verify_widths().begin(), rt_.verify_widths().end()) : 0;
    if (n + std::int64_t(max_new_tokens) + reserve > options_.max_ctx) {
      max_new_tokens = options_.max_ctx - n - reserve;
      if (max_new_tokens <= 0)
        throw std::invalid_argument("prompt of " + std::to_string(n) + " tokens does not fit max_ctx " +
                                    std::to_string(options_.max_ctx));
    }
    GenerateStats stats;
    head::SamplerRng rng;
    rng.seed = sampling.seed ? *sampling.seed : (std::uint64_t(std::random_device{}()) << 32) ^ std::random_device{}();
    std::lock_guard<std::mutex> lock(mutex_);

    auto t = std::chrono::steady_clock::now();
    try {
      prefill(prompt, images, stats);
      if (boot_trace_enabled()) {
        CUDA_CHECK(cudaStreamSynchronize(rt_.stream()));
        spdlog::info("request: prefill of {} tokens {:.1f} ms", n, seconds_since(t) * 1e3);
      }
      stats.prompt_tokens = n;
      if (spec_) {
        std::int32_t token = sample(static_cast<const float*>(logits_.p), sampling, rng);
        if (!sampling.forced.empty()) token = sampling.forced[0];
        stats.prefill_seconds = seconds_since(t);
        t = std::chrono::steady_clock::now();
        if (max_new_tokens > 0) {
          drafter_->set_seed(rng.seed);
          spec_->generate(token, max_new_tokens, sampling, rng, stop_ids, on_tokens, stats, tokens_in_state_);
        }
      } else {
        decode(n, max_new_tokens, sampling, rng, stop_ids, on_tokens, stats, t);
      }
      CUDA_CHECK(cudaStreamSynchronize(rt_.stream()));
    } catch (...) {
      forget_state();
      throw;
    }
    stats.decode_seconds = seconds_since(t);
    return stats;
  }

  ~EngineGenerator() override {
    // nothing queued may outlive the graphs and buffers of the members destroyed below
    if (rt_.stream()) cudaStreamSynchronize(rt_.stream());
    if (copy_stream_) {
      cudaStreamSynchronize(copy_stream_);
      cudaStreamDestroy(copy_stream_);
      cudaEventDestroy(ev_ready_);
      cudaEventDestroy(ev_moved_);
    }
  }

 private:
  // After a failure anywhere in a request the device state may be ahead of or behind tokens_in_state_ (a step
  // in flight, a partial prefill, a round between launch and bookkeeping): wait for the stream and forget
  // everything reusable, so the next request starts from an empty state.
  void forget_state() noexcept {
    cudaStreamSynchronize(rt_.stream());
    tokens_in_state_.clear();
    checkpoint_ids_.clear();
    state_images_.clear();
    checkpoint_images_.clear();
    if (copy_stream_) cudaStreamSynchronize(copy_stream_);
    move_pending_ = false;
    for (OlderCheckpoint& o : older_) o.ids.clear();
    parked_.clear();
    spare_park_buffer_.reset();
    if (drafter_) drafter_->reset();  // stream-ordered; CUDA failures abort (CUDA_CHECK)
  }

  // Plain decoding after the prefill of n tokens (logits in logits_): the token at position p is sampled on
  // the device, chain_advance makes it the width-1 step's input, and the step at p is launched before the
  // host reads that token, so the GPU never waits for the host. A step launched for a token that ends the
  // generation (stop token, cancel) is undone, leaving the state at tokens_in_state_. Sets prefill_seconds
  // (until the first token is known) and restarts `t` for the decode time.
  void decode(int n, int max_new_tokens, const Sampling& sampling, head::SamplerRng& rng,
              std::span<const std::int32_t> stop_ids,
              const std::function<bool(std::span<const std::int32_t>)>& on_tokens, GenerateStats& stats,
              std::chrono::steady_clock::time_point& t) {
    const cudaStream_t stream = rt_.stream();
    const head::SamplingParams params{sampling.temperature, sampling.top_k, sampling.top_p};
    auto* sampled = static_cast<std::int32_t*>(token_.p);
    // forced replay: the forced tokens on the device; each sampled token is replaced by its forced one
    DeviceBuffer forced;
    if (!sampling.forced.empty()) {
      forced = DeviceBuffer(sampling.forced.size_bytes());
      CUDA_CHECK(cudaMemcpyAsync(forced.p, sampling.forced.data(), sampling.forced.size_bytes(),
                                 cudaMemcpyHostToDevice, stream));
    }
    // engine._sample of the logits into `sampled`, then the inputs of the step at `position`
    auto sample_for = [&](const float* logits, int position) {
      head::sample(logits, 1, UNPADDED_VOCAB, UNPADDED_VOCAB, params, rng, sampled, stream);
      if (forced.p)
        CUDA_CHECK(cudaMemcpyAsync(sampled, static_cast<const std::int32_t*>(forced.p) + (position - n),
                                   sizeof(std::int32_t), cudaMemcpyDeviceToDevice, stream));
      rt_.chain_advance(sampled, position);
    };
    // the step at `position` (its input token is on the device), then the next token
    auto launch = [&](int position) { sample_for(rt_.run_step(1), position + 1); };
    auto is_stop = [&](std::int32_t tok) { return std::find(stop_ids.begin(), stop_ids.end(), tok) != stop_ids.end(); };

    sample_for(static_cast<const float*>(logits_.p), n);
    if (max_new_tokens > 1) launch(n);
    std::int32_t token = rt_.chain_token(n);
    stats.prefill_seconds = seconds_since(t);
    t = std::chrono::steady_clock::now();
    for (int i = 0; i < max_new_tokens; ++i) {
      // `token` is at position n + i; its step is in flight unless it is the budget's last token
      const bool last = i + 1 == max_new_tokens;
      stats.generated_tokens = i + 1;
      const std::int32_t out[1] = {token};
      const bool go_on = on_tokens(out);
      if (last || is_stop(token) || !go_on) {
        if (!last) rt_.chain_undo(n + i);
        break;
      }
      tokens_in_state_.push_back(token);
      if (i + 2 < max_new_tokens) launch(n + i + 1);
      token = rt_.chain_token(n + i + 1);
    }
  }

  // engine._prefill with prompt checkpoints: brings the state to `ids` (last logits in logits_) and sets
  // stats.reused_tokens and stats.reuse.
  // * Extend: `ids` extends tokens_in_state_ (the previous prompt and the generated tokens processed since):
  //   prefill the new suffix.
  // * Checkpoint: the prompt of the request that saved the checkpoint (the state right after its prefill) is a
  //   prefix of `ids`: restore it and prefill the rest. This is the agent-loop case: the next prompt
  //   re-renders the previous answer differently from the tokens the model generated.
  // * None: reset and prefill everything.
  // After a prefill the checkpoint moves to the end of `ids`. Images take part in the prefix comparison: a reused
  // prefix must hold the same images (positions, sizes, content hashes) as the new prompt's first `prefix` tokens.
  void prefill(std::span<const std::int32_t> ids, std::span<const PromptImage> images, GenerateStats& stats) {
    const std::vector<std::int32_t>& prev = tokens_in_state_;
    const std::size_t n = ids.size();
    const bool drafter_in_step = !drafter_ || drafter_->length() == int(prev.size());  // else a row is missing
    State& st = rt_.state();
    const cudaStream_t stream = rt_.stream();
    std::size_t start = 0;
    auto fits = [&](const auto& tokens, const auto& held_images) {
      return !tokens.empty() && tokens.size() <= n && std::equal(tokens.begin(), tokens.end(), ids.begin()) &&
             same_images(held_images, images, tokens.size());
    };
    if (drafter_in_step && prev.size() < n && fits(prev, state_images_)) {
      start = prev.size();
      stats.reuse = Reuse::Extend;
    }
    if (checkpoint_ids_.size() > start && fits(checkpoint_ids_, checkpoint_images_)) {
      start = checkpoint_ids_.size();
      stats.reuse = Reuse::Checkpoint;
    }
    const OlderCheckpoint* older = nullptr;
    for (const auto& o : older_)
      if (o.ids.size() > start && fits(o.ids, o.images)) {
        start = o.ids.size();
        older = &o;
        stats.reuse = Reuse::OlderCheckpoint;
      }
    auto parked = parked_.end();
    for (auto it = parked_.begin(); it != parked_.end(); ++it)
      if (it->ids.size() > start && fits(it->ids, it->images)) {
        start = it->ids.size();
        parked = it;
        stats.reuse = Reuse::Parked;
      }
    if (parked != parked_.end()) {
      OlderCheckpoint incoming = std::move(*parked);
      parked_.erase(parked);
      park_live(incoming.buf->size());
      rt_.restore_full_rows(static_cast<const char*>(incoming.buf->data()) + checkpoint_.bytes, int(start));
      restore_older(incoming);
      // The source stays pinned through the stream-ordered restore and can hold the next displaced chat.
      spare_park_buffer_ = std::move(incoming.buf);
      checkpoint_ids_.clear();
      checkpoint_images_.clear();
      for (auto& o : older_) o.ids.clear();  // these depended on the displaced conversation's full KV
    } else if (older) {
      restore_older(*older);
    } else if (stats.reuse == Reuse::Checkpoint) {
      restore_checkpoint();
    } else if (stats.reuse == Reuse::None) {
      park_live(0);
      reset_state(st, stream);
      if (drafter_) drafter_->reset();
      tokens_in_state_.clear();
      state_images_.clear();
    }
    // positions from `start` on are rewritten: checkpoints longer than that no longer match the state
    if (checkpoint_ids_.size() > start) checkpoint_ids_.clear(), checkpoint_images_.clear();
    for (OlderCheckpoint& o : older_)
      if (o.ids.size() > start) o.ids.clear();
    // the VRAM checkpoint is replaced at the end of this prefill: copy it to host memory meanwhile
    if (start < n && !checkpoint_ids_.empty() && !older_.empty()) move_checkpoint_to_host();
    st.length = int(start);  // decode steps advance the state without its host-side length
    const std::size_t chunk = std::size_t(options_.prefill_chunk);
    for (std::size_t s = start; s < n; s += chunk) {
      const int T = int(std::min(chunk, n - s));
      CUDA_CHECK(cudaMemcpyAsync(chunk_tokens_.p, ids.data() + s, std::size_t(T) * sizeof(std::int32_t),
                                 cudaMemcpyHostToDevice, stream));
      Tensor taps;
      if (drafter_) taps = rm(taps_.p, DType::BF16, {T, NUM_TAPS, HIDDEN});
      const RowOverride rows = image_rows(images, int(s), T);
      prefiller_.run(st, static_cast<const std::int32_t*>(chunk_tokens_.p), T, int(s), static_cast<float*>(logits_.p),
                     drafter_ ? &taps : nullptr, rows.n ? &rows : nullptr, s == 0 || s - start >= WINDOW);
      if (drafter_) {
        drafter_->note_tokens(ids.subspan(s, std::size_t(T)));  // the prompt's tokens join the drafter's shortlist
        drafter_->commit(taps, int(s));
      }
    }
    tokens_in_state_.assign(ids.begin(), ids.end());
    state_images_.clear();
    for (const PromptImage& im : images) state_images_.push_back({im.pos, im.tokens, im.hash});
    if (start < n || stats.reuse == Reuse::Parked) save_checkpoint();
    stats.reused_tokens = int(start);
  }

  // The checkpoint (on the device): target state (Runtime::save_checkpoint) | drafter context
  // (Drafter::save_checkpoint) | logits of the last prompt token.
  void save_checkpoint() {
    if (move_pending_) {  // the VRAM checkpoint is still being copied to host memory
      CUDA_CHECK(cudaStreamWaitEvent(rt_.stream(), ev_moved_, 0));
      move_pending_ = false;
    }
    char* p = static_cast<char*>(checkpoint_.p);
    rt_.save_checkpoint(p);
    if (drafter_) drafter_->save_checkpoint(p + checkpoint_drafter_);
    CUDA_CHECK(cudaMemcpyAsync(p + checkpoint_logits_, logits_.p, logits_.bytes, cudaMemcpyDeviceToDevice,
                               rt_.stream()));
    checkpoint_ids_ = tokens_in_state_;
    checkpoint_images_ = state_images_;
  }

  void restore_checkpoint() {
    const char* p = static_cast<const char*>(checkpoint_.p);
    rt_.restore_checkpoint(p);
    if (drafter_) drafter_->restore_checkpoint(p + checkpoint_drafter_);
    CUDA_CHECK(cudaMemcpyAsync(logits_.p, p + checkpoint_logits_, logits_.bytes, cudaMemcpyDeviceToDevice,
                               rt_.stream()));
    tokens_in_state_ = checkpoint_ids_;
    state_images_ = checkpoint_images_;
  }

  // An older checkpoint (host memory, same layout) back into the state. Stream-ordered after the copy that
  // filled it (save_checkpoint waited for it).
  void restore_older(const OlderCheckpoint& o) {
    const char* p = static_cast<const char*>(o.buf->data());
    rt_.restore_checkpoint(p);
    if (drafter_) drafter_->restore_copy(p + checkpoint_drafter_, o.drafter_length, o.drafter_extra);
    CUDA_CHECK(cudaMemcpyAsync(logits_.p, p + checkpoint_logits_, logits_.bytes, cudaMemcpyHostToDevice, rt_.stream()));
    tokens_in_state_ = o.ids;
    state_images_ = o.images;
  }

  // Copies the VRAM checkpoint into a host slot on the copy stream, overlapping the prefill that follows;
  // save_checkpoint waits for it before overwriting. The slot: an unused one, else the oldest other than the one
  // just restored.
  void move_checkpoint_to_host() {
    std::size_t slot = 0;
    bool found = false;
    for (std::size_t i = 0; i < older_.size() && !found; ++i)
      if (older_[i].ids.empty()) slot = i, found = true;
    if (!found) {
      std::size_t keep = older_.size();  // the restored one, if any
      for (std::size_t i = 0; i < older_.size(); ++i)
        if (older_[i].ids.size() == tokens_in_state_.size() && older_[i].ids == tokens_in_state_) keep = i;
      std::uint64_t oldest = ~std::uint64_t(0);
      for (std::size_t i = 0; i < older_.size(); ++i)
        if (i != keep && older_[i].seq < oldest) oldest = older_[i].seq, slot = i;
    }
    OlderCheckpoint& o = older_[slot];
    if (!o.buf) o.buf = std::make_unique<PinnedHostBuffer>(checkpoint_.bytes);
    CUDA_CHECK(cudaEventRecord(ev_ready_, rt_.stream()));
    CUDA_CHECK(cudaStreamWaitEvent(copy_stream_, ev_ready_, 0));
    copy_checkpoint_to_host(o, copy_stream_);
    CUDA_CHECK(cudaEventRecord(ev_moved_, copy_stream_));
    move_pending_ = true;
    o.seq = ++seq_;
  }

  void copy_checkpoint_to_host(OlderCheckpoint& o, cudaStream_t stream) {
    char* dst = static_cast<char*>(o.buf->data());
    const char* src = static_cast<const char*>(checkpoint_.p);
    auto copy = [&](std::size_t off, std::size_t bytes) {
      if (bytes) CUDA_CHECK(cudaMemcpyAsync(dst + off, src + off, bytes, cudaMemcpyDeviceToHost, stream));
    };
    copy(0, rt_.checkpoint_bytes());
    if (drafter_)
      for (const auto& [off, bytes] : drafter_->checkpoint_spans(drafter_->checkpoint_length())) copy(checkpoint_drafter_ + off, bytes);
    copy(checkpoint_logits_, logits_.bytes);
    o.ids = checkpoint_ids_;
    o.images = checkpoint_images_;
    o.drafter_length = drafter_ ? drafter_->checkpoint_length() : 0;
    o.drafter_extra = drafter_ ? drafter_->checkpoint_extra() : std::vector<std::int32_t>{};
  }

  void park_live(std::size_t reserved) {
    // Small prompts are cheaper to re-read than to allocate and transfer a host checkpoint.
    if (!options_.park_host_mib || checkpoint_ids_.size() < 2048) return;
    const std::size_t budget = std::size_t(options_.park_host_mib) << 20;
    const int rows = int(checkpoint_ids_.size());
    const std::size_t bytes = checkpoint_.bytes + rt_.full_rows_bytes(rows);
    if (bytes + reserved > budget) return;
    std::size_t used = reserved;
    for (const auto& p : parked_) used += p.buf->size();
    // A larger spare must not evict another chat just to avoid an allocation.
    if (spare_park_buffer_ && (spare_park_buffer_->size() < bytes ||
        (spare_park_buffer_->size() > bytes && spare_park_buffer_->size() > budget - used)))
      spare_park_buffer_.reset();
    const std::size_t allocation = spare_park_buffer_ ? spare_park_buffer_->size() : bytes;
    while (used + allocation > budget) {
      used -= parked_.front().buf->size();
      parked_.pop_front();
    }
    OlderCheckpoint entry;
    try {
      entry.buf = std::move(spare_park_buffer_);
      if (!entry.buf) entry.buf = std::make_unique<PinnedHostBuffer>(bytes);
    } catch (const PinnedHostOutOfMemory& error) {
      spdlog::warn("conversation parking skipped: {}", error.what());
      return;
    }
    copy_checkpoint_to_host(entry, rt_.stream());
    rt_.save_full_rows(static_cast<char*>(entry.buf->data()) + checkpoint_.bytes, rows);
    parked_.push_back(std::move(entry));  // the target stream orders copies before the next prefill
  }

  // Whether `held` (the images of a state holding `prefix` tokens) are exactly the images among the first
  // `prefix` tokens of a prompt with images `req`.
  static bool same_images(const std::vector<ImageKey>& held, std::span<const PromptImage> req, std::size_t prefix) {
    std::size_t i = 0;
    for (const PromptImage& im : req) {
      if (std::size_t(im.pos) >= prefix) break;
      if (i >= held.size() || !(held[i] == ImageKey{im.pos, im.tokens, im.hash})) return false;
      ++i;
    }
    return i == held.size();
  }

  // The image rows of prompt chunk [s, s + T): their embeddings uploaded (pageable copies, staged before the
  // calls return) into the override buffers, which the previous chunk's prefill has finished reading in stream
  // order.
  RowOverride image_rows(std::span<const PromptImage> images, int s, int T) {
    RowOverride ov;
    std::vector<int> rows;
    for (const PromptImage& im : images) {
      const int a = std::max(im.pos, s), b = std::min(im.pos + im.tokens, s + T);
      if (a >= b) continue;
      if (!image_vals_.p) {
        image_vals_ = DeviceBuffer(std::size_t(options_.prefill_chunk) * HIDDEN * 2);
        image_rows_ = DeviceBuffer(std::size_t(options_.prefill_chunk) * sizeof(int));
      }
      CUDA_CHECK(cudaMemcpyAsync(static_cast<std::uint16_t*>(image_vals_.p) + rows.size() * HIDDEN,
                                 im.embeds->data() + std::size_t(a - im.pos) * HIDDEN,
                                 std::size_t(b - a) * HIDDEN * 2, cudaMemcpyHostToDevice, rt_.stream()));
      for (int r = a; r < b; ++r) rows.push_back(r - s);
    }
    if (rows.empty()) return ov;
    CUDA_CHECK(cudaMemcpyAsync(image_rows_.p, rows.data(), rows.size() * sizeof(int), cudaMemcpyHostToDevice,
                               rt_.stream()));
    ov.rows = static_cast<const int*>(image_rows_.p);
    ov.values = static_cast<const __nv_bfloat16*>(image_vals_.p);
    ov.n = int(rows.size());
    return ov;
  }

  static std::size_t carved(std::size_t bytes) { return (bytes + 255) / 256 * 256; }

  // engine._sample with a host synchronisation (the speculative loop's first anchor): argmax at
  // temperature <= 0, else one draw from engine._probs.
  std::int32_t sample(const float* logits, const Sampling& sampling, head::SamplerRng& rng) {
    head::SamplingParams p;
    p.temperature = sampling.temperature;
    p.top_k = sampling.top_k;
    p.top_p = sampling.top_p;
    head::sample(logits, 1, UNPADDED_VOCAB, UNPADDED_VOCAB, p, rng, static_cast<std::int32_t*>(token_.p),
                 rt_.stream());
    CUDA_CHECK(cudaMemcpyAsync(host_token_.data(), token_.p, sizeof(std::int32_t), cudaMemcpyDeviceToHost,
                               rt_.stream()));
    CUDA_CHECK(cudaStreamSynchronize(rt_.stream()));
    return *static_cast<const std::int32_t*>(host_token_.data());
  }

  RuntimeOptions options_;
  std::unique_ptr<Runtime> runtime_;
  Runtime& rt_;
  Prefiller prefiller_;
  std::unique_ptr<Drafter> drafter_;
  std::unique_ptr<SpecLoop> spec_;
  DeviceBuffer chunk_tokens_, logits_, token_, taps_;
  PinnedHostBuffer host_token_;
  // prompt checkpoint (save_checkpoint) and the prompt it holds (empty: none)
  DeviceBuffer checkpoint_;
  std::size_t checkpoint_drafter_ = 0, checkpoint_logits_ = 0;  // byte offsets in checkpoint_
  std::vector<std::int32_t> checkpoint_ids_;
  std::vector<std::int32_t> tokens_in_state_;
  std::vector<ImageKey> state_images_, checkpoint_images_;  // images in the state / the checkpoint
  std::unique_ptr<PinnedHostBuffer> spare_park_buffer_;  // charged to the same budget
  std::list<OlderCheckpoint> parked_;  // FIFO among displaced prompts; restored entries leave the cache
  std::vector<OlderCheckpoint> older_;  // earlier prompts' checkpoints in pinned host memory
  std::uint64_t seq_ = 0;
  cudaStream_t copy_stream_ = nullptr;
  cudaEvent_t ev_ready_ = nullptr, ev_moved_ = nullptr;
  bool move_pending_ = false;
  DeviceBuffer image_vals_, image_rows_;  // a chunk's image embeddings (bf16 [chunk, H]) and their rows
  std::mutex mutex_;
};

}  // namespace

std::unique_ptr<Generator> make_generator(const RuntimeOptions& options) {
  return std::make_unique<EngineGenerator>(options);
}

std::unique_ptr<Generator> make_generator(const RuntimeOptions& options, Weights weights, Ablation ablation) {
  return std::make_unique<EngineGenerator>(options, std::move(weights), std::move(ablation));
}

}  // namespace ingine::model
