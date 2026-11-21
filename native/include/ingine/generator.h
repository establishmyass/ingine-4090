// The token generator interface the server drives (mirror of engine.Engine.generate). CUDA-free so the
// scheduler, the HTTP layer and their tests build without the model; ingine_model implements make_generator.
#pragma once

#include <cstdint>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace ingine::model {

struct Sampling {
  float temperature = 1.0f;  // 0 => greedy
  float top_p = 1.0f;
  int top_k = 0;  // 0 => disabled
  std::optional<uint64_t> seed;
  // Teacher-forced replay (test/benchmark only; empty = off): generate() emits exactly these tokens
  // (max_new_tokens is capped at their count), running its real loop along them: plain decoding samples as
  // usual and then takes the forced token; speculative decoding accepts a draft iff it equals the forced token
  // and continues with the forced token after the accepted run. Fills GenerateStats::spec_trace.
  std::span<const int32_t> forced;
};

// How the prompt reused the state of earlier requests (reused_tokens were not prefilled again).
enum class Reuse {
  None,        // empty state, the whole prompt prefilled
  Extend,      // the prompt extends the tokens in the state (previous prompt + processed generated tokens)
  Checkpoint,  // restored the checkpoint taken after the previous prompt's prefill (a prefix of this prompt)
};
inline const char* to_string(Reuse reuse) {
  switch (reuse) {
    case Reuse::Extend: return "extend";
    case Reuse::Checkpoint: return "checkpoint";
    default: return "none";
  }
}

// One speculative round of a forced replay: verify width, accepted drafts, the drafts that would have been
// accepted had all 7 been verified, and GPU time of the round's verify (+ acceptance) and of its draft.
struct SpecRoundRecord {
  int width = 0, accepted = 0, accepted_all = 0;
  float verify_ms = 0, draft_ms = 0;
};

struct GenerateStats {
  int prompt_tokens = 0, reused_tokens = 0, generated_tokens = 0;
  Reuse reuse = Reuse::None;
  double prefill_seconds = 0, decode_seconds = 0;
  int spec_rounds = 0, spec_accepted = 0;
  std::vector<SpecRoundRecord> spec_trace;  // forced replay only
};

// An image in the prompt: `tokens` image-token positions from `pos` whose input embeddings are `embeds`
// (BF16 bits [tokens, 2304], the vision tower's output) instead of the placeholder token's embedding. `hash`
// identifies the embeddings: prompt reuse (Extend / Checkpoint) requires equal tokens and equal images.
struct PromptImage {
  int pos = 0, tokens = 0;
  std::uint64_t hash = 0;
  std::shared_ptr<const std::vector<std::uint16_t>> embeds;
};

struct RuntimeOptions {
  std::filesystem::path model_dir, ablation;  // empty ablation: original model
  int max_ctx = 65536;
  int prefill_chunk = 2048;
  bool dense_fp8 = true, kv_fp8 = false, spec = false;
  float spec_threshold = 0.8f;
  // Expert offloading (src/offload/expert_store.h): GiB of routed experts kept only in host memory (0: off;
  // $INGINE_OFFLOAD_GIB overrides), CPU pool threads computing them during decode (0: the GPU reads them over
  // PCIe instead; $INGINE_OFFLOAD_THREADS), adaptive placement ($INGINE_OFFLOAD_ADAPTIVE=0 freezes it).
  double offload_gib = 0.0;
  int offload_threads = 8;
  bool offload_adaptive = true;
};

class Generator {
 public:
  virtual ~Generator() = default;
  virtual std::string model_id() const = 0;  // "inkling-2-mini-abliterated" with an ablation, else "inkling-2-mini"
  virtual int max_model_len() const = 0;     // = max_ctx
  // One request at a time. Emits sampled tokens in order through on_tokens (a speculative round emits several);
  // stops after a token in stop_ids, after max_new_tokens, or when on_tokens returns false (cancel).
  // Throws std::invalid_argument when the prompt does not fit max_ctx (reserve as engine.py does) or an image
  // span lies outside it. `images`: ordered, non-overlapping spans of the prompt (none: text only).
  virtual GenerateStats generate(std::span<const int32_t> prompt, int max_new_tokens, const Sampling& sampling,
                                 std::span<const int32_t> stop_ids,
                                 const std::function<bool(std::span<const int32_t>)>& on_tokens,
                                 std::span<const PromptImage> images = {}) = 0;
};

std::unique_ptr<Generator> make_generator(const RuntimeOptions& options);  // the model (ingine_model)

}  // namespace ingine::model
