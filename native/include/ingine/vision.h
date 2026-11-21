// Image input (CUDA-free interface): an ImageEncoder turns an image file into the language model's input
// embeddings for its image tokens. ingine_vision implements it on the CPU (the Inkling 2 vision tower in host
// memory, AVX-512 BF16 kernels), so images cost no VRAM. The server encodes on the request thread and hands
// the embeddings to the generator as model::PromptImage spans.
#pragma once

#include <cstdint>
#include <filesystem>
#include <memory>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

namespace ingine::vision {

inline constexpr std::int32_t kImageToken = 200054;  // <|unused_200054|>: one per image token
inline constexpr int kTokenPixels = 40;              // one image token covers 40 x 40 pixels (2 x 2 patches of 20)

struct EncodedImage {
  int rows = 0, cols = 0;           // grid in image tokens (rows * cols tokens, raster order)
  int width = 0, height = 0;        // pixels encoded (after any downscale)
  std::uint64_t hash = 0;           // identity of the embeddings: file bytes + encoder settings
  // BF16 bits [rows * cols, 2304]: replace the normed token embeddings of the image tokens
  std::shared_ptr<const std::vector<std::uint16_t>> embeds;
  double seconds = 0;               // decode + encode time (0 when served from the cache)
  bool cached = false;

  int tokens() const { return rows * cols; }
};

// Bad image input (HTTP 400): undecodable data, unsupported format, zero size.
struct ImageError : std::runtime_error {
  using std::runtime_error::runtime_error;
};

class ImageEncoder {
 public:
  virtual ~ImageEncoder() = default;
  // Decodes an image file (JPEG, PNG, BMP, GIF, TGA, PSD, HDR, PNM) and encodes it. Thread-safe; encodings are
  // serialized and recent results cached by content. Throws ImageError for bad input.
  virtual EncodedImage encode(std::span<const std::uint8_t> file) = 0;
  // Image tokens an image of width x height pixels takes (after the max_tokens bound).
  virtual int tokens_for(int width, int height) const = 0;
};

struct VisionOptions {
  std::filesystem::path model_dir;  // checkpoint with model.visual.* tensors
  int threads = 0;                  // CPU threads (0: one per physical core, $INGINE_VISION_THREADS)
  // Upper bound on image tokens per image (40 x 40 px each); larger images are downscaled, keeping the aspect
  // ratio (the reference preprocessor never resizes; llama.cpp bounds it at 4096).
  int max_tokens = 1024;
  std::size_t cache_bytes = std::size_t(256) << 20;  // encodings kept for repeated images (multi-turn chats)
  bool load_in_background = true;  // read the weights on a thread; the first encode waits for them
};

// Whether the CPU runs the vision kernels (AVX-512 F/BW/VL/DQ + AVX512_BF16).
bool vision_supported();
// The checkpoint has a vision tower.
bool has_vision_tower(const std::filesystem::path& model_dir);
std::unique_ptr<ImageEncoder> make_image_encoder(const VisionOptions& options);

}  // namespace ingine::vision
