// Image decoding and the Inkling 2 image preprocessing (transformers Inkling2ImageProcessor):
//   decode -> RGB (RGBA composited onto an 8 px checkerboard of 204 / 153, as convert_to_rgb)
//   -> [downscale when above the token bound; the reference never resizes]
//   -> pad right / bottom to multiples of 40 px with -1 (on the 0-255 scale)
//   -> (x - 127.5) / 127.5 in FP32 (rescale 1/255 and normalize mean 0.5 / std 0.5, fused as transformers does)
//   -> one 40 x 40 tile per image token in raster order, each cut into 2 x 2 patches of 20 x 20 px
//      (Inkling2VisionEmbeddings: reshape(-1, 2, 20, 2, 20, 3).transpose(2, 3)), a patch flattened (row, col, C)
//   -> BF16 (the vision tower's dtype).
#pragma once

#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace ingine::vision {

inline constexpr int kPatch = 20, kMerge = 2, kPatchDim = kPatch * kPatch * 3;  // 1200

struct RgbImage {
  int width = 0, height = 0;
  std::vector<std::uint8_t> rgb;  // [height][width][3]
};

// Throws ImageError (undecodable, unsupported format, empty, above 64 Mpx).
RgbImage decode_image(std::span<const std::uint8_t> file);

inline int token_grid(int pixels) { return (pixels + 39) / 40; }
// The size an image of w x h is encoded at: unchanged when it fits max_tokens image tokens, else the largest
// aspect-preserving downscale that fits.
std::pair<int, int> bounded_size(int w, int h, int max_tokens);
// Antialiased bilinear resampling (triangle filter widened by the scale factor when shrinking, as PIL).
RgbImage resize(const RgbImage& in, int w, int h);

struct Patches {
  int rows = 0, cols = 0;           // image tokens
  std::vector<std::uint16_t> data;  // BF16 bits [rows * cols * 4, 1200]
};
Patches patchify(const RgbImage& img);

}  // namespace ingine::vision
