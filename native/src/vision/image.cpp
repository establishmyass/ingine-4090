#include "vision/image.h"

#include "ingine/vision.h"

#define STB_IMAGE_IMPLEMENTATION
#define STBI_NO_STDIO          // memory only: requests never name server files
#define STBI_FAILURE_USERMSG
#define STBI_MAX_DIMENSIONS (1 << 16)
#include <stb/stb_image.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <string>

namespace ingine::vision {
namespace {

constexpr std::int64_t kMaxPixels = std::int64_t(64) << 20;

std::uint16_t bf16_bits(float f) {
  std::uint32_t u;
  std::memcpy(&u, &f, 4);
  u += 0x7fffu + ((u >> 16) & 1u);  // round to nearest even (finite inputs)
  return std::uint16_t(u >> 16);
}

// Separable resampling weights for one axis: out[i] = sum_j w[i][j] * in[first[i] + j].
struct Axis {
  std::vector<int> first, count;
  std::vector<float> w;  // count[i] values from offset[i]
  std::vector<int> offset;
};

Axis axis_weights(int in, int out) {
  Axis a;
  const double scale = double(in) / double(out);
  const double support = std::max(1.0, scale);  // triangle radius in input pixels
  for (int i = 0; i < out; ++i) {
    const double center = (i + 0.5) * scale;
    const int lo = std::max(0, int(std::floor(center - support)));
    const int hi = std::min(in, int(std::ceil(center + support)));
    double sum = 0;
    std::vector<double> ws;
    for (int j = lo; j < hi; ++j) {
      const double d = std::abs((j + 0.5 - center) / support);
      const double w = d < 1.0 ? 1.0 - d : 0.0;
      ws.push_back(w);
      sum += w;
    }
    a.first.push_back(lo);
    a.count.push_back(int(ws.size()));
    a.offset.push_back(int(a.w.size()));
    for (double w : ws) a.w.push_back(float(sum > 0 ? w / sum : 0.0));
  }
  return a;
}

}  // namespace

RgbImage decode_image(std::span<const std::uint8_t> file) {
  if (file.empty()) throw ImageError("image is empty");
  if (file.size() > 0x7fffffffu) throw ImageError("image file is too large");
  int w = 0, h = 0, n = 0;
  if (!stbi_info_from_memory(file.data(), int(file.size()), &w, &h, &n))
    throw ImageError(std::string("cannot decode image (supported: JPEG, PNG, BMP, GIF, TGA, PSD, HDR, PNM): ") +
                     stbi_failure_reason());
  if (w <= 0 || h <= 0 || std::int64_t(w) * h > kMaxPixels)
    throw ImageError("image of " + std::to_string(w) + " x " + std::to_string(h) + " pixels is not supported");
  std::unique_ptr<stbi_uc, void (*)(void*)> px(
      stbi_load_from_memory(file.data(), int(file.size()), &w, &h, &n, 4), stbi_image_free);
  if (!px) throw ImageError(std::string("cannot decode image: ") + stbi_failure_reason());
  // RGBA -> RGB over the checkerboard (convert_to_rgb); opaque pixels are unchanged
  RgbImage img;
  img.width = w, img.height = h;
  img.rgb.resize(std::size_t(w) * h * 3);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      const stbi_uc* s = px.get() + (std::size_t(y) * w + x) * 4;
      std::uint8_t* d = img.rgb.data() + (std::size_t(y) * w + x) * 3;
      const int a = s[3];
      const int bg = ((y / 8 + x / 8) % 2 == 0) ? 204 : 153;
      for (int c = 0; c < 3; ++c) d[c] = std::uint8_t((s[c] * a + bg * (255 - a) + 127) / 255);
    }
  return img;
}

std::pair<int, int> bounded_size(int w, int h, int max_tokens) {
  if (max_tokens <= 0 || std::int64_t(token_grid(w)) * token_grid(h) <= max_tokens) return {w, h};
  double s = std::min(1.0, std::sqrt(double(max_tokens) * 1600.0 / (double(w) * double(h))));  // never upscale
  for (;;) {
    const int nw = std::max(1, int(std::lround(w * s))), nh = std::max(1, int(std::lround(h * s)));
    if (std::int64_t(token_grid(nw)) * token_grid(nh) <= max_tokens) return {nw, nh};
    s *= 0.995;
  }
}

RgbImage resize(const RgbImage& in, int w, int h) {
  if (w == in.width && h == in.height) return in;
  const Axis ax = axis_weights(in.width, w), ay = axis_weights(in.height, h);
  // horizontal pass into FP32 rows, then vertical
  std::vector<float> tmp(std::size_t(in.height) * w * 3);
  for (int y = 0; y < in.height; ++y)
    for (int x = 0; x < w; ++x) {
      float acc[3] = {0, 0, 0};
      for (int j = 0; j < ax.count[x]; ++j) {
        const float wt = ax.w[ax.offset[x] + j];
        const std::uint8_t* s = in.rgb.data() + (std::size_t(y) * in.width + ax.first[x] + j) * 3;
        for (int c = 0; c < 3; ++c) acc[c] += wt * float(s[c]);
      }
      for (int c = 0; c < 3; ++c) tmp[(std::size_t(y) * w + x) * 3 + c] = acc[c];
    }
  RgbImage out;
  out.width = w, out.height = h;
  out.rgb.resize(std::size_t(w) * h * 3);
  for (int y = 0; y < h; ++y)
    for (int x = 0; x < w; ++x) {
      float acc[3] = {0, 0, 0};
      for (int j = 0; j < ay.count[y]; ++j) {
        const float wt = ay.w[ay.offset[y] + j];
        const float* s = tmp.data() + (std::size_t(ay.first[y] + j) * w + x) * 3;
        for (int c = 0; c < 3; ++c) acc[c] += wt * s[c];
      }
      for (int c = 0; c < 3; ++c)
        out.rgb[(std::size_t(y) * w + x) * 3 + c] = std::uint8_t(std::clamp(std::lround(acc[c]), 0L, 255L));
    }
  return out;
}

Patches patchify(const RgbImage& img) {
  Patches p;
  p.rows = token_grid(img.height), p.cols = token_grid(img.width);
  const std::size_t tokens = std::size_t(p.rows) * p.cols;
  p.data.resize(tokens * 4 * kPatchDim);
  const std::uint16_t pad = bf16_bits((-1.0f - 127.5f) / 127.5f);
  for (int r = 0; r < p.rows; ++r)
    for (int c = 0; c < p.cols; ++c)
      for (int mr = 0; mr < kMerge; ++mr)
        for (int mc = 0; mc < kMerge; ++mc) {
          const std::size_t patch = (std::size_t(r) * p.cols + c) * 4 + std::size_t(mr) * 2 + mc;
          std::uint16_t* dst = p.data.data() + patch * kPatchDim;
          for (int pr = 0; pr < kPatch; ++pr)
            for (int pc = 0; pc < kPatch; ++pc) {
              const int y = r * 40 + mr * kPatch + pr, x = c * 40 + mc * kPatch + pc;
              std::uint16_t* d = dst + (pr * kPatch + pc) * 3;
              if (y >= img.height || x >= img.width) {
                d[0] = d[1] = d[2] = pad;
                continue;
              }
              const std::uint8_t* s = img.rgb.data() + (std::size_t(y) * img.width + x) * 3;
              for (int ch = 0; ch < 3; ++ch) d[ch] = bf16_bits((float(s[ch]) - 127.5f) / 127.5f);
            }
        }
  return p;
}

}  // namespace ingine::vision
