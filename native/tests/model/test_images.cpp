// Image spans in the prompt (model::PromptImage) on the full model: a prompt whose "image" embeddings are the
// GPU embedding kernel's own normed embeddings of text tokens generates exactly what the text prompt generates
// (greedy, the span crossing prefill chunk boundaries), prefix reuse keeps working with images, and an image
// with another content hash at the same positions is never reused. Needs the model (~20 GB of GPU memory).
#include "common/test_util.h"

#include "core/safetensors.h"
#include "ingine/generator.h"
#include "kernels/dense/dense.h"

#include <cuda_bf16.h>

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

using namespace ingine::model;

namespace {

constexpr int H = 2304;
constexpr std::int32_t kImageToken = 200054;

// bf16 bits of model.Model.embed's output rows for `ids`, computed by the engine's embed kernel over a table of
// just these rows (rows are independent; >= 8 rows: the block width prefill chunks of >= 8 tokens use)
std::vector<std::uint16_t> normed_embeddings(const std::filesystem::path& dir, const std::vector<std::int32_t>& ids) {
  ingine::SafetensorsModel m(dir);
  const ingine::SafetensorsRef table = m.at("model.llm.embed.weight");
  const int n = int(ids.size());
  std::vector<std::uint16_t> rows(std::size_t(n) * H);
  for (int i = 0; i < n; ++i)
    table.file->read_range(table->offset + std::uint64_t(ids[std::size_t(i)]) * H * 2, H * 2, rows.data() + std::size_t(i) * H);
  const std::vector<std::byte> nw = m.read("model.llm.embed_norm.weight");
  std::vector<float> norm(H);
  for (int c = 0; c < H; ++c) {
    std::uint16_t b;
    std::memcpy(&b, nw.data() + 2 * c, 2);
    const std::uint32_t u = std::uint32_t(b) << 16;
    std::memcpy(&norm[std::size_t(c)], &u, 4);
  }
  std::vector<std::int64_t> idx(static_cast<std::size_t>(n));
  for (int i = 0; i < n; ++i) idx[std::size_t(i)] = i;
  void *d_table, *d_norm, *d_idx, *d_out;
  INGINE_TEST_CUDA(cudaMalloc(&d_table, rows.size() * 2));
  INGINE_TEST_CUDA(cudaMalloc(&d_norm, H * 4));
  INGINE_TEST_CUDA(cudaMalloc(&d_idx, idx.size() * 8));
  INGINE_TEST_CUDA(cudaMalloc(&d_out, std::size_t(n) * H * 4));
  INGINE_TEST_CUDA(cudaMemcpy(d_table, rows.data(), rows.size() * 2, cudaMemcpyHostToDevice));
  INGINE_TEST_CUDA(cudaMemcpy(d_norm, norm.data(), H * 4, cudaMemcpyHostToDevice));
  INGINE_TEST_CUDA(cudaMemcpy(d_idx, idx.data(), idx.size() * 8, cudaMemcpyHostToDevice));
  ingine::kernels::dense::embed(static_cast<const std::int64_t*>(d_idx), n, static_cast<const __nv_bfloat16*>(d_table),
                                static_cast<const float*>(d_norm), static_cast<float*>(d_out), false, nullptr);
  std::vector<float> out(std::size_t(n) * H);
  INGINE_TEST_CUDA(cudaMemcpy(out.data(), d_out, out.size() * 4, cudaMemcpyDeviceToHost));
  for (void* p : {d_table, d_norm, d_idx, d_out}) cudaFree(p);
  std::vector<std::uint16_t> bits(out.size());
  for (std::size_t i = 0; i < out.size(); ++i) {
    std::uint32_t u;
    std::memcpy(&u, &out[i], 4);
    INGINE_CHECK((u & 0xffffu) == 0);  // the kernel's output is BF16-rounded
    bits[i] = std::uint16_t(u >> 16);
  }
  return bits;
}

}  // namespace

int main() {
  if (ingine::test::skip_without_gpu()) return ingine::test::kSkip;
  const std::filesystem::path dir = INGINE_MODEL_DIR;
  if (!std::filesystem::exists(dir / "mtp.safetensors")) {
    std::printf("skip: no model at %s\n", dir.c_str());
    return ingine::test::kSkip;
  }
  ingine::test::Rng rng(11);
  auto random_ids = [&](int n) {
    std::vector<std::int32_t> v(static_cast<std::size_t>(n));
    for (auto& t : v) t = std::int32_t(1000 + rng.next() % 50000);
    return v;
  };
  // A | X (the "image": 150 tokens over the chunk boundaries at 128 and 192) | B; chunks of 64
  const std::vector<std::int32_t> A = random_ids(100), X = random_ids(150), B = random_ids(60), C = random_ids(30);
  const auto embeds = std::make_shared<const std::vector<std::uint16_t>>(normed_embeddings(dir, X));

  RuntimeOptions opt;
  opt.model_dir = dir;
  opt.max_ctx = 4096;
  opt.dense_fp8 = opt.kv_fp8 = true;
  opt.spec = false;  // plain decoding: bit-exact
  opt.prefill_chunk = 64;
  opt.park_host_mib = 640;
  auto gen = make_generator(opt);

  struct Out {
    std::vector<std::int32_t> tokens;
    GenerateStats stats;
  };
  Sampling greedy;
  greedy.temperature = 0.f;
  auto run = [&](const std::vector<std::int32_t>& ids, int max_new, std::vector<PromptImage> images = {}) {
    Out o;
    o.stats = gen->generate(ids, max_new, greedy, {}, [&](std::span<const std::int32_t> t) {
      o.tokens.insert(o.tokens.end(), t.begin(), t.end());
      return true;
    }, images);
    return o;
  };
  auto cat = [](std::initializer_list<const std::vector<std::int32_t>*> parts) {
    std::vector<std::int32_t> v;
    for (auto* p : parts) v.insert(v.end(), p->begin(), p->end());
    return v;
  };
  const std::vector<std::int32_t> holes(X.size(), kImageToken), other = random_ids(40);
  const std::vector<std::int32_t> text = cat({&A, &X, &B}), with_image = cat({&A, &holes, &B});
  const PromptImage image{int(A.size()), int(X.size()), 0x1234, embeds};
  PromptImage changed = image;
  changed.hash = 0x5678;

  bool ok = true;
  auto report = [&](bool good, const char* what) {
    std::printf("%s %s\n", good ? "PASS" : "FAIL", what);
    ok &= good;
  };
  const Out ref = run(text, 48);
  run(other, 2);
  const Out img = run(with_image, 48, {image});
  report(img.tokens == ref.tokens && img.stats.reuse == Reuse::None,
         "an image of the text's own embeddings generates the text prompt's tokens (span across 3 chunks)");

  // follow-ups: the image prompt + 2 generated tokens + C. The state holds the image prompt + its generation, so
  // the follow-up restores the checkpoint taken after the image prompt; the text equivalent extends a state
  // holding just the text prompt (the same prefill chunk boundaries).
  std::vector<std::int32_t> g2(ref.tokens.begin(), ref.tokens.begin() + 2);
  const std::vector<std::int32_t> text_follow = cat({&text, &g2, &C}), image_follow = cat({&with_image, &g2, &C});
  const Out image_x = run(image_follow, 32, {image});
  run(other, 2);
  run(text, 0);
  const Out text_x = run(text_follow, 32);
  report(image_x.stats.reuse == Reuse::Checkpoint && image_x.stats.reused_tokens == int(with_image.size()) &&
             text_x.stats.reuse == Reuse::Extend && image_x.tokens == text_x.tokens,
         "a follow-up restores the image prompt's checkpoint and generates the text follow-up's tokens");
  run(with_image, 0, {image});
  const Out image_e = run(image_follow, 32, {image});
  report(image_e.stats.reuse == Reuse::Extend && image_e.tokens == text_x.tokens,
         "a follow-up extending the image prompt's state generates the text follow-up's tokens");
  // the same positions and tokens with another image: no reuse of either path
  const Out swap = run(image_follow, 8, {changed});
  report(swap.stats.reuse == Reuse::None, "another image at the same positions is not reused");
  const auto pad = random_ids(2050);
  const auto long_image = cat({&with_image, &pad});
  run(long_image, 2, {image});
  run(other, 2);
  const Out parked = run(long_image, 2, {image});
  run(other, 2);
  const Out replaced = run(long_image, 2, {changed});
  report(parked.stats.reuse == Reuse::Parked && replaced.stats.reuse == Reuse::None,
         "parked image prompts match their image hashes");
  // a span outside the prompt is rejected
  bool threw = false;
  try {
    PromptImage bad = image;
    bad.pos = int(with_image.size()) - 10;
    run(with_image, 1, {bad});
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  report(threw, "an image span outside the prompt is rejected");
  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
