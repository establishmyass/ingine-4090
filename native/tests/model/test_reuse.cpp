// Prompt checkpoints on the full model with speculative decoding (DSpark drafter + verify graphs): a follow-up
// that restores the checkpoint of its prompt prefix generates exactly what the strict-prefix path generates
// (same prefill chunk boundaries), greedy and sampled, with the generation between them longer than the
// 1024-slot rings and crossing drafter context buckets. Needs the model (~22 GB of GPU memory).
#include "common/test_util.h"
#include "model/runtime.h"

#include "ingine/generator.h"

#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace ingine::model;

int main() {
  if (ingine::test::skip_without_gpu()) return ingine::test::kSkip;
  const std::filesystem::path dir = INGINE_MODEL_DIR;
  if (!std::filesystem::exists(dir / "mtp.safetensors")) {
    std::printf("skip: no model at %s\n", dir.c_str());
    return ingine::test::kSkip;
  }
  RuntimeOptions opt;
  opt.model_dir = dir;
  opt.max_ctx = 8192;  // drafter buckets 1024, 2048, 4096, 8192
  opt.dense_fp8 = opt.kv_fp8 = opt.spec = true;
  opt.prefill_chunk = 512;
  auto gen = make_generator(opt);

  ingine::test::Rng rng(3);
  // a prompt of ordinary token ids (no special tokens), 900 long: generation crosses the 1024 bucket
  auto random_ids = [&](int n) {
    std::vector<std::int32_t> v(static_cast<std::size_t>(n));
    for (auto& t : v) t = std::int32_t(1000 + rng.next() % 50000);
    return v;
  };
  const std::vector<std::int32_t> prompt = random_ids(900), other = random_ids(300);
  struct Out {
    std::vector<std::int32_t> tokens;
    GenerateStats stats;
  };
  auto run = [&](const std::vector<std::int32_t>& ids, int max_new, const Sampling& sp) {
    Out o;
    o.stats = gen->generate(ids, max_new, sp, {}, [&](std::span<const std::int32_t> t) {
      o.tokens.insert(o.tokens.end(), t.begin(), t.end());
      return true;
    });
    return o;
  };
  bool ok = true;
  Sampling greedy;
  greedy.temperature = 0.f;
  Sampling sampled;
  sampled.temperature = 1.f;
  sampled.seed = 5;
  for (const auto& [name, sp] : {std::pair{"greedy", greedy}, std::pair{"sampled", sampled}}) {
    Sampling follow_sp = sp;
    follow_sp.seed = 9;
    for (const int g_len : {200, 1300}) {
      run(other, 4, sp);
      const Out g = run(prompt, g_len, sp);
      std::vector<std::int32_t> follow = prompt;
      follow.insert(follow.end(), g.tokens.begin(), g.tokens.begin() + 2);
      const std::vector<std::int32_t> y = random_ids(40);
      follow.insert(follow.end(), y.begin(), y.end());
      const Out x = run(follow, 64, follow_sp);
      run(other, 4, sp);
      run(prompt, 0, sp);
      const Out z = run(follow, 64, follow_sp);
      const bool good = int(g.tokens.size()) == g_len && x.stats.reuse == Reuse::Checkpoint &&
                        z.stats.reuse == Reuse::Extend && x.stats.reused_tokens == int(prompt.size()) &&
                        z.stats.reused_tokens == int(prompt.size()) && x.tokens == z.tokens;
      std::printf("%s checkpoint restore equals the strict-prefix path with spec (%s, G %d tokens): %zu tokens\n",
                  good ? "PASS" : "FAIL", name, g_len, x.tokens.size());
      ok &= good;
    }
  }
  // Older checkpoints: B = A + the start of A's answer + a tool result X; then B2 rewrites the end of X (a
  // client that sends a tool result differently the second time). B2 is no extension of the state nor of B's
  // checkpoint, so it restores A's (moved to host memory when B's was saved) and must generate exactly what a
  // state holding just A generates (the same prefill chunk boundaries). A later prompt extending B must not
  // reuse B's checkpoint: B2's prefill rewrote the positions past A.
  for (const auto& [name, sp] : {std::pair{"greedy", greedy}, std::pair{"sampled", sampled}}) {
    const std::vector<std::int32_t> A = random_ids(700), X = random_ids(120), Xb = random_ids(40), Y = random_ids(30);
    run(other, 4, sp);
    const Out ga = run(A, 48, sp);
    std::vector<std::int32_t> B = A;
    B.insert(B.end(), ga.tokens.begin(), ga.tokens.begin() + 2);
    B.insert(B.end(), X.begin(), X.end());
    const Out gb = run(B, 48, sp);
    std::vector<std::int32_t> B2(B.begin(), B.end() - 40);  // X's last 40 tokens replaced
    B2.insert(B2.end(), Xb.begin(), Xb.end());
    B2.insert(B2.end(), gb.tokens.begin(), gb.tokens.begin() + 2);
    B2.insert(B2.end(), Y.begin(), Y.end());
    const Out x = run(B2, 64, sp);
    std::vector<std::int32_t> C = B;  // extends B (whose checkpoint is stale now)
    C.insert(C.end(), Y.begin(), Y.end());
    const Out c = run(C, 32, sp);
    run(other, 4, sp);
    run(A, 0, sp);
    const Out z = run(B2, 64, sp);
    run(other, 4, sp);
    run(A, 0, sp);
    const Out zc = run(C, 32, sp);
    const bool good = gb.stats.reuse == Reuse::Checkpoint && x.stats.reuse == Reuse::OlderCheckpoint &&
                      x.stats.reused_tokens == int(A.size()) && z.stats.reuse == Reuse::Extend &&
                      z.stats.reused_tokens == int(A.size()) && x.tokens == z.tokens &&
                      c.stats.reuse == Reuse::OlderCheckpoint && c.stats.reused_tokens == int(A.size()) &&
                      c.tokens == zc.tokens;
    std::printf("%s an older checkpoint restores the prompt before a rewritten one and generates the strict-prefix "
                "path's tokens (%s): reuse %s/%s, reused %d/%d of %zu\n",
                good ? "PASS" : "FAIL", name, to_string(x.stats.reuse), to_string(c.stats.reuse), x.stats.reused_tokens,
                c.stats.reused_tokens, A.size());
    ok &= good;
  }
  // An interleaved request overwrites the full KV prefix. Parking must reproduce the same continuation
  // as a live prefix, including after both conversations have taken another turn.
  const auto a = random_ids(2300), b = random_ids(2600);
  auto follow_a = a, follow_b = b;
  follow_a.insert(follow_a.end(), other.begin(), other.begin() + 23);
  follow_b.insert(follow_b.end(), other.begin(), other.begin() + 31);
  for (bool spec : {true, false}) {
    opt.spec = spec;
    opt.park_host_mib = 0;
    gen.reset();
    gen = make_generator(opt);
    run(a, 0, greedy);
    const Out ref_a = run(follow_a, 32, sampled);
    run(b, 0, greedy);
    const Out ref_b = run(follow_b, 32, sampled);
    const Out cold = run(a, 2, greedy);
    opt.park_host_mib = 640;
    gen.reset();
    gen = make_generator(opt);
    run(a, 3, greedy);
    run(b, 4, greedy);
    const Out pa = run(follow_a, 32, sampled);
    const Out pb = run(follow_b, 32, sampled);
    const Out exact = run(follow_a, 32, sampled);
    const Out repeat = run(follow_a, 32, sampled);
    const bool good = cold.stats.reuse == Reuse::None && pa.stats.reuse == Reuse::Parked &&
                      pb.stats.reuse == Reuse::Parked && pa.stats.reused_tokens == int(a.size()) &&
                      pb.stats.reused_tokens == int(b.size()) && exact.stats.reuse == Reuse::Parked &&
                      repeat.stats.reuse == Reuse::Checkpoint && pa.tokens == ref_a.tokens &&
                      pb.tokens == ref_b.tokens && exact.tokens == ref_a.tokens && repeat.tokens == ref_a.tokens;
    std::printf("%s alternating host-parked prompts, spec=%d\n", good ? "PASS" : "FAIL", int(spec));
    ok &= good;
  }
  // Only one of these full-KV snapshots fits this budget. The oldest displaced prompt must be cold.
  opt.park_host_mib = 300;
  gen.reset();
  gen = make_generator(opt);
  const auto third = random_ids(2450);
  run(a, 2, greedy);
  run(b, 2, greedy);
  run(third, 2, greedy);
  const Out evicted = run(a, 2, greedy);
  const Out recent = run(third, 2, greedy);
  const bool bounded = evicted.stats.reuse == Reuse::None && recent.stats.reuse == Reuse::Parked;
  std::printf("%s host budget evicts the oldest displaced prompt\n", bounded ? "PASS" : "FAIL");
  ok &= bounded;
  std::printf("%s\n", ok ? "ALL PASS" : "SOME FAILED");
  return ok ? 0 : 1;
}
