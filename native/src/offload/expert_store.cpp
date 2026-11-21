#include "core/boot_trace.h"
#include "offload/expert_store.h"

#include "core/device.h"
#include "core/file_read.h"
#include "core/safetensors.h"
#include "offload/cpu_pool.h"

#include <immintrin.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <numeric>
#include <stdexcept>

namespace ingine::offload {
namespace moe = kernels::moe;
namespace {

#include "offload/default_profile.inc"

long long now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

constexpr const char* kTensor[4] = {"w13_weight", "w13_weight_scale", "w2_weight", "w2_weight_scale"};
constexpr long long kActiveNs = 50'000'000;  // the leader spins this long after the last step or doorbell

}  // namespace

ExpertStore::ExpertStore(const SafetensorsModel& m, const std::vector<LayerExperts>& layers, const OffloadOptions& o,
                         cudaStream_t stream)
    : o_(o), layers_(layers) {
  if (layers_.size() != std::size_t(kLayers)) throw std::invalid_argument("offload: expected 24 layers");
  if (o_.threads < 1 || o_.swap_every < 3 || o_.spares < 1)
    throw std::invalid_argument("offload: bad options (threads >= 1, swap_every >= 3, spares >= 1)");
  cls_[0].sz = {std::size_t(2) * kI * kH / 2, std::size_t(2) * kI * kH / 16, std::size_t(kH) * kI / 2,
                std::size_t(kH) * kI / 16};
  cls_[1].sz = {std::size_t(2) * kI * kH, std::size_t(2) * kI * kH / 32, std::size_t(kH) * kI,
                std::size_t(kH) * kI / 32};

  // ---- every expert in pinned host memory, in checkpoint layout (four stacked tensors per layer). Each tensor
  // sits at a 4 KiB phase matching its file offset, so cold loads read it with O_DIRECT (core/file_read.h).
  std::size_t cursor = 0;
  std::vector<SafetensorsRef> refs;
  std::vector<std::size_t> dst;
  for (int l = 0; l < kLayers; ++l) {
    cls_of_[l] = layers_[l].fmt == ExpertFmt::NVFP4 ? 0 : 1;
    const Sizes& s = cls_[cls_of_[l]].sz;
    const std::size_t per[4] = {s.w13, s.w13s, s.w2, s.w2s};
    for (int t = 0; t < 4; ++t) {
      const SafetensorsRef r = m.at("model.llm.layers." + std::to_string(l) + ".mlp.experts." + kTensor[t]);
      if (r->bytes != per[t] * kExperts)
        throw std::runtime_error("offload: unexpected size of layer " + std::to_string(l) + " " + kTensor[t]);
      toff_[l][t] = aligned_layout(cursor, r->offset);
      cursor = toff_[l][t] + r->bytes;
      total_bytes_ += r->bytes;
      refs.push_back(r);
      dst.push_back(toff_[l][t]);
    }
  }
  spdlog::info("offload: reading {:.2f} GiB of experts into host memory", double(total_bytes_) / double(1ull << 30));
  BootPhase read_phase("store: read experts");
  // 4 KiB pages unless INGINE_OFFLOAD_HUGEPAGES=1: huge-page faults stall in compaction (see HostRegion)
  const char* hp = std::getenv("INGINE_OFFLOAD_HUGEPAGES");
  host_ = HostRegion(cursor + kDirectAlign, hp != nullptr && *hp == '1');
  prefault(host_.data(), host_.size(), 16);
  {
    std::vector<ReadRange> ranges;
    for (std::size_t i = 0; i < refs.size(); ++i)
      ranges.push_back({refs[i].file, refs[i]->offset, refs[i]->bytes, host_.data() + dst[i]});
    const ReadStats rs = read_ranges(ranges, 16);
    spdlog::info("offload: read {:.2f} GiB in {:.2f} s ({:.1f} GB/s, {}, {:.0f}% was cached)",
                 double(rs.bytes) / double(1ull << 30), rs.seconds, double(rs.bytes) / rs.seconds / 1e9,
                 rs.direct ? "O_DIRECT" : "page cache", 100 * rs.cached);
  }
  read_phase.done();
  BootPhase pin_phase("store: cudaHostRegister");
  if (!host_.register_with_cuda()) throw std::runtime_error("offload: cannot pin the expert memory (cudaHostRegister)");
  pin_phase.done();
  BootPhase upload_phase("store: placement + upload");

  // ---- starting placement: best (count + 1) per byte first, until the VRAM budget is full
  score_.assign(kPairs, 0.0);
  counts_last_.assign(kPairs, 0u);
  pending_.assign(kPairs, 0);
  slot_of_.assign(kPairs, -1);
  std::vector<int> order(kPairs);
  std::iota(order.begin(), order.end(), 0);
  auto bytes_of = [&](int p) { return cls_[cls_of_[p / kExperts]].sz.total(); };
  std::vector<double> prof(kPairs);
  for (int p = 0; p < kPairs; ++p) prof[p] = o_.uniform ? 1.0 : double(kDefaultProfile[p]) + 1.0;
  std::stable_sort(order.begin(), order.end(),
                   [&](int a, int b) { return prof[a] / double(bytes_of(a)) > prof[b] / double(bytes_of(b)); });
  const double spare_bytes = double(o_.spares) * double(cls_[0].sz.total() + cls_[1].sz.total());
  double budget = double(total_bytes_) - o_.gib * double(1ull << 30) - spare_bytes;
  std::array<std::vector<int>, 2> resident;
  for (int p : order) {
    const double b = double(bytes_of(p));
    if (b <= budget) resident[cls_of_[p / kExperts]].push_back(p), budget -= b;
  }
  for (int c = 0; c < 2; ++c) {
    Class& k = cls_[c];
    int in_class = 0;
    for (int l = 0; l < kLayers; ++l) in_class += cls_of_[l] == c ? kExperts : 0;
    k.nslots = std::min(int(resident[c].size()) + o_.spares, in_class);
    if (k.nslots > 0) k.slots = DeviceBuffer(std::size_t(k.nslots) * k.sz.total());
    k.free_ready.assign(k.nslots, nullptr);
    for (int s = 0; s < k.nslots; ++s) CUDA_CHECK(cudaEventCreateWithFlags(&k.free_ready[s], cudaEventDisableTiming));
    for (int s = 0; s < int(resident[c].size()); ++s) {
      const int p = resident[c][s];
      slot_of_[p] = s;
      for (int t = 0; t < 4; ++t)
        CUDA_CHECK(cudaMemcpyAsync(slot_ptr(c, s, t), host_ptr(p, t), t == 0 ? k.sz.w13 : t == 1 ? k.sz.w13s
                                                                           : t == 2 ? k.sz.w2 : k.sz.w2s,
                                   cudaMemcpyHostToDevice, stream));
      resident_bytes_ += k.sz.total();
    }
    for (int s = int(resident[c].size()); s < k.nslots; ++s) k.free_slots.push_back(s);
    for (int s = 0; s < k.nslots; ++s) CUDA_CHECK(cudaEventRecord(k.free_ready[s], stream));
  }

  // ---- placement tables
  std::vector<moe::ExpertTable> tab(kLayers);
  for (int p = 0; p < kPairs; ++p) {
    const moe::TableUpdate u = slot_of_[p] >= 0 ? resident_update(p, slot_of_[p]) : host_update(p);
    moe::ExpertTable& t = tab[u.layer];
    t.w13[u.expert] = u.w13, t.w13s[u.expert] = u.w13s, t.w2[u.expert] = u.w2, t.w2s[u.expert] = u.w2s;
    t.hot[u.expert] = u.hot;
  }
  tables_ = DeviceBuffer(sizeof(moe::ExpertTable) * kLayers);
  CUDA_CHECK(cudaMemcpyAsync(tables_.p, tab.data(), sizeof(moe::ExpertTable) * kLayers, cudaMemcpyHostToDevice, stream));

  // ---- decode hand-off and routing counts
  bell_ = MappedHostBuffer(sizeof(moe::ColdBell));
  done_ = MappedHostBuffer(256);  // done at 0, err at 128 (separate lines)
  y_ = MappedHostBuffer(std::size_t(moe::kMaxSlots) * kH * 2);
  counts_host_ = MappedHostBuffer(std::size_t(kPairs) * 4);
  dev_ = DeviceBuffer(64);
  dev_.fill(0);
  counts_ = DeviceBuffer(std::size_t(kPairs) * 4);
  counts_.fill(0);
  ex_.bell = static_cast<moe::ColdBell*>(bell_.device_ptr());
  ex_.done = static_cast<const volatile std::uint32_t*>(done_.device_ptr());
  ex_.err = reinterpret_cast<volatile std::uint32_t*>(static_cast<char*>(done_.device_ptr()) + 128);
  ex_.y = static_cast<const __nv_bfloat16*>(y_.device_ptr());
  ex_.dev = static_cast<std::uint32_t*>(dev_.p);
  ex_.counts = static_cast<std::uint32_t*>(counts_.p);
  ex_.cpu = o_.cpu;
  CUDA_CHECK(cudaStreamCreateWithPriority(&copy_, cudaStreamNonBlocking, 0));
  if (o_.stage_mib > 0) {
    stage_bytes_ = std::size_t(o_.stage_mib) * (1ull << 20) / 2;
    for (auto& b : stage_buf_) b = DeviceBuffer(stage_bytes_);
    stage_tables_ = DeviceBuffer(sizeof(moe::ExpertTable) * kLayers * 2);
    for (int i = 0; i < 2; ++i) {
      CUDA_CHECK(cudaEventCreateWithFlags(&stage_ready_[i], cudaEventDisableTiming));
      CUDA_CHECK(cudaEventCreateWithFlags(&stage_free_[i], cudaEventDisableTiming));
    }
  }
  CUDA_CHECK(cudaEventCreateWithFlags(&step_done_, cudaEventDisableTiming));
  CUDA_CHECK(cudaEventCreateWithFlags(&counts_ready_, cudaEventDisableTiming));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  upload_phase.done();

  if (o_.cpu) {
    if (!cpu_experts_supported())
      throw std::runtime_error("offload: the CPU expert kernels need AVX-512 (F, BW, VL, DQ, BF16); "
                               "use GPU reads of host memory instead (offload cpu = false)");
    cpus_ = CpuPool::default_cpus(o_.threads);
    pool_ = std::make_unique<CpuPool>(o_.threads, o_.spin_us, cpus_);
    xperm_.assign(std::size_t(8) * kH, 0);
    act_.assign(std::size_t(moe::kMaxSlots) * kI, 0);
    aperm_.assign(std::size_t(moe::kMaxSlots) * kI, 0);
    if (const char* v = std::getenv("INGINE_OFFLOAD_SELFTEST"); v && *v == '1') {  // CPU pool speed, GPU idle
      std::vector<std::uint16_t> x(kH, 0x3f80), xp(kH), act(kI), ap(kI), y(kH);
      prepare_row(x.data(), kH, xp.data());
      const int T = pool_->threads(), iters = 64;
      double up = 0, down = 0, bytes = 0;
      for (int it = 0; it < iters; ++it) {
        const int pair = (it * 37 + 11) % kPairs;
        const ExpertView v = view(pair);
        const PreparedRow xr{x.data(), xp.data()};
        const long long t0 = now_ns();
        pool_->run(T, [&](int j) {
          std::uint16_t* a = act.data();
          expert_up(v, &xr, 1, &a, kI * j / T, kI * (j + 1) / T);
        });
        const long long t1 = now_ns();
        prepare_row(act.data(), kI, ap.data());
        const PreparedRow ar{act.data(), ap.data()};
        const long long t2 = now_ns();
        pool_->run(T, [&](int j) {
          std::uint16_t* yp = y.data();
          expert_down(v, &ar, 1, &yp, kH * j / T, kH * (j + 1) / T);
        });
        const long long t3 = now_ns();
        up += double(t1 - t0), down += double(t3 - t2), bytes += double(cls_[cls_of_[pair / kExperts]].sz.total());
      }
      spdlog::info("offload: CPU pool self-test on the host region, GPU idle: up {:.1f} us, down {:.1f} us per "
                   "expert, {:.1f} GB/s", up / iters / 1e3, down / iters / 1e3, bytes / (up + down));
    }
    leader_ = std::thread([this] { leader(); });
  }
  spdlog::info("offload: {}", summary());
}

ExpertStore::~ExpertStore() {
  stop_.store(true);
  if (leader_.joinable()) leader_.join();
  spdlog::info("offload: {}", summary());
  pool_.reset();
  if (copy_) cudaStreamSynchronize(copy_);
  for (Pending& p : pending_swaps_) cudaEventDestroy(p.copied);
  for (Class& k : cls_)
    for (cudaEvent_t e : k.free_ready)
      if (e) cudaEventDestroy(e);
  for (int i = 0; i < 2; ++i) {
    if (stage_ready_[i]) cudaEventDestroy(stage_ready_[i]);
    if (stage_free_[i]) cudaEventDestroy(stage_free_[i]);
  }
  if (step_done_) cudaEventDestroy(step_done_);
  if (counts_ready_) cudaEventDestroy(counts_ready_);
  if (copy_) cudaStreamDestroy(copy_);
}

const moe::ExpertTable* ExpertStore::table(int layer) const {
  return static_cast<const moe::ExpertTable*>(tables_.p) + layer;
}

const std::uint8_t* ExpertStore::host_ptr(int pair, int t) const {
  const int l = pair / kExperts, e = pair % kExperts;
  const Sizes& s = cls_[cls_of_[l]].sz;
  const std::size_t per[4] = {s.w13, s.w13s, s.w2, s.w2s};
  // UVA: the registered region's device address equals its host address
  return reinterpret_cast<const std::uint8_t*>(host_.data()) + toff_[l][t] + per[t] * e;
}

std::uint8_t* ExpertStore::slot_ptr(int c, int slot, int t) const {
  const Sizes& s = cls_[c].sz;
  const std::size_t in_slot[4] = {0, s.w13, s.w13 + s.w13s, s.w13 + s.w13s + s.w2};
  return static_cast<std::uint8_t*>(cls_[c].slots.p) + std::size_t(slot) * s.total() + in_slot[t];
}

moe::TableUpdate ExpertStore::resident_update(int pair, int slot) const {
  const int c = cls_of_[pair / kExperts];
  return {pair / kExperts, pair % kExperts, 1, slot_ptr(c, slot, 0), slot_ptr(c, slot, 1), slot_ptr(c, slot, 2),
          slot_ptr(c, slot, 3)};
}

moe::TableUpdate ExpertStore::host_update(int pair) const {
  return {pair / kExperts, pair % kExperts, 0, host_ptr(pair, 0), host_ptr(pair, 1), host_ptr(pair, 2),
          host_ptr(pair, 3)};
}

ExpertView ExpertStore::view(int pair) const {
  const int l = pair / kExperts, e = pair % kExperts;
  ExpertView v;
  v.fmt = layers_[l].fmt;
  v.w13 = host_ptr(pair, 0), v.w13s = host_ptr(pair, 1), v.w2 = host_ptr(pair, 2), v.w2s = host_ptr(pair, 3);
  if (v.fmt == ExpertFmt::NVFP4) v.f13 = layers_[l].f13.at(e), v.f2 = layers_[l].f2.at(e);
  return v;
}

// ---------------------------------------------------------------------------------------------- CPU side

void ExpertStore::leader() {
  if (!cpus_.empty()) CpuPool::pin_this_thread(cpus_[0]);
  const volatile std::uint32_t* seq = &static_cast<moe::ColdBell*>(bell_.data())->seq;
  volatile std::uint32_t* done = static_cast<volatile std::uint32_t*>(done_.data());
  std::uint32_t last = *seq;
  int polls = 0;
  while (!stop_.load(std::memory_order_relaxed)) {
    const std::uint32_t s = *seq;
    if (s != last) {
      std::atomic_thread_fence(std::memory_order_acquire);
      const long long t0 = now_ns();
      serve_bell();
      bell_ns_ += now_ns() - t0;
      last = s;
      std::atomic_thread_fence(std::memory_order_release);
      *done = s;
      active_until_ns_.store(now_ns() + kActiveNs, std::memory_order_relaxed);
      continue;
    }
    _mm_pause();
    if (++polls < 512) continue;
    polls = 0;
    if (now_ns() > active_until_ns_.load(std::memory_order_relaxed))  // idle: no decode step for a while
      std::this_thread::sleep_for(std::chrono::microseconds(100));
  }
}

void ExpertStore::serve_bell() {
  const moe::ColdBell& b = *static_cast<const moe::ColdBell*>(bell_.data());
  const int M = int(b.M), n = int(b.n), layer = int(b.layer);
  if (M < 1 || M > 8 || n < 1 || n > moe::kMaxSlots || layer < 0 || layer >= kLayers) return;
  struct Group {
    int pair, nt, entry[8];
  };
  Group groups[moe::kMaxSlots];
  int ng = 0;
  bool used[8] = {};
  for (int i = 0; i < n; ++i) {
    const int pair = layer * kExperts + b.expert[i];
    int g = 0;
    while (g < ng && groups[g].pair != pair) ++g;
    if (g == ng) groups[ng++] = Group{pair, 0, {}};
    if (groups[g].nt < 8) groups[g].entry[groups[g].nt++] = i;
    used[b.slot[i] / 8] = true;
  }
  const std::uint16_t* x = reinterpret_cast<const std::uint16_t*>(&b.x[0][0]);
  for (int t = 0; t < M; ++t)
    if (used[t]) prepare_row(x + std::size_t(t) * kH, kH, &xperm_[std::size_t(t) * kH]);
  // One contiguous, equal slice of all cold rows per thread and phase (Strata's row slices): long sequential
  // streams for the prefetchers and no straggling small jobs.
  const int T = pool_->threads();
  const int up_total = ng * kI, down_total = ng * kH;  // gate/up pairs, then W2 rows, of all cold experts
  const long long t0 = now_ns();
  pool_->run(T, [&](int j) {
    const int a = int((long long)up_total * j / T), z = int((long long)up_total * (j + 1) / T);
    for (int g0 = a / kI; g0 < ng && g0 * kI < z; ++g0) {
      const Group& g = groups[g0];
      const int p0 = std::max(a - g0 * kI, 0), p1 = std::min(z - g0 * kI, kI);
      if (p0 >= p1) continue;
      PreparedRow xr[8];
      std::uint16_t* act[8];
      for (int k = 0; k < g.nt; ++k) {
        const int t = b.slot[g.entry[k]] / 8;
        xr[k] = {x + std::size_t(t) * kH, &xperm_[std::size_t(t) * kH]};
        act[k] = &act_[std::size_t(g.entry[k]) * kI];
      }
      expert_up(view(g.pair), xr, g.nt, act, p0, p1);
    }
  });
  const long long t1 = now_ns();
  for (int i = 0; i < n; ++i) prepare_row(&act_[std::size_t(i) * kI], kI, &aperm_[std::size_t(i) * kI]);
  const long long t2 = now_ns();
  std::uint16_t* y = static_cast<std::uint16_t*>(y_.data());
  pool_->run(T, [&](int j) {
    const int a = int((long long)down_total * j / T), z = int((long long)down_total * (j + 1) / T);
    for (int g0 = a / kH; g0 < ng && g0 * kH < z; ++g0) {
      const Group& g = groups[g0];
      const int r0 = std::max(a - g0 * kH, 0), r1 = std::min(z - g0 * kH, kH);
      if (r0 >= r1) continue;
      PreparedRow ar[8];
      std::uint16_t* yr[8];
      for (int k = 0; k < g.nt; ++k) {
        const int i = g.entry[k];
        ar[k] = {&act_[std::size_t(i) * kI], &aperm_[std::size_t(i) * kI]};
        yr[k] = y + std::size_t(b.slot[i]) * kH;
      }
      expert_down(view(g.pair), ar, g.nt, yr, r0, r1);
    }
  });
  up_ns_ += t1 - t0;
  down_ns_ += now_ns() - t2;
  ++bells_;
  cold_entries_ += n;
  cold_experts_ += ng;
}

// ------------------------------------------------------------------------------------- adaptive placement

void ExpertStore::tick(cudaStream_t stream) {
  if (*reinterpret_cast<volatile std::uint32_t*>(static_cast<char*>(done_.data()) + 128))
    throw std::runtime_error("offload: the GPU timed out waiting for the CPU expert pool");
  active_until_ns_.store(now_ns() + kActiveNs, std::memory_order_relaxed);
  const long long step = ticks_++;
  if (!o_.adaptive) return;
  if (step == 0)  // forget the routing of the runtime's eager warm-up / capture runs: count real steps only
    CUDA_CHECK(cudaMemsetAsync(counts_.p, 0, std::size_t(kPairs) * 4, stream));
  // Step-locked, so a request sequence always sees the same placements (reproducible outputs): at step 0 mod N
  // the routing counts of every earlier step are copied back, at step 1 the swaps are decided and their copies
  // issued, at step 2 the compute stream waits for the copies on the GPU and flips the tables.
  switch (step % o_.swap_every) {
    case 0:
      CUDA_CHECK(cudaEventRecord(step_done_, stream));
      CUDA_CHECK(cudaStreamWaitEvent(copy_, step_done_, 0));
      CUDA_CHECK(cudaMemcpyAsync(counts_host_.data(), counts_.p, std::size_t(kPairs) * 4, cudaMemcpyDeviceToHost,
                                 copy_));
      CUDA_CHECK(cudaEventRecord(counts_ready_, copy_));
      counts_in_flight_ = true;
      break;
    case 1:
      if (counts_in_flight_) {
        CUDA_CHECK(cudaEventSynchronize(counts_ready_));  // a step back: normally complete already
        counts_in_flight_ = false;
        decide_swaps();
      }
      break;
    case 2: {
      if (pending_swaps_.empty()) break;
      for (const Pending& p : pending_swaps_) CUDA_CHECK(cudaStreamWaitEvent(stream, p.copied, 0));
      std::vector<std::pair<int, int>> freed;  // class, slot
      for (std::size_t i = 0; i < pending_swaps_.size();) {
        moe::TableUpdates up;
        for (; i < pending_swaps_.size() && up.n + 2 <= moe::kMaxTableUpdates; ++i) {
          const Pending& p = pending_swaps_[i];
          freed.emplace_back(cls_of_[p.out / kExperts], slot_of_[p.out]);
          up.u[up.n++] = resident_update(p.in, p.slot);
          up.u[up.n++] = host_update(p.out);
          slot_of_[p.in] = p.slot;
          slot_of_[p.out] = -1;
          pending_[p.in] = pending_[p.out] = 0;
        }
        moe::apply_table_updates(static_cast<moe::ExpertTable*>(tables_.p), up, stream);
      }
      for (const Pending& p : pending_swaps_) cudaEventDestroy(p.copied);
      swaps_ += static_cast<long long>(pending_swaps_.size());
      pending_swaps_.clear();
      for (auto [c, sl] : freed) {  // reusable once the compute stream passed the flip
        CUDA_CHECK(cudaEventRecord(cls_[c].free_ready[sl], stream));
        cls_[c].free_slots.push_back(sl);
      }
      break;
    }
    default:
      break;
  }
}

void ExpertStore::decide_swaps() {
  const std::uint32_t* c = static_cast<const std::uint32_t*>(counts_host_.data());
  for (int p = 0; p < kPairs; ++p) {
    score_[p] += double(c[p] - counts_last_[p]);
    counts_last_[p] = c[p];
  }
  for (int k = 0; k < 2; ++k) {
    Class& cl = cls_[k];
    std::vector<int> spare(cl.free_slots.size());  // indices into free_slots
    std::iota(spare.begin(), spare.end(), 0);
    if (spare.empty()) continue;
    std::vector<int> cold, hot;
    for (int p = 0; p < kPairs; ++p) {
      if (cls_of_[p / kExperts] != k || pending_[p]) continue;
      if (slot_of_[p] < 0 && score_[p] >= 2.0) cold.push_back(p);
      else if (slot_of_[p] >= 0) hot.push_back(p);
    }
    std::sort(cold.begin(), cold.end(), [&](int a, int b) { return score_[a] > score_[b]; });
    std::sort(hot.begin(), hot.end(), [&](int a, int b) { return score_[a] < score_[b]; });
    std::vector<int> taken;
    for (std::size_t i = 0; i < cold.size() && i < hot.size() && i < spare.size(); ++i) {
      const int in = cold[i], out = hot[i];
      if (!(score_[in] > double(o_.ratio) * score_[out] + 1e-9)) break;
      const int slot = cl.free_slots[std::size_t(spare[i])];
      taken.push_back(spare[i]);
      CUDA_CHECK(cudaStreamWaitEvent(copy_, cl.free_ready[slot], 0));  // its previous expert is out of use
      for (int t = 0; t < 4; ++t)
        CUDA_CHECK(cudaMemcpyAsync(slot_ptr(k, slot, t), host_ptr(in, t),
                                   t == 0 ? cl.sz.w13 : t == 1 ? cl.sz.w13s : t == 2 ? cl.sz.w2 : cl.sz.w2s,
                                   cudaMemcpyHostToDevice, copy_));
      Pending pend{in, out, slot, nullptr};
      CUDA_CHECK(cudaEventCreateWithFlags(&pend.copied, cudaEventDisableTiming));
      CUDA_CHECK(cudaEventRecord(pend.copied, copy_));
      pending_swaps_.push_back(pend);
      pending_[in] = pending_[out] = 1;
    }
    std::sort(taken.rbegin(), taken.rend());
    for (int i : taken) cl.free_slots.erase(cl.free_slots.begin() + i);
  }
  for (double& s : score_) s *= double(o_.decay);
}

// ------------------------------------------------------------------------------------- prefill staging

namespace {
constexpr int kVisitLayer[48] = {0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16,
                                 17, 6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 6,  7,  8,  9,
                                 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23};
}  // namespace

bool ExpertStore::prefill_begin(int tokens, cudaStream_t stream) {
  staging_ = stage_bytes_ > 0 && tokens >= stage_min_tokens_;
  if (!staging_) return false;
  // the current placement, frozen for the chunk (swaps happen only between decode steps)
  std::vector<moe::ExpertTable> tab(std::size_t(kLayers) * 2);
  for (int l = 0; l < kLayers; ++l) {
    const std::size_t sz = cls_[cls_of_[l]].sz.total();
    staged_[l].clear();
    for (int e = 0; e < kExperts; ++e) {
      const int p = l * kExperts + e;
      if (slot_of_[p] < 0 && (staged_[l].size() + 1) * sz <= stage_bytes_) staged_[l].push_back(p);
    }
    for (int b = 0; b < 2; ++b) {
      moe::ExpertTable& t = tab[std::size_t(l) * 2 + b];
      for (int e = 0; e < kExperts; ++e) {
        const int p = l * kExperts + e;
        const moe::TableUpdate u = slot_of_[p] >= 0 ? resident_update(p, slot_of_[p]) : host_update(p);
        t.w13[e] = u.w13, t.w13s[e] = u.w13s, t.w2[e] = u.w2, t.w2s[e] = u.w2s, t.hot[e] = u.hot;
      }
      for (std::size_t k = 0; k < staged_[l].size(); ++k) {
        const int e = staged_[l][k] % kExperts;
        std::uint8_t* base = static_cast<std::uint8_t*>(stage_buf_[b].p) + k * sz;
        const Sizes& z = cls_[cls_of_[l]].sz;
        t.w13[e] = base, t.w13s[e] = base + z.w13, t.w2[e] = base + z.w13 + z.w13s, t.w2s[e] = base + z.w13 + z.w13s + z.w2;
      }
    }
  }
  // the previous chunk's last use of the buffers and tables comes before this upload
  CUDA_CHECK(cudaEventRecord(step_done_, stream));
  CUDA_CHECK(cudaStreamWaitEvent(copy_, step_done_, 0));
  CUDA_CHECK(cudaMemcpyAsync(stage_tables_.p, tab.data(), sizeof(moe::ExpertTable) * tab.size(), cudaMemcpyHostToDevice,
                             copy_));
  CUDA_CHECK(cudaStreamSynchronize(copy_));  // `tab` is pageable host memory
  for (int b = 0; b < 2; ++b) CUDA_CHECK(cudaEventRecord(stage_free_[b], stream));
  stage_visit(0);
  return true;
}

// Copies visit v's staged experts into buffer v % 2 once the visit two back released it; ready event after.
void ExpertStore::stage_visit(int v) {
  const int b = v & 1, l = kVisitLayer[v];
  CUDA_CHECK(cudaStreamWaitEvent(copy_, stage_free_[b], 0));
  const Sizes& z = cls_[cls_of_[l]].sz;
  for (std::size_t k = 0; k < staged_[l].size(); ++k) {
    std::uint8_t* base = static_cast<std::uint8_t*>(stage_buf_[b].p) + k * z.total();
    const int p = staged_[l][k];
    // the four tensors of an expert are separate in host memory, contiguous in the staging slot
    CUDA_CHECK(cudaMemcpyAsync(base, host_ptr(p, 0), z.w13, cudaMemcpyHostToDevice, copy_));
    CUDA_CHECK(cudaMemcpyAsync(base + z.w13, host_ptr(p, 1), z.w13s, cudaMemcpyHostToDevice, copy_));
    CUDA_CHECK(cudaMemcpyAsync(base + z.w13 + z.w13s, host_ptr(p, 2), z.w2, cudaMemcpyHostToDevice, copy_));
    CUDA_CHECK(cudaMemcpyAsync(base + z.w13 + z.w13s + z.w2, host_ptr(p, 3), z.w2s, cudaMemcpyHostToDevice, copy_));
  }
  CUDA_CHECK(cudaEventRecord(stage_ready_[b], copy_));
}

const moe::ExpertTable* ExpertStore::prefill_table(int v, cudaStream_t stream) {
  const int l = kVisitLayer[v];
  if (!staging_) return table(l);
  CUDA_CHECK(cudaStreamWaitEvent(stream, stage_ready_[v & 1], 0));
  if (v + 1 < 48) stage_visit(v + 1);  // into the other buffer, overlapping this visit
  return static_cast<const moe::ExpertTable*>(stage_tables_.p) + std::size_t(l) * 2 + (v & 1);
}

void ExpertStore::prefill_release(int v, cudaStream_t stream) {
  if (staging_) CUDA_CHECK(cudaEventRecord(stage_free_[v & 1], stream));
}

std::string ExpertStore::summary() const {
  char buf[512];
  std::snprintf(buf, sizeof buf,
                "experts %.2f GiB in VRAM (%d NVFP4 + %d MXFP8 slots incl. %d spares each), %.2f GiB only in host "
                "memory; cold experts on the %s%s; %lld decode steps, %lld swaps, %lld CPU hand-offs (%lld slots, "
                "%lld experts, %.1f us each: up %.1f, down %.1f)",
                double(resident_bytes_) / double(1ull << 30), cls_[0].nslots, cls_[1].nslots, o_.spares,
                double(total_bytes_ - resident_bytes_) / double(1ull << 30),
                o_.cpu ? "CPU" : "GPU over PCIe",
                o_.cpu ? (" (" + std::to_string(o_.threads) + " threads)").c_str() : "", ticks_, swaps_, bells_,
                cold_entries_, cold_experts_, bells_ ? double(bell_ns_) / double(bells_) / 1e3 : 0.0,
                bells_ ? double(up_ns_) / double(bells_) / 1e3 : 0.0, bells_ ? double(down_ns_) / double(bells_) / 1e3 : 0.0);
  return buf;
}

}  // namespace ingine::offload
