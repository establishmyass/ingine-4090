// Expert offloading: every routed expert in pinned host memory, a hot subset in VRAM slots, the rest computed
// by the CPU pool during decode (or read over PCIe by the GPU kernels: prefill, and decode with cpu = false).
//
// Design after Strata's expert cache and fngine's hot/cold tiers (see EXPERT-OFFLOAD.md):
// * Placement: the starting set is the best-ranked experts (routing counts per byte) of a profile - the built-in
//   one recorded on 18.5K tokens, or uniform - filling the VRAM budget. Then a decayed LFU (Strata's adaptive
//   tier): the decode routing counts come back from the GPU every `swap_every` steps; per expert class (NVFP4,
//   MXFP8: different slot sizes) a cold expert with score >= 2 replaces the least-scored resident one when it
//   beats it by `ratio`; scores then decay by `decay`. A swap copies the expert into a spare slot on a copy
//   stream and flips the two table entries on the compute stream once the copy is complete, between decode
//   steps (no host sync, the decode graphs keep running); the evicted slot becomes a spare when the compute
//   stream passes the flip.
// * Decode hand-off: after the router, cold_publish puts the visit's cold slots and x into a mapped doorbell;
//   the leader thread spins on it, runs the experts' W13 rows then W2 rows on the CpuPool (cpu_experts.h),
//   writes the BF16 outputs into mapped memory and stores `done`; moe_down waits for it before its weighted
//   sum, after computing the hot experts.
#pragma once

#include "core/arena.h"
#include "core/host_memory.h"
#include "kernels/moe/moe.h"
#include "offload/cpu_experts.h"

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

namespace ingine {
class SafetensorsModel;
}

namespace ingine::offload {

class CpuPool;

struct OffloadOptions {
  double gib = 0.0;        // expert bytes kept out of VRAM (0: offloading off)
  bool cpu = true;         // decode computes cold experts on the CPU (else the GPU reads them over PCIe)
  int threads = 8;         // CPU pool threads, the leader included
  int spin_us = 3000;      // idle spin of pool threads before they sleep
  bool adaptive = true;    // decayed-LFU swaps during decode
  int swap_every = 4;      // decode steps between placement updates
  float decay = 0.7f;
  float ratio = 1.5f;
  int spares = 4;          // spare VRAM slots per expert class (swaps in flight)
  bool uniform = false;    // start from a uniform profile instead of the built-in one
  int stage_mib = 384;     // prefill staging, both buffers together (0: prefill reads cold experts over PCIe)

  bool enabled() const { return gib > 0.0; }
};

// Per layer: expert format and the NVFP4 factors (amax / outer_scale_denominator, empty for MXFP8).
struct LayerExperts {
  ExpertFmt fmt = ExpertFmt::NVFP4;
  std::vector<float> f13, f2;
};

class ExpertStore {
 public:
  ExpertStore(const SafetensorsModel& m, const std::vector<LayerExperts>& layers, const OffloadOptions& o,
              cudaStream_t stream);
  ~ExpertStore();
  ExpertStore(const ExpertStore&) = delete;
  ExpertStore& operator=(const ExpertStore&) = delete;

  const kernels::moe::ExpertTable* table(int layer) const;
  const kernels::moe::ColdExchange& exchange() const { return ex_; }
  // Before each decode step launch on `stream`: placement updates; throws if a CPU hand-off timed out.
  void tick(cudaStream_t stream);

  // Prefill pipelining (after Strata's): for chunks of at least `stage_min_tokens`, each visit's cold experts
  // are copied into one of two VRAM staging buffers while the previous visit computes, so the expert GEMMs read
  // VRAM instead of host memory over PCIe. Call prefill_begin once per chunk, then for each visit in order
  // prefill_table (waits for its staging, starts the next visit's) before its expert GEMMs and prefill_release
  // after them. Without staging (short chunk, staging off) prefill_table is table(layer).
  bool prefill_begin(int tokens, cudaStream_t stream);
  const kernels::moe::ExpertTable* prefill_table(int visit, cudaStream_t stream);
  void prefill_release(int visit, cudaStream_t stream);
  std::string summary() const;

 private:
  static constexpr int kLayers = 24, kExperts = 128, kPairs = kLayers * kExperts;
  struct Sizes { std::size_t w13, w13s, w2, w2s; std::size_t total() const { return w13 + w13s + w2 + w2s; } };
  struct Class {                       // one expert format: its slots in VRAM
    Sizes sz{};
    DeviceBuffer slots;
    int nslots = 0;
    std::vector<int> free_slots;       // spares; each usable once free_ready[slot] completed
    std::vector<cudaEvent_t> free_ready;
  };
  struct Pending {                     // a swap whose copy is in flight
    int in, out, slot;                 // pair indices (layer * 128 + expert), slot of `in`
    cudaEvent_t copied;
  };

  ExpertView view(int pair) const;
  const std::uint8_t* host_ptr(int pair, int tensor) const;
  std::uint8_t* slot_ptr(int cls, int slot, int tensor) const;
  kernels::moe::TableUpdate resident_update(int pair, int slot) const;
  kernels::moe::TableUpdate host_update(int pair) const;
  void leader();
  void serve_bell();
  void decide_swaps();
  void stage_visit(int visit);

  OffloadOptions o_;
  std::vector<LayerExperts> layers_;
  std::array<int, kLayers> cls_of_{};            // 0 NVFP4, 1 MXFP8
  std::array<Class, 2> cls_;
  HostRegion host_;
  std::array<std::size_t, kLayers> layer_off_{};  // byte offset of the layer's four stacked tensors
  std::vector<int> slot_of_;                      // pair -> slot (-1: host)
  std::vector<std::uint8_t> pending_;             // pair in a pending swap
  std::vector<double> score_;
  std::vector<std::uint32_t> counts_last_;
  DeviceBuffer tables_;                           // ExpertTable [24]
  DeviceBuffer dev_;                              // [2] seq counter, wait seq
  DeviceBuffer counts_;                           // [24 * 128]
  MappedHostBuffer bell_, done_, y_, counts_host_;
  kernels::moe::ColdExchange ex_{};
  cudaStream_t copy_ = nullptr;
  cudaEvent_t step_done_ = nullptr, counts_ready_ = nullptr;
  bool counts_in_flight_ = false;
  std::vector<Pending> pending_swaps_;
  long long ticks_ = 0, swaps_ = 0, bells_ = 0, cold_entries_ = 0, cold_experts_ = 0, bell_ns_ = 0;
  long long up_ns_ = 0, down_ns_ = 0;  // time in the two pool phases
  std::size_t resident_bytes_ = 0, total_bytes_ = 0;
  // CPU side
  // prefill staging
  std::array<DeviceBuffer, 2> stage_buf_;
  std::size_t stage_bytes_ = 0;                  // per buffer
  DeviceBuffer stage_tables_;                    // ExpertTable [24][2]: hot slots, staged cold experts, rest host
  std::array<std::vector<int>, kLayers> staged_; // per layer: the cold experts that fit a staging buffer
  std::array<cudaEvent_t, 2> stage_ready_{}, stage_free_{};
  bool staging_ = false;
  int stage_min_tokens_ = 1024;
  std::unique_ptr<CpuPool> pool_;
  std::vector<int> cpus_;  // leader, then pool workers
  std::thread leader_;
  std::atomic<bool> stop_{false};
  std::atomic<long long> active_until_ns_{0};
  std::vector<std::uint16_t> xperm_, act_, aperm_;  // BF16 bits: x (NVFP4 order), act, act (NVFP4 order)
};

}  // namespace ingine::offload
