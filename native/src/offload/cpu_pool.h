// CPU worker pool for offloaded experts: run(njobs, f) calls f(job) for every job in [0, njobs) on the pool's
// threads and the caller, and returns when all are done.
//
// Claim protocol from Strata (github.com/Niko1221/Strata, include/strata/kernels/cpu/pool.hpp, MIT): one
// 64-bit head word `epoch | index | njobs`; a claim is a CAS that succeeds only for the epoch the worker woke
// for, so a worker that wakes late can never take a job of the next batch or count into its `done`. Workers
// spin for a while after their last job (a decode step brings the next batch within milliseconds), then
// sleep on a futex word the caller bumps only when someone sleeps.
#pragma once

#include <atomic>
#include <cstdint>
#include <thread>
#include <type_traits>
#include <vector>

namespace ingine::offload {

class CpuPool {
 public:
  // `threads` >= 1 counts the caller of run(); spin_us: how long an idle worker spins before sleeping.
  // cpus: worker i is pinned to cpus[i + 1] (cpus[0] is for the caller, see pin_this_thread); empty: no pinning.
  CpuPool(int threads, int spin_us, std::vector<int> cpus = {});
  ~CpuPool();
  CpuPool(const CpuPool&) = delete;
  CpuPool& operator=(const CpuPool&) = delete;

  template <class F>
  void run(int njobs, F&& f) {
    using G = std::remove_reference_t<F>;
    run_impl(njobs, [](void* ctx, int job) { (*static_cast<G*>(ctx))(job); },
             const_cast<void*>(static_cast<const void*>(&f)));
  }
  int threads() const { return int(workers_.size()) + 1; }

  // One physical core per thread, alternating between L3 domains (CCDs: each has its own link to memory),
  // core 0 and SMT siblings last; $INGINE_OFFLOAD_CPUS ("8-15", "1,3,5") overrides. After Strata's
  // pool_affinity_linux.hpp.
  static std::vector<int> default_cpus(int n);
  static void pin_this_thread(int cpu);

 private:
  using Fn = void (*)(void*, int);
  void run_impl(int njobs, Fn fn, void* ctx);
  void claim_and_run(std::uint32_t epoch);
  void worker();

  std::atomic<std::uint64_t> head_{0};  // epoch << 32 | index << 16 | njobs
  std::atomic<std::uint32_t> done_{0};
  std::atomic<std::uint32_t> gen_{0};   // futex word
  std::atomic<int> sleepers_{0};
  std::atomic<bool> stop_{false};
  Fn fn_ = nullptr;
  void* ctx_ = nullptr;
  std::uint32_t epoch_ = 0;
  long long spin_ns_;
  std::vector<std::thread> workers_;
};

}  // namespace ingine::offload
