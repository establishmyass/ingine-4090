#include "offload/cpu_pool.h"

#include <immintrin.h>

#include <pthread.h>
#include <sched.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <map>
#include <sstream>
#include <stdexcept>
#include <string>

namespace ingine::offload {
namespace {
long long now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch())
      .count();
}
}  // namespace

CpuPool::CpuPool(int threads, int spin_us, std::vector<int> cpus) : spin_ns_(1000ll * spin_us) {
  if (threads < 1 || threads > 256) throw std::invalid_argument("CpuPool: threads must be in 1..256");
  for (int i = 1; i < threads; ++i)
    workers_.emplace_back([this, cpu = i < int(cpus.size()) ? cpus[i] : -1] {
      if (cpu >= 0) pin_this_thread(cpu);
      worker();
    });
}

void CpuPool::pin_this_thread(int cpu) {
  if (cpu < 0) return;
  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(cpu, &set);
  (void)pthread_setaffinity_np(pthread_self(), sizeof set, &set);
}

namespace {
int read_int(const std::string& path, int fallback) {
  std::ifstream f(path);
  int v;
  return (f >> v) ? v : fallback;
}
std::vector<int> parse_list(const std::string& s) {  // "8-15,24"
  std::vector<int> out;
  std::stringstream ss(s);
  std::string part;
  while (std::getline(ss, part, ',')) {
    const auto dash = part.find('-');
    if (dash == std::string::npos) out.push_back(std::atoi(part.c_str()));
    else
      for (int c = std::atoi(part.substr(0, dash).c_str()); c <= std::atoi(part.substr(dash + 1).c_str()); ++c)
        out.push_back(c);
  }
  return out;
}
}  // namespace

std::vector<int> CpuPool::default_cpus(int n) {
  if (const char* v = std::getenv("INGINE_OFFLOAD_CPUS"); v && *v) {
    std::vector<int> l = parse_list(v);
    l.resize(std::min<std::size_t>(l.size(), std::size_t(n)));
    return l;
  }
  cpu_set_t allowed;
  CPU_ZERO(&allowed);
  if (sched_getaffinity(0, sizeof allowed, &allowed) != 0) return {};
  std::map<int, std::vector<int>> by_l3;  // L3 id -> first SMT thread of each core
  std::vector<int> rest;                   // SMT siblings
  for (int c = 0; c < CPU_SETSIZE; ++c) {
    if (!CPU_ISSET(c, &allowed)) continue;
    const std::string base = "/sys/devices/system/cpu/cpu" + std::to_string(c);
    std::ifstream sib(base + "/topology/thread_siblings_list");
    std::string list;
    std::getline(sib, list);
    const std::vector<int> sibs = parse_list(list);
    if (!sibs.empty() && sibs.front() != c) rest.push_back(c);
    else by_l3[read_int(base + "/cache/index3/id", 0)].push_back(c);
  }
  std::vector<int> out;
  for (std::size_t i = 0; out.size() < std::size_t(n); ++i) {  // round robin over the L3 domains, core 0 last
    bool any = false;
    for (auto& [l3, cores] : by_l3) {
      std::vector<int> order = cores;
      std::stable_partition(order.begin(), order.end(), [](int c) { return c != 0; });
      if (i < order.size()) out.push_back(order[i]), any = true;
      if (out.size() == std::size_t(n)) break;
    }
    if (!any) break;
  }
  for (int c : rest)
    if (out.size() < std::size_t(n)) out.push_back(c);
  return out;
}

CpuPool::~CpuPool() {
  stop_.store(true, std::memory_order_seq_cst);
  gen_.fetch_add(1, std::memory_order_seq_cst);
  gen_.notify_all();
  for (auto& t : workers_) t.join();
}

void CpuPool::claim_and_run(std::uint32_t epoch) {
  for (;;) {
    std::uint64_t h = head_.load(std::memory_order_acquire);
    if (std::uint32_t(h >> 32) != epoch) return;
    const std::uint32_t idx = std::uint32_t(h >> 16) & 0xffffu, n = std::uint32_t(h) & 0xffffu;
    if (idx >= n) return;
    if (!head_.compare_exchange_weak(h, h + (1ull << 16), std::memory_order_acq_rel, std::memory_order_acquire))
      continue;
    fn_(ctx_, int(idx));  // the batch cannot change before this job's done lands
    done_.fetch_add(1, std::memory_order_release);
  }
}

void CpuPool::run_impl(int njobs, Fn fn, void* ctx) {
  if (njobs <= 0) return;
  if (njobs > 0xffff) throw std::invalid_argument("CpuPool: too many jobs");
  fn_ = fn;
  ctx_ = ctx;
  done_.store(0, std::memory_order_relaxed);
  ++epoch_;
  head_.store((std::uint64_t(epoch_) << 32) | std::uint32_t(njobs), std::memory_order_seq_cst);
  gen_.fetch_add(1, std::memory_order_seq_cst);
  if (sleepers_.load(std::memory_order_seq_cst) > 0) gen_.notify_all();
  claim_and_run(epoch_);
  while (done_.load(std::memory_order_acquire) < std::uint32_t(njobs)) _mm_pause();
}

void CpuPool::worker() {
  std::uint32_t seen = 0;
  long long last = now_ns();
  int polls = 0;
  while (!stop_.load(std::memory_order_relaxed)) {
    const std::uint32_t ep = std::uint32_t(head_.load(std::memory_order_acquire) >> 32);
    if (ep != seen) {
      seen = ep;
      claim_and_run(ep);
      last = now_ns();
      continue;
    }
    _mm_pause();
    if (++polls < 256) continue;  // read the clock now and then
    polls = 0;
    if (now_ns() - last < spin_ns_) continue;
    const std::uint32_t g = gen_.load(std::memory_order_seq_cst);
    sleepers_.fetch_add(1, std::memory_order_seq_cst);
    if (std::uint32_t(head_.load(std::memory_order_seq_cst) >> 32) == seen && !stop_.load())
      gen_.wait(g, std::memory_order_seq_cst);
    sleepers_.fetch_sub(1, std::memory_order_seq_cst);
    last = now_ns();
  }
}

}  // namespace ingine::offload
