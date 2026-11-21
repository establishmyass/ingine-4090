// INGINE_BOOT_TRACE=1: logs the wall time of each startup phase (cold-start / time-to-first-token work).
#pragma once

#include <spdlog/spdlog.h>

#include <chrono>
#include <cstdlib>

namespace ingine {

inline bool boot_trace_enabled() {
  static const bool on = [] {
    const char* v = std::getenv("INGINE_BOOT_TRACE");
    return v != nullptr && *v == '1';
  }();
  return on;
}

// Logs "boot: <name> <seconds>" when it goes out of scope (or at done()).
class BootPhase {
 public:
  explicit BootPhase(const char* name) : name_(name), t0_(std::chrono::steady_clock::now()) {}
  ~BootPhase() { done(); }
  void done() {
    if (!name_) return;
    if (boot_trace_enabled())
      spdlog::info("boot: {:<28} {:6.3f} s", name_,
                   std::chrono::duration<double>(std::chrono::steady_clock::now() - t0_).count());
    name_ = nullptr;
  }

 private:
  const char* name_;
  std::chrono::steady_clock::time_point t0_;
};

}  // namespace ingine
