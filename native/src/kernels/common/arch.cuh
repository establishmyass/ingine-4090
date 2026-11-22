// Target switches shared by host and device code. INGINE_AMPERE selects the SM8x path on RTX 3090 and RTX 4090:
// no programmatic dependent launch (PDL) or thread-block clusters.
//
// Without PDL every kernel of a stream starts after its predecessor completed, so griddepcontrol.wait /
// launch_dependents become no-ops (the wait keeps a compiler memory barrier, so program order around it is
// unchanged) and launchers drop the programmatic-stream-serialization attribute.
#pragma once

namespace ingine::kernels {

#ifdef INGINE_AMPERE
inline constexpr bool kHasPdl = false;
#else
inline constexpr bool kHasPdl = true;
#endif

}  // namespace ingine::kernels

#ifdef INGINE_AMPERE
#define INGINE_GRIDDEP_WAIT() asm volatile("" ::: "memory")
#define INGINE_GRIDDEP_LAUNCH() asm volatile("" ::: "memory")
#else
#define INGINE_GRIDDEP_WAIT() asm volatile("griddepcontrol.wait;" ::: "memory")
#define INGINE_GRIDDEP_LAUNCH() asm volatile("griddepcontrol.launch_dependents;" ::: "memory")
#endif
