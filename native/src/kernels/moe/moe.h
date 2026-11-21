// Routed-expert kernels (the PyTorch extension's kernels ported to raw-pointer launchers; contracts in
// native/KERNELS.md, area `moe`) and the expert decode to BF16. Launchers run on `stream`, never
// allocate or synchronise, read routing from device memory and are CUDA-graph capturable. Shapes are
// row-major (Python order); H = 2304, I = 1152, 128 experts, top-8. Invalid arguments throw
// std::invalid_argument.
#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace ingine::kernels::moe {

enum class Format : int { NVFP4 = 0, MXFP8 = 1 };

inline constexpr int kExperts = 128;
inline constexpr int kHidden = 2304;
inline constexpr int kMaxSlots = 64;  // M * 8 routed slots per call, M <= 8

// Expert offloading (offload/expert_store.h). Per layer, in device memory: where each expert's four checkpoint
// tensors live - a VRAM slot, or pinned host memory mapped into the device address space - and whether the
// expert is in VRAM. The tables change only between decode steps (stream-ordered updates), never during one.
struct ExpertTable {
  const std::uint8_t* w13[kExperts];   // codes of the expert's [2I, H] W13 rows
  const std::uint8_t* w13s[kExperts];  // their block scales
  const std::uint8_t* w2[kExperts];    // codes of the [H, I] W2 rows
  const std::uint8_t* w2s[kExperts];
  std::int32_t hot[kExperts];          // 1: in VRAM
};

// Decode hand-off of the experts that are not in VRAM to the CPU expert pool (offload/cpu_pool.h).
// cold_publish (after the router) lists a visit's cold slots and copies x into the doorbell, then bumps
// `seq` last; the pool computes those experts' outputs into `y` and stores `done = seq`; moe_down computes
// the hot experts meanwhile and waits for `done` only before its weighted sum.
struct ColdBell {                       // host memory, written by the GPU
  volatile std::uint32_t seq;
  std::uint32_t visit, layer, M, n;     // n cold slots
  std::int32_t slot[kMaxSlots];         // routed slot (token * 8 + k) of each cold entry
  std::int32_t expert[kMaxSlots];
  __nv_bfloat16 x[8][kHidden];          // the visit's MoE input rows (moe_up's x)
};
struct ColdExchange {
  ColdBell* bell = nullptr;                       // device alias of the host-mapped doorbell
  const volatile std::uint32_t* done = nullptr;   // host-mapped: last seq the pool finished
  const __nv_bfloat16* y = nullptr;               // host-mapped [kMaxSlots, H]: CPU outputs by routed slot
  std::uint32_t* dev = nullptr;                   // device [2]: seq counter; seq the current visit waits for
  std::uint32_t* counts = nullptr;                // device [24, 128]: decode routing counts (placement)
  volatile std::uint32_t* err = nullptr;          // host-mapped: 1 once a wait timed out
  bool cpu = false;                               // cold experts on the CPU (else read from host memory)
};

// cold_publish: count the visit's selections (ex.counts[layer]) and, with ex.cpu, hand its cold slots to
// the CPU pool. x bf16 [M, H], idx int32 [M, 8]. One small kernel; capturable.
void cold_publish(const ColdExchange& ex, const ExpertTable* table, const __nv_bfloat16* x, const int* idx, int M,
                  int visit, int layer, cudaStream_t stream);
// apply_table_updates: n entries {layer, expert, slot pointers or host pointers, hot} into tables [24]
// (stream-ordered between decode steps).
struct TableUpdate {
  std::int32_t layer, expert, hot;
  const std::uint8_t *w13, *w13s, *w2, *w2s;
};
inline constexpr int kMaxTableUpdates = 32;
struct TableUpdates {
  int n = 0;
  TableUpdate u[kMaxTableUpdates];
};
void apply_table_updates(ExpertTable* tables, const TableUpdates& updates, cudaStream_t stream);

// moe_up: W13 GEMV of the selected experts + softcapped SwiGLU. x bf16 [M, H], idx int32 [M, 8],
// out bf16 [M, 8, I]. NVFP4: w13 u8 [128, 2I, H/2], w13_scale e4m3 [128, 2I, H/16], w13_factor fp32 [128];
// MXFP8: w13 e4m3 [128, 2I, H], w13_scale u8 E8M0 [128, 2I, H/32], factor unused (may be null).
// pdl: the kernel waits for its predecessor before reading anything.
// table: expert placement (w13 / w13_scale ignored); skip_cold: experts not in VRAM are left to the CPU
// pool (their act rows are not written).
void moe_up(const __nv_bfloat16* x, int M, const int* idx, const void* w13, const void* w13_scale,
            const float* w13_factor, Format fmt, __nv_bfloat16* out, bool pdl, cudaStream_t stream,
            const ExpertTable* table = nullptr, bool skip_cold = false);

// moe_down: W2 GEMV + routing-weighted sum + post-MLP conv with state update. act bf16 [M, 8, I],
// wts fp32 [M, 8], w2 [128, H, I(/2)] + scales [128, H, I/16 | I/32] (+ factor), conv_w fp32 [H, 4],
// conv_state fp32 [3, H], out fp32 [M, H], conv_hist fp32 [M, H] (BF16-rounded conv inputs) or null.
// pdl: valid only right after moe_up (idx is read before the PDL wait).
// table: expert placement (w2 / w2_scale ignored); cold (with cold->cpu): experts not in VRAM come from the
// CPU pool's outputs, waited for before the weighted sum.
void moe_down(const __nv_bfloat16* act, int M, const int* idx, const float* wts, const void* w2,
              const void* w2_scale, const float* w2_factor, Format fmt, const float* conv_w, float* conv_state,
              float* out, float* conv_hist, bool pdl, cudaStream_t stream, const ExpertTable* table = nullptr,
              const ColdExchange* cold = nullptr);

// dequant_experts: decode experts[0..n) (int32 device) of one weight ([128, rows, cols] in checkpoint
// encoding: NVFP4 codes u8 [128, rows, cols/2] + e4m3 scales [128, rows, cols/16] + factor fp32 [128];
// MXFP8 codes e4m3 [128, rows, cols] + E8M0 scales [128, rows, cols/32]) into out bf16 [n, rows, cols],
// exactly as model.decode_experts (FP32 product, one BF16 rounding).
void dequant_experts(const void* codes, const void* scales, const float* factor, const std::int32_t* experts, int n,
                     int rows, int cols, Format fmt, __nv_bfloat16* out, cudaStream_t stream);

}  // namespace ingine::kernels::moe
