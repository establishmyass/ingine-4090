// Optional routing trace for expert-placement studies (RTX 3090 port, expert offloading analysis).
//
// With INGINE_ROUTE_TRACE=<file> set when the decode graphs are captured, every router call is followed by
// a one-block kernel that appends its M x 8 selections to a device log; Runtime's destructor writes the log
// to <file>: little-endian uint16 records (visit << 10) | (token << 7) | expert, one call's records
// contiguous, calls in stream order. Without the variable nothing is launched or allocated beyond the log
// symbol. Prefill routing is not traced.
#include "kernels/dense/dense.h"

#include "core/device.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace ingine::kernels::dense {
namespace {

constexpr unsigned kCap = 8u << 20;  // records: ~21K plain decode tokens (48 visits x 8)
__device__ std::uint16_t g_trace[kCap];
__device__ unsigned g_cursor;

__global__ void route_trace_kernel(const int* __restrict__ idx, int n, int visit) {
  __shared__ unsigned base;
  if (threadIdx.x == 0) base = atomicAdd(&g_cursor, unsigned(n));
  __syncthreads();
  const int t = threadIdx.x;
  if (t < n && base + t < kCap)
    g_trace[base + t] = std::uint16_t((visit << 10) | ((t / 8) << 7) | (idx[t] & 127));
}

}  // namespace

const char* route_trace_path() {
  const char* p = std::getenv("INGINE_ROUTE_TRACE");
  return p != nullptr && *p != '\0' ? p : nullptr;
}

void route_trace(const int* idx, int M, int visit, cudaStream_t stream) {
  route_trace_kernel<<<1, 64, 0, stream>>>(idx, M * 8, visit);
  CUDA_CHECK(cudaGetLastError());
}

void route_trace_dump() {
  const char* path = route_trace_path();
  if (path == nullptr) return;
  unsigned n = 0;
  if (cudaDeviceSynchronize() != cudaSuccess || cudaMemcpyFromSymbol(&n, g_cursor, sizeof n) != cudaSuccess) return;
  if (n > kCap) {
    std::fprintf(stderr, "route trace: log full, %u of %u records kept\n", kCap, n);
    n = kCap;
  }
  std::vector<std::uint16_t> host(n);
  if (n > 0 && cudaMemcpyFromSymbol(host.data(), g_trace, n * sizeof(std::uint16_t)) != cudaSuccess) return;
  if (std::FILE* f = std::fopen(path, "wb")) {
    std::fwrite(host.data(), sizeof(std::uint16_t), n, f);
    std::fclose(f);
    std::fprintf(stderr, "route trace: %u records -> %s\n", n, path);
  }
}

}  // namespace ingine::kernels::dense
