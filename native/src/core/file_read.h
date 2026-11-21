// Fast bulk reads of file ranges into host memory, for startup (cold time to first token).
//
// A cold read through the page cache runs at the kernel's readahead pace and copies every byte twice; O_DIRECT
// with many large requests in flight reads straight into the destination at the device's speed (an NVMe
// drive needs several parallel streams to reach it). A warm page cache is faster to copy from than to bypass.
// read_ranges picks per call: O_DIRECT when less than half of the bytes are cached (mincore), else buffered
// pread; both on `threads` threads in 16 MiB pieces. $INGINE_LOAD_DIRECT=1 / 0 forces one.
//
// O_DIRECT needs 4 KiB alignment of file offsets, lengths and destinations, so each range is read as its
// aligned superset: the caller places `dst` so that dst % 4096 == offset % 4096 and leaves the bytes between
// align_down(dst) and align_up(dst + bytes) writable (see aligned_layout). Without that the range is read
// buffered.
#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <string>

namespace ingine {

class SafetensorsFile;

struct ReadRange {
  const SafetensorsFile* file = nullptr;
  std::uint64_t offset = 0;  // absolute file offset
  std::uint64_t bytes = 0;
  std::byte* dst = nullptr;
};

inline constexpr std::size_t kDirectAlign = 4096;

// Where to put a range in a buffer filled front to back: the returned offset is >= cursor, congruent to
// `file_offset` modulo 4 KiB, and the next range may start at cursor = returned + bytes.
inline std::size_t aligned_layout(std::size_t cursor, std::uint64_t file_offset) {
  const std::size_t block = (cursor + kDirectAlign - 1) & ~(kDirectAlign - 1);
  return block + std::size_t(file_offset % kDirectAlign);
}

// Fraction of the ranges' bytes in the page cache (0 .. 1).
double cached_fraction(std::span<const ReadRange> ranges);

struct ReadStats {
  std::uint64_t bytes = 0;
  double seconds = 0;
  bool direct = false;
  double cached = 0;  // fraction found in the page cache
};

ReadStats read_ranges(std::span<const ReadRange> ranges, int threads = 16);

// Faults in [p, p + bytes) (writing one byte per page) on `threads` threads: the kernel zeroes fresh anonymous
// pages at fault time, which is cheap in parallel and slow when O_DIRECT faults them one I/O at a time.
void prefault(std::byte* p, std::size_t bytes, int threads = 16);

}  // namespace ingine
