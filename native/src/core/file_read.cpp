#include "core/file_read.h"

#include "core/safetensors.h"

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <atomic>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace ingine {
namespace {

constexpr std::uint64_t kPiece = std::uint64_t(16) << 20;

std::uint64_t align_down(std::uint64_t x) { return x & ~std::uint64_t(kDirectAlign - 1); }
std::uint64_t align_up(std::uint64_t x) { return (x + kDirectAlign - 1) & ~std::uint64_t(kDirectAlign - 1); }

void pread_full(int fd, std::byte* dst, std::uint64_t bytes, std::uint64_t offset, std::uint64_t need,
                const std::string& path) {
  std::uint64_t done = 0;
  while (done < bytes) {
    const ssize_t n = ::pread(fd, dst + done, std::size_t(bytes - done), off_t(offset + done));
    if (n < 0) {
      if (errno == EINTR) continue;
      throw std::runtime_error("read of " + path + " failed: " + std::strerror(errno));
    }
    if (n == 0) break;  // end of file: the aligned tail of the last range may run past it
    done += std::uint64_t(n);
  }
  if (done < need) throw std::runtime_error("short read of " + path);
}

}  // namespace

double cached_fraction(std::span<const ReadRange> ranges) {
  std::uint64_t total = 0, resident = 0;
  const long page = ::sysconf(_SC_PAGESIZE);
  std::map<const SafetensorsFile*, std::pair<void*, std::size_t>> maps;
  for (const ReadRange& r : ranges) {
    if (r.bytes == 0) continue;
    auto it = maps.find(r.file);
    if (it == maps.end()) {
      const std::size_t len = std::size_t(r.file->file_size());
      void* p = ::mmap(nullptr, len, PROT_READ, MAP_SHARED, r.file->fd(), 0);
      it = maps.emplace(r.file, std::make_pair(p == MAP_FAILED ? nullptr : p, len)).first;
    }
    total += r.bytes;
    if (it->second.first == nullptr) continue;
    const std::uint64_t a = r.offset & ~std::uint64_t(page - 1), z = r.offset + r.bytes;
    std::vector<unsigned char> vec(std::size_t((z - a + std::uint64_t(page) - 1) / std::uint64_t(page)));
    if (::mincore(static_cast<char*>(it->second.first) + a, std::size_t(z - a), vec.data()) != 0) continue;
    std::uint64_t n = 0;
    for (unsigned char c : vec) n += c & 1u;
    resident += std::min<std::uint64_t>(n * std::uint64_t(page), r.bytes);
  }
  for (auto& [f, m] : maps)
    if (m.first) ::munmap(m.first, m.second);
  return total ? double(resident) / double(total) : 1.0;
}

ReadStats read_ranges(std::span<const ReadRange> ranges, int threads) {
  ReadStats st;
  const auto t0 = std::chrono::steady_clock::now();
  st.cached = cached_fraction(ranges);
  bool direct = st.cached < 0.5;
  if (const char* v = std::getenv("INGINE_LOAD_DIRECT"); v && *v) direct = *v == '1';

  struct Piece {
    std::size_t range;
    std::uint64_t offset, bytes;  // file range of this piece (aligned in direct mode)
    std::byte* dst;
    bool direct;
  };
  std::vector<Piece> pieces;
  for (std::size_t i = 0; i < ranges.size(); ++i) {
    const ReadRange& r = ranges[i];
    st.bytes += r.bytes;
    const bool aligned = reinterpret_cast<std::uintptr_t>(r.dst) % kDirectAlign == r.offset % kDirectAlign;
    if (direct && aligned) {
      const std::uint64_t a = align_down(r.offset), z = align_up(r.offset + r.bytes);
      std::byte* base = r.dst - (r.offset - a);
      for (std::uint64_t o = a; o < z; o += kPiece) pieces.push_back({i, o, std::min(kPiece, z - o), base + (o - a), true});
    } else {
      for (std::uint64_t o = 0; o < r.bytes; o += kPiece)
        pieces.push_back({i, r.offset + o, std::min(kPiece, r.bytes - o), r.dst + o, false});
    }
  }
  std::map<const SafetensorsFile*, int> direct_fd;
  if (direct)
    for (const ReadRange& r : ranges)
      if (!direct_fd.count(r.file)) {
        const int fd = ::open(r.file->path().c_str(), O_RDONLY | O_DIRECT | O_CLOEXEC);
        direct_fd[r.file] = fd;  // -1: this file reads buffered
      }
  std::atomic<std::size_t> next{0};
  std::exception_ptr err;
  std::mutex mu;
  std::vector<std::thread> pool;
  for (int k = 0; k < std::max(1, threads); ++k)
    pool.emplace_back([&] {
      for (std::size_t j; (j = next++) < pieces.size();) {
        const Piece& p = pieces[j];
        const ReadRange& r = ranges[p.range];
        try {
          const int dfd = p.direct ? direct_fd[r.file] : -1;
          if (dfd >= 0) {
            // the bytes of the range inside this piece must arrive; the aligned tail may hit end of file
            const std::uint64_t real_end = std::min(p.offset + p.bytes, r.offset + r.bytes);
            pread_full(dfd, p.dst, p.bytes, p.offset, real_end > p.offset ? real_end - p.offset : 0, r.file->path());
          } else if (p.direct) {  // O_DIRECT unavailable for this file: the range's bytes only, buffered
            const std::uint64_t a = std::max(p.offset, r.offset), z = std::min(p.offset + p.bytes, r.offset + r.bytes);
            if (z > a) r.file->read_range(a, z - a, r.dst + (a - r.offset));
          } else {
            r.file->read_range(p.offset, p.bytes, p.dst);
          }
        } catch (...) {
          std::lock_guard<std::mutex> lk(mu);
          if (!err) err = std::current_exception();
        }
      }
    });
  for (auto& t : pool) t.join();
  for (auto& [f, fd] : direct_fd)
    if (fd >= 0) ::close(fd);
  if (err) std::rethrow_exception(err);
  st.direct = direct;
  st.seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
  return st;
}

void prefault(std::byte* p, std::size_t bytes, int threads) {
  const std::size_t page = std::size_t(::sysconf(_SC_PAGESIZE));
  std::vector<std::thread> pool;
  const int n = std::max(1, threads);
  for (int k = 0; k < n; ++k)
    pool.emplace_back([=] {
      const std::size_t a = bytes * std::size_t(k) / std::size_t(n) / page * page;
      const std::size_t z = bytes * std::size_t(k + 1) / std::size_t(n);
      for (std::size_t o = a; o < z; o += page) *reinterpret_cast<volatile std::byte*>(p + o) = std::byte{0};
    });
  for (auto& t : pool) t.join();
}

}  // namespace ingine
