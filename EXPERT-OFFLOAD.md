# Expert offloading to system RAM (RTX 3090 build)

`--offload-experts GIB` keeps GIB of Inkling-2-Mini's routed experts only in system RAM, freeing that much VRAM
for context, spare memory, or other programs. All 3,072 experts (14.6 GiB) are held in pinned host memory; the
most-used ones also sit in a VRAM slot cache that adapts to the conversation; the rest run on the CPU during
decode and are streamed to the GPU during long prefills. The design follows Strata's expert cache and pipelining
and fngine's hot/cold tiers (see NOTICE).

```bash
ingine-serve --engine --model-dir models/Inkling-2-Mini --ablation models/Inkling-2-Mini/inkling-2-mini-abliterated.safetensors \
  --max-ctx 262144 --dense-fp8 --kv-fp8 --spec --spec-threshold 0.8 --offload-experts 4
```

That is upstream's full serving profile (256K context, DSpark) on a 24 GB RTX 3090: 20.5 GB peak for the whole
card including ~1.5 GB of desktop.

## Results (RTX 3090, Ryzen 9 9950X3D, DDR5; `ingine-bench`, greedy, 32K context unless noted)

| Offloaded | Engine VRAM | Plain decode | DSpark decode (chat / code) | Prefill, 30K-token prompt |
|---:|---:|---:|---:|---:|
| 0 | 18.2 GiB | 131 tok/s | 191 / 247 tok/s | 4,350 tok/s |
| 2 GiB | 16.1 GiB | 127 tok/s | | |
| 4 GiB | 14.1 GiB | 116 tok/s | 152-167 / 179-181 tok/s | 3,327 tok/s (3,821 with 1 GiB staging) |
| 6 GiB | 12.0 GiB | 98 tok/s | | |

- Cold experts read by the GPU over PCIe instead of the CPU (`--offload-threads 0`): 85 tok/s at 4 GiB.
- A fixed placement instead of the adaptive cache (`--offload-static`): 97 tok/s at 4 GiB.
- With ~0 GiB offloaded the machinery costs nothing (132 tok/s): the cost is the CPU hand-offs.
- At 4 GiB, a decode token hands about 5.6 layer visits (9 expert slots) to the CPU, ~180 µs each, while the GPU
  needs ~60 µs for the hot experts of the visit.

## How it works

**Placement.** The starting VRAM set ranks experts by routing count per byte from a built-in profile
(`src/offload/default_profile.inc`: 18.5K tokens of 24 varied requests, recorded with `INGINE_ROUTE_TRACE`).
During decode the cache adapts like Strata's adaptive tier: every 4 steps (step-locked) the GPU's routing
counts come back; per expert class (NVFP4, MXFP8: different slot sizes) a cold expert scoring >= 2 replaces the
least-scored resident one when it beats it by 1.5x, then scores decay by 0.7. A swap copies the expert into a
spare VRAM slot on a copy stream; two steps later the compute stream waits for the copy on the GPU and flips both
placement-table entries. The decode graphs never stop and the host never waits for a copy.

**Kernels.** `moe_up` / `moe_down` and the prefill expert GEMMs address experts through a per-layer
`ExpertTable` (pointer per expert: VRAM slot or pinned host memory, plus a hot flag). In decode, after the router
`cold_publish` lists the visit's cold slots and copies its input rows into a mapped doorbell, bumping a sequence
number last (fenced to system scope). `moe_up` skips cold experts; `moe_down` computes the hot ones, then waits
for the CPU's sequence number (backed-off polling, 2 s timeout that raises an error flag) before its weighted sum,
reading the CPU's BF16 expert outputs from mapped memory.

**CPU side.** A leader thread spins on the doorbell and runs the cold experts' W13 rows, then their W2 rows, on a
pool of pinned threads (one physical core each, alternating CCDs) that claim row slices with Strata's epoch-tagged
CAS protocol. The AVX-512 kernels (`src/offload/cpu_experts.cpp`) decode NVFP4 / MXFP8 to the exact BF16 weights
the GPU uses and multiply with VDPBF16PS (BF16 activations, FP32 accumulation): against the GPU kernels on real
experts, 0.01-0.04% of activations differ, by a summation-order rounding flip (`tests/offload`). One NVFP4
expert takes ~76 µs on 8 threads alone (59 GB/s of DRAM); during decode, ~100-110 µs.

**Prefill.** For chunks of >= 1,024 tokens, each layer visit's cold experts are copied into one of two VRAM
staging buffers (384 MiB together by default) while the previous visit computes; experts that do not fit and
shorter chunks read host memory over PCIe. Prefill is bit-identical to the all-in-VRAM engine.

## Options

| Flag / variable | Default | |
|---|---|---|
| `--offload-experts GIB` / `INGINE_OFFLOAD_GIB` | 0 (off) | expert bytes kept only in RAM |
| `--offload-threads N` / `INGINE_OFFLOAD_THREADS` | 8 | CPU pool threads (0: the GPU reads cold experts over PCIe) |
| `--offload-static` / `INGINE_OFFLOAD_ADAPTIVE=0` | adaptive | keep the starting placement |
| `INGINE_OFFLOAD_STAGE_MIB` | 384 | prefill staging buffers (0: no staging) |
| `INGINE_OFFLOAD_CPUS` | automatic | CPU list for the pool, e.g. `1-7,9-15` (first: the leader) |
| `INGINE_OFFLOAD_SPARES`, `_SWAP_EVERY`, `_DECAY`, `_RATIO`, `_UNIFORM`, `_SPIN_US` | 4, 4, 0.7, 1.5, 0, 3000 | cache tuning |
| `INGINE_OFFLOAD_SELFTEST=1` | | log the CPU pool's speed at startup |
| `INGINE_ROUTE_TRACE=FILE` | | record decode routing (`tools/route_offload_study.py`) |

Startup reads all 14.6 GiB of experts into pinned RAM (about 15 GiB stays pinned while the server runs): ~3.7 s
from disk (O_DIRECT, 6.5 GB/s), ~1.5 s from the page cache; boot to ready 5.8-6.1 s cold, 3.8 s warm (see
PORT-3090.md, Startup). Short prompts' first token takes ~240 ms (75 ms without offloading): their prefill reads
the cold experts they use over PCIe.

## Caveats

- **Reproducibility.** CPU and GPU compute an expert with different FP32 summation orders, so outputs depend on
  where each expert ran. With the adaptive cache that depends on the engine's request history: a fresh engine
  given the same requests reproduces its outputs, but replaying a request later, or restoring a prompt checkpoint,
  can differ in near-tie tokens (as with DSpark's wider kernels). `--offload-static` is bit-reproducible across
  requests. Under offloading, `model_reuse` and `spec_forced` therefore fail with the adaptive cache and pass with
  `INGINE_OFFLOAD_ADAPTIVE=0`; every other test passes either way (parity: KL mean 2.5e-3, top-1 0.964).
- **CPU load.** During decode the pool spins on 8 cores (3 ms after the last hand-off, then sleeps).
- **Requirements.** AVX-512 F/BW/VL/DQ + AVX512_BF16 for the CPU path (Zen 4/5, recent Xeons); otherwise use
  `--offload-threads 0`.

## Where the time goes, and what could still help

The CPU reaches ~60% of DRAM bandwidth during decode (vs 59 GB/s alone): the hand-off and memory-system
interference with the running GPU. Ideas not done: splitting large hand-offs between the CPU and GPU PCIe reads
(Strata's `--pcie-frac`), prefill staging sized per layer instead of a fixed buffer, persisting the learned
placement across restarts (Strata's `--expert-profile-save`).
