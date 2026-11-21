#!/usr/bin/env python3
"""Expert-offloading study for ingine on an RTX 3090: routing skew, static and adaptive (decayed-LFU) expert
placement hit rates, and a decode-time model, from a routing trace.

Record a trace (plain decode, so one record group per token):
    INGINE_ROUTE_TRACE=/tmp/route.bin native/build-sm86/apps/serve/ingine-serve --engine ...   (send requests,
    then stop the server: the trace is written at shutdown)
Then:
    python3 tools/route_offload_study.py /tmp/route.bin

Placement is learned on the first half of the tokens and scored on the second half. Constants measured on the
development machine (RTX 3090 PCIe 4.0 x16, Ryzen 9 9950X3D, DDR5): RAM read 58 GB/s, PCIe 26 GB/s, GPU MoE time
per visit 62 us (NVFP4) / 97 us (MXFP8), GPU<->CPU hand-off 20 us per visit with a cold expert.
"""
import sys

import numpy as np
rec = np.fromfile(sys.argv[1], dtype=np.uint16)
NV = 48; ntok = len(rec) // (NV * 8); rec = rec[: ntok * NV * 8]
visit = (rec >> 10).astype(np.int64); expert = (rec & 127).astype(np.int64)
LAYER = np.array(list(range(6)) + [6 + (v % 12) for v in range(36)] + list(range(18, 24)))
pair = (LAYER[visit] * 128 + expert).reshape(ntok, NV, 8)
SIZE = np.where(np.arange(24) >= 20, 8211456, 4478976).astype(float); PSIZE = SIZE[np.arange(3072) // 128]
GiB = 2 ** 30; total = PSIZE.sum()
half = ntok // 2; train, test = pair[:half], pair[half:]
# measured on this machine: RAM 58 GB/s (16 threads), PCIe 26 GB/s; GPU MoE time per visit (nsys): 62 us NVFP4,
# 97 us MXFP8; hand-off GPU->CPU->GPU per visit with a miss: 20 us (fngine 13 us at width 1, Strata 5 us + host)
T_BASE = 7.4e-3; GPU_VISIT = np.where(LAYER >= 20, 97e-6, 62e-6); SYNC = 20e-6
def extra_time(resident, data, bw):
    miss = ~resident[data]                                  # [tok, visit, 8]
    mbytes = (PSIZE[data] * miss).sum(axis=2)               # cold bytes per visit
    any_ = miss.any(axis=2)
    # cold experts on the CPU/PCIe overlap the GPU's hot experts within the visit
    gpu_saved = GPU_VISIT[None, :] * miss.sum(axis=2) / 8 * 0.5  # GPU skips cold rows (half of time is fixed cost)
    t = np.where(any_, np.maximum(0, mbytes / bw + SYNC - (GPU_VISIT[None, :] - gpu_saved)), 0)
    return t.sum(axis=1).mean(), 1 - miss.mean()
def static(train, budget):
    cnt = np.bincount(train.ravel(), minlength=3072).astype(float)
    order = np.argsort(-cnt / PSIZE, kind="stable"); res = np.zeros(3072, bool)
    res[order[np.cumsum(PSIZE[order]) <= budget]] = True
    return res
def adaptive_hit(train, data, budget, every=4, decay=0.7, ratio=1.5, max_swaps=96):
    # Strata-style decayed LFU: start from the static set, every `every` tokens swap in cold experts used
    # >= 2x that beat the least-used resident one by `ratio` (same byte class), at most max_swaps per round.
    res = static(train, budget); cnt = np.zeros(3072); hits = 0; n = 0; swapped = 0
    for t in range(len(data)):
        sel = data[t].ravel(); hits += res[sel].sum(); n += len(sel)
        np.add.at(cnt, sel, 1.0)
        if (t + 1) % every == 0:
            for cls in (False, True):
                m = (np.arange(3072) // 128 >= 20) == cls
                cold = np.where(m & ~res & (cnt >= 2))[0]; hot = np.where(m & res)[0]
                if len(cold) == 0 or len(hot) == 0: continue
                cold = cold[np.argsort(-cnt[cold])][:max_swaps]; hot = hot[np.argsort(cnt[hot])][:len(cold)]
                ok = cnt[cold] > ratio * cnt[hot] + 1e-9
                res[cold[ok]] = True; res[hot[ok]] = False; swapped += ok.sum()
            cnt *= decay
    return hits / n, swapped / len(data)
print(f"held-out tokens {len(test)}; baseline {T_BASE*1e3:.1f} ms/token = {1/T_BASE:.0f} tok/s")
print(" offload   static hit   CPU 58 GB/s        CPU+PCIe 84 GB/s     adaptive hit (swaps/token)")
for off in [0.5, 1, 2, 3, 4, 6]:
    res = static(train, total - off * GiB)
    e1, hit = extra_time(res, test, 58e9); e2, _ = extra_time(res, test, 84e9)
    ah, sw = adaptive_hit(train, test[:3000], total - off * GiB)
    print(f" {off:4.1f} GiB   {hit:.4f}     +{e1*1e3:4.2f} ms -> {1/(T_BASE+e1):5.0f} tok/s   "
          f"+{e2*1e3:4.2f} ms -> {1/(T_BASE+e2):5.0f} tok/s     {ah:.4f} ({sw:.1f})")

def adaptive_time(train, data, budget, bw, every=4, decay=0.7, ratio=1.5, max_swaps=96):
    res = static(train, budget); cnt = np.zeros(3072); extra = 0.0
    for t in range(len(data)):
        sel = data[t]; miss = ~res[sel]
        if miss.any():
            mbytes = (PSIZE[sel] * miss).sum(axis=1); any_ = miss.any(axis=1)
            gpu = GPU_VISIT - GPU_VISIT * miss.sum(axis=1) / 8 * 0.5
            extra += np.where(any_, np.maximum(0, mbytes / bw + SYNC - gpu), 0).sum()
        np.add.at(cnt, sel.ravel(), 1.0)
        if (t + 1) % every == 0:
            for cls in (False, True):
                m = (np.arange(3072) // 128 >= 20) == cls
                cold = np.where(m & ~res & (cnt >= 2))[0]; hot = np.where(m & res)[0]
                if len(cold) == 0 or len(hot) == 0: continue
                cold = cold[np.argsort(-cnt[cold])][:max_swaps]; hot = hot[np.argsort(cnt[hot])][:len(cold)]
                ok = cnt[cold] > ratio * cnt[hot] + 1e-9
                res[cold[ok]] = True; res[hot[ok]] = False
            cnt *= decay
    return extra / len(data)
print("\nadaptive placement, all held-out tokens:")
for off in [1, 2, 3, 4, 6]:
    e1 = adaptive_time(train, test, total - off * GiB, 58e9); e2 = adaptive_time(train, test, total - off * GiB, 84e9)
    print(f" {off:4.1f} GiB   CPU: {1/(T_BASE+e1):5.0f} tok/s   CPU+PCIe: {1/(T_BASE+e2):5.0f} tok/s")

c = np.bincount(pair.ravel(), minlength=3072).astype(float).reshape(24, 128)
cs = -np.sort(-c, axis=1); share = np.cumsum(cs, axis=1) / cs.sum(axis=1, keepdims=True)
print(f"\nper-layer share of selections held by the top 32 / 64 / 96 experts: "
      f"{share[:, 31].mean():.3f} / {share[:, 63].mean():.3f} / {share[:, 95].mean():.3f}; unused experts: {(c == 0).sum()}")
