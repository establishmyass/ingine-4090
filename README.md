# ingine on one RTX 4090

This Linux port serves Inkling 2 Mini through a native C++/CUDA engine built for **SM89**. On the test machine all routed experts, the target KV cache, and the DSpark drafter fit on one **24 GB RTX 4090 at 262,144 total context tokens**. The optional image encoder runs on the CPU.

## Install and run

```bash
git clone --branch perf/4090-latency https://github.com/establishmyass/ingine-4090.git
cd ingine-4090
bash setup.sh
```

Choose option 1 to build the engine and download the model. After setup, run `./scripts/ingine serve` for the abliterated model or `INGINE_ABLATION= ./scripts/ingine serve` for the original.

Use an **OpenAI-compatible** client with base URL `http://127.0.0.1:8080/v1` and model ID `inkling-2-mini-abliterated`. Stop a foreground server with **Ctrl-C**. For background use, run `./scripts/ingine start`, `./scripts/ingine status` and `./scripts/ingine stop`. The server accepts one request at a time, serves chat, reasoning, streaming, base64 images and tools, and defaults to loopback with no authentication.

| Setting | Local default |
|---|---|
| Model | Pinned upstream checkpoint and supplied ablation file |
| Context | 262,144 input and output tokens combined |
| Dense weights / full-attention KV | FP8 / FP8 |
| Routed experts | Original mixed NVFP4 / MXFP8, resident on GPU |
| Speculation | Native DSpark, confidence threshold 0.8 |
| Prefill chunk | 4,096 tokens |
| Displaced conversations | Up to 9,216 MiB of pinned system RAM, allocated on demand |
| Images | CPU vision tower, AVX-512 BF16 capable CPU |

The original model is served as `inkling-2-mini`. `PORT`, `HOST`, `INGINE_MODEL_DIR` and `INGINE_CTX` override the local settings. Stop other model servers before using the full window. For example, `INGINE_CTX=65536 ./scripts/ingine serve` frees memory on a busier desktop. On CPUs supporting the expert offload path, `./scripts/ingine serve --offload-experts 4` is a lower-VRAM option.

## Performance branch

The 4090 profile shares prefill workspace between operations that run at different times and uses larger chunks. For full-attention KV, verification reads two FP8 tiles ahead of the value computation. The dense gates reuse their inputs across verified tokens.

When another chat displaces a prompt of at least 2,048 tokens, the server can save its full-attention KV and checkpoint in system RAM. Returning to a matching token and image prefix restores it. The first switch pays for allocating pinned memory and transferring the prefix. The server can reuse that allocation on subsequent switches. Set the budget with `./scripts/ingine serve --park-host-mib 4096`, or use `--park-host-mib 0` to disable conversation parking. The server log reports `reuse=parked` and the number of tokens restored. To recover desktop VRAM, use `--prefill-chunk 2048`.

The verification kernels and full-row checkpoint copies are adapted from satellitedown's original engine. [Narrow verification](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine/commit/ac5f704e176b387ea8c1d377b332887a2344e722) · [FP8 attention](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine/commit/866e908c9f0eaefb238ddcd205349ac026a77bed) · [Conversation parking](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine/commit/8210872574b9607f59639da49bb483faf5b37523)

| Workload | Published `rtx4090` | Performance branch |
|---|---:|---:|
| First token, 33,018 fresh tokens | 3.23 s | 3.04 s |
| First token, 251,504 fresh tokens | 35.27 s | 33.89 s |
| Fixed-output decode, 2,914 timed tokens across eight workloads | 271.1 tok/s | 275.0 tok/s |
| Fixed-output decode after 130,351 input tokens, 194 timed tokens | 226.2 tok/s | 233.1 tok/s |
| First token on return to the same 251,529-token chat after another chat | 35.33 s | 0.19 s |

On that first switch, the intervening short request took 1.67 s on the branch versus 43 ms on the published build. Repeating the switch with the reusable host allocation took 0.21 s to leave and 0.19 s to return. The final launcher also retrieved all three codes from 261,145 fresh tokens in 35.84 s to first output. After loading, the desktop had 629 MiB of free VRAM. The cold-chat numbers are individual requests. Both decode comparisons replay identical output IDs across engines.

Using 4,096-token chunks, the speculative fidelity check recorded 3 of 12,000 emitted tokens more than 0.5 logits below the width-one greedy argmax (2.50 per 10,000, limit 15). The maximum gap was 1.4407. Kernel selection and chunking can change floating-point rounding and generated text.

## Published baseline measurements

Measured on one RTX 4090, a Ryzen 9 7950X3D and CUDA 13.5. Each server was already loaded. Time to first token includes reasoning. Cold documents contained three passcodes placed at 2%, 50% and 96% depth. The supplied ablation was active.

| Workload | Initial 4090 port (`172c767`) | Published `rtx4090` baseline |
|---|---:|---:|
| Short chat, median first token across nine requests | 57.3 ms | 45.4 ms |
| 33,018-token cold prompt, first token | 3.25 s | 3.23 s |
| 251,504-token cold prompt, first token | 35.40 s | 35.31 s |
| Decode after that prompt, same 249-token output | 224.5 tok/s | 228.8 tok/s |
| Same chat, 25-token follow-up, first token | 150 ms | 141 ms |
| Fixed-output replay, 2,914 timed tokens over eight workloads | 266.8 tok/s | 270.8 tok/s |

The near-limit test recovered all three codes from 261,145 fresh tokens. It reached first output in 37.24 s and decoded at 220.3 tok/s. Baseline figures average two runs, and the updated column is one final run. Speculative acceptance varies with the text. After startup, the card reported 1,069 MiB free. Both original and ablated launchers answered a sampled chat request. The ablated launcher also passed the image and tool checks.

## Rebuild

The build needs a CUDA 13 toolkit, CMake 3.28+, Ninja and a C++20 compiler accepted by nvcc. Python 3 at configure time writes the shell launch profile. This is the source-build command used on the test host:

```bash
cmake -S native -B native/build-sm89 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_CXX_COMPILER=g++-15 \
  -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-15 -DINGINE_BUILD_TESTS=ON
cmake --build native/build-sm89 --parallel 12
python3 scripts/download_model.py --verify-only
./scripts/ingine serve
```

`bash setup.sh` opens the installer menu for building with a private CUDA toolkit or downloading the pinned weights into a fresh checkout. For the native tests, stop the server first, then run `ctest --test-dir native/build-sm89 --output-on-failure -E bench`. The optional trace test uses external chat fixtures. The published baseline recorded nine of 12,000 emitted tokens more than 0.5 logits below the width-one greedy argmax. The upstream gate allows 15 per 10,000.

## Port lineage

Based on [skyne98’s RTX 3090 adaptation](https://github.com/skyne98/inkling-ingine-3090/tree/616ee271282dd84417e1b38dea69825c4fe341d5) of [satellitedown’s RTX 5090 engine](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine). This branch uses the 3090 adaptation’s cluster-free kernels, prompt checkpoints and CPU vision implementation. Ada uses hardware E4M3-to-FP16 conversion and the portable E2M1 path. Its eight KV heads use 32 decode splits for one row and 16 for verification, sized for the 4090’s 128 SMs. At 131,064 tokens the eight-row full-attention kernel took 316 µs, down from 493 µs with the 3090 split.

The upstream model and license notices are in [NOTICE](NOTICE). The [3090 port notes](PORT-3090.md), [expert offload guide](EXPERT-OFFLOAD.md) and [image guide](IMAGE-INPUT.md) document the shared implementation.

This port also incorporates satellitedown's tensor-core DSpark attention, FP8 draft projections and global draft KV ring, adapted to the 3090 branch's host checkpoint layout. On SM89, resident expert batches up to 128 prompt tokens use a 32-row tile derived from the upstream grouped expert GEMM. Full chunks use the 3090 branch's warp-specialized kernel. [Upstream DSpark implementation](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine/commit/57b2dc74777205718e76fe121beeb9433898251c) · [Upstream prefill implementation](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine/commit/58377f443e0b987c3057cce3b0671557ddd0c324)
