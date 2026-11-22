# ingine on one RTX 4090

This Linux port serves Inkling 2 Mini through a native C++/CUDA engine built for **SM89**. On the test machine all routed experts, the target KV cache, and the DSpark drafter fit on one **24 GB RTX 4090 at 262,144 total context tokens**. The optional image encoder runs on the CPU.

## Install and run

```bash
git clone https://github.com/establishmyass/ingine-4090.git
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
| Prefill chunk | 2,048 tokens |
| Images | CPU vision tower, AVX-512 BF16 capable CPU |

The original model is served as `inkling-2-mini`. `PORT`, `HOST`, `INGINE_MODEL_DIR` and `INGINE_CTX` override the local settings. Stop other model servers before using the full window. For example, `INGINE_CTX=65536 ./scripts/ingine serve` frees memory on a busier desktop. On CPUs supporting the expert offload path, `./scripts/ingine serve --offload-experts 4` is a lower-VRAM option.

## Measurements

Measured on one RTX 4090, a Ryzen 9 7950X3D and CUDA 13.5. Each server was already loaded. Time to first token includes reasoning. Cold documents contained three passcodes placed at 2%, 50% and 96% depth. The supplied ablation was active.

| Workload | Initial 4090 port (`172c767`) | This build |
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

`bash setup.sh` opens the installer menu for building with a private CUDA toolkit or downloading the pinned weights into a fresh checkout. For the native tests, stop the server first, then run `ctest --test-dir native/build-sm89 --output-on-failure -E bench`. The optional trace test uses external chat fixtures. The updated build passed the upstream speculative fidelity gate. Nine of 12,000 emitted tokens exceeded a 0.5-logit gap from width-one greedy decoding, compared with three on the initial 4090 port. The upstream limit is 15 per 10,000.

## Port lineage

Based on [skyne98’s RTX 3090 adaptation](https://github.com/skyne98/inkling-ingine-3090/tree/616ee271282dd84417e1b38dea69825c4fe341d5) of [satellitedown’s RTX 5090 engine](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine). This branch uses the 3090 adaptation’s cluster-free kernels, prompt checkpoints and CPU vision implementation. Ada uses hardware E4M3-to-FP16 conversion and the portable E2M1 path. Its eight KV heads use 32 decode splits for one row and 16 for verification, sized for the 4090’s 128 SMs. At 131,064 tokens the eight-row full-attention kernel took 316 µs, down from 493 µs with the 3090 split.

The upstream model and license notices are in [NOTICE](NOTICE). The [3090 port notes](PORT-3090.md), [expert offload guide](EXPERT-OFFLOAD.md) and [image guide](IMAGE-INPUT.md) document the shared implementation.

This port also incorporates satellitedown's tensor-core DSpark attention, FP8 draft projections and global draft KV ring, adapted to the 3090 branch's host checkpoint layout. On SM89, resident expert batches up to 128 prompt tokens use a 32-row tile derived from the upstream grouped expert GEMM. Full chunks use the 3090 branch's warp-specialized kernel. [Upstream DSpark implementation](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine/commit/57b2dc74777205718e76fe121beeb9433898251c) · [Upstream prefill implementation](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine/commit/58377f443e0b987c3057cce3b0671557ddd0c324)
