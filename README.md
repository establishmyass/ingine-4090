# ingine on one RTX 4090

This local Linux port serves Inkling 2 Mini through a native C++/CUDA engine built for **SM89**. On the test machine all routed experts, the target KV cache, and the DSpark drafter fit on one **24 GB RTX 4090 at 262,144 total context tokens**. The optional image encoder runs on the CPU.

## Run the local checkout

```bash
cd /home/me/Developments/ingine-4090
./scripts/ingine serve
```

Use an **OpenAI-compatible** client with base URL `http://127.0.0.1:8003/v1` and model ID `inkling-2-mini-abliterated`. Stop a foreground server with **Ctrl-C**. For background use, run `./scripts/ingine start`, `./scripts/ingine status` and `./scripts/ingine stop`. The server accepts one request at a time, serves chat, reasoning, streaming, base64 images and tools, and defaults to loopback with no authentication.

| Setting | Local default |
|---|---|
| Model | Pinned upstream checkpoint and supplied ablation file |
| Context | 262,144 input and output tokens combined |
| Dense weights / full-attention KV | FP8 / FP8 |
| Routed experts | Original mixed NVFP4 / MXFP8, resident on GPU |
| Speculation | Native DSpark, confidence threshold 0.8 |
| Prefill chunk | 2,048 tokens |
| Images | CPU vision tower, AVX-512 BF16 capable CPU |

The binary and verified weights are already present in this checkout. `INGINE_ABLATION= ./scripts/ingine serve` serves the original target instead, as `inkling-2-mini`. `PORT`, `HOST`, `INGINE_MODEL_DIR` and `INGINE_CTX` override the local settings. Stop other model servers before using the full window. For example, `INGINE_CTX=65536 ./scripts/ingine serve` frees memory on a busier desktop. On CPUs supporting the expert offload path, `./scripts/ingine serve --offload-experts 4` is a lower-VRAM option.

## Measurements

One RTX 4090, Ryzen 9 7950X3D and CUDA 13.5. The server was ready before timing each request. The long documents carried fresh inventory records and three passcodes placed at 2%, 50% and 96% depth. The client measured first output including reasoning and counted completion tokens after the first for decode.

| Uncached input tokens | Time to first token | Decode at that depth | Output tokens |
|---:|---:|---:|---:|
| 32,993 | 3.24 s | 265.7 tok/s | 163 |
| 245,215 | 34.14 s | 230.5 tok/s | 204 |
| 261,583 | 37.46 s | 225.1 tok/s | 470 |

All three codes were recovered on both long requests. Two image requests returned the colors in the correct order, and a separate request returned a structured weather tool call. Each timing is one request; DSpark acceptance depends on the text. The median of three short, 128-output-token coding requests was 272.9 tok/s. A 3,072-token prefill chunk gave nearly the same 245K first-token time while leaving roughly 380 MiB less headroom. Moving 4 GiB of experts to host memory increased the 32,993-token first-token time from about 3.2 to 6.6 seconds on this machine.

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

`bash setup.sh` opens the installer menu for building with a private CUDA toolkit or downloading the pinned weights into a fresh checkout. For the native tests, stop the server first, then run `ctest --test-dir native/build-sm89 --output-on-failure -E bench`. The optional trace test uses external chat fixtures. The current upstream speculative fidelity gate passed: three of 12,000 emitted tokens differed from the width-one argmax by more than 0.5 logits (2.5 per 10,000; upstream limit 15).

## Port lineage

Based on [skyne98’s RTX 3090 adaptation](https://github.com/skyne98/inkling-ingine-3090/tree/616ee271282dd84417e1b38dea69825c4fe341d5) of [satellitedown’s RTX 5090 engine](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine). This branch uses the 3090 adaptation’s cluster-free kernels, prompt checkpoints and CPU vision implementation. Ada uses hardware E4M3-to-FP16 conversion and the portable E2M1 path. Its eight KV heads use 32 decode splits for one row and 16 for verification, sized for the 4090’s 128 SMs. At 131,064 tokens the eight-row full-attention kernel took 316 µs, down from 493 µs with the 3090 split.

The upstream model and license notices are in [NOTICE](NOTICE). The [3090 port notes](PORT-3090.md), [expert offload guide](EXPERT-OFFLOAD.md) and [image guide](IMAGE-INPUT.md) document the shared implementation.
