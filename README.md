# Inkling-2-Mini abliterated on an RTX 3090

**ingine**, a C++/CUDA inference engine for the abliterated
[Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini), ported from the RTX 5090 to a 24 GB
**RTX 3090**, with routed experts offloaded to system RAM, fast cold starts and image input on the CPU. It
serves an OpenAI-compatible API for chat, tool calls, reasoning and images.

This is a fork of [satellitedown/inkling-2-mini-abliterated-ingine](https://github.com/satellitedown/inkling-2-mini-abliterated-ingine),
which made the engine, the abliteration and the 5090 installer; its README is kept as
[README.upstream.md](README.upstream.md) (abliteration results, 5090 numbers, the installer).

| RTX 3090, Ryzen 9 9950X3D | |
|---|---|
| decode, greedy, 32K context | 131 tok/s (llama.cpp with the UD-Q4_K_XL GGUF on the same card: 56) |
| decode with DSpark speculative decoding | 191 tok/s chat, 247 tok/s code |
| prefill, 30K-token prompt | 4,350 tok/s |
| context | 64K with everything in VRAM; 256K with `--offload-experts 4` |
| boot to ready / first token | 2.5 s warm, 4.3-4.9 s cold / 74 ms |
| image encoding (CPU) | 70-100 ms for a typical photo, 0.7 s for a 1280×1280 image |

## What this fork adds

- **sm_86 port** ([PORT-3090.md](PORT-3090.md)): Blackwell-only features (FP4/FP8 conversion instructions,
  thread-block clusters, programmatic dependent launch) replaced with exact Ampere code paths; parity with the
  Python reference unchanged (KL 2.2e-3, top-1 0.979). The `120a` (RTX 5090) build still compiles.
- **Expert offloading** ([EXPERT-OFFLOAD.md](EXPERT-OFFLOAD.md)): `--offload-experts GIB` keeps that many GiB of
  experts only in pinned RAM, with an adaptive VRAM expert cache, AVX-512 CPU expert kernels during decode and
  staged prefill. Frees VRAM for long context: upstream's 256K-token DSpark profile fits the 3090.
- **Fast startup**: O_DIRECT parallel loading, page-cache re-warming, a cached drafter fold and a warm-up
  generation (PORT-3090.md, Startup).
- **Image input** ([IMAGE-INPUT.md](IMAGE-INPUT.md)): the vision tower runs on the CPU (AVX-512 BF16 kernels,
  0.6 GiB of RAM, no VRAM); images in chat requests as base64 `data:` URLs.

## Requirements

- An NVIDIA RTX 3090 (sm_86; other GPUs need their own port), a driver for CUDA 13.
- To build: CUDA 13 (nvcc, cudart, cuBLAS, NVTX), CMake 3.28+, Ninja, GCC 12-16.
- 20.2 GB for the model, 24 GB of RAM or more (32+ with `--offload-experts`).
- For offloading's CPU path and for images: AVX-512 with BF16 (Zen 4 / Zen 5, Sapphire Rapids and later).
  Without it, image input is off and offloading needs `--offload-threads 0` (the GPU reads cold experts over
  PCIe).

## Build

```bash
git clone https://github.com/skyne98/inkling-ingine-3090
cd inkling-ingine-3090/native
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86
ninja -C build    # build/apps/serve/ingine-serve, build/apps/bench/ingine-bench, build/apps/validate/ingine-validate
```

Pass `-DCMAKE_CUDA_COMPILER=/path/to/nvcc -DCUDAToolkit_ROOT=/path/to/cuda` when nvcc is not on `PATH`.

**Nix**: [nix/package.nix](nix/package.nix) builds the three programs with nixpkgs' CUDA (unfree):

```bash
nix-build -E 'with import <nixpkgs> { config.allowUnfree = true; }; callPackage ./nix/package.nix { }'
```

## Get the model

The pinned Hugging Face revision of
[satellitedown/Inkling-2-Mini-abliterated-ingine](https://huggingface.co/satellitedown/Inkling-2-Mini-abliterated-ingine)
(Inkling-2-Mini's files unchanged, the DSpark drafter and the abliteration file), checked file by file:

```bash
uv run --with huggingface_hub python scripts/download_model.py --model-dir models/Inkling-2-Mini
```

## Run

```bash
M=models/Inkling-2-Mini
native/build/apps/serve/ingine-serve --engine --model-dir $M --ablation $M/inkling-2-mini-abliterated.safetensors \
  --host 127.0.0.1 --port 8003 --max-ctx 65536 --dense-fp8 --kv-fp8 --spec --spec-threshold 0.8
```

Without `--ablation` it serves the original model (model id `inkling-2-mini`). Useful flags:

| Flag | |
|---|---|
| `--max-ctx N` | context positions; 65536 fits 24 GB with everything in VRAM |
| `--dense-fp8`, `--kv-fp8` | FP8 dense weights / full-attention KV cache (needed on 24 GB) |
| `--spec --spec-threshold 0.8` | DSpark speculative decoding |
| `--offload-experts GIB` | experts kept only in RAM; `4` runs `--max-ctx 262144` on the 3090 |
| `--image-max-tokens N`, `--vision-threads N`, `--no-vision` | image input (default 1024 tokens per image) |
| `--temperature`, `--top-p`, `--top-k` | defaults for requests that omit them (else `generation_config.json`) |
| `--max-queue N`, `--model-id NAME` | waiting requests (default 4), the id reported by the API |

## API

OpenAI-compatible at `http://127.0.0.1:8003/v1`: `POST /v1/chat/completions` (streaming and not),
`GET /v1/models`, `GET /health`. No authentication: keep it on loopback. One request runs at a time.

- `reasoning_effort`: `none` … `max` or 0-0.99; `enable_thinking: false` answers without reasoning.
- `tools` (function calling), `stop`, `max_tokens`, `temperature`, `top_p`, `top_k`, `seed`, `stream`.
- Images: `image_url` parts with base64 `data:` URLs (JPEG, PNG, BMP, GIF, ...), several per message.

```bash
curl -s http://127.0.0.1:8003/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What is in this picture?"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$(base64 -w0 photo.jpg)"'"}}]}],
  "reasoning_effort": "low"
}'
```

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8003/v1", api_key="none")
stream = client.chat.completions.create(
    model="inkling-2-mini-abliterated",
    messages=[{"role": "user", "content": "Write a Python LRU cache with tests."}],
    reasoning_effort="medium",
    stream=True,
)
for chunk in stream:
    delta = chunk.choices[0].delta
    print(getattr(delta, "reasoning_content", None) or delta.content or "", end="", flush=True)
```

Reasoning arrives as `reasoning_content` before the answer's `content`.

**Follow-up requests** reuse the processed conversation: a prompt that extends the previous one (plus the answer
as generated) only prefills the new part; otherwise the engine restores the longest prompt checkpoint that is a
prefix of the new prompt: the last prompt's (in VRAM) or one of the 3 before it (pinned host memory, up to
~430 MiB each, copied while the next prompt prefills). Agent clients that resend a recent tool result differently
then re-read only the last turns instead of the whole conversation. `INGINE_PROMPT_CHECKPOINTS` (default 4, VRAM
one included; 1: only the last) sets how many; the log's `reuse=` field shows the path each request took.

## Tests

```bash
cd native
cmake -S . -B build -G Ninja -DCMAKE_CUDA_ARCHITECTURES=86 -DINGINE_BUILD_TESTS=ON && ninja -C build
ctest --test-dir build -LE model -E bench   # kernels, server, chat format, vision kernels
ctest --test-dir build -L model             # full model (INGINE_MODEL_DIR or ../models/Inkling-2-Mini)
```

## License and credits

Apache-2.0 ([LICENSE](LICENSE)); see [NOTICE](NOTICE) for everything this builds on. The engine, the
abliteration and the installer are [satellitedown](https://github.com/satellitedown)'s; the expert offloading
follows [Strata](https://github.com/Niko1221/Strata); the vision path follows Hugging Face transformers and
llama.cpp. Inkling-2-Mini is by Thinking Machines (Apache-2.0); its weights are not in this repository. The
abliteration removes the model's safety refusals: you are responsible for how you use it. Unofficial, not
endorsed by Thinking Machines.
