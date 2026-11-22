# Image input (CPU vision tower)

`ingine-serve --engine` accepts images in chat requests. Inkling-2-Mini's vision tower (a 23-layer ViT plus a
2×2 patch merger, 0.59 GiB BF16, `model.visual.*` in the checkpoint) runs on the CPU, so images cost no VRAM
beyond a 9 MiB staging buffer. The image embeddings replace the image placeholder tokens' embeddings in the GPU
prefill. Nothing to download or configure: the tower is read from the model folder when the server starts.

```bash
curl -s http://127.0.0.1:8080/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "messages": [{"role": "user", "content": [
    {"type": "text", "text": "What is in this picture?"},
    {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,'"$(base64 -w0 photo.jpg)"'"}}]}],
  "reasoning_effort": "low"
}'
```

- Content parts of type `image_url` (`{"url": ...}` or a string), `input_image` and `image` (transformers'
  `image` / `url` / `base64` fields) are accepted in user and tool messages, any number per request, in any
  order with text parts. Each image becomes `<|content_image|>`, then one `<|unused_200054|>` per 40×40 pixels,
  then `<|end_message|>` (the checkpoint's chat template and processor). An image in a system or assistant
  message is rejected, as in the template.
- Images must be sent inline as `data:` URLs (base64). The server does not fetch `http(s)://` URLs or read
  files; such requests get HTTP 400 with a message saying so.
- Formats: JPEG, PNG (with transparency), BMP, GIF (first frame), TGA, PSD, HDR, PNM, via stb_image. WebP and
  HEIC are not supported.
- `GET /v1/models` lists `"input_modalities": ["text", "image"]`. Image tokens count as prompt tokens in
  `usage`.

| Flag | Default | |
|---|---|---|
| `--image-max-tokens N` | 1024 | image tokens per image (at most 4096); larger images are downscaled, keeping the aspect ratio |
| `--vision-threads N` / `INGINE_VISION_THREADS` | one per physical core | CPU threads encoding images |
| `--no-vision` | | do not load the tower; requests with images get HTTP 400 |

The checkpoint's processor never resizes: a 4032×3024 photo would be 7,676 image tokens. 1024 tokens is
1280×1280 pixels' worth (a 4032×3024 photo is encoded at 1434×1076, 972 tokens), about 0.7 s on the CPU.
llama.cpp bounds Inkling 2 images at 4096 tokens.

## Results (Ryzen 9 9950X3D, 16 threads, RTX 3090)

| Image | Tokens | CPU encode |
|---|---:|---:|
| 451×300 (photo of a cat) | 96 | 67 ms |
| 600×400 | 150 | 80 ms |
| 512×512 | 169 | 79 ms |
| 640×488 (newspaper front page) | 208 | 100 ms |
| 2875×1500 diagram, downscaled to 1754×915 | 1012 | 707 ms |
| 1280×1280 | 1024 | 690-740 ms (850 ms while another request decodes) |

- Decoding the file takes 2-70 ms more; the prefill of the image tokens is the same as for text tokens.
- Encoding runs on the request's HTTP thread before the request is queued, so the GPU keeps serving other
  requests. Three 1024-token encodes during an 800-token decode under `--offload-experts 4` (where both use CPU
  cores) left its speed unchanged (131 vs 126 tok/s alone, within DSpark's variation).
- Encodings are cached by content (256 MiB, about 55 images of 1024 tokens), so chat clients that resend the
  whole conversation each turn do not re-encode. The prompt checkpoint / prefix reuse compares images by
  content: a follow-up question about an image reuses the cached prefill, and a different image with the same
  token count is not mistaken for the previous one.
- Host memory: 608 MiB of weights (read in ~0.1 s on a background thread when the server starts; boot to ready
  stays 2.5 s warm), plus ~110 MiB of scratch for a 1024-token image. VRAM: unchanged, plus the 9 MiB
  staging buffer allocated by the first image request.

Answers on public test images (greedy, `reasoning_effort` low): scikit-image's `chelsea.png` "a tan tabby cat
with large green-yellow eyes", `coffee.png` "a reddish-orange espresso cup and saucer ... a spoon resting on
the saucer", `astronaut.png` "an orange NASA-style spacesuit ... an American flag and a space shuttle model",
llama.cpp's `tools/mtmd/test-1.jpeg` read as the New York Times front page "MEN WALK ON MOON", and two images in
one message told apart by position.

## How it works

`native/src/vision/`:

- **Preprocessing** (`image.cpp`), as transformers' `Inkling2ImageProcessor`: RGBA composited onto an 8 px
  checkerboard of 204 / 153, padded right and bottom to multiples of 40 px with -1 (on the 0-255 scale),
  `(x - 127.5) / 127.5` in FP32, cut into one 40×40 tile per image token and 2×2 patches of 20×20 per tile, BF16.
  Bit-identical to the reference on the test images.
- **Vision tower** (`vit.cpp`), as `Inkling2VisionModel`: patch embedding (Linear 1200 → 1024) plus the 32×32
  position table resampled bilinearly to the patch grid, a class token, pre-LayerNorm, 23 layers of
  `x += ls_1 * attn(LN1(x)); x += ls_2 * mlp(LN2(x))` (16 heads of 64, full attention within the image, axial
  2D RoPE on q / k with patches at (row + 1, col + 1) and the class token at (0, 0), exact-erf GELU MLP), then
  the class token dropped and each token's 4 patches concatenated into `proj_in` → GELU → `proj_out` (2304).
- **Kernels.** One AVX-512 BF16 GEMM: weights packed once into 32-column panels of BF16 pairs, a 12×32
  microkernel of VDPBF16PS (BF16 products, FP32 accumulation) over K blocks of 256, 96×128 tiles as jobs on a
  CPU pool, each finished by a fused epilogue (bias, GELU, layer-scale residual, BF16 store, or RoPE and the
  per-head q / k / v layouts). Attention per (head, 48 query rows): `q k^T`, an FP32 softmax with a vector exp,
  then `p v`, both on the same microkernel. Activations stay FP32 between operations and are rounded to BF16
  at GEMM inputs, as the BF16 model does. ~5.4 TFLOP/s on 16 cores.
- **Serving** (`encoder.cpp`, `serve/openai.cpp`): base64 `data:` URLs decoded, images encoded (one at a time,
  cached), each placeholder token expanded to the image's tokens and the spans passed to the generator
  (`model::PromptImage`: position, size, content hash, BF16 embeddings).
- **Prefill** (`model/generator.cpp`, `model/prefill.cpp`): per chunk, the rows of image tokens are uploaded and
  `set_rows` overwrites their normed token embeddings (`embed_norm` output) before the first layer, as
  transformers' `masked_scatter`. The DSpark drafter sees the images through the target model's taps.

## Verification

- `tools/vision_reference.py`: an FP64 NumPy port of the transformers code (processor, embeddings with the
  interpolation helper, rotary embedding, attention, layers, merger) that writes the test images and goldens in
  `native/tests/golden/vision/`.
- `ctest -R vision_kernels`: erf (max error 3.3e-7) and exp (2.2e-7 relative) against libm, the GEMM against
  FP64 on shapes with every tail (error 1.3e-7 of the sum of |products|), preprocessing bit-exact.
- `ctest -R vision_tower` (model): the tower's output against the FP64 reference: relative RMS error 0.3-1.0%,
  worst per-token cosine 0.99985 (three images: 2×3, 1×2 with transparency and padding, 3×5 tokens); prints the
  encode time of a 1024-token image.
- `ctest -R model_images` (model): a prompt whose "image" carries the embedding kernel's own normed embeddings of
  150 text tokens (spanning three 64-token prefill chunks) generates exactly the text prompt's tokens; follow-ups
  through the checkpoint and through extension match the text equivalents exactly; another image at the same
  positions is not reused.
- `serve_server` (`images`): data URLs, placeholder expansion and spans with a stub encoder, error statuses;
  `text_chat`: template renders of image parts.
- The rest of the suite passes unchanged (`ctest -E bench`, including model parity, generation, DSpark, reuse and
  offloading).

## Not done

- Remote image URLs (would need an HTTP(S) client and an SSRF policy) and WebP.
- OpenAI's `detail: low` is ignored (images are encoded up to `--image-max-tokens`).
- Video (the config's `temporal_patch_size` is unused by the image path).
- The tower on the GPU would be ~10× faster but costs 0.6 GiB of VRAM; this build keeps VRAM for context.
- A changed image inside the reused prefix disables reuse for that request (full prefill) instead of reusing
  the text before it.
