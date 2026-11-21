#!/usr/bin/env python3
"""FP64 NumPy reference of the Inkling 2 image path, a line-by-line port of transformers'
Inkling2ImageProcessor (image_processing_inkling2.py), Inkling2VisionModel / Inkling2VisionEmbeddings /
Inkling2VisionRotaryEmbedding / Inkling2VisionAttention / Inkling2VisionEncoderLayer / Inkling2VisionMerger
(modeling_inkling2.py) and the vision_utils helpers (get_vision_interpolation_indices_and_weights,
get_vision_position_ids). Weights are the checkpoint's BF16 values; every operation runs in FP64; the pixel
values are rounded to BF16 as the model receives them (pixel_values.to(bf16)).

Writes the test images and goldens of tests/vision (`--write-golden DIR`), or prints the reference for one raw
image. NumPy only (no torch / PIL): the images are generated here, PNG / PPM written with zlib.

  python3 tools/vision_reference.py --model-dir ../models/Inkling-2-Mini --write-golden tests/golden/vision
"""
import argparse
import json
import math
import os
import struct
import zlib

import numpy as np


# ---- checkpoint -------------------------------------------------------------------------------------------------

class Checkpoint:
    def __init__(self, model_dir):
        idx = os.path.join(model_dir, "model.safetensors.index.json")
        if os.path.exists(idx):
            files = sorted(set(json.load(open(idx))["weight_map"].values()))
        else:
            files = ["model.safetensors"]
        self.where = {}
        for f in files:
            path = os.path.join(model_dir, f)
            with open(path, "rb") as fh:
                n = struct.unpack("<Q", fh.read(8))[0]
                header = json.loads(fh.read(n))
            for k, v in header.items():
                if k != "__metadata__":
                    self.where[k] = (path, 8 + n + v["data_offsets"][0], v["data_offsets"][1] - v["data_offsets"][0],
                                     v["dtype"], v["shape"])

    def get(self, name):
        path, off, nbytes, dtype, shape = self.where[name]
        assert dtype == "BF16", (name, dtype)
        with open(path, "rb") as fh:
            fh.seek(off)
            raw = np.frombuffer(fh.read(nbytes), dtype=np.uint16)
        return (raw.astype(np.uint32) << 16).view(np.float32).astype(np.float64).reshape(shape)


def to_bf16(x):
    """Round FP64 -> FP32 -> BF16 (nearest even), back to FP64."""
    f = np.asarray(x, dtype=np.float32)
    u = f.view(np.uint32).astype(np.uint64)
    u = ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint32) << 16
    return u.astype(np.uint32).view(np.float32).astype(np.float64)


def bf16_bits(x):
    f = np.asarray(x, dtype=np.float32)
    u = f.view(np.uint32).astype(np.uint64)
    return ((u + 0x7FFF + ((u >> 16) & 1)) >> 16).astype(np.uint16)


# ---- Inkling2ImageProcessor ------------------------------------------------------------------------------------

def convert_to_rgb(image):  # image: uint8 [C, H, W]
    if image.shape[0] != 4:
        return image
    height, width = image.shape[-2:]
    rows = np.arange(height)[:, None] // 8
    columns = np.arange(width)[None, :] // 8
    background = np.where((rows + columns) % 2 == 0, 204, 153)
    rgba = image.astype(np.int64)
    alpha = rgba[3:4]
    return ((rgba[:3] * alpha + background * (255 - alpha) + 127) // 255).astype(np.uint8)


def preprocess(image_hwc):
    image = convert_to_rgb(np.ascontiguousarray(image_hwc.transpose(2, 0, 1)))
    channels, height, width = image.shape
    size = 40
    rows, columns = math.ceil(height / size), math.ceil(width / size)
    padded = np.full((channels, rows * size, columns * size), -1.0)
    padded[:, :height, :width] = image.astype(np.float64)
    # rescale_and_normalize fused: (x - 255 * mean) / (255 * std), in FP32
    image = ((padded.astype(np.float32) - np.float32(127.5)) / np.float32(127.5)).astype(np.float64)
    image = image.reshape(channels, rows, size, columns, size).transpose(1, 3, 2, 4, 0)
    pixel_values = image.reshape(rows * columns, size, size, channels)
    return pixel_values, (1, rows, columns)


# ---- vision_utils --------------------------------------------------------------------------------------------

def axis_taps_weights(index, size, side):
    index = index.astype(np.float32)
    src = (index + np.float32(0.5)) * np.float32(side) / size.astype(np.float32) - np.float32(0.5)
    floor = np.floor(src)
    offsets = np.arange(0, 2)
    raw_taps = floor.astype(np.int64)[:, None] + offsets
    taps = np.clip(raw_taps, 0, side - 1)
    distance = np.abs(src[:, None] - floor[:, None] - offsets)
    weights = np.clip(1 - distance, 0, None)
    return taps, weights.astype(np.float64)


def interpolation_indices_and_weights(grid_thw, side, merge):
    t, h, w = grid_thw
    counts = t * h * w
    within = np.arange(counts) % (h * w)
    blocks_w = w // merge
    in_col = within % merge
    in_row = (within // merge) % merge
    block_col = (within // (merge * merge)) % blocks_w
    block_row = within // (merge * merge * blocks_w)
    row = block_row * merge + in_row
    col = block_col * merge + in_col
    h_taps, h_weights = axis_taps_weights(row, np.full(counts, h), side)
    w_taps, w_weights = axis_taps_weights(col, np.full(counts, w), side)
    indices = (h_taps[:, :, None] * side + w_taps[:, None, :]).reshape(-1, 4)
    weights = (h_weights[:, :, None] * w_weights[:, None, :]).reshape(-1, 4)
    return indices, weights


def vision_position_ids(grid_thw, merge):
    t, h, w = grid_thw
    hpos, wpos = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    shape = (h // merge, merge, w // merge, merge)
    hpos = hpos.reshape(shape).transpose(0, 2, 1, 3).flatten()
    wpos = wpos.reshape(shape).transpose(0, 2, 1, 3).flatten()
    return np.tile(np.stack([hpos, wpos], axis=-1), (t, 1))


# ---- Inkling2VisionModel ----------------------------------------------------------------------------------------

def layer_norm(x, w, b, eps=1e-5):
    mean = x.mean(-1, keepdims=True)
    var = ((x - mean) ** 2).mean(-1, keepdims=True)
    return (x - mean) / np.sqrt(var + eps) * w + b


def gelu(x):
    return 0.5 * x * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def rotate_pairwise(x):
    x = x.reshape(*x.shape[:-1], -1, 2)
    x1, x2 = x[..., 0], x[..., 1]
    return np.stack((-x2, x1), axis=-1).reshape(*x.shape[:-2], -1)


class VisionTower:
    def __init__(self, ck):
        B = "model.visual.backbone."
        self.patch_w = ck.get(B + "embeddings.patch_embedding.weight")
        self.pos = ck.get(B + "embeddings.position_embedding")
        self.cls = ck.get(B + "embeddings.class_embedding") + ck.get(B + "embeddings.cls_position_embedding")
        self.ln_pre = (ck.get(B + "ln_pre.weight"), ck.get(B + "ln_pre.bias"))
        self.layers = []
        for l in range(23):
            L = f"{B}layers.{l}."
            g = lambda n: ck.get(L + n)
            self.layers.append(dict(
                ln1=(g("ln_1.weight"), g("ln_1.bias")), ln2=(g("ln_2.weight"), g("ln_2.bias")),
                ls1=g("ls_1.gamma"), ls2=g("ls_2.gamma"),
                q=(g("self_attn.q_proj.weight"), g("self_attn.q_proj.bias")),
                k=(g("self_attn.k_proj.weight"), g("self_attn.k_proj.bias")),
                v=(g("self_attn.v_proj.weight"), g("self_attn.v_proj.bias")),
                o=(g("self_attn.out_proj.weight"), g("self_attn.out_proj.bias")),
                fc1=(g("mlp.c_fc.weight"), g("mlp.c_fc.bias")), fc2=(g("mlp.c_proj.weight"), g("mlp.c_proj.bias"))))
        self.proj_in = ck.get("model.visual.proj_in.weight")
        self.proj_out = ck.get("model.visual.proj_out.weight")

    def forward(self, pixel_values, grid_thw):
        merge, patch = 2, 20
        patch_grid = (grid_thw[0], grid_thw[1] * merge, grid_thw[2] * merge)
        num_patches = pixel_values.shape[0] * merge ** 2
        # Inkling2VisionEmbeddings
        patches = pixel_values.reshape(-1, merge, patch, merge, patch, pixel_values.shape[-1]).transpose(0, 1, 3, 2, 4, 5)
        patch_embeddings = patches.reshape(-1, 1200) @ self.patch_w.T
        idx, wts = interpolation_indices_and_weights(patch_grid, 32, merge)
        position_embeddings = (self.pos[idx] * wts[:, :, None]).sum(1)
        patch_embeddings = patch_embeddings + position_embeddings
        x = np.concatenate([self.cls[None], patch_embeddings], 0)
        x = layer_norm(x, *self.ln_pre)
        # rotary: patches 1-indexed (row, col), class token (0, 0)
        pos_ids = np.zeros((num_patches + 1, 2))
        pos_ids[1:] = vision_position_ids(patch_grid, merge) + 1
        inv_freq = 1.0 / (10000.0 ** (np.arange(0, 32, 2, dtype=np.float64) / 32))
        freqs = pos_ids[..., None] * inv_freq  # (N, 2, 16)
        cos, sin = np.cos(freqs), np.sin(freqs)
        recompose = lambda f: np.repeat(np.concatenate([f[:, 1], f[:, 0]], -1), 2, axis=-1)
        cos, sin = recompose(cos)[:, None, :], recompose(sin)[:, None, :]
        n = x.shape[0]
        for L in self.layers:
            h = layer_norm(x, *L["ln1"])
            q = (h @ L["q"][0].T + L["q"][1]).reshape(n, 16, 64)
            k = (h @ L["k"][0].T + L["k"][1]).reshape(n, 16, 64)
            v = (h @ L["v"][0].T + L["v"][1]).reshape(n, 16, 64)
            q = q * cos + rotate_pairwise(q) * sin
            k = k * cos + rotate_pairwise(k) * sin
            s = np.einsum("qhd,khd->hqk", q, k) * 64 ** -0.5
            s = np.exp(s - s.max(-1, keepdims=True))
            p = s / s.sum(-1, keepdims=True)
            a = np.einsum("hqk,khd->qhd", p, v).reshape(n, 1024)
            x = x + L["ls1"] * (a @ L["o"][0].T + L["o"][1])
            h = layer_norm(x, *L["ln2"])
            x = x + L["ls2"] * (gelu(h @ L["fc1"][0].T + L["fc1"][1]) @ L["fc2"][0].T + L["fc2"][1])
        patch_states = x[1:]
        return gelu(patch_states.reshape(-1, 4 * 1024) @ self.proj_in.T) @ self.proj_out.T


# ---- test images ----------------------------------------------------------------------------------------------

def png_bytes(img):
    h, w, c = img.shape
    raw = b"".join(b"\x00" + img[y].tobytes() for y in range(h))
    chunk = lambda t, d: struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    ihdr = struct.pack(">IIBBBBB", w, h, 8, 6 if c == 4 else 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", ihdr) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


def ppm_bytes(img):
    h, w, _ = img.shape
    return f"P6\n{w} {h}\n255\n".encode() + img.tobytes()


def test_images():
    rng = np.random.default_rng(1234)
    yy, xx = np.mgrid[0:70, 0:100]
    a = np.stack([xx * 255 // 99, yy * 255 // 69, (xx + yy) % 256], -1).astype(np.uint8)
    a[10:40, 20:60] = [220, 30, 30]  # red box
    b = np.zeros((37, 53, 4), np.uint8)
    b[..., :3] = rng.integers(0, 256, (37, 53, 3))
    b[..., 3] = (np.arange(53)[None, :] * 255 // 52).astype(np.uint8)  # alpha ramp: checkerboard shows through
    b[5:15, 5:15, 3] = 0
    yy, xx = np.mgrid[0:120, 0:200]
    c = (rng.normal(128, 40, (120, 200, 3))).clip(0, 255).astype(np.uint8)
    c[(xx - 100) ** 2 + (yy - 60) ** 2 < 900] = [30, 200, 60]  # green disc
    return [("gradient.ppm", a, ppm_bytes(a)), ("alpha.png", b, png_bytes(b)), ("noise.png", c, png_bytes(c))]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--write-golden", required=True, help="directory for the images, goldens and manifest")
    args = ap.parse_args()
    os.makedirs(args.write_golden, exist_ok=True)
    tower = VisionTower(Checkpoint(args.model_dir))
    manifest = []
    for name, pixels, data in test_images():
        open(os.path.join(args.write_golden, name), "wb").write(data)
        pv, grid = preprocess(pixels)
        pv = to_bf16(pv)  # pixel_values.to(bf16)
        out = tower.forward(pv, grid)
        stem = name.split(".")[0]
        # the patch embedding's input order: [tokens, 2, 2, 20, 20, 3] (Inkling2VisionEmbeddings' transpose(2, 3))
        patches = pv.reshape(-1, 2, 20, 2, 20, 3).transpose(0, 1, 3, 2, 4, 5)
        bf16_bits(np.ascontiguousarray(patches)).tofile(os.path.join(args.write_golden, stem + ".pixels.bf16"))
        out.astype(np.float32).tofile(os.path.join(args.write_golden, stem + ".embeds.f32"))
        manifest.append({"image": name, "rows": grid[1], "cols": grid[2], "pixels": stem + ".pixels.bf16",
                         "embeds": stem + ".embeds.f32", "rms": float(np.sqrt((out ** 2).mean()))})
        print(name, grid, "rms", manifest[-1]["rms"])
    json.dump(manifest, open(os.path.join(args.write_golden, "manifest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
