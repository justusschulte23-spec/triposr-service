#!/usr/bin/env python3

import sys
import os
import gc
import torch
from PIL import Image

def main():
    if len(sys.argv) < 3:
        print("Usage: reconstruct.py <input_image> <output_glb>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Step 1: background removal BEFORE loading TripoSR to reduce peak RAM
    print("[TripoSR] Step 1: background removal...")
    image = Image.open(input_path).convert("RGBA")
    alpha_min, alpha_max = image.split()[3].getextrema()
    if alpha_max == 255 and alpha_min == 255:
        from rembg import remove as rembg_remove
        image = rembg_remove(image)
        gc.collect()

    # Step 2: load model
    print("[TripoSR] Step 2: loading model...")
    from tsr.system import TSR
    from tsr.utils import resize_foreground

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TripoSR] device: {device}")

    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model = model.to(device)
    model.eval()

    image = resize_foreground(image, 0.85)

    print("[TripoSR] Step 3: inference...")
    with torch.no_grad():
        scene_codes = model([image], device=device)

    # resolution=64 uses ~8x less RAM than 192 on CPU (64^3 vs 192^3 voxels)
    print("[TripoSR] Step 4: mesh extraction (resolution=64)...")
    meshes = model.extract_mesh(scene_codes, resolution=64)

    mesh = meshes[0]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    mesh.export(output_path)
    print(f"[TripoSR] saved: {output_path} ({os.path.getsize(output_path)} bytes)")

if __name__ == "__main__":
    main()
