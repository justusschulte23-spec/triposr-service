#!/usr/bin/env python3

import sys
import os
import torch
from PIL import Image

def main():
    if len(sys.argv) < 3:
        print("Usage: reconstruct.py <input_image> <output_glb>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    print(f"[TripoSR] Loading model...")

    # Lazy import after pip install
    from tsr.system import TSR
    from tsr.utils import remove_background, resize_foreground

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[TripoSR] Device: {device}")

    model = TSR.from_pretrained(
        "stabilityai/TripoSR",
        config_name="config.yaml",
        weight_name="model.ckpt",
    )
    model = model.to(device)
    model.eval()

    print(f"[TripoSR] Processing image: {input_path}")
    image = Image.open(input_path).convert("RGBA")

    # Remove background if no alpha
    if image.mode != "RGBA" or image.split()[3].getextrema() == (255, 255):
        image = remove_background(image)
    image = resize_foreground(image, 0.85)

    with torch.no_grad():
        scene_codes = model([image], device=device)

    print(f"[TripoSR] Extracting mesh...")
    meshes = model.extract_mesh(scene_codes, resolution=192)

    mesh = meshes[0]
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    mesh.export(output_path)

    print(f"[TripoSR] Saved GLB: {output_path} ({os.path.getsize(output_path)} bytes)")

if __name__ == "__main__":
    main()
