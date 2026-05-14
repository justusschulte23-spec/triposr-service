#!/usr/bin/env python3

import sys
import os
import gc
import subprocess
import tempfile

# Set before any torch/onnxruntime import.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import torch
from PIL import Image
import numpy as np

MAX_INPUT_PX = 512  # TripoSR's vision encoder processes at this resolution anyway

_REMBG_SCRIPT = """\
import os, sys
os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['OMP_NUM_THREADS']='1'
from rembg import remove
from PIL import Image
img = Image.open(sys.argv[1]).convert('RGBA')
result = remove(img)
result.save(sys.argv[2])
"""


def _remove_bg_rembg(input_path):
    """Run rembg in isolation — if onnxruntime SIGSEGVs, only the child dies."""
    fd, tmp = tempfile.mkstemp(suffix='.png')
    os.close(fd)
    try:
        r = subprocess.run(
            ['python3', '-c', _REMBG_SCRIPT, input_path, tmp],
            timeout=120, capture_output=True
        )
        if r.returncode == 0 and os.path.getsize(tmp) > 0:
            img = Image.open(tmp).convert('RGBA').copy()
            print(f"[TripoSR] rembg OK")
            return img
        print(f"[TripoSR] rembg exit={r.returncode} stderr={r.stderr[-400:]}")
    except subprocess.TimeoutExpired:
        print("[TripoSR] rembg timed out (120 s)")
    except Exception as e:
        print(f"[TripoSR] rembg error: {e}")
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return None


def _remove_bg_pil(image, threshold=240):
    """Threshold-based white-background removal — fallback for product shots."""
    arr = np.array(image)
    bright = arr[:, :, :3].astype(np.float32).mean(axis=2)
    arr[:, :, 3] = np.where(bright > threshold, 0, 255).astype(np.uint8)
    return Image.fromarray(arr, 'RGBA')


def _rgba_to_rgb_white(image):
    """Composite RGBA over white background → RGB (required by TripoSR image tokenizer)."""
    bg = Image.new("RGBA", image.size, (255, 255, 255, 255))
    bg.paste(image, mask=image.split()[3])
    return bg.convert("RGB")


def main():
    if len(sys.argv) < 3:
        print("Usage: reconstruct.py <input_image> <output_glb>")
        sys.exit(1)

    input_path = sys.argv[1]
    output_path = sys.argv[2]

    # Step 1: background removal BEFORE loading TripoSR to reduce peak RAM
    print("[TripoSR] Step 1: background removal...")
    image = Image.open(input_path).convert("RGBA")

    # Resize early — limits memory in rembg subprocess and main process alike.
    # rembg's u2net runs at 320×320 internally so no quality loss past ~512px.
    w, h = image.size
    if max(w, h) > MAX_INPUT_PX:
        print(f"[TripoSR] resizing {w}x{h} → max {MAX_INPUT_PX}px")
        image.thumbnail((MAX_INPUT_PX, MAX_INPUT_PX), Image.LANCZOS)

    alpha_min, alpha_max = image.split()[3].getextrema()
    if alpha_max == 255 and alpha_min == 255:
        # Save already-resized image to a temp PNG so rembg subprocess gets the small version
        fd, tmp_input = tempfile.mkstemp(suffix='.png')
        os.close(fd)
        try:
            image.save(tmp_input)
            print("[TripoSR] no transparency — trying rembg subprocess...")
            result = _remove_bg_rembg(tmp_input)
        finally:
            try:
                os.unlink(tmp_input)
            except OSError:
                pass

        if result is not None:
            image = result
        else:
            print("[TripoSR] rembg failed — falling back to PIL white-bg removal")
            image = _remove_bg_pil(image)
        gc.collect()

    print(f"[TripoSR] image: {image.size} mode={image.mode}")

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
    print(f"[TripoSR] after resize_foreground: {image.size}")

    # TripoSR's image_tokenizer normalises with 3-channel mean/std → must be RGB
    image = _rgba_to_rgb_white(image)
    print(f"[TripoSR] converted to RGB")

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
