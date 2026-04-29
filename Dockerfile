# ── Stage: full runtime ──────────────────────────────────────────────────────
FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl wget xz-utils ca-certificates \
    libgl1-mesa-glx libglib2.0-0 libgomp1 \
    libxi6 libxrender1 libxfixes3 libxxf86vm1 \
    libsm6 libxext6 libxcomposite1 libxcursor1 \
    libxdamage1 libxrandr2 libxss1 libxtst6 \
    && rm -rf /var/lib/apt/lists/*

# Node.js 18
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash - \
    && apt-get install -y nodejs \
    && rm -rf /var/lib/apt/lists/*

# Blender 3.6 LTS (headless)
RUN wget -q https://mirrors.ocf.berkeley.edu/blender/release/Blender3.6/blender-3.6.12-linux-x64.tar.xz \
    && tar -xf blender-3.6.12-linux-x64.tar.xz \
    && mv blender-3.6.12-linux-x64 /opt/blender \
    && ln -s /opt/blender/blender /usr/local/bin/blender \
    && rm blender-3.6.12-linux-x64.tar.xz

WORKDIR /app

# Python deps (CPU PyTorch — no GPU needed for TripoSR inference at 192 res)
RUN pip install --no-cache-dir \
    torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir \
    tsr \
    trimesh[all] \
    Pillow \
    omegaconf \
    einops \
    transformers \
    huggingface-hub

# Node deps
COPY package.json ./
RUN npm install

# Pre-download TripoSR model weights at build time (avoids cold-start delay)
RUN python3 -c "\
from huggingface_hub import hf_hub_download; \
hf_hub_download('stabilityai/TripoSR', 'model.ckpt'); \
hf_hub_download('stabilityai/TripoSR', 'config.yaml'); \
print('TripoSR weights cached.')" || echo "Model pre-download skipped (will download at runtime)"

# App files
COPY . .

EXPOSE 3001
CMD ["node", "server.js"]
