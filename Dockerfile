# Pin to Debian 12 bookworm — stable package names
FROM python:3.10-slim-bookworm

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends     curl wget xz-utils ca-certificates     libgl1 libglib2.0-0 libgomp1     libxi6 libxrender1 libxfixes3     libsm6 libxext6     xvfb     && rm -rf /var/lib/apt/lists/*

# Node.js 18
RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash -     && apt-get install -y nodejs     && rm -rf /var/lib/apt/lists/*

# Blender 3.6.9 LTS headless
RUN wget -q https://mirrors.ocf.berkeley.edu/blender/release/Blender3.6/blender-3.6.9-linux-x64.tar.xz     && tar -xf blender-3.6.9-linux-x64.tar.xz     && mv blender-3.6.9-linux-x64 /opt/blender     && ln -s /opt/blender/blender /usr/local/bin/blender     && rm blender-3.6.9-linux-x64.tar.xz

WORKDIR /app

# PyTorch CPU — split for better layer caching
RUN pip install --no-cache-dir     torch torchvision --index-url https://download.pytorch.org/whl/cpu

# TripoSR + deps
RUN pip install --no-cache-dir     git+https://github.com/VAST-AI-Research/TripoSR.git     trimesh[all] Pillow omegaconf einops transformers huggingface-hub

# Node deps
COPY package.json ./
RUN npm install

# Pre-cache TripoSR model weights (saves cold-start time)
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('stabilityai/TripoSR', 'model.ckpt'); hf_hub_download('stabilityai/TripoSR', 'config.yaml'); print('TripoSR weights cached.')" || echo "Model pre-cache skipped"

COPY . .

EXPOSE 3001
CMD ["node", "server.js"]
