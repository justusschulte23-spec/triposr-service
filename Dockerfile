FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends     curl wget xz-utils ca-certificates git     libgl1 libglib2.0-0 libgomp1     libxi6 libxrender1 libxfixes3     libsm6 libxext6 xvfb     && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash -     && apt-get install -y nodejs     && rm -rf /var/lib/apt/lists/*

RUN wget -q https://mirrors.ocf.berkeley.edu/blender/release/Blender3.6/blender-3.6.9-linux-x64.tar.xz     && tar -xf blender-3.6.9-linux-x64.tar.xz     && mv blender-3.6.9-linux-x64 /opt/blender     && ln -s /opt/blender/blender /usr/local/bin/blender     && rm blender-3.6.9-linux-x64.tar.xz

WORKDIR /app

RUN pip install --no-cache-dir     torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir     git+https://github.com/VAST-AI-Research/TripoSR.git     trimesh[all] Pillow omegaconf einops transformers huggingface-hub

COPY package.json ./
RUN npm install

RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('stabilityai/TripoSR', 'model.ckpt'); hf_hub_download('stabilityai/TripoSR', 'config.yaml'); print('TripoSR weights cached.')" || echo "Pre-cache skipped"

COPY . .

EXPOSE 3001
CMD ["node", "server.js"]
