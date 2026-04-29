FROM python:3.10-slim-bookworm

RUN apt-get update && apt-get install -y --no-install-recommends     curl wget xz-utils ca-certificates git     libgl1 libglib2.0-0 libgomp1     libxi6 libxrender1 libxfixes3     libsm6 libxext6 xvfb     && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL https://deb.nodesource.com/setup_18.x | bash -     && apt-get install -y nodejs     && rm -rf /var/lib/apt/lists/*

RUN wget -q https://mirrors.ocf.berkeley.edu/blender/release/Blender3.6/blender-3.6.9-linux-x64.tar.xz     && tar -xf blender-3.6.9-linux-x64.tar.xz     && mv blender-3.6.9-linux-x64 /opt/blender     && ln -s /opt/blender/blender /usr/local/bin/blender     && rm blender-3.6.9-linux-x64.tar.xz

WORKDIR /app

# PyTorch CPU
RUN pip install --no-cache-dir     torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu

# Clone TripoSR — do NOT use requirements.txt (contains unpinned gradio → pip timeout)
RUN git clone --depth 1 https://github.com/VAST-AI-Research/TripoSR.git /opt/TripoSR

# Install only the packages reconstruct.py actually needs (no gradio, no demo deps)
RUN pip install --no-cache-dir     "omegaconf>=2.3"     "einops>=0.7"     "transformers>=4.35"     "huggingface-hub>=0.20"     "trimesh[easy]>=4.0"     "rembg>=2.0"     "Pillow>=10.0"     "numpy>=1.24,<2.0"     "scipy>=1.11"     "skimage"     scikit-image

ENV PYTHONPATH="/opt/TripoSR"

COPY package.json ./
RUN npm install

# Pre-cache model weights
RUN python3 -c "from huggingface_hub import hf_hub_download; hf_hub_download('stabilityai/TripoSR', 'model.ckpt'); hf_hub_download('stabilityai/TripoSR', 'config.yaml'); print('weights cached')" || echo "pre-cache skipped"

COPY . .
EXPOSE 3001
CMD ["node", "server.js"]
