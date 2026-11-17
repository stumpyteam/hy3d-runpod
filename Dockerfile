# Hunyuan3D-2.1 (Tencent), ready to run on RunPod GPUs: everything installed and compiled once, at build time.
# Model weights are NOT in the image; they download from Hugging Face when the container starts (~1 min on RunPod).
FROM nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 \
    TORCH_CUDA_ARCH_LIST="8.0;8.6;8.9;9.0+PTX" FORCE_CUDA=1
RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev git wget curl build-essential \
      libgl1 libglib2.0-0 libxrender1 libxi6 libxkbcommon0 libsm6 libxext6 \
    && rm -rf /var/lib/apt/lists/* && ln -sf /usr/bin/python3 /usr/bin/python \
    && python -m pip install --upgrade pip setuptools wheel
RUN pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu124
RUN git clone --depth 1 https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1.git /app && rm -rf /app/.git
WORKDIR /app
# the upstream requirements list two China-hosted mirrors that are very slow elsewhere
# bpy==4.0 (Python 3.10 only) is no longer on PyPI: use Blender 4.2 LTS, the closest release for Python 3.11
RUN sed -i '/extra-index-url/d; s/^bpy==4.0$/bpy==4.2.0/' requirements.txt && pip install -r requirements.txt
RUN cd hy3dpaint/custom_rasterizer && pip install --no-build-isolation .
RUN cd hy3dpaint/DifferentiableRenderer && bash compile_mesh_painter.sh
RUN wget -q https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth -P hy3dpaint/ckpt
COPY start.sh /start.sh
RUN chmod +x /start.sh
EXPOSE 7860
CMD ["/start.sh"]
