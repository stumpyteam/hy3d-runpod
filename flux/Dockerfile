# FLUX.1 schnell + FLUX.1 Kontext [dev] picture server for RunPod (Asset Forge picture jobs: concepts, views, edits).
# Weights are NOT in the image; they download from Hugging Face when the pod starts (hf_transfer, a few minutes).
FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HF_HUB_ENABLE_HF_TRANSFER=1 HF_HOME=/workspace/hf
RUN apt-get update && apt-get install -y --no-install-recommends curl && rm -rf /var/lib/apt/lists/*
RUN pip install diffusers==0.35.2 transformers==4.52.4 accelerate==1.8.1 sentencepiece==0.2.0 protobuf==5.29.3 \
      "huggingface_hub[hf_transfer]==0.34.4" pillow \
    && python -c "from diffusers import FluxPipeline, FluxKontextPipeline; print('diffusers OK')"
COPY flux_server.py /srv/flux_server.py
EXPOSE 8000
CMD ["python", "/srv/flux_server.py"]
