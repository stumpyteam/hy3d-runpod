#!/bin/bash
# Download the Hunyuan3D-2.1 weights (cached in the container) and start the official Gradio app on port 7860.
set -e
cd /app
python -c "from huggingface_hub import snapshot_download; snapshot_download('tencent/Hunyuan3D-2.1', max_workers=16)"
exec python gradio_app.py --model_path tencent/Hunyuan3D-2.1 --subfolder hunyuan3d-dit-v2-1 \
     --texgen_model_path tencent/Hunyuan3D-2.1 --host 0.0.0.0 --port 7860 "$@"
