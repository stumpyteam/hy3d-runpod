# hy3d-runpod

A ready-to-run container for **[Hunyuan3D-2.1](https://github.com/Tencent-Hunyuan/Hunyuan3D-2.1)** (image → textured 3D model) on rented GPUs such as RunPod.
Everything is installed and compiled at build time, so a GPU machine is usable a few minutes after it starts.

- Image: `ghcr.io/stumpyteam/hy3d-runpod:latest` (built by GitHub Actions from this repo)
- Needs an NVIDIA GPU with ~48 GB for shape + texture (24 GB with `--low_vram_mode`)
- On start it downloads the model weights from Hugging Face and serves the official Gradio app on port **7860**

```
docker run --gpus all -p 7860:7860 ghcr.io/stumpyteam/hy3d-runpod:latest
```

## Licence
This repo only contains build files. The image includes code from Tencent's Hunyuan3D-2.1, which is distributed under the
**Tencent Hunyuan 3D 2.1 Community License** (see the upstream repository). That licence does not apply in the European Union,
the United Kingdom and South Korea, and has further conditions; read it before any use.
