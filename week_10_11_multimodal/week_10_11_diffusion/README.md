# Week 10–11: Multimodal (Diffusion)

Minimal **text-to-image** demos: core grading uses **Hugging Face diffusers** locally; **fal.ai** is optional bonus.

## Layout

```
week_10_11_diffusion/
├── generate_image_diffusers.py   # CORE — CLI, Stable Diffusion via diffusers
├── ui_diffusers.py               # CORE — Gradio UI
├── generate_image_falai.py       # OPTIONAL — hosted API (flux schnell)
├── ui_falai.py                   # OPTIONAL — Gradio for fal.ai
├── text_to_video_experiment.ipynb  # OPTIONAL — T2V experiment
├── outputs/                      # generated images/videos (gitignored except .gitkeep)
└── README.md
```

## Setup (core)

Python 3.10+ recommended. From this folder:

```bash
pip install torch diffusers transformers accelerate safetensors gradio pillow requests
```

On **CPU**, the first run downloads weights (~4 GB for SD 1.5) and generation is slow. **CUDA** is strongly preferred.

## Core: CLI

```bash
cd week_10_11_diffusion
python generate_image_diffusers.py "a watercolor owl reading a newspaper"
```

Images are written to `outputs/` as `sd_<timestamp>_<prompt_snippet>.png`.

Options: `--model`, `--steps`, `--guidance`, `--seed`, `--device`.

## Core: Gradio UI

```bash
python ui_diffusers.py
```

Opens a browser with prompt, steps, guidance, and optional seed.

## Optional: fal.ai

1. Create an API key at [fal.ai dashboard](https://fal.ai/dashboard/keys).
2. Set `FAL_KEY` in your environment.
3. Install the client: `pip install fal-client`

```bash
python generate_image_falai.py "neon alley in the rain, cinematic"
python ui_falai.py
```

Outputs use the prefix `fal_` in `outputs/`. Default model is `fal-ai/flux/schnell` (override with `--model`).

## Optional: Text-to-video

Open `text_to_video_experiment.ipynb`. It uses a diffusers text-to-video pipeline and **needs a capable GPU** and significant disk/VRAM; treat it as an experiment, not a graded minimum.

## Notes

- This package is **standalone** (no integration with other course agents).
- Do not commit API keys or large generated files; keep `outputs/` for local runs only.
