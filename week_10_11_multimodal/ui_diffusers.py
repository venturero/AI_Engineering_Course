"""
Minimal Gradio UI for Stable Diffusion (diffusers).
Run: python ui_diffusers.py
"""
from __future__ import annotations

import os

import gradio as gr

from generate_image_diffusers import (
    build_pipeline,
    default_device,
    generate_with_pipeline,
)

_pipe = None
_model_id = "runwayml/stable-diffusion-v1-5"
_device = default_device()


def get_pipeline():
    global _pipe
    if _pipe is None:
        _pipe = build_pipeline(_model_id, _device)
    return _pipe


def generate(prompt: str, steps: int, guidance: float, seed: str):
    if not prompt.strip():
        return None, "Enter a prompt."
    seed_val = None
    if seed.strip().isdigit():
        seed_val = int(seed.strip())
    pipe = get_pipeline()
    image, path = generate_with_pipeline(
        pipe,
        prompt.strip(),
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        seed=seed_val,
        device=_device,
    )
    return image, f"Saved: {path}"


def main():
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    with gr.Blocks(title="Stable Diffusion (diffusers)") as demo:
        gr.Markdown("## Text → Image (Hugging Face diffusers)")
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", lines=2, value="a watercolor landscape at sunset")
            out_img = gr.Image(label="Output", type="pil")
        with gr.Row():
            steps = gr.Slider(10, 50, value=30, step=1, label="Steps")
            guidance = gr.Slider(1.0, 15.0, value=7.5, step=0.5, label="Guidance")
            seed = gr.Textbox(label="Seed (optional)", placeholder="e.g. 42")
        status = gr.Textbox(label="Status", interactive=False)
        btn = gr.Button("Generate")
        btn.click(fn=generate, inputs=[prompt, steps, guidance, seed], outputs=[out_img, status])

    demo.launch()


if __name__ == "__main__":
    main()
