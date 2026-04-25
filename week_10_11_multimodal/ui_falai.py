"""
Optional Gradio UI for fal.ai image generation. Set FAL_KEY before running.
Run: python ui_falai.py
"""
from __future__ import annotations

import gradio as gr

from generate_image_falai import generate_image_fal


def run(prompt: str, image_size: str):
    if not prompt.strip():
        return None, "Enter a prompt."
    try:
        path = generate_image_fal(prompt.strip(), image_size=image_size or "square_hd")
        from PIL import Image

        img = Image.open(path)
        return img, f"Saved: {path}"
    except Exception as e:
        return None, str(e)


def main():
    with gr.Blocks(title="fal.ai Text → Image") as demo:
        gr.Markdown("## Text → Image (fal.ai) — optional bonus")
        gr.Markdown("Set `FAL_KEY` in your environment ([fal.ai keys](https://fal.ai/dashboard/keys)).")
        with gr.Row():
            prompt = gr.Textbox(label="Prompt", lines=2, value="neon city at night, cinematic")
            out_img = gr.Image(label="Output", type="pil")
        image_size = gr.Dropdown(
            choices=["square_hd", "landscape_4_3", "portrait_4_3", "square"],
            value="square_hd",
            label="Image size",
        )
        status = gr.Textbox(label="Status", interactive=False)
        btn = gr.Button("Generate")
        btn.click(fn=run, inputs=[prompt, image_size], outputs=[out_img, status])

    demo.launch()


if __name__ == "__main__":
    main()
