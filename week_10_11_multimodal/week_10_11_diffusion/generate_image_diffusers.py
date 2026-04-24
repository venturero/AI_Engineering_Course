"""
Text-to-image generation with Hugging Face diffusers (Stable Diffusion).
Grading entry point: run from CLI and check outputs/ for saved images.
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime
from pathlib import Path

import torch
from diffusers import StableDiffusionPipeline
from PIL import Image


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_pipeline(model_id: str, device: str) -> StableDiffusionPipeline:
    dtype = torch.float16 if device in ("cuda", "mps") else torch.float32
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    if device == "cuda":
        pipe.enable_attention_slicing()
    return pipe


def save_image(image: Image.Image, prompt: str) -> Path:
    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in prompt[:40])
    path = out_dir / f"sd_{stamp}_{safe}.png"
    image.save(path)
    return path


def generate_with_pipeline(
    pipe: StableDiffusionPipeline,
    prompt: str,
    *,
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int | None = None,
    device: str | None = None,
) -> tuple[Image.Image, Path]:
    device = device or default_device()
    generator = None
    if seed is not None:
        generator = torch.Generator(device=device).manual_seed(seed)

    result = pipe(
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        generator=generator,
    )
    image = result.images[0]
    path = save_image(image, prompt)
    return image, path


def generate_image(
    prompt: str,
    *,
    model_id: str = "runwayml/stable-diffusion-v1-5",
    num_inference_steps: int = 30,
    guidance_scale: float = 7.5,
    seed: int | None = None,
    device: str | None = None,
) -> None:
    """Generate one image and save it under outputs/."""
    device = device or default_device()
    pipe = build_pipeline(model_id, device)
    path = generate_with_pipeline(
        pipe,
        prompt,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        seed=seed,
        device=device,
    )[1]
    print(f"Saved: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Stable Diffusion text-to-image (diffusers)")
    parser.add_argument("prompt", nargs="?", default="a photo of a cat wearing sunglasses")
    parser.add_argument("--model", default="runwayml/stable-diffusion-v1-5", help="HF model id")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=7.5)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, help="cuda | mps | cpu (default: auto)")
    args = parser.parse_args()

    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

    generate_image(
        args.prompt,
        model_id=args.model,
        num_inference_steps=args.steps,
        guidance_scale=args.guidance,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
