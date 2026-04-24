"""
Optional: text-to-image via fal.ai (hosted API). Requires FAL_KEY in the environment.
Install: pip install fal-client
"""
from __future__ import annotations

import argparse
import io
import os
from datetime import datetime
from pathlib import Path

from PIL import Image
import requests


def _load_fal_key_from_dotenv() -> None:
    """Load FAL_KEY from workspace root .env if it is not already set."""
    if os.environ.get("FAL_KEY"):
        return

    env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if not env_path.exists():
        return

    for line in env_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        if key.strip() == "FAL_KEY":
            os.environ["FAL_KEY"] = value.strip().strip("'\"")
            return


def default_model() -> str:
    return "fal-ai/flux/schnell"


def generate_image_fal(
    prompt: str,
    *,
    model_id: str | None = None,
    image_size: str = "square_hd",
) -> Path:
    _load_fal_key_from_dotenv()
    api_key = os.environ.get("FAL_KEY")
    if not api_key:
        raise RuntimeError("Set FAL_KEY (see https://fal.ai/dashboard/keys)")

    try:
        import fal_client  # type: ignore
    except ImportError as e:
        raise RuntimeError("Install fal-client: pip install fal-client") from e

    model_id = model_id or default_model()
    result = fal_client.subscribe(
        model_id,
        arguments={
            "prompt": prompt,
            "image_size": image_size,
            "num_inference_steps": 4,
            "num_images": 1,
            "enable_safety_checker": True,
        },
    )

    # fal-ai/flux/schnell returns images with url fields
    images = result.get("images") or result.get("image") or []
    if isinstance(images, dict):
        images = [images]
    if not images:
        raise RuntimeError(f"Unexpected API response: {result!r}")

    first = images[0]
    url = first.get("url") if isinstance(first, dict) else str(first)
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    pil = Image.open(io.BytesIO(r.content)).convert("RGB")

    out_dir = Path(__file__).resolve().parent / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in prompt[:40])
    path = out_dir / f"fal_{stamp}_{safe}.png"
    pil.save(path)
    print(f"Saved: {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="fal.ai text-to-image (optional bonus)")
    parser.add_argument("prompt", nargs="?", default="a robot reading a book in a library")
    parser.add_argument("--model", default=None, help="fal model id (default: fal-ai/flux/schnell)")
    parser.add_argument("--image-size", default="square_hd", dest="image_size")
    args = parser.parse_args()

    generate_image_fal(args.prompt, model_id=args.model, image_size=args.image_size)


if __name__ == "__main__":
    main()
