import os
import re
import textwrap
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import requests

from .config import STRATEGY_COVER_MODE

BCG_GREEN = "#147B58"
TEXT_DARK = "#1E1E1E"
TEXT_MUTED = "#5C6670"
COVER_BG = "#0a1014"
COVER_ACCENT = "#5ec4b4"
COVER_TEXT = "#f2f5f7"


def compact_visual_topic(user_query: str, max_len: int = 110) -> str:
    """
    Short, image-friendly topic (not a full assignment prompt) for FAL / cover art.
    Strips instruction prefixes, turns "… and analyze …" into one briefing-style phrase, then trims.
    """
    q = (user_query or "").strip().replace("\n", " ")
    q = re.sub(
        r"^\s*(research|analyze|analyse|please|i want you to|could you|help me)\s+",
        "",
        q,
        flags=re.I,
    )
    q = re.sub(r"\s+", " ", q).strip()
    # "Topic A and analyze topic B" -> "Topic A — topic B"
    parts = re.split(r"\s+and\s+analyze\s+", q, maxsplit=1, flags=re.I)
    if len(parts) == 2:
        left, right = parts[0].strip(), parts[1].strip()
        right = re.sub(r"^(their|the)\s+impact\s+on\s+", "", right, flags=re.I)
        q = f"{left} - {right}" if right else left
    if len(q) > max_len:
        q = q[:max_len].rsplit(" ", 1)[0] + "…"
    return q or "Strategy topic"


def _score_call(call: Dict[str, object]) -> int:
    direction = str(call.get("direction", "mixed")).lower()
    confidence = int(call.get("confidence", 0))
    sign = 1 if direction == "up" else -1 if direction == "down" else 0
    return sign * confidence


def generate_stock_chart(stock_calls: List[Dict[str, object]], output_dir: Path) -> Optional[Path]:
    if not stock_calls:
        return None
    output_dir.mkdir(parents=True, exist_ok=True)
    chart_path = output_dir / "stock_impact_chart.png"

    ranked = sorted(stock_calls, key=_score_call, reverse=True)[:12]
    labels = [str(item.get("ticker") or item.get("company", "unknown")) for item in ranked]
    scores = [_score_call(item) for item in ranked]
    colors = ["#147B58" if value >= 0 else "#A61B1B" for value in scores]

    plt.figure(figsize=(12, 5))
    plt.bar(labels, scores, color=colors)
    plt.axhline(0, color="#1E1E1E", linewidth=1)
    plt.title("Stock Impact Signal (Direction x Confidence)")
    plt.ylabel("Impact Score")
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    plt.savefig(chart_path, dpi=180)
    plt.close()
    return chart_path


def _strip_light_markdown(text: str) -> str:
    """Remove common markdown markers for plain-text exhibit labels."""
    t = (text or "").strip()
    t = re.sub(r"\*\*(.+?)\*\*", r"\1", t)
    t = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"\1", t)
    return t


def generate_bcg_cover_exhibit(visual_topic: str, output_dir: Path) -> Path:
    """
    Dark consulting-style cover (BCG-inspired): title + topic only — executive takeaways
    stay in the PDF body once (avoids duplicating the same bullets on cover and below).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    image_path = output_dir / "strategy_visual.png"

    from datetime import datetime

    from matplotlib.patches import Circle, Ellipse

    topic = _strip_light_markdown(visual_topic).replace("\n", " ")
    if len(topic) > 200:
        topic = topic[:197] + "…"

    fig = plt.figure(figsize=(12, 6.75), dpi=200)
    fig.patch.set_facecolor(COVER_BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_facecolor(COVER_BG)

    # Abstract organic shapes (right side), green → cyan — no text, no fake data
    for i, (cx, cy, w, h, ang, alpha) in enumerate(
        [
            (0.78, 0.52, 0.42, 0.55, 12, 0.35),
            (0.88, 0.48, 0.28, 0.4, -25, 0.25),
            (0.72, 0.35, 0.22, 0.3, 40, 0.2),
        ]
    ):
        ax.add_patch(
            Ellipse(
                (cx, cy),
                w,
                h,
                angle=ang,
                facecolor=(0.1 + i * 0.05, 0.55 + i * 0.08, 0.5, alpha),
                edgecolor="none",
                transform=ax.transAxes,
            )
        )
    ax.add_patch(
        Circle(
            (0.82, 0.55),
            0.12,
            facecolor=(0.15, 0.7, 0.55, 0.15),
            edgecolor="none",
            transform=ax.transAxes,
        )
    )

    ax.text(
        0.06,
        0.78,
        "Strategy perspectives",
        transform=ax.transAxes,
        color=COVER_TEXT,
        fontsize=11,
        va="top",
        ha="left",
        alpha=0.95,
    )
    wrapped_title = textwrap.fill(topic, width=44)
    ax.text(
        0.06,
        0.68,
        wrapped_title,
        transform=ax.transAxes,
        color=COVER_TEXT,
        fontsize=15,
        fontweight="bold",
        va="top",
        ha="left",
        linespacing=1.2,
    )
    ax.text(
        0.06,
        0.38,
        "Strategy memorandum",
        transform=ax.transAxes,
        color=COVER_ACCENT,
        fontsize=10.5,
        va="top",
        ha="left",
    )
    ax.text(
        0.06,
        0.09,
        datetime.now().strftime("%B %Y").upper(),
        transform=ax.transAxes,
        color="#8a9399",
        fontsize=9,
        va="bottom",
        ha="left",
    )

    fig.savefig(image_path, dpi=220, bbox_inches="tight", pad_inches=0.2, facecolor=COVER_BG)
    plt.close(fig)
    return image_path


def _fal_dark_cover_prompt(visual_topic: str) -> str:
    """Prompt tuned for consulting-style dark covers (abstract art only; no readable text in-image)."""
    return (
        f"Premium corporate strategy briefing cover, ultra-wide 16:9. Subject mood: {visual_topic}. "
        "Very dark charcoal and deep forest green background, minimalist high contrast. "
        "Large abstract organic blob shape on the right third of the frame with fine film grain texture, "
        "smooth gradient from emerald green to cyan blue, soft glow, subtle vignette. "
        "Absolutely no text, no letters, no numbers, no charts, no logos, no watermark, no UI, no people."
    )


def generate_fal_visual(visual_topic: str, output_dir: Path, abstract_only: bool = True) -> Dict[str, Optional[str]]:
    """
    Generate a cover image via FAL (Flux). Image models are poor at legible text/numbers;
    when abstract_only=True (default), the prompt forbids text and charts so output stays illustrative only.
    """
    fal_key = os.getenv("FAL_KEY")
    if not fal_key:
        return {"visual_path": None, "error": "FAL_KEY is missing.", "caption": None}

    try:
        import fal_client
    except ImportError:
        return {"visual_path": None, "error": "fal_client package is not installed.", "caption": None}

    topic = (visual_topic or "").strip() or "strategy topic"
    if abstract_only:
        visual_prompt = _fal_dark_cover_prompt(topic)
    else:
        visual_prompt = (
            f"Create a premium management-consulting report visual for: {topic}. "
            "Style: clean white background, professional business infographic, "
            "minimalist layout, green accent color, no logos, no watermark, "
            "suitable for embedding in a strategy PDF."
        )

    try:
        result = fal_client.run(
            "fal-ai/flux/schnell",
            arguments={
                "prompt": visual_prompt,
                "image_size": "landscape_4_3",
                "num_inference_steps": 12,
            },
        )
        image_url = result.get("images", [{}])[0].get("url")
        if not image_url:
            return {"visual_path": None, "error": "FAL returned no image URL.", "caption": None}

        output_dir.mkdir(parents=True, exist_ok=True)
        image_path = output_dir / "strategy_visual.png"

        response = requests.get(image_url, timeout=30)
        response.raise_for_status()
        image_path.write_bytes(response.content)
        cap = (
            "Cover visual (abstract illustration; FAL — details are in the report body)."
            if abstract_only
            else "AI-generated visual (FAL AI)."
        )
        return {"visual_path": str(image_path), "error": None, "caption": cap}
    except Exception as exc:
        return {"visual_path": None, "error": f"FAL generation failed: {exc}", "caption": None}


def generate_strategy_cover_visual(visual_topic: str, output_dir: Path) -> Dict[str, Optional[str]]:
    """
    Primary cover: dark BCG-inspired title slide (default) — no duplicate of executive bullets.
    Pass compact_visual_topic(user_query) so FAL/matplotlib get a short, understandable theme.

    Set STRATEGY_COVER_MODE=fal_abstract to use FAL only (dark abstract cover prompt).
    """
    mode = STRATEGY_COVER_MODE
    if mode == "fal_abstract":
        return generate_fal_visual(visual_topic, output_dir, abstract_only=True)
    if mode == "fal_infographic":
        return generate_fal_visual(visual_topic, output_dir, abstract_only=False)
    try:
        path = generate_bcg_cover_exhibit(visual_topic, output_dir)
        return {"visual_path": str(path), "error": None, "caption": None}
    except Exception as exc:
        return {"visual_path": None, "error": str(exc), "caption": None}
