"""
Perplexity evaluation utilities for causal language models.

What this metric measures:
- Average next-token negative log-likelihood on unseen text.
- Perplexity (exp(loss)) as an interpretable uncertainty proxy.

When useful:
- Comparing language-modeling fluency between checkpoints.
- Detecting catastrophic degradation after fine-tuning.

How to read scores:
- Lower is better.
- Good score: fine-tuned model has similar or lower perplexity than base model.
- Bad score: perplexity rises significantly, suggesting overfitting or drift.
"""

from __future__ import annotations

import math
from typing import Iterable

import torch


def get_unseen_eval_texts() -> list[str]:
    """Return a small unseen LLM/GenAI-style text set for perplexity evaluation."""
    return [
        (
            "Large language models are trained with self-supervised objectives such as "
            "next-token prediction, then adapted with supervised fine-tuning on curated instructions."
        ),
        (
            "Transformer blocks interleave multi-head attention with position-wise feed-forward "
            "layers, using residual connections and normalization to stabilize optimization."
        ),
        (
            "Causal masking ensures each position only attends to prior tokens, which preserves "
            "the autoregressive factorization used during both training and sampling."
        ),
        (
            "Retrieval-augmented generation grounds answers in external documents, reducing "
            "hallucination when factual updates are required beyond the model cutoff."
        ),
        (
            "Alignment methods such as preference optimization or reinforcement learning from "
            "human feedback adjust behavior to better match safety and usefulness constraints."
        ),
    ]


@torch.inference_mode()
def compute_average_loss(
    model,
    tokenizer,
    texts: Iterable[str],
    max_length: int = 256,
) -> float:
    """Compute token-weighted average loss on unseen text."""
    model.eval()
    total_weighted_loss = 0.0
    total_tokens = 0

    for text in texts:
        enc = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
        )
        input_ids = enc["input_ids"]
        attn_mask = enc.get("attention_mask")

        out = model(input_ids=input_ids, attention_mask=attn_mask, labels=input_ids)
        token_count = int(attn_mask.sum().item()) if attn_mask is not None else int(input_ids.numel())
        total_weighted_loss += float(out.loss.item()) * token_count
        total_tokens += token_count

    if total_tokens == 0:
        raise ValueError("No valid tokens for perplexity evaluation.")

    return total_weighted_loss / total_tokens


def loss_to_perplexity(avg_loss: float) -> float:
    """Convert average negative log-likelihood to perplexity."""
    return math.exp(avg_loss)


def evaluate_perplexity_comparison(base_model, fine_tuned_model, tokenizer) -> dict:
    """Evaluate and compare perplexity between base and fine-tuned models."""
    texts = get_unseen_eval_texts()

    base_loss = compute_average_loss(base_model, tokenizer, texts)
    ft_loss = compute_average_loss(fine_tuned_model, tokenizer, texts)

    base_ppl = loss_to_perplexity(base_loss)
    ft_ppl = loss_to_perplexity(ft_loss)

    return {
        "base_avg_loss": round(base_loss, 6),
        "base_perplexity": round(base_ppl, 6),
        "fine_tuned_avg_loss": round(ft_loss, 6),
        "fine_tuned_perplexity": round(ft_ppl, 6),
        "perplexity_delta_ft_minus_base": round(ft_ppl - base_ppl, 6),
    }
