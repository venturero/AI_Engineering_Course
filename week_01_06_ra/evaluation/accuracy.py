"""
Semantic instruction-following accuracy for domain-specific LLM / GenAI QA.

What this metric measures:
- Whether generated answers cover required domain concepts for each prompt.

When useful:
- Assessing instruction quality where wording can vary but meaning must be correct.
- Validating technical answers about transformers, training, and alignment beyond surface overlap.

How to read scores:
- Higher is better (0.0 to 1.0).
- Good score: most required concepts are present with correct semantic cues.
- Bad score: missing key technical distinctions despite fluent language.
"""

from __future__ import annotations

import re
from statistics import mean


def build_generation_prefix(instruction: str) -> str:
    """Build the same instruction-response prefix used during SFT."""
    return f"### Instruction:\n{instruction.strip()}\n\n### Response:\n"


def _contains_any(text: str, patterns: list[str]) -> bool:
    return any(re.search(pattern, text) for pattern in patterns)


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower().strip())


def _concepts() -> dict[str, dict[str, list[str]]]:
    return {
        "Explain what instruction fine-tuning is and why it is used in large language models.": {
            "supervised instruction-response training": [
                r"\bsupervised\b",
                r"\binstruction[-– ]response\b",
                r"\binstruction\b.*\btrain",
            ],
            "goal: follow instructions / usefulness": [
                r"\bfollow\b",
                r"\busability\b",
                r"\bcontrollability\b",
                r"\balignment\b",
                r"\buseful\b",
            ],
            "pretrained model as starting point": [r"\bpretrain\w*\b", r"\bpre-trained\b", r"\bbase model\b", r"\blanguage model\b"],
        },
        "Describe the role of the transformer architecture in modern large language models.": {
            "self-attention vs recurrence": [r"\bself-attention\b", r"\battention\b", r"\brecurr\w*\b", r"\brnn\b"],
            "long-range dependencies / context": [r"\blong[- ]range\b", r"\bdependenc\w*\b", r"\bcontext\b"],
            "scalability / sequence efficiency": [r"\bscal\w+\b", r"\befficien\w*\b", r"\bparallel\b"],
        },
        "What is self-attention and how does it help language models understand context?": {
            "tokens attend to other tokens": [r"\btoken\b", r"\battend\b", r"\bweigh\w*\b", r"\bimportance\b"],
            "contextual representation": [r"\bcontext\w*\b", r"\brepresentation\b", r"\bsequence\b"],
            "full input / pairwise": [r"\ball\b", r"\bentire\b", r"\beach\b", r"\bother tokens\b"],
        },
        "Explain the difference between pretraining and fine-tuning in large language models.": {
            "pretraining: unlabeled / next-token": [r"\bpretrain\w*\b", r"\bunlabeled\b", r"\bnext[- ]token\b", r"\blarge\b.*\btext\b"],
            "fine-tuning: smaller labeled / adaptation": [r"\bfine[- ]tun\w*\b", r"\blabeled\b", r"\bsmaller\b", r"\badapt\b"],
            "task or behavior specialization": [r"\btask\b", r"\binstruction\b", r"\bclassif\w*\b", r"\bbehavior\b"],
        },
        "What problem does causal attention solve in autoregressive language models?": {
            "blocks future tokens": [r"\bfuture\b", r"\bnot.*see\b", r"\bmask\b", r"\bprevent\b"],
            "autoregressive property": [r"\bautoregressive\b", r"\btoken by token\b", r"\bgenerat\w*\b"],
            "depends only on past context": [r"\bprevious\b", r"\bpast\b", r"\bearlier\b", r"\bonly\b"],
        },
        "Describe the purpose of positional embeddings in transformer-based language models.": {
            "encode order / position": [r"\bposition\w*\b", r"\border\b", r"\bsequence\b"],
            "attention is permutation-sensitive fix": [r"\bposition[- ]agnostic\b", r"\bpermutation\b", r"\borderless\b", r"\bwithout\b.*\border\b"],
            "added to token embeddings": [r"\badd\w*\b", r"\btoken embedd\w*\b", r"\bembedd\w*\b"],
        },
        "What is multi-head attention and why is it used instead of single-head attention?": {
            "multiple parallel heads": [r"\bmulti[- ]head\b", r"\bparallel\b", r"\bseveral\b.*\bhead\b"],
            "diverse relationships / subspaces": [r"\bdiverse\b", r"\bdifferent\b.*\baspect\b", r"\brelationship\b"],
            "stronger than single head": [r"\bcompared\b", r"\bsingle[- ]head\b", r"\bmore\b.*\bcapacity\b", r"\brepresentational\b"],
        },
        "Explain the role of layer normalization in training deep transformer models.": {
            "stabilizes training": [r"\bstabil\w*\b", r"\btrain\w*\b"],
            "normalize activations / features": [r"\bnorm\w+\b", r"\bactivat\w*\b", r"\bfeature\b"],
            "gradients / deeper networks": [r"\bgradient\b", r"\bdeep\w*\b", r"\bcovariate\b"],
        },
        "What is a feed-forward network in a transformer block and why is it needed?": {
            "per-token MLP after attention": [r"\bfeed[- ]forward\b", r"\bmlp\b", r"\bafter attention\b", r"\beach token\b"],
            "nonlinearity / capacity": [r"\bnonlinear\b", r"\bcapacity\b", r"\btransform\b"],
            "expand and contract hidden dim": [
                r"\bexpand\w*\b",
                r"\bcontract\w*\b",
                r"\bdimension\b",
                r"\bhidden\b",
                r"\bembedding space\b",
            ],
        },
        "Explain why instruction datasets typically exclude rejected or incorrect answers during supervised fine-tuning.": {
            "high-quality correct demonstrations": [r"\bhigh[- ]quality\b", r"\bcorrect\b", r"\bdemonstrat\w*\b", r"\bexclude\b"],
            "rejected answers harm signal": [r"\breject\w*\b", r"\bconfus\w*\b", r"\bnoise\b", r"\bincorrect\b"],
            "preference methods named": [r"\bdpo\b", r"\brlhf\b", r"\bpreference\b"],
        },
    }


CONCEPTS_BY_PROMPT = _concepts()


def generate_model_answer(model, tokenizer, prompt: str, max_new_tokens: int = 120) -> str:
    """Generate deterministic answer text for one instruction prompt."""
    prefix = build_generation_prefix(prompt)
    enc = tokenizer(prefix, return_tensors="pt")
    input_len = enc["input_ids"].shape[1]
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        repetition_penalty=1.05,
    )
    new_tokens = out[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def evaluate_prompt_accuracy(prompt: str, model_answer: str) -> dict:
    """Score concept coverage for a single prompt."""
    concept_map = CONCEPTS_BY_PROMPT.get(prompt, {})
    normalized = _normalize(model_answer)

    matched = []
    missed = []
    for concept_name, patterns in concept_map.items():
        if _contains_any(normalized, patterns):
            matched.append(concept_name)
        else:
            missed.append(concept_name)

    total = len(concept_map)
    score = (len(matched) / total) if total else 0.0
    return {
        "score": round(score, 4),
        "matched_concepts": matched,
        "missed_concepts": missed,
        "total_concepts": total,
    }


def _row_prompt(row: dict) -> str:
    return str(row.get("prompt") or row.get("instruction") or "").strip()


def _row_reference(row: dict) -> str:
    return str(row.get("chosen") or row.get("output") or row.get("completion") or "").strip()


def evaluate_instruction_accuracy(dataset_rows: list[dict], model, tokenizer) -> dict:
    """Run semantic concept-based accuracy over a list of instruction rows."""
    per_question = []
    for row in dataset_rows:
        prompt = _row_prompt(row)
        if not prompt:
            continue
        reference = _row_reference(row)
        answer = generate_model_answer(model, tokenizer, prompt)
        score = evaluate_prompt_accuracy(prompt, answer)

        per_question.append(
            {
                "prompt": prompt,
                "reference_chosen": reference,
                "model_answer": answer,
                "accuracy": score["score"],
                "matched_concepts": score["matched_concepts"],
                "missed_concepts": score["missed_concepts"],
                "total_concepts": score["total_concepts"],
            }
        )

    avg = mean([x["accuracy"] for x in per_question]) if per_question else 0.0
    return {
        "average_accuracy": round(avg, 4),
        "num_prompts": len(per_question),
        "per_question": per_question,
    }
