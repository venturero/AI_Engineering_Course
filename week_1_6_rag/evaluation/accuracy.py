"""
Semantic instruction-following accuracy for domain-specific regulatory QA.

What this metric measures:
- Whether generated answers cover required domain concepts for each prompt.

When useful:
- Assessing instruction quality where wording can vary but meaning must be correct.
- Validating policy/regulation answers beyond surface text overlap.

How to read scores:
- Higher is better (0.0 to 1.0).
- Good score: most required concepts are present with correct semantic cues.
- Bad score: missing key regulatory distinctions despite fluent language.
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
        "Under Basel IV, under what conditions is a 0% risk weight applied to a sovereign counterparty? Please specify the distinction between local and foreign currency.": {
            "local currency condition": [r"\blocal currency\b", r"\bdomestic currency\b"],
            "foreign currency distinction": [r"\bforeign currency\b", r"\bnon[- ]local currency\b"],
            "sovereign floor / elevated treatment": [r"\bsovereign floor\b", r"\bhigher risk weight\b", r"\belevated risk\b"],
            "risk weight context": [r"\brisk weight\b", r"\b0%\b"],
        },
        "In the CRM_GUARANTEE_BSL4 process, what is the approach to including collateral in the ERD calculation? Please explain the common practice across banks.": {
            "collateral included for ERD context": [r"\bcollateral\b", r"\berd\b"],
            "post-ERD allocation logic": [r"\bpost[- ]erd\b", r"\bafter erd\b", r"\bdistribut\w+\b"],
            "common banking practice mention": [r"\bcommon practice\b", r"\bmajority\b", r"\bmost banks\b"],
        },
        "What does the EXP_FLG field represent in exposure tables? Explain the impact of values 1 and 0 on risk calculation.": {
            "exp_flg = 1 primary exposure": [r"\bexp_flg\b", r"\b1\b", r"\bprimary\b", r"\bmain\b"],
            "exp_flg = 0 collateral/mitigant": [r"\b0\b", r"\bcollateral\b", r"\bmitigant\b"],
            "risk impact distinction": [r"\breduce risk\b", r"\brwa\b", r"\bmain source of risk\b"],
        },
        "How are maturity and risk weight determined in repo transactions? What method is used if the original maturity is not available?": {
            "repo date window": [r"\brepo inception\b", r"\brepo maturity\b", r"\bdifference between\b"],
            "fallback when original maturity missing": [r"\bif .*maturity.*not available\b", r"\bfallback\b", r"\bcalculated\b"],
            "short-term vs long-term classification": [r"\bshort[- ]term\b", r"\blong[- ]term\b", r"\bclassification\b"],
        },
        "What is the risk weight multiplier applied to retail customers under Basel IV, and under what circumstances is it applied?": {
            "retail scope": [r"\bretail\b"],
            "1.5 multiplier": [r"\b1\.5\b", r"\bmultiplier\b"],
            "conditional application": [r"\bunder certain conditions\b", r"\bconservative\b", r"\bcircumstances\b"],
        },
        "Why is 'future collateral (fc)' not considered in the CRM Value calculation? Which value is ultimately used?": {
            "future collateral excluded": [r"\bfuture collateral\b", r"\bnot considered\b", r"\bexcluded\b"],
            "regulatory reason": [r"\bregulator\w*\b", r"\bregulatory rules\b", r"\bnot allowed\b"],
            "final value usage": [r"\bcrm value\b", r"\bas[- ]is\b", r"\bcrm value fc = crm value\b"],
        },
        "How is the risk weight treated for a foreign currency repo transaction of a bank with missing rating information?": {
            "foreign currency repo context": [r"\bforeign currency\b", r"\brepo\b"],
            "missing rating implication": [r"\bmissing rating\b", r"\bno rating\b"],
            "sovereign floor/country risk conservative treatment": [r"\bsovereign floor\b", r"\bcountry risk\b", r"\b100%\b", r"\bhigher risk\b"],
        },
        "What is the purpose of the SIM-FLAG field and why is it critical in the production (prod) process?": {
            "simulation vs production distinction": [r"\bsim[- ]flag\b", r"\bsimulation\b", r"\bproduction\b"],
            "prevent contamination of prod outputs": [r"\bprevent\b", r"\bproduction output\b", r"\baccident\w+\b"],
        },
        "What do Currency Type values 'A' and 'D' represent? What are their differences in risk calculation?": {
            "A as average exchange rate": [r"\bcurrency type\b", r"\b'a'\b", r"\baverage exchange rate\b"],
            "D as daily exchange rate": [r"\b'd'\b", r"\bdaily exchange rate\b"],
            "calculation usage distinction": [r"\breference date\b", r"\bspecific day\b", r"\bdifference\b"],
        },
        "Why are COUNTERPARTY_RK or ENTITY_ID critical fields when joining exposure and CRM tables?": {
            "primary key/customer uniqueness": [r"\bcounterparty_rk\b", r"\bentity_id\b", r"\bprimary key\b", r"\buniqueness\b"],
            "join correctness": [r"\bjoin\b", r"\bincorrect or missing\b", r"\bduplication\b"],
            "risk/provisioning impact": [r"\brwa\b", r"\boverstated\b", r"\bprovision\w+\b"],
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


def evaluate_instruction_accuracy(dataset_rows: list[dict], model, tokenizer) -> dict:
    """Run semantic concept-based accuracy over a list of instruction rows."""
    per_question = []
    for row in dataset_rows:
        prompt = row["prompt"]
        reference = row.get("chosen", "")
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
