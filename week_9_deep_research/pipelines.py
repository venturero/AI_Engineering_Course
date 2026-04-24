from __future__ import annotations

from dataclasses import dataclass

from .llm import LLM


@dataclass(frozen=True)
class DeepResearchResult:
    draft: str
    critique: str
    revised: str


def naive_chain(llm: LLM, question: str) -> str:
    system = (
        "You are a helpful assistant. Answer the user's question clearly and concisely. "
        "If assumptions are required, state them briefly."
    )
    return llm.chat(system=system, user=question, temperature=0.2).text


def deep_research_chain(llm: LLM, question: str) -> DeepResearchResult:
    draft_system = (
        "You are a helpful assistant. Produce a strong first draft answer. "
        "Be clear and structured, but do not overthink."
    )
    draft = llm.chat(system=draft_system, user=question, temperature=0.3).text

    critique_system = (
        "CRITIQUE MODE. You will be given a draft answer. Critique it.\n"
        "- Identify what is missing, unclear, incorrect, or weak.\n"
        "- Give concrete improvement suggestions.\n"
        "- Do NOT rewrite the full answer.\n"
        "Respond as bullet points."
    )
    critique_user = f"Question:\n{question}\n\nDraft answer:\n{draft}"
    critique = llm.chat(system=critique_system, user=critique_user, temperature=0.2).text

    revise_system = (
        "REVISE MODE. You will be given a question, a draft answer, and a critique.\n"
        "- Rewrite the answer to address the critique.\n"
        "- Keep it factual and clear.\n"
        "- Preserve what's good; fix what's weak.\n"
        "- Output ONLY the revised answer."
    )
    revise_user = f"Question:\n{question}\n\nDraft:\n{draft}\n\nCritique:\n{critique}"
    revised = llm.chat(system=revise_system, user=revise_user, temperature=0.2).text

    return DeepResearchResult(draft=draft, critique=critique, revised=revised)

