from typing import List

from .schemas import EvaluationResult, RetrievedDoc


REQUIRED_HEADINGS = [
    "## Executive Summary",
    "## Winners Right Now",
    "## Market & Technology Overview",
    "## Stock Impact (Up/Down)",
    "## Risks & What Could Break The Thesis",
    "## Action Agenda",
]


def evaluate_report(report: str, docs: List[RetrievedDoc]) -> EvaluationResult:
    doc_titles = [doc.title.lower() for doc in docs if doc.title]
    doc_urls = [doc.source.lower() for doc in docs if doc.source]
    report_lower = report.lower()

    title_hits = sum(1 for title in doc_titles if title in report_lower)
    url_hits = sum(1 for url in doc_urls if url in report_lower)
    factual_grounding = min(1.0, (title_hits + url_hits) / max(1, len(doc_titles) + len(doc_urls)))

    coverage_hits = sum(1 for heading in REQUIRED_HEADINGS if heading.lower() in report_lower)
    coverage = coverage_hits / len(REQUIRED_HEADINGS)

    has_stock_direction = (" up" in report_lower) or (" down" in report_lower)
    coherence = 1.0 if all(h.lower() in report_lower for h in REQUIRED_HEADINGS) and has_stock_direction else 0.5

    notes = {
        "factual_grounding": (
            "Measures whether source titles/URLs are explicitly referenced, indicating traceable evidence use."
        ),
        "coverage": "Measures whether all mandatory decision-oriented sections are present.",
        "coherence": "Penalizes reports that miss explicit stock direction signals (up/down).",
    }
    return EvaluationResult(
        factual_grounding_score=round(factual_grounding, 3),
        coverage_score=round(coverage, 3),
        coherence_score=round(coherence, 3),
        notes=notes,
    )

