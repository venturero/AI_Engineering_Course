"""
Dry-run benchmark utilities.

What this benchmark measures:
- Practical instruction-following quality on a small slice (10-20 prompts).
- Per-prompt diagnostics that highlight missing regulatory concepts.

When useful:
- Fast iteration loop during model improvements.
- Regression checks before broader evaluation runs.
"""

from __future__ import annotations

import json
from pathlib import Path

from .accuracy import evaluate_instruction_accuracy


def _diagnostic_note(missed_concepts: list[str]) -> str:
    if not missed_concepts:
        return "covered expected concepts"
    first = missed_concepts[0]
    if "currency" in first:
        return "missed currency distinction"
    if "risk weight" in first or "sovereign floor" in first:
        return "missed risk-weight treatment detail"
    if "join" in first or "primary key" in first:
        return "missed data-join integrity concept"
    return f"missed concept: {first}"


def run_dry_run_benchmark(
    dataset_rows: list[dict],
    model,
    tokenizer,
    max_prompts: int = 10,
    output_path: str | Path = "evaluation/dry_run_benchmark.json",
) -> dict:
    """Run small benchmark and save prompt-level outputs as JSON."""
    subset = dataset_rows[:max_prompts]
    accuracy_results = evaluate_instruction_accuracy(subset, model, tokenizer)

    benchmark_rows = []
    for row in accuracy_results["per_question"]:
        benchmark_rows.append(
            {
                "prompt": row["prompt"],
                "model_answer": row["model_answer"],
                "accuracy": row["accuracy"],
                "diagnostic_note": _diagnostic_note(row["missed_concepts"]),
            }
        )

    payload = {
        "num_prompts": len(benchmark_rows),
        "average_accuracy": accuracy_results["average_accuracy"],
        "results": benchmark_rows,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return payload


def print_benchmark_summary(benchmark_payload: dict) -> None:
    """Print concise, readable benchmark summary to console."""
    print("\n=== Dry-Run Benchmark Summary ===")
    print(f"Prompts: {benchmark_payload['num_prompts']}")
    print(f"Average accuracy: {benchmark_payload['average_accuracy']:.4f}")
    for idx, row in enumerate(benchmark_payload["results"], start=1):
        print(f"{idx:02d}. score={row['accuracy']:.4f} | {row['diagnostic_note']}")
