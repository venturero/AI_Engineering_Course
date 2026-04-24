"""
Week-4 Evaluation Pipeline entry point.

This module provides a production-oriented, explainable and extensible evaluation flow:
1) Perplexity comparison (base vs fine-tuned)
2) Semantic instruction-following accuracy
3) Dry-run benchmark with diagnostics and JSON output

README-style notes:
- Perplexity alone is insufficient because it measures token prediction confidence,
  not whether a model follows compliance instructions correctly.
- Instruction accuracy is essential in domain LLMs because regulatory answers must
  cover specific concepts even when wording varies.
- Together, these metrics prepare the model for RAG/agent workflows by validating
  both linguistic stability (perplexity) and task alignment (instruction accuracy).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

try:
    from .accuracy import evaluate_instruction_accuracy
    from .benchmark import print_benchmark_summary, run_dry_run_benchmark
    from .perplexity import evaluate_perplexity_comparison
except ImportError:
    # Supports direct execution: `python evaluation/run_eval.py`
    from accuracy import evaluate_instruction_accuracy
    from benchmark import print_benchmark_summary, run_dry_run_benchmark
    from perplexity import evaluate_perplexity_comparison


def load_instruction_dataset(dataset_path: str | Path) -> list[dict]:
    """Load prompt/chosen/rejected dataset from JSON list file."""
    path = Path(dataset_path)
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list):
        raise ValueError(f"Expected a JSON list in {path}")
    return rows


def run_full_evaluation(
    base_model_id: str,
    fine_tuned_model_path: str,
    dataset_path: str,
    dry_run_prompts: int,
    benchmark_output_json: str,
) -> dict:
    """Run full Week-4 evaluation pipeline and return all results."""
    print("\n[eval] Loading tokenizer and models...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_id, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(base_model_id)

    ft_path = Path(fine_tuned_model_path)
    if ft_path.exists() and any(ft_path.iterdir()):
        try:
            ft_model = AutoModelForCausalLM.from_pretrained(str(ft_path))
        except Exception as exc:  # pragma: no cover - defensive fallback
            print(f"[eval][warn] Could not load fine-tuned model from {ft_path}: {exc}")
            print("[eval][warn] Falling back to base model for pipeline smoke-check.")
            ft_model = AutoModelForCausalLM.from_pretrained(base_model_id)
    else:
        print(f"[eval][warn] Fine-tuned path missing/empty: {ft_path}")
        print("[eval][warn] Falling back to base model for pipeline smoke-check.")
        ft_model = AutoModelForCausalLM.from_pretrained(base_model_id)
    base_model.eval()
    ft_model.eval()

    rows = load_instruction_dataset(dataset_path)

    print("[eval] Running perplexity comparison...")
    ppl = evaluate_perplexity_comparison(base_model, ft_model, tokenizer)
    print(
        "[eval][perplexity] base={:.4f} | fine_tuned={:.4f} | delta(ft-base)={:.4f}".format(
            ppl["base_perplexity"],
            ppl["fine_tuned_perplexity"],
            ppl["perplexity_delta_ft_minus_base"],
        )
    )

    print("[eval] Running semantic instruction accuracy...")
    acc = evaluate_instruction_accuracy(rows, ft_model, tokenizer)
    print(
        "[eval][accuracy] average={:.4f} over {} prompts".format(
            acc["average_accuracy"], acc["num_prompts"]
        )
    )

    print(f"[eval] Running dry-run benchmark on {dry_run_prompts} prompts...")
    benchmark = run_dry_run_benchmark(
        dataset_rows=rows,
        model=ft_model,
        tokenizer=tokenizer,
        max_prompts=dry_run_prompts,
        output_path=benchmark_output_json,
    )
    print_benchmark_summary(benchmark)
    print(f"[eval] Benchmark JSON saved: {benchmark_output_json}")

    return {"perplexity": ppl, "accuracy": acc, "benchmark": benchmark}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Week-4 model evaluation pipeline.")
    root = Path(__file__).resolve().parent.parent
    parser.add_argument("--base-model-id", default="distilgpt2")
    parser.add_argument("--fine-tuned-model-path", default=str(root / "sft-output-distilgpt2"))
    parser.add_argument("--dataset-path", default=str(root / "Instruction_Style_Dataset.json"))
    parser.add_argument("--dry-run-prompts", type=int, default=10)
    parser.add_argument(
        "--benchmark-output-json", default=str(root / "evaluation" / "dry_run_benchmark.json")
    )
    args = parser.parse_args()

    run_full_evaluation(
        base_model_id=args.base_model_id,
        fine_tuned_model_path=args.fine_tuned_model_path,
        dataset_path=args.dataset_path,
        dry_run_prompts=args.dry_run_prompts,
        benchmark_output_json=args.benchmark_output_json,
    )


if __name__ == "__main__":
    main()
