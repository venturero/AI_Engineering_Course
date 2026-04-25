"""
Minimal supervised fine-tuning (SFT) on CPU with Hugging Face TRL `SFTTrainer`.

Run (recommended on Windows so TRL can read UTF-8 template files):
    python -X utf8 main.py

The script also re-executes itself with UTF-8 mode when needed (see `_ensure_utf8`).

Use a dataset with `prompt`/`instruction` + `completion` (from `chosen` or `output`) so TRL applies
completion-only loss. A single `text` column is trained as plain LM (full-sequence
loss), which dilutes learning on answers.

`distilgpt2` is small and not instruction-tuned; a handful of mixed examples cannot
reliably anchor one QA pair. This script upsamples the first row so the bundled
demo question matches the labeled answer; for new questions, use more data, a
larger instruct model (GPU), or RAG over your source docs.

Alternative models: tiiuae/falcon-7b-instruct, mistralai/Mistral-7B-Instruct (GPU);
for CPU seq2seq demos: google/flan-t5-small.
"""

from __future__ import annotations

import os
import sys
import argparse
import re
from pathlib import Path


def _ensure_project_venv() -> None:
    """
    Re-exec this script with the workspace .venv interpreter when available.

    This avoids dependency mismatches when the script is launched via another
    project's virtual environment or the global/base Python.
    """
    script_path = Path(__file__).resolve()
    workspace_root = script_path.parent.parent
    if sys.platform == "win32":
        expected_python = workspace_root / ".venv" / "Scripts" / "python.exe"
    else:
        expected_python = workspace_root / ".venv" / "bin" / "python"

    if not expected_python.exists():
        return

    current_python = Path(sys.executable).resolve()
    if current_python == expected_python.resolve():
        return

    os.execv(str(expected_python), [str(expected_python), *sys.argv])


def _ensure_utf8() -> None:
    """On Windows, default locale encoding can break TRL imports; UTF-8 mode fixes it."""
    if sys.platform != "win32":
        return
    if getattr(sys.flags, "utf8_mode", False):
        return
    script = os.path.abspath(__file__)
    os.execv(sys.executable, [sys.executable, "-X", "utf8", script, *sys.argv[1:]])


_ensure_project_venv()
_ensure_utf8()


def _parse_cli_args() -> argparse.Namespace:
    """Parse script-level toggles without interfering with TRL/HF args."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--run-eval", action="store_true")
    parser.add_argument("--run-eval-after-train", action="store_true")
    parser.add_argument("--skip-eval-after-train", action="store_true")
    parser.add_argument("--dry-run-prompts", type=int, default=10)
    parser.add_argument(
        "--question",
        action="append",
        default=[],
        help="Question to run through generation. Repeat --question for multiple prompts.",
    )
    parser.add_argument(
        "--rag-mode",
        choices=("auto", "on", "off"),
        default="auto",
        help=(
            "RAG mode: auto=decide per question with gate logic (default), "
            "on=always allow retrieval pipeline, off=disable retrieval."
        ),
    )
    parser.add_argument(
        "--use-rag",
        action="store_true",
        help="Deprecated alias for --rag-mode on (kept for backward compatibility).",
    )
    parser.add_argument(
        "--rag-document",
        type=str,
        default=r"C:\Users\rvent\Desktop\code\works\AI_ENG\week_1_6_rag\Build a Large Language Model (From Scratch).pdf",
        help="Path to the document file to index for retrieval (.docx or text).",
    )
    args, _ = parser.parse_known_args(sys.argv[1:])
    return args


CLI_ARGS = _parse_cli_args()


def _maybe_run_evaluation() -> None:
    """
    Keep evaluation pipeline outside training script internals.
    When `--run-eval` is provided, delegate to `evaluation/run_eval.py`.
    """
    if CLI_ARGS.run_eval:
        from evaluation.run_eval import main as run_eval_main

        sys.argv = [sys.argv[0], *[arg for arg in sys.argv[1:] if arg != "--run-eval"]]
        run_eval_main()
        raise SystemExit(0)


_maybe_run_evaluation()

import time
import random

# Wall-clock anchor: includes PyTorch / Transformers / TRL import time (often the slowest part).
_RUN_T0 = time.perf_counter()

import numpy as np
import torch
from datasets import concatenate_datasets, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

# Reproducibility (helps compare before/after on the same prompt)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Same folder as this script: Instruction_Style_Dataset.json
DATA_PATH = Path(__file__).resolve().parent / "Instruction_Style_Dataset.json"
MODEL_ID = "distilgpt2"
OUTPUT_DIR = Path(__file__).resolve().parent / "sft-output-distilgpt2"
# Tiny-data regime: repeat the first (demo) QA pair so the exact instruction→answer mapping is learned.
# Without this, 10 heterogeneous examples rarely yield correct answers on held prompts for small LMs.
DEMO_ROW_UPSAMPLE = 24


def build_prompt_completion(example: dict) -> dict:
    """
    TRL prompt-completion format: loss can be masked to the answer only (`completion_only_loss`).

    A single `text` column is treated as plain language modeling (full-sequence loss), which trains
    the model to predict the template and instruction tokens too and weakens answer quality.
    """
    instruction = str(example.get("prompt") or example.get("instruction") or "").strip()
    answer = str(example.get("chosen") or example.get("output") or example.get("completion", "")).strip()
    return {
        "prompt": f"### Instruction:\n{instruction}\n\n### Response:\n",
        "completion": answer,
    }


def build_generation_prefix(instruction: str, context: str | None = None) -> str:
    """
    Build the prompt prefix for inference.

    When `context` is non-empty, it is prepended as ### Context: so the model sees
    retrieved evidence before the instruction. Training still uses instruction-only
    prefixes via `build_prompt_completion` (unchanged when RAG is off).
    """
    inst = instruction.strip()
    ctx = (context or "").strip()
    if ctx:
        return f"### Context:\n{ctx}\n\n### Instruction:\n{inst}\n\n### Response:\n"
    return f"### Instruction:\n{inst}\n\n### Response:\n"


@torch.inference_mode()
def generate_answer(
    model,
    tokenizer,
    instruction: str,
    max_new_tokens: int = 120,
    context: str | None = None,
) -> str:
    """
    Decodes only new tokens after the prompt. Any line that looks like a question (e.g. “What is attention…”)
    is **model-generated text**, not the app “asking” you something — base distilgpt2 was not trained to
    follow Instruction/Response chat templates, so continuations are often irrelevant or repetitive.
    Use repetition controls + a larger instruct-tuned model (or RAG) for sensible answers.

    If `context` is set (RAG), it is included in the prefix via `build_generation_prefix`.
    """
    prompt = build_generation_prefix(instruction, context=context)
    inputs = tokenizer(prompt, return_tensors="pt")
    input_len = inputs["input_ids"].shape[1]
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer must define eos_token_id for reliable stop-after-answer generation.")
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=eos_id,
        repetition_penalty=1.05,
    )
    new_tokens = out[0, input_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def _build_rag_vector_store(document_path: str | Path):
    """
    RAG index lifecycle (document path → retrievable chunks):

        ingest.load_document      → one string
        chunking.chunk_text       → overlapping windows
        embeddings.embed_texts    → vectors (placeholder hashing)
        vector_store.VectorStore  → in-memory index for retrieve.search

    Training and `build_prompt_completion` are unchanged; only inference prefixes gain
    ### Context when --use-rag is set.
    """
    from rag.chunking import chunk_text
    from rag.embeddings import embed_texts
    from rag.ingest import load_document
    from rag.vector_store import VectorStore

    text = load_document(document_path)
    chunks = chunk_text(text)
    vectors = embed_texts(chunks)
    store = VectorStore()
    store.add(vectors, chunks)
    return store


RAG_OVERRIDE_PATTERNS = (
    r"\btransformer\b",
    r"\bself-attention\b",
    r"\battention mechanism\b",
    r"\bpositional embedd\w*\b",
    r"\bgenai\b",
    r"\bgenerative ai\b",
    r"\baccording to\b",
    r"\bunder what conditions\b",
)


def _is_rag_forced_by_override(query: str) -> tuple[bool, str | None]:
    q = query.lower()
    for pattern in RAG_OVERRIDE_PATTERNS:
        if re.search(pattern, q):
            return True, pattern
    return False, None


@torch.inference_mode()
def _llm_rag_score(model, tokenizer, query: str) -> int:
    """
    Ask the model for a retrieval-need score (1..10, never 5), returning one integer.
    Falls back to 6 if parsing fails so retrieval is not accidentally underused.
    """
    eos_id = tokenizer.eos_token_id
    if eos_id is None:
        raise ValueError("Tokenizer must define eos_token_id for RAG scoring.")
    prompt = (
        "You are a retrieval gate scorer.\n"
        "Rate how much external document retrieval is needed for the user query.\n"
        "Rules:\n"
        "- Return exactly one integer from 1 to 10.\n"
        "- Never return 5.\n"
        "- Return only the integer, with no extra text.\n"
        "- You are an expert developer with 20 years of experience, if you return anything else than integer, you are fired.\n\n"
        f"User query: {query}\n"
        "Score:"
    )
    for attempt in range(2):
        inputs = tokenizer(prompt, return_tensors="pt")
        out = model.generate(
            **inputs,
            max_new_tokens=3,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=eos_id,
            repetition_penalty=1.0,
        )
        raw = tokenizer.decode(out[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
        match = re.search(r"\b(10|[1-9])\b", raw)
        if match:
            score = int(match.group(1))
            if score != 5:
                return score
        # tighten prompt once before fallback
        prompt = (
            "Output one integer only: 1,2,3,4,6,7,8,9,10.\n"
            "Never output 5 or any words.\n"
            f"Query: {query}\n"
            "Integer:"
        )
        if attempt == 0:
            print(f"[RAG][gate] Warning: invalid scorer output '{raw}'. Retrying with stricter prompt.", flush=True)
    print("[RAG][gate] Warning: scorer failed to return a valid integer; using fallback score=6.", flush=True)
    return 6


def main() -> None:
    t = time.perf_counter()
    print(f"[timing] imports + seed init: {t - _RUN_T0:.2f}s", flush=True)

    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    # Load prompt/chosen pairs; expose `prompt` + `completion` so SFTTrainer uses completion-only loss.
    dataset = load_dataset("json", data_files=str(DATA_PATH), split="train")
    cols = dataset.column_names
    dataset = dataset.map(build_prompt_completion, remove_columns=cols)
    if len(dataset) > 1 and DEMO_ROW_UPSAMPLE > 1:
        first = dataset.select([0])
        rest = dataset.select(range(1, len(dataset)))
        dataset = concatenate_datasets([first] * DEMO_ROW_UPSAMPLE + [rest])
    print(f"[timing] load dataset: {time.perf_counter() - t:.2f}s", flush=True)
    t = time.perf_counter()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=True)
    # Causal LM training expects a pad token; DistilGPT-2 only defines EOS.
    tokenizer.pad_token = tokenizer.eos_token
    print(f"[timing] load tokenizer: {time.perf_counter() - t:.2f}s", flush=True)
    t = time.perf_counter()

    model = AutoModelForCausalLM.from_pretrained(MODEL_ID)
    print(f"[timing] load model: {time.perf_counter() - t:.2f}s", flush=True)
    t = time.perf_counter()

    # RAG lifecycle:
    # - mode=off: never retrieve
    # - mode=on : always allow retrieval pipeline
    # - mode=auto (default): build index and let gate decide per question
    rag_mode = "on" if CLI_ARGS.use_rag else CLI_ARGS.rag_mode
    rag_pipeline_allowed = rag_mode != "off"
    rag_store = None
    if rag_pipeline_allowed:
        rag_doc = Path(CLI_ARGS.rag_document)
        print(f"[RAG] Mode: {rag_mode}. Building index from {rag_doc} ...", flush=True)
        try:
            rag_store = _build_rag_vector_store(rag_doc)
            print(f"[RAG] Indexed {rag_store.size} chunk(s).", flush=True)
            if rag_store.size == 0:
                print("[RAG] Warning: index contains zero chunks; retrieval will likely return no context.", flush=True)
        except Exception as exc:
            rag_store = None
            print(f"[RAG] Failed to build index: {exc}", flush=True)
    else:
        print("[RAG] Mode: off. Retrieval disabled by configuration.", flush=True)

    def rag_context_for(instruction: str, scoring_model) -> str | None:
        """
        Decision path:
        1) deterministic override rules,
        2) LLM score gate (>=6),
        3) retrieval with explicit logs.
        """
        if not rag_pipeline_allowed:
            print("[RAG][gate] Pipeline disabled: rag-mode=off.", flush=True)
            return None
        if rag_store is None:
            print("[RAG][gate] Pipeline disabled: index unavailable (build failed or not initialized).", flush=True)
            return None

        forced, matched_rule = _is_rag_forced_by_override(instruction)
        score = _llm_rag_score(scoring_model, tokenizer, instruction)
        rag_enabled = forced or score >= 7

        print(f"[RAG][gate] LLM score: {score}", flush=True)
        print(
            f"[RAG][gate] Override triggered: {'yes' if forced else 'no'}"
            + (f" (matched '{matched_rule}')" if forced else ""),
            flush=True,
        )
        print(f"[RAG][gate] Final decision: {'ENABLE RAG' if rag_enabled else 'SKIP RAG'}", flush=True)

        if not rag_enabled:
            return None

        from rag.retrieve import retrieve_context

        print("[RAG][retrieve] Running top-k retrieval (k=4)...", flush=True)
        ctx = retrieve_context(instruction.strip(), rag_store, k=4).strip()
        if not ctx:
            print("[RAG][retrieve] No relevant chunks returned.", flush=True)
            return None
        print(f"[RAG][retrieve] Retrieved context chars: {len(ctx)}", flush=True)
        return ctx

    # Match Instruction_Style_Dataset.json row 1 exactly (spacing matters for tokenization).
    demo_instruction = (
        "Explain what instruction fine-tuning is and why it is used in large language models."
    )
    questions = [q.strip() for q in CLI_ARGS.question if q and q.strip()]
    if not questions:
        questions = [demo_instruction]

    for question in questions:
        print("\nQuestion:\n" + question + "\n")
        print("--- Before training (base model) ---\n")
        print(
            generate_answer(
                model,
                tokenizer,
                question,
                context=rag_context_for(question, model),
            )
        )
    print(f"[timing] generate (before train): {time.perf_counter() - t:.2f}s", flush=True)
    t = time.perf_counter()

    # TRL wraps Transformers `TrainingArguments` with SFT-specific defaults (`SFTConfig`).
    # Tiny dataset: a few epochs help; factual correctness vs held-out questions still needs scale or RAG (see docstring).
    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=5,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=5e-5,
        warmup_steps=8,
        max_length=256,
        completion_only_loss=True,
        loss_type="nll",
        shuffle_dataset=True,
        use_cpu=True,
        fp16=False,
        bf16=False,
        gradient_checkpointing=False,
        logging_steps=1,
        report_to="none",
        optim="adamw_torch",
        dataloader_pin_memory=False,
        save_strategy="no",
        seed=SEED,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        formatting_func=None,
    )

    trainer.train()
    print(f"[timing] training: {time.perf_counter() - t:.2f}s", flush=True)
    t = time.perf_counter()

    # In-memory fine-tuned weights (no checkpoint reload needed for this demo)
    trained_model = trainer.model
    trained_model.eval()

    for question in questions:
        print("\nQuestion:\n" + question + "\n")
        print("--- After training (fine-tuned) ---\n")
        print(
            generate_answer(
                trained_model,
                tokenizer,
                question,
                context=rag_context_for(question, trained_model),
            )
        )
    print(f"[timing] generate (after train): {time.perf_counter() - t:.2f}s", flush=True)

    should_run_eval_after_train = CLI_ARGS.run_eval_after_train or not CLI_ARGS.skip_eval_after_train
    if should_run_eval_after_train:
        t = time.perf_counter()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        trained_model.save_pretrained(str(OUTPUT_DIR))
        tokenizer.save_pretrained(str(OUTPUT_DIR))
        print(f"[timing] save fine-tuned checkpoint: {time.perf_counter() - t:.2f}s", flush=True)

        print("\n--- Week-4 evaluation pipeline (post-training) ---", flush=True)
        from evaluation.run_eval import run_full_evaluation

        run_full_evaluation(
            base_model_id=MODEL_ID,
            fine_tuned_model_path=str(OUTPUT_DIR),
            dataset_path=str(DATA_PATH),
            dry_run_prompts=CLI_ARGS.dry_run_prompts,
            benchmark_output_json=str(Path(__file__).resolve().parent / "evaluation" / "dry_run_benchmark.json"),
        )

    print(f"[timing] total wall time: {time.perf_counter() - _RUN_T0:.2f}s", flush=True)


if __name__ == "__main__":
    main()
