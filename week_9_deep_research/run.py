from __future__ import annotations

import json
import os
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv

from .llm import LLM
from .pipelines import deep_research_chain, naive_chain


HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
DATA_PATH = HERE / "data" / "questions.json"
OUTPUT_DIR = HERE / "outputs"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_questions() -> list[dict[str, str]]:
    data = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, list) or len(data) != 20:
        raise ValueError(f"Expected 20 questions in {DATA_PATH}, got {len(data) if isinstance(data, list) else type(data)}")
    for item in data:
        if "id" not in item or "question" not in item:
            raise ValueError("Each question must have keys: 'id', 'question'")
    return data


def main() -> Path:
    # Load shared secrets from the workspace root.
    load_dotenv(dotenv_path=ROOT / ".env", override=False)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = OUTPUT_DIR / f"run_{run_id}.jsonl"

    llm = LLM()
    questions = load_questions()

    meta = {
        "run_id": run_id,
        "started_at": _utc_now_iso(),
        "model": llm.model,
        "base_url_set": bool(os.getenv("OPENAI_BASE_URL")),
        "is_mock": llm.is_mock,
        "num_questions": len(questions),
    }

    with out_path.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "run_meta", **meta}, ensure_ascii=False) + "\n")

        for idx, q in enumerate(questions, start=1):
            qid = q["id"]
            question = q["question"]

            naive = naive_chain(llm, question)
            deep = deep_research_chain(llm, question)

            row = {
                "type": "result",
                "run_id": run_id,
                "created_at": _utc_now_iso(),
                "question_id": qid,
                "question": question,
                "naive_answer": naive,
                "deep": asdict(deep),
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            print(f"[{idx:02d}/20] {qid} done")

    print(f"\nWrote: {out_path}")
    if llm.is_mock:
        print("NOTE: OPENAI_API_KEY was not set, so a deterministic mock LLM was used.")

    # Quick sanity check: expect 1 meta line + 20 result lines.
    try:
        num_lines = sum(1 for _ in out_path.open("r", encoding="utf-8"))
        expected_lines = 1 + len(questions)
        print(f"Output lines: {num_lines} (expected {expected_lines})")
    except OSError:
        print("WARNING: Could not count output lines.")
    return out_path


if __name__ == "__main__":
    main()

