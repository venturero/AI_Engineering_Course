import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from .config import LOGS_DIR


class CapstoneLogger:
    def __init__(self) -> None:
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.session_id = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
        self.agent_steps_path = LOGS_DIR / f"agent_steps_{self.session_id}.jsonl"
        self.retrieval_path = LOGS_DIR / f"retrieval_{self.session_id}.jsonl"
        self.eval_path = LOGS_DIR / f"evaluation_{self.session_id}.json"

    def _append_jsonl(self, path: Path, row: Dict[str, Any]) -> None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, ensure_ascii=True) + "\n")

    def log_agent_step(self, agent_name: str, step: str, payload: Dict[str, Any]) -> None:
        self._append_jsonl(
            self.agent_steps_path,
            {
                "ts_utc": datetime.now(tz=timezone.utc).isoformat(),
                "agent": agent_name,
                "step": step,
                "payload": payload,
            },
        )

    def log_retrieval(self, query: str, docs: List[Dict[str, Any]]) -> None:
        self._append_jsonl(
            self.retrieval_path,
            {
                "ts_utc": datetime.now(tz=timezone.utc).isoformat(),
                "query": query,
                "documents": docs,
            },
        )

    def log_evaluation(self, payload: Dict[str, Any]) -> None:
        with self.eval_path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=True, indent=2)

