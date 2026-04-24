import json
import os
from typing import Any, Dict, List

from openai import OpenAI

from .config import LIVE_NEWS_DOC_LIMIT, RETRIEVAL_TOP_K
from .schemas import RetrievedDoc


def _build_openai_client() -> OpenAI:
    base_url = (os.getenv("OPENAI_BASE_URL") or "").strip()
    if not base_url:
        os.environ.pop("OPENAI_BASE_URL", None)
        return OpenAI()
    if not base_url.startswith(("http://", "https://")):
        raise RuntimeError(
            "OPENAI_BASE_URL must start with http:// or https://, or be unset."
        )
    return OpenAI(base_url=base_url)


def _safe_json_load(raw: str) -> Dict[str, Any]:
    cleaned = (raw or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    if not cleaned:
        raise RuntimeError("OpenAI returned empty retrieval output.")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"OpenAI returned non-JSON retrieval output: {exc}") from exc


class CorpusRetriever:
    """
    Live retrieval powered by OpenAI web search.
    """

    def __init__(self) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for live web/news ingestion.")
        self.client = _build_openai_client()

    def retrieve(self, query: str, top_k: int = RETRIEVAL_TOP_K) -> List[RetrievedDoc]:
        doc_limit = max(top_k, LIVE_NEWS_DOC_LIMIT)
        retrieval_prompt = f"""
You are a research retriever.
Task: gather current, high-signal evidence for the user query from credible public web sources.

User query: {query}

Return strict JSON with this shape:
{{
  "documents": [
    {{
      "title": "source title",
      "url": "https://...",
      "published_at": "YYYY-MM-DD or null",
      "summary": "2-4 sentence factual summary",
      "relevance_score": 0-100
    }}
  ]
}}

Requirements:
- Focus on latest developments, company announcements, major business press, and regulatory updates.
- Keep only diverse, non-duplicate sources.
- Include at most {doc_limit} documents sorted by relevance_score descending.
- Do not include commentary outside JSON.
""".strip()
        response = self.client.responses.create(
            model=os.getenv("OPENAI_RESEARCH_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1")),
            tools=[{"type": "web_search_preview"}],
            input=retrieval_prompt,
        )
        payload = _safe_json_load(response.output_text)
        docs: List[RetrievedDoc] = []
        for item in payload.get("documents", [])[:doc_limit]:
            docs.append(
                RetrievedDoc(
                    source=item.get("url", ""),
                    title=item.get("title", "Untitled Source"),
                    text=item.get("summary", ""),
                    score=float(item.get("relevance_score", 0.0)),
                    published_at=item.get("published_at"),
                )
            )
        return docs[:top_k]

