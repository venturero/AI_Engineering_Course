from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse


@dataclass(frozen=True)
class LLMResponse:
    text: str
    raw: Optional[dict[str, Any]] = None


class LLM:
    """
    Minimal OpenAI-compatible chat wrapper.

    - No frameworks
    - One method: chat(prompt/system) -> text
    """
    DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(self, model: str | None = None, base_url: str | None = None, api_key: str | None = None):
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
        self.base_url = self._normalize_base_url(base_url or os.getenv("OPENAI_BASE_URL"))
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")

        self._client = None
        self._is_mock = False

        if not self.api_key:
            self._is_mock = True
            return

        from openai import OpenAI  # imported lazily to keep failures simple

        kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            # Avoid SDK falling back to malformed/empty OPENAI_BASE_URL from env.
            "base_url": self.base_url or self.DEFAULT_BASE_URL,
        }
        self._client = OpenAI(**kwargs)

    @staticmethod
    def _normalize_base_url(value: str | None) -> str | None:
        """
        Accept empty values and user-provided hosts without a scheme.
        Examples:
          - "" -> None (use OpenAI default endpoint)
          - "api.openai.com/v1" -> "https://api.openai.com/v1"
          - "https://api.openai.com/v1" -> unchanged
        """
        if not value:
            return None
        cleaned = value.strip()
        if not cleaned:
            return None

        parsed = urlparse(cleaned)
        if parsed.scheme:
            return cleaned
        return f"https://{cleaned}"

    @property
    def is_mock(self) -> bool:
        return self._is_mock

    def chat(self, *, system: str, user: str, temperature: float = 0.2) -> LLMResponse:
        if self._is_mock:
            text = self._mock_answer(system=system, user=user)
            return LLMResponse(text=text, raw={"mock": True})

        assert self._client is not None
        resp = self._client.chat.completions.create(
            model=self.model,
            temperature=temperature,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        text = (resp.choices[0].message.content or "").strip()
        return LLMResponse(text=text, raw=resp.model_dump())

    def _mock_answer(self, *, system: str, user: str) -> str:
        """
        Fallback so the assignment can run without keys.
        This is intentionally simple and deterministic (not "smart").
        """
        q = user.strip().replace("\r", "").splitlines()[-1]
        if "REVISE" in system.upper():
            return (
                "Revised answer (mock):\n"
                f"- Restate question: {q}\n"
                "- Provide a clear definition.\n"
                "- Give 3 actionable steps.\n"
                "- Mention 2 edge cases.\n"
                "- Add a tiny example.\n"
                "- End with a one-line takeaway."
            )
        if "CRITIQUE" in system.upper():
            return (
                "Missing: concrete steps, edge cases, and assumptions.\n"
                "Unclear: definitions and constraints.\n"
                "Weak: lacks a short example and a crisp conclusion."
            )
        return f"Naive answer (mock): {q} — provide a direct, brief response with 3 bullets."

