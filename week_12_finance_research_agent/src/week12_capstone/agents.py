import json
import os
from typing import Any, Dict, List

from openai import OpenAI
from .schemas import AgentOutput, RetrievedDoc


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


def _build_citations(docs: List[RetrievedDoc]) -> List[str]:
    return [f"{doc.title} ({doc.source})" for doc in docs if doc.source]


def _docs_context(docs: List[RetrievedDoc]) -> str:
    rows = []
    for idx, doc in enumerate(docs, start=1):
        rows.append(
            (
                f"[{idx}] title={doc.title}\n"
                f"url={doc.source}\n"
                f"published_at={doc.published_at}\n"
                f"summary={doc.text}\n"
            )
        )
    return "\n".join(rows)


def _safe_json(text: str, agent_name: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:].strip()
    if not cleaned:
        raise RuntimeError(f"{agent_name} returned empty output.")
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{agent_name} returned invalid JSON: {exc}") from exc


class _OpenAIJSONAgent:
    def __init__(self, agent_name: str) -> None:
        if not os.getenv("OPENAI_API_KEY"):
            raise RuntimeError("OPENAI_API_KEY is required for OpenAI agents.")
        self.agent_name = agent_name
        self.client = _build_openai_client()
        self.model = os.getenv("OPENAI_ANALYSIS_MODEL", os.getenv("OPENAI_MODEL", "gpt-4.1"))

    def ask_json(self, prompt: str) -> Dict[str, Any]:
        response = self.client.responses.create(model=self.model, input=prompt)
        return _safe_json(response.output_text, self.agent_name)


class ResearchAgent:
    name = "research_agent"

    def __init__(self) -> None:
        self.agent = _OpenAIJSONAgent(self.name)

    def run(self, user_query: str, docs: List[RetrievedDoc]) -> AgentOutput:
        prompt = f"""
You are a senior strategy research analyst.
Use only the provided evidence. No hardcoded entities. No invented facts.

User query:
{user_query}

Evidence:
{_docs_context(docs)}

Return strict JSON:
{{
  "market_overview_markdown": "markdown section with concise bullets",
  "critical_developments": ["...", "..."],
  "winners_now_summary": "2-3 sentence direct answer on who is leading now and why"
}}
""".strip()
        payload = self.agent.ask_json(prompt)
        content = payload.get("market_overview_markdown", "").strip()
        return AgentOutput(
            agent_name=self.name,
            content=content,
            citations=_build_citations(docs),
            structured_data={
                "critical_developments": payload.get("critical_developments", []),
                "winners_now_summary": payload.get("winners_now_summary", ""),
            },
        )


class FinancialAnalystAgent:
    name = "financial_analyst_agent"

    def __init__(self) -> None:
        self.agent = _OpenAIJSONAgent(self.name)

    def run(self, user_query: str, docs: List[RetrievedDoc]) -> AgentOutput:
        prompt = f"""
You are an equity strategy analyst.
Goal: identify public stocks likely to rise/fall based on the evidence.
Do not hardcode companies. Discover from evidence only.
If evidence is weak for a claim, set lower confidence.
Whenever you mention a public company in any markdown text, format it as Company ($TICKER).

User query:
{user_query}

Evidence:
{_docs_context(docs)}

Return strict JSON:
{{
  "stock_calls": [
    {{
      "company": "name",
      "ticker": "exchange:ticker or unknown",
      "direction": "up|down|mixed",
      "confidence": 0-100,
      "time_horizon": "near-term|medium-term|long-term",
      "thesis": "why",
      "key_catalysts": ["...", "..."],
      "key_risks": ["...", "..."],
      "evidence_refs": [1,2]
    }}
  ],
  "portfolio_implications_markdown": "markdown bullets for portfolio positioning"
}}
""".strip()
        payload = self.agent.ask_json(prompt)
        content = payload.get("portfolio_implications_markdown", "").strip()
        return AgentOutput(
            agent_name=self.name,
            content=content,
            citations=_build_citations(docs),
            structured_data={"stock_calls": payload.get("stock_calls", [])},
        )


class StrategySynthesisAgent:
    name = "strategy_synthesis_agent"

    def __init__(self) -> None:
        self.agent = _OpenAIJSONAgent(self.name)

    def run(
        self,
        user_query: str,
        research_output: AgentOutput,
        finance_output: AgentOutput,
        docs: List[RetrievedDoc],
    ) -> AgentOutput:
        prompt = f"""
You are a Boston Consulting Group (BCG)–style strategy partner writing a short board memo.
Use only provided agent outputs and evidence. Tone: precise, neutral, decision-oriented; avoid marketing hype.

Structure and voice (follow closely):
- Pyramid principle: each ## section must open with one bold headline sentence (**...**) stating the single most important point, then supporting bullets.
- Executive Summary: at most 6 sentences — situation, critical insight, implications, and what to do next. Do not paste the executive_takeaways bullets verbatim here; write integrated prose so the summary does not duplicate the takeaway list word-for-word.
- Prefer short section titles in Title Case (e.g., "Winners right now", "Stock impact").
- Action Agenda: numbered list 1., 2., 3.; each line starts with an imperative verb (e.g., "Monitor…", "Reallocate…").
- Use ### subheadings inside Stock Impact where you split "Likely to rise" vs "Likely to underperform".

Must explicitly answer:
1) which companies are winning now
2) which stocks likely rise/fall and why
3) what could invalidate the thesis
Formatting rule: whenever a public company is mentioned, use Company ($TICKER) notation.

User query:
{user_query}

Research agent output:
{research_output.content}
Research metadata:
{json.dumps(research_output.structured_data, ensure_ascii=True)}

Financial analyst output:
{finance_output.content}
Financial metadata:
{json.dumps(finance_output.structured_data, ensure_ascii=True)}

Evidence:
{_docs_context(docs)}

Return strict JSON:
{{
  "report_markdown": "with sections: Executive Summary, Winners Right Now, Market & Technology Overview, Stock Impact (Up/Down), Risks & What Could Break The Thesis, Action Agenda, Sources",
  "executive_takeaways": ["...", "...", "..."]
}}

Sources section rule:
- Include every evidence URL with title in markdown bullet format.
- Use explicit URL text (not hidden hyperlinks) so traceability can be audited.
""".strip()
        payload = self.agent.ask_json(prompt)
        return AgentOutput(
            agent_name=self.name,
            content=payload.get("report_markdown", "").strip(),
            citations=_build_citations(docs),
            structured_data={"executive_takeaways": payload.get("executive_takeaways", [])},
        )

