from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class RetrievedDoc:
    source: str
    title: str
    text: str
    score: float
    published_at: Optional[str] = None


@dataclass
class AgentOutput:
    agent_name: str
    content: str
    citations: List[str] = field(default_factory=list)
    structured_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationResult:
    factual_grounding_score: float
    coverage_score: float
    coherence_score: float
    notes: Dict[str, str]

