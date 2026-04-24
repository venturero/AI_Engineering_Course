import json
from pathlib import Path
from typing import Dict

from .agents import FinancialAnalystAgent, ResearchAgent, StrategySynthesisAgent
from .config import OUTPUT_DIR
from .evaluator import evaluate_report
from .logger import CapstoneLogger
from .pdf_renderer import render_strategy_pdf
from .retriever import CorpusRetriever
from .visuals import compact_visual_topic, generate_stock_chart, generate_strategy_cover_visual


class StrategyReportOrchestrator:
    def __init__(self) -> None:
        self.retriever = CorpusRetriever()
        self.research_agent = ResearchAgent()
        self.finance_agent = FinancialAnalystAgent()
        self.strategy_agent = StrategySynthesisAgent()
        self.logger = CapstoneLogger()
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    def run(self, user_query: str, include_visuals: bool = True) -> Dict[str, str]:
        self.logger.log_agent_step("orchestrator", "start", {"query": user_query})

        docs = self.retriever.retrieve(user_query)
        self.logger.log_retrieval(
            user_query,
            [
                {"title": d.title, "source": d.source, "score": d.score, "preview": d.text[:220]}
                for d in docs
            ],
        )

        research = self.research_agent.run(user_query=user_query, docs=docs)
        self.logger.log_agent_step(research.agent_name, "analysis_complete", {"content": research.content})

        finance = self.finance_agent.run(user_query=user_query, docs=docs)
        self.logger.log_agent_step(finance.agent_name, "analysis_complete", {"content": finance.content})

        strategy = self.strategy_agent.run(
            user_query=user_query, research_output=research, finance_output=finance, docs=docs
        )
        self.logger.log_agent_step(strategy.agent_name, "report_generated", {"content": strategy.content})

        eval_result = evaluate_report(strategy.content, docs)
        self.logger.log_evaluation(
            {
                "query": user_query,
                "factual_grounding_score": eval_result.factual_grounding_score,
                "coverage_score": eval_result.coverage_score,
                "coherence_score": eval_result.coherence_score,
                "notes": eval_result.notes,
            }
        )

        stock_calls = finance.structured_data.get("stock_calls", [])
        chart_path = generate_stock_chart(stock_calls, OUTPUT_DIR)
        # Short topic for cover art / FAL — reads like a briefing title, not a full assignment prompt
        visual_topic = compact_visual_topic(user_query)
        if include_visuals:
            visual_result = generate_strategy_cover_visual(visual_topic, OUTPUT_DIR)
        else:
            visual_result = {"visual_path": None, "error": "Visual generation disabled by CLI flag.", "caption": None}
        visual_path = Path(visual_result["visual_path"]) if visual_result.get("visual_path") else None

        report_path = OUTPUT_DIR / f"strategy_report_{self.logger.session_id}.pdf"
        render_strategy_pdf(
            report_markdown=strategy.content,
            user_query=user_query,
            output_path=report_path,
            executive_takeaways=strategy.structured_data.get("executive_takeaways", []),
            stock_calls=stock_calls,
            chart_path=chart_path,
            visual_path=visual_path,
            visual_error=visual_result.get("error"),
            visual_caption=visual_result.get("caption"),
        )

        meta_path = OUTPUT_DIR / f"run_meta_{self.logger.session_id}.json"
        meta_path.write_text(
            json.dumps(
                {
                    "query": user_query,
                    "visual_topic": visual_topic,
                    "retrieved_docs": [d.source for d in docs],
                    "agent_logs": str(self.logger.agent_steps_path),
                    "retrieval_logs": str(self.logger.retrieval_path),
                    "evaluation_log": str(self.logger.eval_path),
                    "report_path": str(report_path),
                    "visual_path": str(visual_path) if visual_path else None,
                    "visual_error": visual_result.get("error"),
                    "stock_chart_path": str(chart_path) if chart_path else None,
                },
                indent=2,
                ensure_ascii=True,
            ),
            encoding="utf-8",
        )

        return {
            "report_path": str(report_path),
            "meta_path": str(meta_path),
            "agent_logs": str(self.logger.agent_steps_path),
            "retrieval_logs": str(self.logger.retrieval_path),
            "evaluation_log": str(self.logger.eval_path),
        }

