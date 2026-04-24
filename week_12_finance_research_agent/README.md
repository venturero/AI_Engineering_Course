# Week 12 Capstone - AI Strategy Report Generator

This project implements a Finance + Research strategy intelligence workflow designed to produce consulting-grade outputs with grounded retrieval.

## What This System Does

- Accepts a natural-language investment/strategy research prompt
- Retrieves relevant source documents using local RAG
- Runs a mandatory multi-agent pipeline:
  - `ResearchAgent`
  - `FinancialAnalystAgent`
  - `StrategySynthesisAgent`
- Produces a structured strategy report with required sections:
  - Executive Summary
  - Market & Technology Overview
  - Competitive Landscape (named companies)
  - Company-Level Strategic Implications
  - Risks & Uncertainties
  - Forward-Looking Outlook
- Logs:
  - agent steps
  - retrieved documents
  - evaluation scores (factual grounding, coverage, coherence)

## Project Structure

`Week_12/`  
- `demo.py` - minimal end-to-end run (now outputs PDF)
- `requirements.txt`
- `data/sample_corpus/` - demo retrieval corpus
- `logs/` - runtime logs
- `outputs/` - generated reports and metadata
- `src/week12_capstone/`
  - `retriever.py`
  - `agents.py`
  - `orchestrator.py`
  - `evaluator.py`
  - `logger.py`
  - `schemas.py`
  - `config.py`

## Run

```bash
cd Week_12
python demo.py
```

Generate a different topic (example: space mining):

```bash
python demo.py --query "Research the commercial readiness of space mining and analyze the impact on publicly traded companies."
```

Disable visuals:

```bash
python demo.py --query "..." --no-visual
```

## Demo Prompt

`Research the evolution of autonomous driving technologies and analyze their impact on publicly traded companies.`

## Where to Change the Report Topic/Prompt

- Runtime prompt (recommended): pass a new value to `--query` in `demo.py`.
- Default fallback prompt in code: edit the default value in `demo.py` argument parser.
- Programmatic use: call `StrategyReportOrchestrator().run("your new prompt")`.

## Output Artifacts

After execution, inspect:

- `outputs/strategy_report_<timestamp>.pdf`
- `outputs/run_meta_<timestamp>.json`
- `logs/agent_steps_<timestamp>.jsonl`
- `logs/retrieval_<timestamp>.jsonl`
- `logs/evaluation_<timestamp>.json`

If `FAL_KEY` is present and visuals are enabled, a generated image is embedded into the PDF and the run metadata includes `visual_path`.

## PDF Styling Notes

The PDF renderer follows a BCG-inspired style based on public design cues:
- green accent headers (`#147B58`)
- generous whitespace and clean section hierarchy
- concise executive-format paragraphs and bullets
- minimal visual noise with an optional single hero visual

## Notes on Credibility

- Reports are retrieval-grounded and explicitly cite retrieved source titles.
- Evaluation scores are transparent and stored for each run.
- The architecture is modular so stronger retrievers/LLMs can be swapped in without changing the workflow contract.

