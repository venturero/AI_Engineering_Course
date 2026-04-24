# AI Engineering (12-Week Build Program)

This repository is a hands-on **AI Engineering learning + implementation track** that moves from LLM foundations to production-style systems:

- prompt engineering and structured generation
- retrieval-augmented generation (RAG)
- ReAct-style agent tool use
- deep research with critique-and-revise loops
- multimodal diffusion experiments
- a multi-agent finance research capstone

The goal is simple: by the end of this repo, a newcomer can understand how modern AI products are built end-to-end, and run each stage locally.

---

## Repository Map (Clickable)

### Week-by-week links

- [Week 1-6: Foundations, SFT, and RAG](./week_1_6_rag/)
- [Week 7-8: ReAct Agent from Scratch](./week_7_8_agent/)
- [Week 9: Deep Research Pipeline](./week_9_deep_research/)
- [Week 10-11: Multimodal Diffusion](./week_10_11_multimodal/week_10_11_diffusion/)
- [Week 12: Finance Research Agent Capstone](./week_12_finance_research_agent/)

### Suggested learning flow

- Week 1-6 builds core LLM and retrieval intuition.
- Week 7-8 adds planning + tool calling.
- Week 9 introduces iterative reasoning quality improvements.
- Week 10-11 expands into multimodal generation.
- Week 12 combines everything into a portfolio-grade capstone.

---

## Quick Start (Using `uv`)

This repo is easiest to run with [`uv`](https://earthly.dev/blog/python-uv/).

### 1) Install `uv`

On Windows PowerShell:

```powershell
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Verify:

```powershell
uv --version
```

### 2) Create and activate virtual environment

From repo root:

```powershell
uv venv .venv
.\.venv\Scripts\Activate.ps1
```

### 3) Install dependencies per module

There is no single monolithic requirements file at root, so install what you need by week:

```powershell
pip install -r .\week_9_deep_research\requirements.txt
pip install -r .\week_12_finance_research_agent\requirements.txt
```

For Week 10-11 diffusion experiments, install from that README:

- [Week 10-11 setup instructions](./week_10_11_multimodal/week_10_11_diffusion/README.md)

---

## Environment Variables (`.env.example` -> `.env`)

Use a template file and fill your own keys.

1. Keep a committed template: `.env.example`
2. Create local runtime file: `.env`
3. Add your own keys (never commit real secrets)

PowerShell:

```powershell
Copy-Item .env.example .env
```

Minimum variable:

- `OPENAI_API_KEY=your_key_here`

Common optional variables used in this repo:

- `OPENAI_MODEL`
- `OPENAI_RESEARCH_MODEL`
- `OPENAI_ANALYSIS_MODEL`
- `OPENAI_BASE_URL`
- `LIVE_NEWS_DOC_LIMIT`
- `STRATEGY_COVER_MODE`
- `FAL_KEY` (only for fal.ai image generation features)

---

## Week Details

### Week 1-6: Foundations, SFT, and RAG

Folder: [`week_1_6_rag`](./week_1_6_rag/)

Focus:

- instruction-style data formatting
- tokenizer and generation intuition (temperature/top-k/top-p style concepts)
- supervised fine-tuning flow with TRL (`SFTTrainer`)
- optional retrieval augmentation from local documents (toward production RAG patterns)

What you build:

- a small instruction-tuned text generation workflow
- repeatable local training + inference script
- optional retrieval-supported answer generation pipeline

Main entry:

- [`week_1_6_rag/main.py`](./week_1_6_rag/main.py)

Typical run:

```powershell
python .\week_1_6_rag\main.py
```

Recommended learning resources:

- Andrej Karpathy - [Let's build GPT: from scratch, in code, spelled out](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- Andrej Karpathy - [Let's reproduce GPT-2 (124M)](https://www.youtube.com/watch?v=l8pRSuU81PU)
- Sebastian Raschka - [LLMs-from-scratch (book repo)](https://github.com/rasbt/LLMs-from-scratch)
- Hugging Face TRL - [SFTTrainer docs](https://huggingface.co/docs/trl/main/en/sft_trainer)

---

### Week 7-8: ReAct Agent from Scratch

Folder: [`week_7_8_agent`](./week_7_8_agent/)

Focus:

- explicit ReAct loop (`Thought -> Action -> Observation -> Final`)
- simple tool routing logic
- safe calculator and Wikipedia lookup tool integration

What you build:

- a transparent single-agent planner with deterministic, inspectable behavior
- tool-calling flow that mirrors production agent orchestration patterns

Main entry:

- [`week_7_8_agent/main.py`](./week_7_8_agent/main.py)

Example run:

```powershell
python .\week_7_8_agent\main.py "What is 125*16?"
python .\week_7_8_agent\main.py "Who is Alan Turing?"
```

Recommended learning resources:

- ReAct prompting guide - [Prompting Guide: ReAct](https://www.promptingguide.ai/techniques/react)
- Agent ecosystem references - [Awesome AI Agents](https://github.com/jim-schwoebel/awesome_ai_agents)

---

### Week 9: Deep Research Pipeline

Folder: [`week_9_deep_research`](./week_9_deep_research/)

Focus:

- compare naive single-pass answers vs iterative quality loops
- implement `draft -> critique -> revise`
- evaluate runs over a fixed question set

What you build:

- an iterative research chain with self-critique pass
- side-by-side output logs for quality benchmarking and analysis

Main entry:

- `python -m week_9_deep_research.run`

Useful docs:

- [Week 9 README](./week_9_deep_research/README.md)

---

### Week 10-11: Multimodal Diffusion

Folder: [`week_10_11_multimodal/week_10_11_diffusion`](./week_10_11_multimodal/week_10_11_diffusion/)

Focus:

- local text-to-image generation with Diffusers
- optional hosted generation with fal.ai
- optional text-to-video experimentation notebook

What you build:

- local text-to-image CLI + Gradio UI
- optional cloud-backed image generation variant
- optional text-to-video experimental workflow

Useful docs:

- [Week 10-11 README](./week_10_11_multimodal/week_10_11_diffusion/README.md)
- Hugging Face Diffusers - [Official docs](https://huggingface.co/docs/diffusers/index)

---

### Week 12: Finance Research Agent Capstone

Folder: [`week_12_finance_research_agent`](./week_12_finance_research_agent/)

Focus:

- retrieval-grounded research pipeline
- multi-agent orchestration (`ResearchAgent`, `FinancialAnalystAgent`, `StrategySynthesisAgent`)
- evaluation and logging for report quality
- generated strategy report output (PDF + metadata + logs)

What you build:

- a consulting-style strategy report generator
- traceable agent outputs, retrieval logs, and evaluation artifacts
- presentation-ready PDF output with structured sections

Main entry:

```powershell
python .\week_12_finance_research_agent\demo.py
```

Useful docs:

- [Week 12 README](./week_12_finance_research_agent/README.md)

---

## External Learning Reference

This project follows the progression described in your course PDF and uses `uv` setup guidance based on:

- [How to Create a Python Virtual Environment with uv](https://earthly.dev/blog/python-uv/)

---

## Common Troubleshooting

- If imports fail, confirm your virtual environment is active (`.venv`).
- If OpenAI calls fail, verify `OPENAI_API_KEY` is set in `.env`.
- If image generation is slow, use a CUDA-enabled setup for diffusion.
- If a script works in one folder but not another, run commands from repo root and use explicit paths.
- If folder links do not open locally, push to GitHub first; relative markdown links are designed for GitHub browsing.

---

## Outcome

After completing all weeks, you will have a portfolio showing:

- LLM training/fine-tuning fundamentals
- practical RAG implementation
- agentic reasoning and tool use
- deep research quality loops
- multimodal generation workflows
- a complete capstone with evaluation, logging, and professional output artifacts

