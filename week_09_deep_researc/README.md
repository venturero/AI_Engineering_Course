## Week 9 — Deep Research (Minimal)

This module compares:

- **Naive chain**: question → single answer
- **Deep research chain**: draft → critique → revise

No LangChain / AutoGen / external agent frameworks are used. The logic is intentionally small and explainable.

### Setup

Create a virtual environment (optional) and install deps:

```bash
pip install -r week_9_deep_research/requirements.txt
```

Set an API key (recommended):

- **Windows PowerShell**

```powershell
$env:OPENAI_API_KEY="YOUR_KEY"
```

Optional overrides:

- `OPENAI_BASE_URL` (OpenAI-compatible endpoint)
- `OPENAI_MODEL` (default: `gpt-4.1-mini`)

### Run

From the repo's `AI_ENG/week_9_deep_research` workspace root:

```bash
python -m week_9_deep_research.run
```

### Outputs

The runner executes both pipelines on the same **20 questions** from `week_9_deep_research/data/questions.json` and writes:

- `week_9_deep_research/outputs/run_YYYYMMDD_HHMMSS.jsonl`

Each JSONL row includes the **draft, critique, and revised** content (for the deep research chain) plus the naive answer for comparison.

