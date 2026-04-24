from pathlib import Path
import os


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
CORPUS_DIR = DATA_DIR / "sample_corpus"
LOGS_DIR = ROOT_DIR / "logs"
OUTPUT_DIR = ROOT_DIR / "outputs"
INDEX_DIR = ROOT_DIR / "vector_store"


EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RETRIEVAL_TOP_K = 8
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4.1")
OPENAI_RESEARCH_MODEL = os.getenv("OPENAI_RESEARCH_MODEL", OPENAI_MODEL)
OPENAI_ANALYSIS_MODEL = os.getenv("OPENAI_ANALYSIS_MODEL", OPENAI_MODEL)
LIVE_NEWS_DOC_LIMIT = int(os.getenv("LIVE_NEWS_DOC_LIMIT", "12"))

# Cover: "bcg_exhibit" = dark BCG-inspired title slide (matplotlib; topic only, no duplicate bullets).
# "fal_abstract" = FAL dark abstract cover if FAL_KEY is set (uses compact_visual_topic in the prompt).
# "fal_infographic" = legacy white infographic prompt (may show gibberish text in-image).
STRATEGY_COVER_MODE = os.getenv("STRATEGY_COVER_MODE", "bcg_exhibit").strip().lower()

