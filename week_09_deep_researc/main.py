from __future__ import annotations

import sys
import traceback
from pathlib import Path
import os


def _add_import_path() -> None:
    """
    Allow running this file directly with:
      python main.py

    so that `week_9_deep_research` can be imported as a package.
    """
    here = Path(__file__).resolve().parent  # .../AI_ENG/week_9_deep_research
    ai_eng_root = here.parent  # .../AI_ENG
    sys.path.insert(0, str(ai_eng_root))


def main() -> int:
    _add_import_path()
    try:
        if "--mock" in sys.argv:
            # Force mock mode even if a .env is present.
            os.environ["OPENAI_API_KEY"] = ""
            print("Running in MOCK mode (OPENAI_API_KEY forced to empty).")

        # Import as a module to avoid relative-import issues.
        from week_9_deep_research.run import main as runner_main

        out_path = runner_main()
        if out_path.exists():
            print(f"\nSUCCESS: output written to: {out_path}")
            return 0

        print(f"\nFAILED: output path not found: {out_path}")
        return 1
    except Exception as exc:
        print(f"\nERROR: {exc.__class__.__name__}: {exc}")
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

