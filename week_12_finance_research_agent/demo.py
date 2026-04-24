import argparse
from pathlib import Path

from dotenv import load_dotenv

from src.week12_capstone.orchestrator import StrategyReportOrchestrator


def main() -> None:
    project_dir = Path(__file__).resolve().parent
    load_dotenv(project_dir.parent / ".env")

    parser = argparse.ArgumentParser(description="Generate strategy report PDF.")
    parser.add_argument(
        "--query",
        default=(
            "Research the evolution of autonomous driving technologies and analyze "
            "their impact on publicly traded companies."
        ),
        help="The full strategy research prompt used by the agents.",
    )
    parser.add_argument(
        "--no-visual",
        action="store_true",
        help="Disable FAL-generated visuals in the PDF output.",
    )
    args = parser.parse_args()

    user_prompt = args.query

    orchestrator = StrategyReportOrchestrator()
    result = orchestrator.run(user_prompt, include_visuals=not args.no_visual)

    print("=== USER PROMPT ===")
    print(user_prompt)
    print("\n=== GENERATED STRATEGY REPORT (PDF) ===")
    print(Path(result["report_path"]).resolve())
    print("\n=== ARTIFACTS ===")
    for key, value in result.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()

