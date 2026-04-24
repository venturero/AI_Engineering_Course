import argparse

from agent import ReActAgent, load_default_prompt
from tools import TOOL_REGISTRY


def main() -> None:
    parser = argparse.ArgumentParser(description="Single-agent ReAct-style planner (from scratch).")
    parser.add_argument("question", type=str, help="User question to solve.")
    parser.add_argument("--max-steps", type=int, default=6, help="Maximum ReAct steps.")
    parser.add_argument("--quiet", action="store_true", help="Reduce trace printing.")
    args = parser.parse_args()

    prompt = load_default_prompt()
    agent = ReActAgent(tools=TOOL_REGISTRY, prompt_template=prompt, max_steps=args.max_steps)
    final = agent.run(args.question, verbose=not args.quiet)
    if not args.quiet:
        print(f"Final Output: {final}")
    else:
        print(final)


if __name__ == "__main__":
    main()

