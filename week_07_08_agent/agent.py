import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


ActionInput = Union[str, int, float, Dict[str, Any], List[Any]]


@dataclass
class ReActStep:
    thought: str
    action: str
    action_input_raw: str
    observation: Optional[str] = None

    def to_scratchpad_block(self) -> str:
        if self.observation is None:
            return (
                f"Thought: {self.thought}\n"
                f"Action: {self.action}\n"
                f"Action Input: {self.action_input_raw}"
            )
        return (
            f"Thought: {self.thought}\n"
            f"Action: {self.action}\n"
            f"Action Input: {self.action_input_raw}\n"
            f"Observation: {self.observation}"
        )


class ReActAgent:
    """
    A minimal single-agent ReAct-style tool-use system implemented from scratch.

    Notes:
    - No LLM framework is used. Instead, `policy_predict()` implements a simple,
      deterministic planner that produces ReAct-formatted text.
    - The agent still performs the explicit Thought/Action/Action Input/Observation
      loop, and parses Action/Action Input using regex.
    """

    ACTION_RE = re.compile(
        r"Action:\s*(?P<action>[A-Za-z_][A-Za-z0-9_-]*|FINISH)\s*$",
        re.MULTILINE,
    )
    ACTION_INPUT_RE = re.compile(
        r"Action Input:\s*(?P<input>.+)\s*$",
        re.MULTILINE,
    )
    FINAL_RE = re.compile(r"Final Answer:\s*(?P<final>[\s\S]*)\Z")

    def __init__(
        self,
        tools: Dict[str, Callable[..., str]],
        prompt_template: str,
        max_steps: int = 6,
    ) -> None:
        self.tools = tools
        self.prompt_template = prompt_template
        self.max_steps = max_steps

    def _format_scratchpad(self, steps: List[ReActStep]) -> str:
        blocks = [s.to_scratchpad_block() for s in steps]
        return "\n\n".join(blocks).strip()

    def _policy_predict(self, question: str, scratchpad: str) -> str:
        """
        Deterministic ReAct "planner" that decides:
          - which tool to call next
          - when to finish

        This produces text following the required schema, e.g.:
          Thought: ...
          Action: calculator
          Action Input: "2+2"
        """
        # Look for recent observations to decide when to finish.
        last_obs = None
        last_obs_first_line = None
        if "Observation:" in scratchpad:
            last_obs = scratchpad.split("Observation:")[-1].strip()
            last_obs_first_line = last_obs.splitlines()[0].strip() if last_obs else ""

        # If a tool failed, finish immediately with the error.
        if last_obs_first_line is not None and last_obs_first_line.startswith("ERROR:"):
            return (
                "Thought: A tool returned an error, so I should report it.\n"
                "Action: FINISH\n"
                "Action Input: \"\"\n"
                f"Final Answer: {last_obs_first_line}"
            )

        # If we've already produced a plan, we can finish by returning it.
        if last_obs_first_line is not None and last_obs_first_line.startswith("PLAN:"):
            return (
                "Thought: I have a clear plan; I can stop now.\n"
                "Action: FINISH\n"
                "Action Input: \"\"\n"
                f"Final Answer: {last_obs}"
            )

        # If we've already observed a calculator result, finish with it.
        if last_obs_first_line is not None and re.fullmatch(r"-?\d+(\.\d+)?", last_obs_first_line.strip()):
            return (
                "Thought: I have the numeric result, so I can answer.\n"
                "Action: FINISH\n"
                "Action Input: \"\"\n"
                f"Final Answer: {last_obs_first_line.strip()}"
            )

        # If we've already observed Wikipedia results, finish using the top line.
        if last_obs_first_line is not None and last_obs_first_line.startswith("RESULT:"):
            # Extract the first bullet's title to produce a more "answer-like" final output.
            result_lines = last_obs.splitlines() if last_obs is not None else []
            summary = "No answer found in Wikipedia results."
            if len(result_lines) >= 2:
                first_bullet = result_lines[1].strip()
                m = re.match(r"-\s*(?P<title>[^:]+):", first_bullet)
                if m:
                    summary = m.group("title").strip()
                else:
                    summary = first_bullet
            return (
                "Thought: I have enough Wikipedia context to answer.\n"
                "Action: FINISH\n"
                "Action Input: \"\"\n"
                f"Final Answer: {summary}"
            )

        q = question.strip()

        # Heuristic tool routing:
        # 1) Calculator if question looks like math.
        math_candidate = self._extract_math_expression(q)
        if math_candidate is not None:
            return (
                "Thought: The question requires arithmetic, so I should use calculator.\n"
                "Action: calculator\n"
                f"Action Input: {json.dumps(math_candidate)}"
            )

        # 2) Wikipedia if question requests general knowledge.
        if self._looks_like_wikipedia_question(q):
            return (
                "Thought: This looks like a general-knowledge question, so I should use wikipedia_search.\n"
                "Action: wikipedia_search\n"
                f"Action Input: {json.dumps(q)}"
            )

        # 3) Otherwise, we are not confident which tool applies.
        #    Do NOT finish immediately; produce a plan first.
        return (
            "Thought: I can't confidently choose a tool, so I should produce a plan.\n"
            "Action: PLAN\n"
            f"Action Input: {json.dumps(q)}"
        )

    @staticmethod
    def _make_plan(goal: str) -> str:
        g = goal.strip()
        if not g:
            g = "(empty goal)"

        # Minimal, deterministic planning fallback (no tools, no LLM).
        # Keep it generic and actionable; avoid implicitly selecting tools.
        steps = [
            f"Restate the goal in one sentence: {g}",
            "Define what a correct final answer/output should look like (format, constraints, scope).",
            "List the key facts/inputs needed to solve it; note which are missing.",
            "Decide which approach fits: arithmetic/derivation, factual lookup, or procedural steps.",
            "If it's arithmetic, identify the exact expression and required precision.",
            "If it's factual, identify the specific entities/terms and the exact question to look up.",
            "If it's procedural, break it into sub-tasks that can be answered one-by-one.",
            "Execute the chosen approach and draft the final answer; then sanity-check for completeness.",
        ]

        lines = ["PLAN:"]
        for i, s in enumerate(steps, start=1):
            lines.append(f"{i}. {s}")
        return "\n".join(lines)

    @staticmethod
    def _extract_math_expression(question: str) -> Optional[str]:
        """
        Extract a plausible arithmetic expression from the user's question.
        """
        q = question.lower()
        if any(k in q for k in ["calculate", "what is", "compute", "evaluate"]):
            # Try "what is <expr>" or "calculate <expr>" patterns.
            m = re.search(r"(?:what is|calculate|compute|evaluate)\s+(.+)$", q)
            if m:
                expr = m.group(1).strip()
                expr = re.sub(r"[^0-9+\-*/^().%\s]", "", expr)
                expr = expr.replace(" ", "")
                if re.search(r"\d", expr) and re.search(r"[+\-*/^%]", expr):
                    return expr

        # Otherwise: if the whole question is mostly an expression.
        stripped = re.sub(r"[^0-9+\-*/^().%\s]", "", q).replace(" ", "")
        if re.search(r"\d", stripped) and re.search(r"[+\-*/^%]", stripped):
            # Require some operator presence to avoid just "42".
            if len(stripped) >= 3:
                return stripped
        return None

    @staticmethod
    def _looks_like_wikipedia_question(question: str) -> bool:
        q = question.lower()
        if any(k in q for k in ["wikipedia", "capital of", "who is", "what is", "when was", "where is", "define "]):
            return True
        # Very small fallback: "x is y" patterns often map to general knowledge.
        if len(q) <= 80 and (" is " in q or q.startswith("who ") or q.startswith("what ")):
            return True
        return False

    @staticmethod
    def _parse_action_and_input(text: str) -> Tuple[str, str]:
        """
        Regex-based parsing of Action and Action Input from the generated text.
        """
        action_match = ReActAgent.ACTION_RE.search(text)
        input_match = ReActAgent.ACTION_INPUT_RE.search(text)

        if not action_match:
            raise ValueError(f"Could not parse Action from text:\n{text}")
        if not input_match:
            raise ValueError(f"Could not parse Action Input from text:\n{text}")

        action = action_match.group("action").strip()
        action_input_raw = input_match.group("input").strip()
        return action, action_input_raw

    @staticmethod
    def _coerce_action_input(action_input_raw: str) -> ActionInput:
        """
        Best-effort coercion of Action Input into structured data.
        """
        s = action_input_raw.strip()

        # If it looks like JSON, parse it.
        if (s.startswith("{") and s.endswith("}")) or (s.startswith("[") and s.endswith("]")):
            try:
                return json.loads(s)
            except Exception:
                return s

        # If it's a JSON string literal, parse it.
        if (s.startswith('"') and s.endswith('"')) or (s.startswith("'") and s.endswith("'")):
            try:
                return json.loads(s)  # handles double quotes; single quotes likely won't.
            except Exception:
                # Strip quotes as fallback.
                return s.strip("'").strip('"')

        # Otherwise return as plain string.
        return s

    @staticmethod
    def _call_tool(tool: Callable[..., str], action_input: ActionInput) -> str:
        if isinstance(action_input, dict):
            return tool(**action_input)
        if isinstance(action_input, list):
            return tool(*action_input)
        return tool(action_input)  # positional single argument

    def run(self, question: str, verbose: bool = True) -> str:
        steps: List[ReActStep] = []

        for _ in range(self.max_steps):
            scratchpad = self._format_scratchpad(steps)
            prompt = self.prompt_template.format(question=question, scratchpad=scratchpad)
            _ = prompt  # For a real LLM, you'd send `prompt` to the model.

            policy_text = self._policy_predict(question=question, scratchpad=scratchpad)

            action, action_input_raw = self._parse_action_and_input(policy_text)
            thought_match = re.search(r"Thought:\s*(?P<thought>.+?)\s*$", policy_text, re.MULTILINE)
            thought = thought_match.group("thought").strip() if thought_match else ""

            action_input = self._coerce_action_input(action_input_raw)

            # FINISH step: return final answer (or fall back to the thought).
            if action == "FINISH":
                final_match = self.FINAL_RE.search(policy_text)
                final = final_match.group("final").strip() if final_match else thought
                if verbose:
                    # Ensure the trace includes the final "schema".
                    trace = (
                        f"Thought: {thought}\n"
                        "Action: FINISH\n"
                        f"Action Input: {action_input_raw}\n"
                        f"Final Answer: {final}"
                    )
                    print(trace)
                return final

            # PLAN step: generate an internal plan as an observation (no tools).
            if action == "PLAN":
                if isinstance(action_input, str):
                    goal = action_input
                else:
                    goal = question
                observation = self._make_plan(goal)
                step = ReActStep(
                    thought=thought,
                    action=action,
                    action_input_raw=action_input_raw,
                    observation=observation,
                )
                steps.append(step)
                if verbose:
                    print(
                        f"Thought: {thought}\n"
                        f"Action: {action}\n"
                        f"Action Input: {action_input_raw}\n"
                        f"Observation: {observation}\n"
                    )
                continue

            if action not in self.tools:
                observation = f"ERROR: Unknown tool '{action}'. Available: {sorted(self.tools.keys())}"
            else:
                try:
                    observation = self._call_tool(self.tools[action], action_input)
                except Exception as e:
                    observation = f"ERROR: Tool '{action}' crashed: {e}"

            step = ReActStep(
                thought=thought,
                action=action,
                action_input_raw=action_input_raw,
                observation=observation,
            )
            steps.append(step)

            if verbose:
                print(
                    f"Thought: {thought}\n"
                    f"Action: {action}\n"
                    f"Action Input: {action_input_raw}\n"
                    f"Observation: {observation}\n"
                )

        # If we exhaust the loop without FINISH.
        return "Reached max steps without producing FINISH."


def load_default_prompt(prompt_path: Optional[Union[str, Path]] = None) -> str:
    if prompt_path is None:
        prompt_path = Path(__file__).resolve().parent / "prompts" / "react_prompt.txt"
    else:
        prompt_path = Path(prompt_path)
    return prompt_path.read_text(encoding="utf-8")

