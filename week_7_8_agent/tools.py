import ast
import json
import re
import urllib.parse
import urllib.request
from typing import Any, Callable, Dict, Optional


def _safe_eval_arithmetic(expr: str) -> float:
    """
    Safely evaluate a basic arithmetic expression.

    Allowed:
      - numbers (int/float)
      - +, -, *, /, %, **, // (floor div)
      - parentheses
      - unary +/-
    """
    # Normalize a few common user inputs.
    expr = expr.strip()
    expr = expr.replace("^", "**")

    node = ast.parse(expr, mode="eval")

    allowed_binops = {
        ast.Add: lambda a, b: a + b,
        ast.Sub: lambda a, b: a - b,
        ast.Mult: lambda a, b: a * b,
        ast.Div: lambda a, b: a / b,
        ast.Mod: lambda a, b: a % b,
        ast.Pow: lambda a, b: a ** b,
        ast.FloorDiv: lambda a, b: a // b,
    }
    allowed_unops = {
        ast.UAdd: lambda a: +a,
        ast.USub: lambda a: -a,
    }

    def walk(n: ast.AST) -> float:
        if isinstance(n, ast.Expression):
            return walk(n.body)
        if isinstance(n, ast.Num):  # py<3.8
            return float(n.n)
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return float(n.value)
        if isinstance(n, ast.BinOp) and type(n.op) in allowed_binops:
            left = walk(n.left)
            right = walk(n.right)
            return allowed_binops[type(n.op)](left, right)
        if isinstance(n, ast.UnaryOp) and type(n.op) in allowed_unops:
            return allowed_unops[type(n.op)](walk(n.operand))
        if isinstance(n, ast.Call):
            raise ValueError("Function calls are not allowed.")
        if isinstance(n, ast.Name):
            raise ValueError("Variables are not allowed.")
        raise ValueError(f"Unsupported expression element: {type(n).__name__}")

    return walk(node)


def calculator(expression: str) -> str:
    """
    Evaluate a simple arithmetic expression.
    """
    if not isinstance(expression, str) or not expression.strip():
        return "ERROR: calculator expected a non-empty expression string."

    try:
        value = _safe_eval_arithmetic(expression)
    except Exception as e:
        return f"ERROR: {e}"

    # Friendly formatting: show integers without trailing .0 when close.
    if abs(value - round(value)) < 1e-12:
        return str(int(round(value)))
    return str(value)


def wikipedia_search(query: str, max_results: int = 3) -> str:
    """
    Search Wikipedia (English) for the given query using the REST API.
    Returns top results in a compact, tool-friendly format.
    """
    if not isinstance(query, str) or not query.strip():
        return "ERROR: wikipedia_search expected a non-empty query string."

    max_results = int(max(1, min(5, max_results)))
    q = query.strip()

    # MediaWiki search API.
    url = (
        "https://en.wikipedia.org/w/api.php"
        "?action=query"
        "&list=search"
        f"&srsearch={urllib.parse.quote_plus(q)}"
        "&utf8=1"
        "&format=json"
        f"&srlimit={max_results}"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "react-agent/1.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except Exception as e:
        return f"ERROR: wikipedia_search failed: {e}"

    results = data.get("query", {}).get("search", []) or []
    if not results:
        return "RESULT: No Wikipedia results found."

    lines = ["RESULT:"]
    for r in results[:max_results]:
        title = r.get("title", "").strip()
        snippet = (r.get("snippet", "") or "").strip()
        snippet = re.sub(r"<[^>]+>", "", snippet)  # remove html tags
        lines.append(f"- {title}: {snippet}")
    return "\n".join(lines)


# Tool registry used by the agent.
TOOL_REGISTRY: Dict[str, Callable[..., str]] = {
    "calculator": calculator,
    "wikipedia_search": wikipedia_search,
}

