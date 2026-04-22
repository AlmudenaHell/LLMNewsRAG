"""Agent tools: think_and_write, web_search, calculator, and document lookup."""

from __future__ import annotations

import ast
import math
import operator
from datetime import datetime
from pathlib import Path

from langchain_core.tools import tool
from loguru import logger

project_root = Path(__file__).parent.parent.parent

# ---------------------------------------------------------------------------
# Allowed safe operators for the calculator tool
# ---------------------------------------------------------------------------
_SAFE_OPERATORS: dict[type, any] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.Pow: operator.pow,
    ast.USub: operator.neg,
    ast.UAdd: operator.pos,
}

_SAFE_NAMES: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
    "inf": math.inf,
    "sqrt": math.sqrt,
    "log": math.log,
    "log10": math.log10,
    "exp": math.exp,
    "abs": abs,
    "round": round,
}


def _safe_eval(node: ast.AST) -> float:
    """Recursively evaluate a parsed AST node using only safe operations."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, (int, float)):
            return float(node.value)
        raise ValueError(f"Unsupported constant type: {type(node.value)}")
    if isinstance(node, ast.Name):
        if node.id in _SAFE_NAMES:
            val = _SAFE_NAMES[node.id]
            return val if isinstance(val, float) else float(val)
        raise ValueError(f"Unknown name: {node.id!r}")
    if isinstance(node, ast.Call):
        if isinstance(node.func, ast.Name) and node.func.id in _SAFE_NAMES:
            func = _SAFE_NAMES[node.func.id]
            if callable(func):
                args = [_safe_eval(a) for a in node.args]
                return float(func(*args))
        raise ValueError(f"Unsupported function call: {ast.dump(node)}")
    if isinstance(node, ast.BinOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported operator: {op_type.__name__}")
        left = _safe_eval(node.left)
        right = _safe_eval(node.right)
        return float(_SAFE_OPERATORS[op_type](left, right))
    if isinstance(node, ast.UnaryOp):
        op_type = type(node.op)
        if op_type not in _SAFE_OPERATORS:
            raise ValueError(f"Unsupported unary operator: {op_type.__name__}")
        operand = _safe_eval(node.operand)
        return float(_SAFE_OPERATORS[op_type](operand))
    raise ValueError(f"Unsupported AST node: {type(node).__name__}")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@tool(parse_docstring=True)
def think_and_write(reasoning: str, company_name: str, topic: str) -> str:
    """Write down your current thinking, analysis, and next steps.

    Use this tool to:
    - Document your reasoning before performing searches or extractions
    - Explain why you are checking a particular source
    - Record observations about evidence quality or consistency
    - Track your decision-making process

    All reasoning is timestamped and written to a local log for auditability.

    Args:
        reasoning: Detailed reasoning, observations, or analysis. Be specific.
        company_name: The company currently being analysed.
        topic: A brief label for the current reasoning step (e.g. "carbon targets").

    Returns:
        Confirmation that reasoning was recorded.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_dir = project_root / "data" / "reasoning_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f"{company_name.lower().replace(' ', '_')}_reasoning.md"
    entry = f"## [{timestamp}] {topic}\n\n{reasoning}\n\n---\n\n"

    with log_file.open("a", encoding="utf-8") as fh:
        fh.write(entry)

    logger.debug(f"Reasoning recorded for '{company_name}' → {topic}")
    return f"Reasoning recorded for company='{company_name}', topic='{topic}'."


@tool(parse_docstring=True)
def web_search(query: str) -> str:
    """Search the web for current ESG news, events, or company information.

    Use this to supplement the RAG context with up-to-date information when the
    retrieved document chunks do not contain sufficient detail.

    Args:
        query: A focused search query. Include the company name and ESG topic.

    Returns:
        A text summary of the most relevant search results.
    """
    try:
        from tavily import TavilyClient

        from esg_agent.config import settings

        client = TavilyClient()
        response = client.search(query=query, max_results=5, search_depth="advanced")
        results = response.get("results", [])
        if not results:
            return "No results found."

        lines: list[str] = []
        for r in results:
            title = r.get("title", "")
            url = r.get("url", "")
            content = r.get("content", "")
            lines.append(f"**{title}** ({url})\n{content}")

        return "\n\n".join(lines)
    except Exception as exc:
        logger.warning(f"Web search failed: {exc}")
        return f"Web search unavailable: {exc}"


@tool(parse_docstring=True)
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely.

    Supports: +, -, *, /, ** (power), sqrt(), log(), log10(), exp(), abs(), round(),
    and the constants pi, e.

    Use this tool to compute emissions reductions, growth rates, ratios, or any ESG
    metric that requires arithmetic.

    Args:
        expression: A valid arithmetic expression string, e.g. "sqrt(144) * 2.5".

    Returns:
        The numeric result as a string, or an error message if evaluation fails.
    """
    try:
        tree = ast.parse(expression, mode="eval")
        result = _safe_eval(tree.body)
        logger.debug(f"Calculator: {expression!r} = {result}")
        return str(result)
    except Exception as exc:
        logger.warning(f"Calculator error for expression {expression!r}: {exc}")
        return f"Error evaluating expression: {exc}"


@tool(parse_docstring=True)
def lookup_esg_database(company_name: str, category: str | None = None) -> str:
    """Query the in-memory ESG event database for previously extracted events.

    Use this tool to check whether an event for a company has already been recorded
    in this session, avoiding duplicate extractions.

    Args:
        company_name: The company whose records to look up.
        category: Optional ESG category filter (e.g. "Environmental", "Social").

    Returns:
        A formatted summary of matching records, or a message if none are found.
    """
    from esg_agent._db import get_session_db

    db = get_session_db()
    records = db.query(company_name, category)
    if not records:
        return f"No records found for company='{company_name}'" + (
            f", category='{category}'" if category else ""
        ) + "."

    lines = [f"Found {len(records)} record(s) for '{company_name}':"]
    for r in records:
        lines.append(
            f"  • [{r['category']}] {r['event']} (confidence={r['confidence']:.2f})"
        )
    return "\n".join(lines)
