"""ESG agent: LLM agent with RAG + tools for sustainability report analysis."""

from __future__ import annotations

from typing import Optional, Type

from deepagents import create_deep_agent
from deepagents.backends.filesystem import FilesystemBackend
from langchain.agents.middleware import ModelRetryMiddleware, ToolCallLimitMiddleware
from pydantic import BaseModel

from esg_agent.config import ModelType, settings
from esg_agent.models import ModelFactory
from esg_agent.tools import calculator, lookup_esg_database, think_and_write, web_search

_DATA_DIR = None


def _get_data_dir():
    from pathlib import Path

    global _DATA_DIR
    if _DATA_DIR is None:
        _DATA_DIR = Path(__file__).parent.parent.parent / "data"
    return _DATA_DIR


def create_esg_agent(
    model_type: ModelType | None = None,
    system_prompt: Optional[str] = None,
    output_schema: Optional[Type[BaseModel]] = None,
    log_reasoning: bool = True,
    max_web_searches: int | None = None,
):
    """Create a configured ESG deep agent with RAG-enhanced tools.

    Args:
        model_type: LLM backend to use. Defaults to the value of $ESG_DEFAULT_MODEL.
        system_prompt: Override the default system prompt.
        output_schema: Optional Pydantic schema for structured output.
        log_reasoning: Include the think_and_write tool when True.
        max_web_searches: Cap on web_search tool calls per run.

    Returns:
        A compiled LangGraph agent ready to invoke.
    """
    if model_type is None:
        model_type = ModelFactory.get_model_type()
    if max_web_searches is None:
        max_web_searches = settings.max_web_searches

    tools = [web_search, calculator, lookup_esg_database]
    if log_reasoning:
        tools.insert(0, think_and_write)

    middleware = [
        ModelRetryMiddleware(max_retries=2),
        ToolCallLimitMiddleware(
            tool_name="web_search",
            run_limit=max_web_searches,
            exit_behavior="continue",
        ),
    ]

    agent_kwargs = {
        "model": ModelFactory.create_model(model_type),
        "tools": tools,
        "system_prompt": system_prompt,
        "backend": FilesystemBackend(
            root_dir=str(_get_data_dir()),
            virtual_mode=True,
        ),
        "response_format": output_schema,
        "middleware": middleware,
    }

    return create_deep_agent(**agent_kwargs)
