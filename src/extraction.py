"""LLM-based ESG event extraction from RAG-retrieved context."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import ValidationError

from esg_agent.config import ModelType
from esg_agent.models import ESGEvent, ESGExtractionResult, ModelFactory
from esg_agent.prompts.extraction_prompt import EXTRACTION_SYSTEM_PROMPT
from esg_agent.rag.retriever import ESGRetriever


_EXTRACTION_USER_TEMPLATE = """\
Company: {company}
Query: {query}

Retrieved context from sustainability reports:
---
{context}
---

Extract all ESG events for this company that are supported by the context above.
Return a JSON array of event objects with keys:
  company, event, category, confidence, source_excerpt, validated (always false at extraction time).
"""


class ESGExtractor:
    """Extracts structured ESG events using an LLM with structured output."""

    def __init__(
        self,
        model_type: ModelType | None = None,
        retriever: ESGRetriever | None = None,
        top_k: int = 5,
    ) -> None:
        self._model_type = model_type or ModelFactory.get_model_type()
        self._retriever = retriever
        self._top_k = top_k
        self._model = ModelFactory.create_model(self._model_type)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def extract(self, company: str, query: str, documents: list[str] | None = None) -> ESGExtractionResult:
        """Extract ESG events for *company* using optional *documents* for RAG.

        Args:
            company: Company name to analyse.
            query: ESG topic or question guiding the extraction.
            documents: Optional list of raw document texts to index for RAG. If
                       omitted the retriever must have been pre-populated.

        Returns:
            :class:`ESGExtractionResult` containing extracted events and the
            retrieved context that was supplied to the LLM.
        """
        context = self._get_context(company, query, documents)

        prompt = _EXTRACTION_USER_TEMPLATE.format(
            company=company, query=query, context=context
        )
        messages = [
            SystemMessage(content=EXTRACTION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        logger.debug(f"Extracting ESG events for '{company}' | query={query!r}")
        response = self._model.invoke(messages)
        events = self._parse_response(response.content)

        logger.debug(f"Extracted {len(events)} event(s) for '{company}'")
        return ESGExtractionResult(events=events, raw_context=context)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_context(
        self, company: str, query: str, documents: list[str] | None
    ) -> str:
        if documents:
            if self._retriever is None:
                self._retriever = ESGRetriever()
            self._retriever.index_documents(documents)

        if self._retriever and self._retriever.is_indexed:
            combined_query = f"{company} {query}"
            return self._retriever.retrieve_as_text(combined_query, top_k=self._top_k)

        logger.warning("No RAG retriever available; proceeding without context.")
        return ""

    @staticmethod
    def _parse_response(raw: str) -> list[ESGEvent]:
        """Parse the LLM's JSON response into a list of :class:`ESGEvent`."""
        try:
            # Strip markdown code fences if present
            cleaned = raw.strip()
            if cleaned.startswith("```"):
                cleaned = cleaned.split("```")[1]
                if cleaned.startswith("json"):
                    cleaned = cleaned[4:]

            data: Any = json.loads(cleaned.strip())
            if isinstance(data, dict) and "events" in data:
                data = data["events"]
            if not isinstance(data, list):
                data = [data]

            events: list[ESGEvent] = []
            for item in data:
                try:
                    events.append(ESGEvent(**item))
                except (ValidationError, TypeError) as exc:
                    logger.warning(f"Skipping malformed event: {exc} | raw={item}")
            return events
        except (json.JSONDecodeError, Exception) as exc:
            logger.error(f"Failed to parse extraction response: {exc}")
            return []
