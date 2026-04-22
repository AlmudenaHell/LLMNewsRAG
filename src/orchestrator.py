"""Top-level orchestrator for ESG analysis."""

from __future__ import annotations

import sys
import os
from typing import Any

from loguru import logger

from esg_agent._db import reset_session_db
from esg_agent.config import ModelType
from esg_agent.extraction import ESGExtractor
from esg_agent.models import ESGEvent, ESGExtractionResult, ModelFactory
from esg_agent.rag.retriever import ESGRetriever
from esg_agent.validation import ESGValidator

# Write logs to stdout so they surface correctly in any execution environment.
logger.remove()
logger.add(sys.stdout, level=os.getenv("LOGURU_LEVEL", "INFO"))


class ESGOrchestrator:
    """Coordinates extraction and validation of ESG events from sustainability reports."""

    def __init__(
        self,
        model_type: ModelType | None = None,
        top_k: int = 5,
    ) -> None:
        self._model_type = model_type or ModelFactory.get_model_type()
        self._retriever = ESGRetriever()
        self._extractor = ESGExtractor(
            model_type=self._model_type,
            retriever=self._retriever,
            top_k=top_k,
        )
        self._validator = ESGValidator(model_type=self._model_type)

    def run(
        self,
        company: str,
        query: str,
        documents: list[str],
    ) -> ESGExtractionResult:
        """Run the full pipeline: RAG → extraction → validation.

        Args:
            company: Company name being analysed.
            query: ESG topic or free-text question.
            documents: List of raw sustainability report texts.

        Returns:
            :class:`ESGExtractionResult` with all events marked as validated or not.
        """
        reset_session_db()

        # Step 1: Extract
        logger.info(f"Step 1 – Extracting ESG events | company={company!r} | query={query!r}")
        result = self._extractor.extract(company, query, documents=documents)

        if not result.events:
            logger.info("No events extracted, returning empty result.")
            return result

        logger.info(f"Extracted {len(result.events)} event(s). Running validation…")

        # Step 2: Validate
        validated_events = self._validator.validate_all(result.events)
        result.events = validated_events

        n_valid = sum(1 for e in validated_events if e.validated)
        logger.info(
            f"Validation complete | {n_valid}/{len(validated_events)} event(s) grounded"
        )
        return result


async def orchestrate_esg_analysis(
    company: str,
    query: str,
    documents: list[str],
    model_type: ModelType | None = None,
    top_k: int = 5,
) -> ESGExtractionResult:
    """Convenience async entry point for the ESG pipeline.

    Args:
        company: Company name being analysed.
        query: ESG topic or free-text question.
        documents: Raw sustainability report text(s) to analyse.
        model_type: LLM to use. Falls back to $ESG_DEFAULT_MODEL.
        top_k: Number of RAG chunks retrieved per query.

    Returns:
        Validated :class:`ESGExtractionResult`.
    """
    orchestrator = ESGOrchestrator(model_type=model_type, top_k=top_k)
    return orchestrator.run(company=company, query=query, documents=documents)
