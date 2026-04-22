"""Validation layer: verifies extracted ESG events are grounded in their source texts."""

from __future__ import annotations

from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import ValidationError

from esg_agent.config import ModelType
from esg_agent.models import ESGEvent, ModelFactory, ValidationResult
from esg_agent.prompts.extraction_prompt import VALIDATION_SYSTEM_PROMPT


_VALIDATION_USER_TEMPLATE = """\
Event to validate:
  company:        {company}
  event:          {event}
  category:       {category}
  confidence:     {confidence}
  source_excerpt: {source_excerpt}

Is the event description clearly and specifically supported by the source_excerpt above?

Respond with a JSON object with keys:
  is_grounded (bool), grounding_explanation (str).
"""


class ESGValidator:
    """Validates ESG events against their source excerpts using an LLM."""

    def __init__(self, model_type: ModelType | None = None) -> None:
        self._model_type = model_type or ModelFactory.get_model_type()
        self._model = ModelFactory.create_structured_model(
            self._model_type, schema=_GroundingResponse
        )

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def validate(self, event: ESGEvent) -> ValidationResult:
        """Validate a single *event* against its source excerpt.

        Args:
            event: The extracted ESG event to validate.

        Returns:
            :class:`ValidationResult` with is_grounded and an explanation.
        """
        prompt = _VALIDATION_USER_TEMPLATE.format(
            company=event.company,
            event=event.event,
            category=event.category.value,
            confidence=event.confidence,
            source_excerpt=event.source_excerpt,
        )
        messages = [
            SystemMessage(content=VALIDATION_SYSTEM_PROMPT),
            HumanMessage(content=prompt),
        ]

        try:
            response: _GroundingResponse = self._model.invoke(messages)
            return ValidationResult(
                event=event,
                is_grounded=response.is_grounded,
                grounding_explanation=response.grounding_explanation,
            )
        except (ValidationError, Exception) as exc:
            logger.warning(f"Validation failed for event '{event.event}': {exc}")
            return ValidationResult(
                event=event,
                is_grounded=False,
                grounding_explanation=f"Validation error: {exc}",
            )

    def validate_all(self, events: list[ESGEvent]) -> list[ESGEvent]:
        """Validate every event in *events* and mark each with its result.

        Args:
            events: List of extracted :class:`ESGEvent` instances.

        Returns:
            The same list with the ``validated`` field updated in-place.
        """
        results: list[ESGEvent] = []
        for event in events:
            validation = self.validate(event)
            updated = event.model_copy(update={"validated": validation.is_grounded})
            logger.debug(
                f"Validated '{updated.event[:60]}' → grounded={validation.is_grounded}"
            )
            results.append(updated)
        return results


# ---------------------------------------------------------------------------
# Internal schema for structured LLM output during validation
# ---------------------------------------------------------------------------

from pydantic import BaseModel, Field


class _GroundingResponse(BaseModel):
    is_grounded: bool = Field(
        description="True if the event is clearly supported by the source excerpt."
    )
    grounding_explanation: str = Field(
        description="Explanation referencing specific parts of the excerpt."
    )
