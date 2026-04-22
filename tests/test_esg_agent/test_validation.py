"""Unit tests for esg_agent.validation."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from esg_agent.config import ESGCategory, ModelType
from esg_agent.models import ESGEvent, ValidationResult
from esg_agent.validation import ESGValidator, _GroundingResponse


def _make_event(
    company: str = "Tesla",
    event: str = "launched solar roof program",
    category: ESGCategory = ESGCategory.environmental,
    confidence: float = 0.85,
    source_excerpt: str = "Tesla launched its solar roof program.",
) -> ESGEvent:
    return ESGEvent(
        company=company,
        event=event,
        category=category,
        confidence=confidence,
        source_excerpt=source_excerpt,
    )


def _make_validator_with_mock(is_grounded: bool, explanation: str) -> ESGValidator:
    validator = ESGValidator.__new__(ESGValidator)
    validator._model_type = ModelType.OPENAI_GPT4O_MINI

    mock_model = MagicMock()
    mock_model.invoke.return_value = _GroundingResponse(
        is_grounded=is_grounded,
        grounding_explanation=explanation,
    )
    validator._model = mock_model
    return validator


class TestESGValidator:
    def test_validate_returns_validation_result(self):
        validator = _make_validator_with_mock(True, "The excerpt confirms the event.")
        event = _make_event()
        result = validator.validate(event)

        assert isinstance(result, ValidationResult)
        assert result.is_grounded is True
        assert "confirms" in result.grounding_explanation

    def test_validate_false_when_not_grounded(self):
        validator = _make_validator_with_mock(False, "The excerpt does not mention it.")
        event = _make_event()
        result = validator.validate(event)

        assert result.is_grounded is False

    def test_validate_returns_false_on_model_error(self):
        validator = ESGValidator.__new__(ESGValidator)
        validator._model_type = ModelType.OPENAI_GPT4O_MINI

        mock_model = MagicMock()
        mock_model.invoke.side_effect = RuntimeError("connection error")
        validator._model = mock_model

        event = _make_event()
        result = validator.validate(event)

        assert result.is_grounded is False
        assert "Validation error" in result.grounding_explanation

    def test_validate_all_marks_validated_flag(self):
        validator = _make_validator_with_mock(True, "Grounded.")
        events = [_make_event(), _make_event(event="different event")]
        updated = validator.validate_all(events)

        assert len(updated) == 2
        assert all(e.validated for e in updated)

    def test_validate_all_original_events_unchanged(self):
        validator = _make_validator_with_mock(True, "Grounded.")
        event = _make_event()
        original_validated = event.validated

        validator.validate_all([event])

        assert event.validated == original_validated  # model_copy does not mutate

    def test_validate_all_with_mixed_results(self):
        validator = ESGValidator.__new__(ESGValidator)
        validator._model_type = ModelType.OPENAI_GPT4O_MINI

        responses = [
            _GroundingResponse(is_grounded=True, grounding_explanation="yes"),
            _GroundingResponse(is_grounded=False, grounding_explanation="no"),
        ]
        mock_model = MagicMock()
        mock_model.invoke.side_effect = responses
        validator._model = mock_model

        events = [_make_event(), _make_event(event="unrelated claim")]
        updated = validator.validate_all(events)

        assert updated[0].validated is True
        assert updated[1].validated is False
