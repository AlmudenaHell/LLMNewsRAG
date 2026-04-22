"""Unit tests for esg_agent.models."""

from __future__ import annotations

import pytest

from esg_agent.config import ESGCategory, ModelType
from esg_agent.models import ESGEvent, ESGExtractionResult, ModelFactory


class TestESGEvent:
    def test_valid_event_creation(self):
        event = ESGEvent(
            company="Tesla",
            event="announced new manufacturing facility",
            category=ESGCategory.expansion,
            confidence=0.91,
            source_excerpt="Tesla announced a new manufacturing facility in Texas.",
        )
        assert event.company == "Tesla"
        assert event.confidence == 0.91
        assert event.validated is False

    def test_confidence_bounds(self):
        with pytest.raises(ValueError):
            ESGEvent(
                company="X",
                event="e",
                category=ESGCategory.other,
                confidence=1.5,
                source_excerpt="s",
            )
        with pytest.raises(ValueError):
            ESGEvent(
                company="X",
                event="e",
                category=ESGCategory.other,
                confidence=-0.1,
                source_excerpt="s",
            )

    def test_validated_flag_defaults_false(self):
        event = ESGEvent(
            company="BP",
            event="committed to net zero by 2050",
            category=ESGCategory.commitment,
            confidence=0.85,
            source_excerpt="BP committed to achieving net zero by 2050.",
        )
        assert event.validated is False

    def test_model_copy_updates_validated(self):
        event = ESGEvent(
            company="Shell",
            event="reduced scope 1 emissions by 10%",
            category=ESGCategory.environmental,
            confidence=0.8,
            source_excerpt="Shell reduced its Scope 1 emissions by 10% year-on-year.",
        )
        updated = event.model_copy(update={"validated": True})
        assert updated.validated is True
        assert event.validated is False  # original unchanged


class TestESGExtractionResult:
    def test_empty_factory(self):
        result = ESGExtractionResult.empty()
        assert result.events == []
        assert result.raw_context == ""

    def test_construction_with_events(self):
        event = ESGEvent(
            company="Apple",
            event="pledged 100% renewable energy",
            category=ESGCategory.environmental,
            confidence=0.9,
            source_excerpt="Apple pledged to use 100% renewable energy.",
        )
        result = ESGExtractionResult(events=[event], raw_context="context text")
        assert len(result.events) == 1
        assert result.raw_context == "context text"


class TestModelFactory:
    def test_get_model_type_returns_default_on_bad_env(self, monkeypatch):
        monkeypatch.setenv("ESG_DEFAULT_MODEL", "not-a-real-model")
        model_type = ModelFactory.get_model_type()
        assert model_type == ModelType.OPENAI_GPT4O_MINI

    def test_get_model_type_from_env(self, monkeypatch):
        monkeypatch.setenv("ESG_DEFAULT_MODEL", "gpt-4o")
        model_type = ModelFactory.get_model_type()
        assert model_type == ModelType.OPENAI_GPT4O

    def test_create_model_raises_on_unsupported(self):
        with pytest.raises(ValueError, match="Unsupported model type"):
            # Force invalid value using string cast
            ModelFactory.create_model("invalid-model-xyz")  # type: ignore[arg-type]
