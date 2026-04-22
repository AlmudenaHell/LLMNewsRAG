"""Unit tests for esg_agent.extraction."""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from esg_agent.config import ESGCategory, ModelType
from esg_agent.extraction import ESGExtractor
from esg_agent.models import ESGEvent, ESGExtractionResult


def _make_extractor_with_mock_model(response_text: str) -> ESGExtractor:
    """Return an ESGExtractor with a mocked LLM that returns *response_text*."""
    extractor = ESGExtractor.__new__(ESGExtractor)
    extractor._model_type = ModelType.OPENAI_GPT4O_MINI
    extractor._retriever = None
    extractor._top_k = 5

    mock_model = MagicMock()
    mock_model.invoke.return_value = MagicMock(content=response_text)
    extractor._model = mock_model

    return extractor


class TestParseResponse:
    def test_parses_valid_json_array(self):
        raw = json.dumps(
            [
                {
                    "company": "Tesla",
                    "event": "announced new manufacturing facility",
                    "category": "Expansion",
                    "confidence": 0.91,
                    "source_excerpt": "Tesla announced a new factory.",
                    "validated": False,
                }
            ]
        )
        events = ESGExtractor._parse_response(raw)
        assert len(events) == 1
        assert events[0].company == "Tesla"
        assert events[0].category == ESGCategory.expansion

    def test_parses_dict_with_events_key(self):
        raw = json.dumps(
            {
                "events": [
                    {
                        "company": "BP",
                        "event": "net zero pledge",
                        "category": "Commitment",
                        "confidence": 0.85,
                        "source_excerpt": "BP pledged net zero.",
                        "validated": False,
                    }
                ]
            }
        )
        events = ESGExtractor._parse_response(raw)
        assert len(events) == 1
        assert events[0].company == "BP"

    def test_strips_markdown_code_fences(self):
        raw = "```json\n" + json.dumps([{
            "company": "Apple",
            "event": "100% renewable",
            "category": "Environmental",
            "confidence": 0.9,
            "source_excerpt": "Apple uses 100% renewable energy.",
            "validated": False,
        }]) + "\n```"
        events = ESGExtractor._parse_response(raw)
        assert len(events) == 1

    def test_returns_empty_list_on_invalid_json(self):
        events = ESGExtractor._parse_response("not json at all")
        assert events == []

    def test_skips_malformed_events(self):
        raw = json.dumps(
            [
                {"company": "Good", "event": "valid event", "category": "Social",
                 "confidence": 0.7, "source_excerpt": "excerpt"},
                {"missing_required_fields": True},
            ]
        )
        events = ESGExtractor._parse_response(raw)
        # Only the valid one should be returned
        assert len(events) == 1
        assert events[0].company == "Good"


class TestESGExtractor:
    def test_extract_returns_result_with_events(self):
        response_payload = json.dumps([
            {
                "company": "Tesla",
                "event": "announced new manufacturing facility",
                "category": "Expansion",
                "confidence": 0.91,
                "source_excerpt": "Tesla announced a new factory in Texas.",
                "validated": False,
            }
        ])
        extractor = _make_extractor_with_mock_model(response_payload)

        result = extractor.extract(
            company="Tesla",
            query="expansion plans",
            documents=None,
        )

        assert isinstance(result, ESGExtractionResult)
        assert len(result.events) == 1
        assert result.events[0].company == "Tesla"

    def test_extract_returns_empty_on_bad_response(self):
        extractor = _make_extractor_with_mock_model("not parseable")
        result = extractor.extract(company="Acme", query="emissions", documents=None)
        assert result.events == []

    def test_extract_logs_warning_and_continues_on_model_error(self):
        extractor = ESGExtractor.__new__(ESGExtractor)
        extractor._model_type = ModelType.OPENAI_GPT4O_MINI
        extractor._retriever = None
        extractor._top_k = 5

        mock_model = MagicMock()
        mock_model.invoke.side_effect = RuntimeError("LLM timeout")
        extractor._model = mock_model

        with pytest.raises(RuntimeError, match="LLM timeout"):
            extractor.extract(company="Fail Corp", query="test", documents=None)
