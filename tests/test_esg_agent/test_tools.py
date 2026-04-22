"""Unit tests for esg_agent.tools."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from esg_agent._db import reset_session_db
from esg_agent.tools import calculator, lookup_esg_database, think_and_write


# ---------------------------------------------------------------------------
# calculator
# ---------------------------------------------------------------------------


class TestCalculator:
    def test_addition(self):
        assert calculator.invoke({"expression": "2 + 3"}) == "5.0"

    def test_subtraction(self):
        assert calculator.invoke({"expression": "10 - 4"}) == "6.0"

    def test_multiplication(self):
        assert calculator.invoke({"expression": "3 * 4"}) == "12.0"

    def test_division(self):
        assert calculator.invoke({"expression": "9 / 2"}) == "4.5"

    def test_power(self):
        assert calculator.invoke({"expression": "2 ** 8"}) == "256.0"

    def test_sqrt(self):
        assert calculator.invoke({"expression": "sqrt(144)"}) == "12.0"

    def test_log10(self):
        result = calculator.invoke({"expression": "log10(1000)"})
        assert float(result) == pytest.approx(3.0)

    def test_combined_expression(self):
        result = calculator.invoke({"expression": "round(sqrt(2) * 100)"})
        assert float(result) == pytest.approx(141.0)

    def test_error_on_invalid_expression(self):
        result = calculator.invoke({"expression": "import os"})
        assert "Error" in result

    def test_error_on_disallowed_builtin(self):
        result = calculator.invoke({"expression": "open('file')"})
        assert "Error" in result

    def test_error_on_string_input(self):
        result = calculator.invoke({"expression": "'hello' + 'world'"})
        assert "Error" in result


# ---------------------------------------------------------------------------
# think_and_write
# ---------------------------------------------------------------------------


class TestThinkAndWrite:
    def test_writes_to_file(self, tmp_path, monkeypatch):
        # Redirect the log dir to tmp_path
        import esg_agent.tools as tools_module

        monkeypatch.setattr(tools_module, "project_root", tmp_path)

        result = think_and_write.invoke(
            {
                "reasoning": "Checking emission targets.",
                "company_name": "Tesla",
                "topic": "carbon targets",
            }
        )

        assert "Tesla" in result
        assert "carbon targets" in result

        log_file = tmp_path / "data" / "reasoning_logs" / "tesla_reasoning.md"
        assert log_file.exists()
        content = log_file.read_text()
        assert "Checking emission targets." in content

    def test_appends_multiple_entries(self, tmp_path, monkeypatch):
        import esg_agent.tools as tools_module

        monkeypatch.setattr(tools_module, "project_root", tmp_path)

        for i in range(3):
            think_and_write.invoke(
                {
                    "reasoning": f"Step {i}",
                    "company_name": "BP",
                    "topic": f"step-{i}",
                }
            )

        log_file = tmp_path / "data" / "reasoning_logs" / "bp_reasoning.md"
        content = log_file.read_text()
        assert content.count("Step ") == 3


# ---------------------------------------------------------------------------
# lookup_esg_database
# ---------------------------------------------------------------------------


class TestLookupESGDatabase:
    def setup_method(self):
        reset_session_db()

    def test_returns_no_records_when_empty(self):
        result = lookup_esg_database.invoke({"company_name": "Unknown Corp"})
        assert "No records found" in result

    def test_returns_inserted_records(self):
        from esg_agent._db import get_session_db

        db = get_session_db()
        db.insert("Acme", "launched green bond", "Environmental", 0.88)

        result = lookup_esg_database.invoke({"company_name": "Acme"})
        assert "launched green bond" in result
        assert "Environmental" in result

    def test_filters_by_category(self):
        from esg_agent._db import get_session_db

        db = get_session_db()
        db.insert("Acme", "green bond", "Environmental", 0.9)
        db.insert("Acme", "diversity program", "Social", 0.7)

        result = lookup_esg_database.invoke(
            {"company_name": "Acme", "category": "Social"}
        )
        assert "diversity program" in result
        assert "green bond" not in result

    def test_case_insensitive_company_lookup(self):
        from esg_agent._db import get_session_db

        db = get_session_db()
        db.insert("Tesla", "solar roof announcement", "Environmental", 0.75)

        result = lookup_esg_database.invoke({"company_name": "tesla"})
        assert "solar roof announcement" in result
