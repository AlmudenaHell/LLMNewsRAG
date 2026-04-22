"""Lightweight in-process session database for storing ESG events during a run."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class _SessionDB:
    _records: list[dict] = field(default_factory=list)

    def insert(self, company: str, event: str, category: str, confidence: float) -> None:
        self._records.append(
            {
                "company": company,
                "event": event,
                "category": category,
                "confidence": confidence,
            }
        )

    def query(
        self, company_name: str, category: str | None = None
    ) -> list[dict]:
        matches = [
            r for r in self._records
            if r["company"].lower() == company_name.lower()
        ]
        if category:
            matches = [
                r for r in matches
                if r["category"].lower() == category.lower()
            ]
        return matches

    def clear(self) -> None:
        self._records.clear()


_DB_INSTANCE: _SessionDB | None = None


def get_session_db() -> _SessionDB:
    global _DB_INSTANCE
    if _DB_INSTANCE is None:
        _DB_INSTANCE = _SessionDB()
    return _DB_INSTANCE


def reset_session_db() -> None:
    global _DB_INSTANCE
    _DB_INSTANCE = _SessionDB()
