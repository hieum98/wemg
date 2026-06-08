"""Pytest root: register custom markers used across the suite."""

from __future__ import annotations


def pytest_configure(config) -> None:
    markers = [
        ("requires_wikidata", "needs live Wikidata/Wikipedia APIs"),
        ("integration", "real external services"),
        ("slow_integration", "long-running real-system flows"),
    ]
    for name, desc in markers:
        config.addinivalue_line("markers", f"{name}: {desc}")
