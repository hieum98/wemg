"""Tests for structured LLM response parsing fallbacks."""

import logging

from wemg.llm.roles import SUBQUESTION_GENERATOR, parse_response


def test_parse_response_logs_fallback_warning(caplog):
    with caplog.at_level(logging.WARNING):
        result = parse_response(SUBQUESTION_GENERATOR, {})

    assert result is None
    assert "Falling back to partial-field parsing" in caplog.text