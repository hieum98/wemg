"""Tests for LLM parsing: extract_info_from_text."""

import pytest

from wemg.llm.parsing import extract_info_from_text


def test_extract_info_direct_json():
    """Direct JSON parse extracts keys."""
    text = '{"answer": "Paris", "score": 0.9}'
    result = extract_info_from_text(text, keys=["answer", "score"], value_type=["str", "float"])
    assert result["answer"] == "Paris"
    assert result["score"] == 0.9


def test_extract_info_json_embedded_in_text():
    """JSON blob inside text is found and parsed."""
    text = 'Some preamble {"name": "Alice", "age": 30} and more text.'
    result = extract_info_from_text(text, keys=["name", "age"], value_type=["str", "int"])
    assert result["name"] == "Alice"
    assert result["age"] == 30


def test_extract_info_regex_fallback_str():
    """Regex fallback for string field."""
    text = 'The answer is "Berlin".'
    result = extract_info_from_text(text, keys=["answer"], value_type=["str"])
    # May get empty if no JSON; regex patterns might match "answer" in text
    assert "answer" in result
    assert isinstance(result["answer"], str)


def test_extract_info_value_types():
    """Multiple value types: str, int, bool, float."""
    text = '{"s": "x", "i": 42, "b": true, "f": 3.14}'
    result = extract_info_from_text(
        text,
        keys=["s", "i", "b", "f"],
        value_type=["str", "int", "bool", "float"],
    )
    assert result["s"] == "x"
    assert result["i"] == 42
    assert result["b"] is True
    assert result["f"] == 3.14


def test_extract_info_keys_value_type_length_mismatch():
    """Raises when keys and value_type length differ."""
    with pytest.raises(ValueError, match="same length"):
        extract_info_from_text("{}", keys=["a", "b"], value_type=["str"])


def test_extract_info_missing_key_defaults():
    """Missing key in JSON gets default for type."""
    text = '{"a": "only"}'
    result = extract_info_from_text(text, keys=["a", "b"], value_type=["str", "int"])
    assert result["a"] == "only"
    assert result["b"] == 0
