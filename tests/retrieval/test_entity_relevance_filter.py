"""Unit tests for known-entity relevance filtering (no API calls)."""

from wemg.retrieval.wikidata import (
    WikidataEntity,
    entity_surface_mentioned_in_text,
    filter_entities_relevant_to_text,
)


def test_filter_multiword_label_substring():
    e = WikidataEntity(qid="Q1", label="New York", description="city")
    assert filter_entities_relevant_to_text([e], "I live in New York.") == [e]
    assert filter_entities_relevant_to_text([e], "I live in Paris.") is None


def test_filter_single_token_word_boundary():
    e = WikidataEntity(qid="Q1", label="US", description="country")
    assert filter_entities_relevant_to_text([e], "The US voted.") is not None
    assert filter_entities_relevant_to_text([e], "focus") is None


def test_filter_alias_match():
    e = WikidataEntity(qid="Q1", label="Foo", description="x", aliases=["Bar"])
    assert filter_entities_relevant_to_text([e], "Bar is here") == [e]


def test_filter_empty_or_none_inputs():
    e = WikidataEntity(qid="Q1", label="Berlin", description="city")
    assert filter_entities_relevant_to_text(None, "hello") is None
    assert filter_entities_relevant_to_text([], "hello") is None
    assert filter_entities_relevant_to_text([e], "") is None
    assert filter_entities_relevant_to_text([e], "   ") is None


def test_entity_surface_mentioned_case_insensitive():
    e = WikidataEntity(qid="Q64", label="Berlin", description="capital")
    assert entity_surface_mentioned_in_text(e, "BERLIN is large")
    assert not entity_surface_mentioned_in_text(e, "Munich only")
