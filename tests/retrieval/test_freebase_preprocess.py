"""Unit tests for Freebase subgraph preprocessing helpers.

All tests are purely in-memory — no network, LLM, or file I/O.
"""
from __future__ import annotations

import pytest

from wemg.retrieval.freebase_preprocess import (
    _humanize,
    _is_mid,
    _is_schema_relation,
    _process_sample,
    _topological_cvt_order,
)


# ---------------------------------------------------------------------------
# _humanize
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "path, expected",
    [
        ("baseball.baseball_team.roster", "roster"),
        ("baseball.baseball_team.team_stats", "team stats"),
        ("people.person.place_of_birth", "place of birth"),
        ("film.film.directed_by", "directed by"),
        # Compound flattened relation (stored as R1::R2)
        (
            "baseball.baseball_player.batting_stats::baseball.baseball_player_stats.season",
            "batting stats season",
        ),
        (
            "baseball.baseball_team.team_stats::baseball.baseball_team_stats.year",
            "team stats year",
        ),
        # Deeply nested compound
        ("a.b.c_d::e.f.g_h", "c d g h"),
    ],
)
def test_humanize(path, expected):
    assert _humanize(path) == expected


# ---------------------------------------------------------------------------
# _is_mid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "node, expected",
    [
        ("m.02nzb8", True),
        ("m.0h6m4", True),
        ("g.11bcqdfmhd", True),
        ("m.abc123", True),
        ("San Francisco Giants", False),
        ("Alex Rodriguez", False),
        ("2005", False),
        ("New York Yankees", False),
        ("m.", False),           # no alphanumeric after dot
        ("x.02nzb8", False),    # wrong prefix letter
        ("M.02nzb8", False),    # uppercase M
    ],
)
def test_is_mid(node, expected):
    assert _is_mid(node) == expected


# ---------------------------------------------------------------------------
# _is_schema_relation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "rel, expected",
    [
        ("rdf-schema#label", True),
        ("rdf-schema#type", True),
        ("type.type.instance", True),
        ("type.type.key", True),
        ("freebase.type_profile.strict_unique_name", True),
        ("baseball.baseball_player.batting_stats", False),
        ("people.person.place_of_birth", False),
        ("common.topic.notable_types", False),
    ],
)
def test_is_schema_relation(rel, expected):
    assert _is_schema_relation(rel) == expected


# ---------------------------------------------------------------------------
# _topological_cvt_order
# ---------------------------------------------------------------------------


def test_topological_order_leaf_first():
    """CVT with no nested CVTs (leaf) should come before its parent."""
    cvt_nodes = {"M1", "M2"}
    nested_map = {
        "M1": [("attr", "M2")],  # M1 depends on M2
        "M2": [],
    }
    order = _topological_cvt_order(cvt_nodes, nested_map)
    assert order.index("M2") < order.index("M1")


def test_topological_order_no_nesting():
    cvt_nodes = {"M1", "M2"}
    nested_map = {"M1": [], "M2": []}
    order = _topological_cvt_order(cvt_nodes, nested_map)
    assert set(order) == {"M1", "M2"}


def test_topological_order_chain():
    """M3 → M2 → M1 (M1 is leaf)."""
    cvt_nodes = {"M1", "M2", "M3"}
    nested_map = {
        "M3": [("a", "M2")],
        "M2": [("b", "M1")],
        "M1": [],
    }
    order = _topological_cvt_order(cvt_nodes, nested_map)
    assert order.index("M1") < order.index("M2") < order.index("M3")


# ---------------------------------------------------------------------------
# _process_sample — schema filtering
# ---------------------------------------------------------------------------


def test_schema_triples_are_filtered():
    sample = {
        "id": "s1",
        "question": "Q?",
        "graph": [
            ["Alex Rodriguez", "rdf-schema#label", "Alex Rodriguez"],
            ["Alex Rodriguez", "type.type.instance", "BaseballPlayer"],
            ["Alex Rodriguez", "freebase.type_profile.strict_unique_name", "Alex Rodriguez"],
            ["Alex Rodriguez", "people.person.place_of_birth", "New York"],
        ],
        "q_entity": ["Alex Rodriguez"],
        "answer": ["New York"],
    }
    result = _process_sample(sample)
    # Only the non-schema triple should produce output
    rel_labels = [info["label"] for info in result["property_dict"].values()]
    assert any("place of birth" in lbl for lbl in rel_labels)
    assert not any("label" in lbl.lower() and "rdf" in lbl.lower() for lbl in rel_labels)


# ---------------------------------------------------------------------------
# _process_sample — no CVT nodes
# ---------------------------------------------------------------------------


def test_no_cvt_preserves_triples():
    """When graph has only readable entities, triples pass through unchanged."""
    sample = {
        "id": "s2",
        "question": "Where is Berlin?",
        "graph": [
            ["Berlin", "location.country", "Germany"],
            ["Germany", "location.continent", "Europe"],
        ],
        "q_entity": ["Berlin"],
        "answer": ["Germany"],
    }
    result = _process_sample(sample)

    entity_labels = {info["label"] for info in result["entity_dict"].values()}
    assert "Berlin" in entity_labels
    assert "Germany" in entity_labels
    assert "Europe" in entity_labels

    assert len(result["triples"]) == 2
    assert result["q_entity_qids"]


def test_no_cvt_q_entity_maps_to_qid():
    sample = {
        "id": "s3",
        "question": "Q?",
        "graph": [["Berlin", "capital_of", "Germany"]],
        "q_entity": ["Berlin"],
        "answer": ["Germany"],
    }
    result = _process_sample(sample)
    # q_entity_qids must be non-empty and point to an entity labelled "Berlin"
    assert result["q_entity_qids"]
    qid = result["q_entity_qids"][0]
    assert result["entity_dict"][qid]["label"] == "Berlin"


# ---------------------------------------------------------------------------
# _process_sample — basic CVT (one parent, multiple attrs)
# ---------------------------------------------------------------------------


def _batting_sample():
    """Alex Rodriguez → m.CVT → {season: 2005, team: Yankees}."""
    return {
        "id": "cvt1",
        "question": "Which team in 2005?",
        "graph": [
            ["Alex Rodriguez", "baseball.baseball_player.batting_stats", "m.cvt01"],
            ["m.cvt01", "baseball.baseball_player_stats.season", "2005"],
            ["m.cvt01", "baseball.baseball_player_stats.team", "New York Yankees"],
            ["Alex Rodriguez", "people.person.place_of_birth", "Miami"],
        ],
        "q_entity": ["Alex Rodriguez"],
        "answer": ["New York Yankees"],
    }


def test_cvt_produces_flattened_triples():
    result = _process_sample(_batting_sample())
    # Expect Rule-1 flattened triples for each attribute
    rel_labels = {info["label"] for info in result["property_dict"].values()}
    assert any("batting stats season" in lbl for lbl in rel_labels)
    assert any("batting stats team" in lbl for lbl in rel_labels)


def test_cvt_original_cvt_triples_dropped():
    """The raw (entity, rel, CVT_MID) and (CVT_MID, attr, val) triples must not appear."""
    result = _process_sample(_batting_sample())
    for subj_id, _prop_id, obj_id in result["triples"]:
        # CVT MID node ids should not appear in the triple endpoints
        assert not obj_id.startswith("m."), f"CVT MID leaked as object: {obj_id}"
        assert not subj_id.startswith("m."), f"CVT MID leaked as subject: {subj_id}"


def test_cvt_synthesis_string_appears_as_object():
    """The synthesized CVT string should appear as an object in the output triples."""
    result = _process_sample(_batting_sample())
    # All QCVT entities should be flagged
    cvt_qids = {
        qid for qid, info in result["entity_dict"].items()
        if info.get("is_cvt_record")
    }
    # At least one CVT record with batting stats info
    assert cvt_qids
    for qid in cvt_qids:
        label = result["entity_dict"][qid]["label"]
        assert "2005" in label or "Yankees" in label or "batting" in label.lower()


def test_cvt_readable_entities_assigned_sequential_qids():
    result = _process_sample(_batting_sample())
    non_cvt = [
        qid for qid, info in result["entity_dict"].items()
        if not info.get("is_cvt_record")
    ]
    assert all(qid.startswith("Q") for qid in non_cvt)


def test_cvt_place_of_birth_triple_preserved():
    """Non-CVT triples (place of birth) must survive CVT processing."""
    result = _process_sample(_batting_sample())
    rel_labels = {info["label"] for info in result["property_dict"].values()}
    assert any("place of birth" in lbl for lbl in rel_labels)


# ---------------------------------------------------------------------------
# _process_sample — multi-parent CVT (Rule 2 cross-entity)
# ---------------------------------------------------------------------------


def _multi_parent_sample():
    """Award ceremony: two players share the same performance CVT node."""
    return {
        "id": "cvt2",
        "question": "Who shared the award?",
        "graph": [
            ["Player A", "award.award_winner.award", "m.cvt_award"],
            ["Player B", "award.award_winner.award", "m.cvt_award"],
            ["m.cvt_award", "award.award_honor.year", "2010"],
        ],
        "q_entity": ["Player A"],
        "answer": ["Player B"],
    }


def test_rule2_cross_entity_triples_emitted():
    """Two parents of the same CVT must produce (A, rel_B, B) and (B, rel_A, A)."""
    result = _multi_parent_sample()
    out = _process_sample(result)

    # Find label for Player A and Player B
    qid_a = next(
        qid for qid, info in out["entity_dict"].items()
        if info.get("label") == "Player A"
    )
    qid_b = next(
        qid for qid, info in out["entity_dict"].items()
        if info.get("label") == "Player B"
    )

    # Expect a triple (A, something, B) or (B, something, A)
    ab_triples = [
        t for t in out["triples"]
        if (t[0] == qid_a and t[2] == qid_b) or (t[0] == qid_b and t[2] == qid_a)
    ]
    assert ab_triples, "No cross-entity Rule-2 triple found between Player A and Player B"


# ---------------------------------------------------------------------------
# _process_sample — nested CVTs
# ---------------------------------------------------------------------------


def _nested_cvt_sample():
    """m.outer CVT has m.inner as a nested CVT attribute."""
    return {
        "id": "nested",
        "question": "test",
        "graph": [
            ["Team X", "team.roster", "m.outer"],
            ["m.outer", "roster.player", "Player Y"],
            ["m.outer", "roster.season_stats", "m.inner"],
            ["m.inner", "stats.goals", "42"],
            ["m.inner", "stats.year", "2020"],
        ],
        "q_entity": ["Team X"],
        "answer": ["Player Y"],
    }


def test_nested_cvt_synthesized_bottom_up():
    """Inner CVT synthesized string should appear in outer CVT's label."""
    result = _process_sample(_nested_cvt_sample())
    cvt_labels = [
        info["label"]
        for info in result["entity_dict"].values()
        if info.get("is_cvt_record")
    ]
    # Outer CVT label should include inner CVT content (goals or year)
    outer_labels = [lbl for lbl in cvt_labels if "42" in lbl or "2020" in lbl or "goals" in lbl]
    assert outer_labels, f"Nested CVT content not propagated; CVT labels: {cvt_labels}"


# ---------------------------------------------------------------------------
# _process_sample — empty graph
# ---------------------------------------------------------------------------


def test_empty_graph_returns_valid_structure():
    sample = {
        "id": "empty",
        "question": "Q?",
        "graph": [],
        "q_entity": ["Some Entity"],
        "answer": [],
    }
    result = _process_sample(sample)
    assert result["id"] == "empty"
    assert result["entity_dict"] == {}
    assert result["property_dict"] == {}
    assert result["triples"] == []
    assert result["q_entity_qids"] == []


# ---------------------------------------------------------------------------
# _process_sample — output schema
# ---------------------------------------------------------------------------


def test_output_has_required_keys():
    sample = {
        "id": "schema_check",
        "question": "Who?",
        "graph": [["A", "rel", "B"]],
        "q_entity": ["A"],
        "answer": ["B"],
    }
    result = _process_sample(sample)
    for key in ("id", "question", "q_entity_qids", "entity_dict", "property_dict", "triples"):
        assert key in result, f"Missing key: {key}"


def test_entity_dict_entries_have_required_fields():
    sample = {
        "id": "fields_check",
        "question": "Q?",
        "graph": [["Alpha", "r", "Beta"]],
        "q_entity": ["Alpha"],
        "answer": ["Beta"],
    }
    result = _process_sample(sample)
    for qid, info in result["entity_dict"].items():
        assert "qid" in info
        assert "label" in info
        assert "description" in info
        assert "is_cvt_record" in info


def test_property_dict_entries_have_required_fields():
    sample = {
        "id": "prop_fields",
        "question": "Q?",
        "graph": [["X", "some.relation.path", "Y"]],
        "q_entity": ["X"],
        "answer": ["Y"],
    }
    result = _process_sample(sample)
    for pid, info in result["property_dict"].items():
        assert "pid" in info
        assert "label" in info


def test_triples_are_serialisable_as_lists():
    sample = {
        "id": "serial",
        "question": "Q?",
        "graph": [["A", "r", "B"]],
        "q_entity": ["A"],
        "answer": ["B"],
    }
    result = _process_sample(sample)
    import json
    # Must not raise
    json.dumps(result)


def test_triples_have_three_elements():
    sample = {
        "id": "t3",
        "question": "Q?",
        "graph": [["A", "r", "B"], ["B", "r2", "C"]],
        "q_entity": ["A"],
        "answer": ["C"],
    }
    result = _process_sample(sample)
    for t in result["triples"]:
        assert len(t) == 3


# ---------------------------------------------------------------------------
# _process_sample — q_entity case-insensitive fallback
# ---------------------------------------------------------------------------


def test_q_entity_case_insensitive_fallback():
    """q_entity text with different case should still find the entity node."""
    sample = {
        "id": "ci",
        "question": "Q?",
        "graph": [["San Francisco Giants", "r", "B"]],
        "q_entity": ["san francisco giants"],   # lowercase variant
        "answer": ["B"],
    }
    result = _process_sample(sample)
    assert result["q_entity_qids"], "q_entity_qids should not be empty for case-insensitive match"
