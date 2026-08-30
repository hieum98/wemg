"""Sub-EM's wrapper blind spot, and the guard that keeps the relaxed form honest.

``compute_sub_em`` requires the gold string to appear in the prediction verbatim. Three gold
answers in ``datasets/musique_depth.jsonl`` arrive wrapped — ``"at the city of Cairo,
Illinois"``, ``"The Australian Ballet"``, ``"four-year"`` — so a correctly concise answer can
never match them. Measured across **27 runs / 3,240 questions**, that scored 58 answers wrong
that were right: **+1 to +3 in every single run**, mean +2.15, i.e. **+1.79 points**.

Because the loss is uniform across arms it never flipped an A/B conclusion, but it did keep
up to 3 questions out of every paired comparison's discordant pool, which is sign-test power
spent on a string-matching artifact.

The relaxed form must stay *conservative*: it strips only the gold's wrapper, never its
content, and refuses a one-token residue. Most of these tests exist to pin that down, because
a metric that is too permissive would silently inflate every number in the project.
"""

from __future__ import annotations

import pytest

from langgraph_coe.evaluation.metrics import compute_sub_em, compute_sub_em_relaxed


# The three real cases, with the exact strings taken from the evaluation logs.
@pytest.mark.parametrize(
    "predicted, gold",
    [
        ("Cairo, Illinois", "at the city of Cairo, Illinois"),
        ("Australian Ballet", "The Australian Ballet"),
        ("Australian Ballet (TAB)", "The Australian Ballet"),
        ("four years", "four-year"),
        ("Four years", "four-year"),
    ],
)
def test_a_wrapped_gold_no_longer_scores_a_correct_answer_wrong(predicted, gold):
    assert compute_sub_em(predicted, [gold]) == 0.0, "the blind spot being fixed"
    assert compute_sub_em_relaxed(predicted, [gold]) == 1.0


@pytest.mark.parametrize(
    "predicted, gold, why",
    [
        ("Washington D.C.", "the state of Washington", "one-token residue is refused"),
        ("Mississippi River", "the Mississippi River Delta", "'delta' is content"),
        ("1929", "11 February 1929", "underspecified date"),
        ("1970s", "From the 1950s to the 1970s", "underspecified range"),
        ("Latin", "Medieval Latin", "'Medieval' is content"),
        ("Lexington County", "Richland County", "a different county"),
        ("Paris", "in the country of France", "wrapper stripped, content absent"),
        ("Gulf of Mexico", "the Mississippi River Delta", "unrelated"),
    ],
)
def test_a_genuinely_wrong_answer_stays_wrong(predicted, gold, why):
    assert compute_sub_em(predicted, [gold]) == 0.0
    assert compute_sub_em_relaxed(predicted, [gold]) == 0.0, why


def test_the_one_token_guard_is_what_separates_cairo_from_washington():
    """Both golds are ``<head noun> of <name>``; only the residue length differs."""
    assert compute_sub_em_relaxed("Cairo, Illinois", ["at the city of Cairo, Illinois"]) == 1.0
    assert compute_sub_em_relaxed("Washington D.C.", ["the state of Washington"]) == 0.0


def test_it_never_scores_lower_than_sub_em():
    """It is a relaxation, so it must dominate — otherwise it is not an upper bound."""
    pairs = [
        ("Treaty of Paris was signed in 1783", ["Treaty of Paris"]),
        ("Tucson Raceway Park", ["Tucson Raceway Park"]),
        ("Cairo, Illinois", ["at the city of Cairo, Illinois"]),
        ("nothing relevant", ["Richland County"]),
        ("Meg Ryan", ["Meg Ryan", "Margaret Mary Emily Anne Hyra"]),
    ]
    for pred, gold in pairs:
        assert compute_sub_em_relaxed(pred, gold) >= compute_sub_em(pred, gold), (pred, gold)


def test_degenerate_inputs_match_sub_em():
    for pred, gold in (("", ["x"]), ("x", []), ("", []), ("x", [""])):
        assert compute_sub_em_relaxed(pred, gold) == compute_sub_em(pred, gold) == 0.0
    # A bare string gold is accepted, as in sub_em.
    assert compute_sub_em_relaxed("Australian Ballet", "The Australian Ballet") == 1.0


def test_any_one_of_several_golds_suffices():
    golds = ["The Australian Ballet", "Sydney Dance Company"]
    assert compute_sub_em_relaxed("Australian Ballet", golds) == 1.0


def test_the_runner_records_it_beside_sub_em_not_instead_of_it():
    """A replaced metric would silently restate every historical number."""
    import inspect

    from langgraph_coe.evaluation.runner import DatasetEvaluator

    src = inspect.getsource(DatasetEvaluator)
    assert '"sub_em_short": sub_em_short,' in src
    assert '"sub_em_short_relaxed": sub_em_short_relaxed,' in src
