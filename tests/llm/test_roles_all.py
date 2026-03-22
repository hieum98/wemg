"""Live LLM tests: every `Role` returns valid structured output that matches prompts.

Inspecting output locally:
  pytest -s tests/llm/test_roles_all.py -m requires_llm

Pause / interact after each case (set ``WEMG_ROLE_TEST_PAUSE``):

- ``1`` / ``yes`` — wait for Enter only (no extra interaction).
- ``pdb`` — drop into the debugger: inspect ``parsed``, ``raw``, ``role``, ``inp``,
  ``log_data``, ``results``; type ``c`` (continue) to run the rest of the test and
  move to the next case.
- ``repl`` — open an interactive Python session with those names bound in locals;
  exit with Ctrl-D (EOF) or ``exit()`` to continue.

Choose which cases to run (default: all). Comma-separated **param ids** (see
``_all_execute_role_cases`` in this file, or ``pytest --collect-only -q``)::

  WEMG_ROLE_TEST_IDS=answer_generator,evaluator pytest -s tests/llm/test_roles_all.py -m requires_llm

For both NER scenarios (same underlying ``Role.name`` ``named_entity_recognition``)::

  WEMG_ROLE_TEST_IDS=named_entity_recognition
  # or: ner, ner_all

Examples::

  WEMG_ROLE_TEST_PAUSE=pdb pytest -s tests/llm/test_roles_all.py -m requires_llm -k answer_generator
  WEMG_ROLE_TEST_PAUSE=repl pytest -s tests/llm/test_roles_all.py -m requires_llm -k answer_generator
"""

from __future__ import annotations

import code
import json
import os

import pytest

from tests.conftest import requires_llm_credentials
from wemg.llm.client import LLMClient
from wemg.llm.roles import (
    ANSWER_GENERATOR,
    CONSENSUS_EVALUATOR,
    EVALUATOR,
    EXTRACTOR,
    FINAL_ANSWER_SYNTHESIZER,
    MAJORITY_VOTER,
    MEMORY_CONSOLIDATOR,
    NER,
    QUERY_GENERATOR,
    QUESTION_REPHRASER,
    REASONING_SYNTHESIZER,
    RELATION_EXTRACTOR,
    Role,
    SELF_CORRECTOR,
    STRUCTURED_QUERY_GENERATOR,
    SUBQUESTION_GENERATOR,
    TRIPLE_PRUNER,
    AnswerGenerationInput,
    AnswerEvaluationInput,
    ConsensusEvaluationInput,
    ExtractionInput,
    FinalAnswerSynthesisInput,
    MajorityVoteInput,
    MemoryConsolidationInput,
    NERInput,
    QueryGeneratorInput,
    QuestionRephraserInput,
    ReasoningSynthesizeInput,
    RelationExtractionInput,
    SelfCorrectionInput,
    SubquestionGenerationInput,
    TriplePruneInput,
    QueryGraphGeneratorInput,
    execute_role,
    format_messages,
    set_structured_query_prompt,
)
from wemg.retrieval.wikidata import WikidataEntity


@pytest.fixture(scope="module", autouse=True)
def _structured_query_prompt():
    """Fill `{reference_relations}` so STRUCTURED_QUERY_GENERATOR has a valid system prompt."""
    set_structured_query_prompt(
        {
            "P31": {"label": "instance of", "description": "class of which this entity is a member"},
            "P36": {"label": "capital", "description": "administrative capital of a jurisdiction"},
        }
    )


def _make_client(wemg_config):
    requires_llm_credentials(wemg_config)
    return LLMClient(
        model_name=wemg_config.llm.model_name,
        url=wemg_config.llm.url,
        api_key=wemg_config.llm.api_key,
        max_retries=1,
        max_tokens=min(wemg_config.llm.generation.max_tokens, 8192),
        temperature=0.0,
        cache_config={"enabled": False},
    )


_NER_PARAM_IDS = frozenset({"ner_open_text", "ner_known_entities"})
_NER_GROUP_ALIASES = frozenset({"named_entity_recognition", "ner", "ner_all"})


def _all_execute_role_cases():
    """Full list of pytest.param(...): (role, input, assertion context). IDs must be unique."""
    triples = [
        "France - capital - Paris",
        "France - continent - Europe",
        "Japan - capital - Tokyo",
    ]
    return [
        pytest.param(
            SUBQUESTION_GENERATOR,
            SubquestionGenerationInput(
                question="What is the population of the capital city of France?",
                context="France is a country in Europe.",
            ),
            None,
            id="subquestion_generator",
        ),
        pytest.param(
            ANSWER_GENERATOR,
            AnswerGenerationInput(
                question="What is the capital of France?",
                context="Use general geographic knowledge.",
            ),
            {"expect_substrings": ("paris",)},
            id="answer_generator",
        ),
        pytest.param(
            QUERY_GENERATOR,
            QueryGeneratorInput(input_text="What is the capital of France?"),
            {"min_queries": 1},
            id="query_generator",
        ),
        pytest.param(
            SELF_CORRECTOR,
            SelfCorrectionInput(
                question="What is the capital of France?",
                proposed_answer="Paris",
                context="France is a country in Western Europe.",
            ),
            {"expect_status_in": ("correct", "partial")},
            id="self_corrector",
        ),
        pytest.param(
            QUESTION_REPHRASER,
            QuestionRephraserInput(
                original_question="that big EU country capital city",
                context="We mean France.",
            ),
            {"expect_substrings": ("france", "capital")},
            id="question_rephraser",
        ),
        pytest.param(
            REASONING_SYNTHESIZER,
            ReasoningSynthesizeInput(
                question="What is the capital of France?",
                context="France is a country in Europe. Its capital is Paris.",
            ),
            {"expect_answerable": True},
            id="reasoning_synthesizer",
        ),
        pytest.param(
            EVALUATOR,
            AnswerEvaluationInput(
                user_question="What is the capital of France?",
                system_answer="Paris",
                correct_answer="Paris",
            ),
            {"min_rating": 8.0},
            id="evaluator",
        ),
        pytest.param(
            MAJORITY_VOTER,
            MajorityVoteInput(
                question="What is the capital of France?",
                answers=["Paris", "Paris, France", "Paris is the capital"],
            ),
            {"expect_substrings": ("paris",)},
            id="majority_voter",
        ),
        pytest.param(
            FINAL_ANSWER_SYNTHESIZER,
            FinalAnswerSynthesisInput(
                question="What is the capital of France?",
                candidate_answers=["Paris", "Lyon"],
            ),
            {"expect_substrings": ("paris",)},
            id="final_answer_synthesizer",
        ),
        pytest.param(
            CONSENSUS_EVALUATOR,
            ConsensusEvaluationInput(
                question="What is the capital of France?",
                candidate_answers=[
                    "Paris is the capital of France.",
                    "The capital city is Paris.",
                ],
            ),
            {"min_rating": 7.0},
            id="consensus_evaluator",
        ),
        pytest.param(
            EXTRACTOR,
            ExtractionInput(
                question="What is the capital of France?",
                raw_data="France is a republic in Western Europe. Its capital and largest city is Paris.",
            ),
            {"min_bullets": 1},
            id="extractor",
        ),
        pytest.param(
            MEMORY_CONSOLIDATOR,
            MemoryConsolidationInput(
                question="What is the capital of France?",
                memory=(
                    "[Retrieval] France is in Europe.\n"
                    "[Retrieval] Paris is the capital of France.\n"
                    "[System Prediction] Lyon might be relevant."
                ),
            ),
            {"min_items": 1},
            id="memory_consolidation",
        ),
        pytest.param(
            EXTRACTOR,
            ExtractionInput(
                question=(
                    "According to the archival notes below, which cities hosted the "
                    "Summer Olympic Games in 2012 and 2016, respectively?"
                ),
                raw_data=(
                    "Internal memo (confidential draft). Sponsorship contacts should use the IOC portal.\n\n"
                    "Historical summary: the 2012 Summer Olympics were held in London, United Kingdom. "
                    "Transport upgrades for those Games included the Jubilee line extension.\n\n"
                    "The 2016 Summer Olympics took place in Rio de Janeiro, Brazil, with venues spread "
                    "across Barra, Copacabana, and Maracanã regions.\n\n"
                    "Note for scheduling: the 2020 Summer Olympics were awarded to Tokyo but postponed; "
                    "the 2018 Winter Olympics were in Pyeongchang (not relevant to Summer hosting cities).\n\n"
                    "Legacy reporting: Rio faced budget scrutiny; London's legacy included park redevelopment."
                ),
            ),
            {
                "min_bullets": 2,
                "expect_relevant_substrings": ("london", "rio"),
            },
            id="extractor_complex",
        ),
        pytest.param(
            MEMORY_CONSOLIDATOR,
            MemoryConsolidationInput(
                question="What is the chemical symbol for gold and its atomic number?",
                memory=(
                    "[Retrieval] Gold is a chemical element; its atomic number is 79.\n"
                    "[Retrieval] The symbol for gold is Au, from Latin aurum.\n"
                    "[System Prediction] The symbol for gold is Go (uncertain automated parse).\n"
                    "[Retrieval] The symbol for gold is Au (duplicate line for verification).\n"
                    "[Retrieval] Silver has atomic number 47 and symbol Ag.\n"
                    "[Retrieval] Copper has atomic number 29 and symbol Cu."
                    "[Retrieval] Gold is a chemical element; its atomic number is 79."
                ),
            ),
            {
                "min_items": 1,
                "expect_consolidated_substrings": ("au", "79"),
            },
            id="memory_consolidation_complex",
        ),
        pytest.param(
            NER,
            NERInput(text="Barack Obama spoke at a summit in Paris, France."),
            {"expect_entity_substrings": ("paris", "obama")},
            id="ner_open_text",
        ),
        pytest.param(
            NER,
            NERInput(
                text="Emmanuel Macron is the president of France.",
                known_entities=[
                    WikidataEntity(
                        qid="Q3052772",
                        label="Emmanuel Macron",
                        description="President of France",
                    )
                ],
            ),
            {"expect_entity_substrings": ("macron",), "optional_qid": "Q3052772"},
            id="ner_known_entities",
        ),
        pytest.param(
            RELATION_EXTRACTOR,
            RelationExtractionInput(
                text="Paris is the capital and largest city of France.",
            ),
            {"min_relations": 1},
            id="relation_extraction",
        ),
        pytest.param(
            TRIPLE_PRUNER,
            TriplePruneInput(
                question="What is the capital of France?",
                triples=triples,
            ),
            {"triple_count": len(triples), "expect_index_in_keeps": 0},
            id="triple_pruner",
        ),
    ]


def _parse_role_test_id_filter() -> frozenset[str] | None:
    """If ``WEMG_ROLE_TEST_IDS`` is set, return allowed param ids (lowercase); else None = run all."""
    raw = os.environ.get("WEMG_ROLE_TEST_IDS", "").strip()
    if not raw:
        return None
    all_params = _all_execute_role_cases()
    valid_ids = {getattr(p, "id", "").lower() for p in all_params}
    wanted: set[str] = set()
    for token in raw.split(","):
        t = token.strip().lower()
        if not t:
            continue
        if t in _NER_GROUP_ALIASES:
            wanted.update(_NER_PARAM_IDS)
        else:
            wanted.add(t)
    unknown = wanted - valid_ids
    if unknown:
        raise ValueError(
            "WEMG_ROLE_TEST_IDS unknown id(s): "
            f"{sorted(unknown)}. Valid: {', '.join(sorted(valid_ids))}. "
            f"NER aliases (both NER cases): {', '.join(sorted(_NER_GROUP_ALIASES))}."
        )
    return frozenset(wanted)


def _execute_role_cases():
    """Param list for tests: all cases, or filtered by ``WEMG_ROLE_TEST_IDS``."""
    all_params = _all_execute_role_cases()
    filt = _parse_role_test_id_filter()
    if filt is None:
        return all_params
    out = [p for p in all_params if getattr(p, "id", "").lower() in filt]
    if not out:
        raise RuntimeError(
            "WEMG_ROLE_TEST_IDS produced an empty selection (internal error)."
        )
    return out


def _assert_prompt_follows(role: Role, parsed, ctx: dict | None):
    """Light semantic checks on top of Pydantic validation."""
    if not ctx:
        return
    text_blob = " ".join(
        str(getattr(parsed, name, "")).lower()
        for name in parsed.__class__.model_fields
        if isinstance(getattr(parsed, name, None), str)
    )
    if subs := ctx.get("expect_substrings"):
        assert any(s in text_blob for s in subs), f"{role.name}: expected one of {subs} in output fields"
    if "expect_answerable" in ctx:
        assert parsed.is_answerable is ctx["expect_answerable"]
    if opts := ctx.get("expect_status_in"):
        assert parsed.status.strip().lower() in opts
    if "min_rating" in ctx:
        assert parsed.rating >= ctx["min_rating"]
    if "min_queries" in ctx:
        assert len(parsed.queries) >= ctx["min_queries"]
    if "min_bullets" in ctx:
        assert len(parsed.relevant_information) >= ctx["min_bullets"]
    if subs := ctx.get("expect_relevant_substrings"):
        bullets = getattr(parsed, "relevant_information", None)
        assert bullets is not None, f"{role.name}: expected relevant_information"
        blob = " ".join(b.lower() for b in bullets)
        for s in subs:
            assert s.lower() in blob, f"{role.name}: expected {s!r} in relevant_information"
    if "min_items" in ctx:
        assert len(parsed.consolidated_memory) >= ctx["min_items"]
    if subs := ctx.get("expect_consolidated_substrings"):
        items = getattr(parsed, "consolidated_memory", None)
        assert items is not None, f"{role.name}: expected consolidated_memory"
        blob = " ".join(m.content.lower() for m in items)
        for s in subs:
            assert s.lower() in blob, f"{role.name}: expected {s!r} in consolidated_memory content"
    if ents := ctx.get("expect_entity_substrings"):
        names = " ".join(e.name.lower() for e in parsed.entities)
        assert any(x in names for x in ents)
    if qid := ctx.get("optional_qid"):
        linked = [e for e in parsed.entities if getattr(e, "id", None) == qid]
        if not linked:
            names = " ".join(e.name.lower() for e in parsed.entities)
            assert "macron" in names, f"{role.name}: expected Wikidata link or Macron in names"
    if "min_relations" in ctx:
        assert len(parsed.relations) >= ctx["min_relations"]
    if tc := ctx.get("triple_count"):
        for i in parsed.keep_indices:
            assert 0 <= i < tc, f"{role.name}: invalid index {i} for {tc} triples"
    if (want := ctx.get("expect_index_in_keeps")) is not None:
        assert want in parsed.keep_indices, f"{role.name}: expected triple index {want} to be kept"


def _role_test_pause_mode() -> str | None:
    """Return pause mode: None, 'input', 'pdb', or 'repl'."""
    v = os.environ.get("WEMG_ROLE_TEST_PAUSE", "").strip().lower()
    if not v or v in ("0", "no", "false", "off", "n"):
        return None
    if v in ("pdb", "debug", "breakpoint"):
        return "pdb"
    if v in ("repl", "python", "shell", "ipython"):
        return "repl"
    if v in ("1", "yes", "true", "y", "enter"):
        return "input"
    return "input"


def _interactive_pause_after_print(
    *,
    role: Role,
    inp,
    parsed,
    log_data: dict,
    results: list,
) -> None:
    """Let the user inspect objects: Enter, pdb, or embedded REPL."""
    entries = log_data.get(role.name, [])
    raw = entries[0][1] if entries else ""
    mode = _role_test_pause_mode()
    if mode is None:
        return
    if mode == "input":
        input(f"[{role.name}] Press Enter to continue to the next role test... ")
        return
    if mode == "pdb":
        breakpoint()
        return
    banner = (
        f"\n--- interactive REPL [{role.name}] ---\n"
        "Locals: role, inp, parsed, raw, log_data, results\n"
        "Exit: Ctrl-D or exit()\n"
    )
    code.interact(
        banner=banner,
        local={
            "role": role,
            "inp": inp,
            "parsed": parsed,
            "raw": raw,
            "log_data": log_data,
            "results": results,
        },
    )


def _print_role_inspection(role: Role, inp, log_data: dict, parsed) -> None:
    """Print input, raw model output, and parsed object for manual inspection (use pytest -s)."""
    entries = log_data.get(role.name, [])
    raw = entries[0][1] if entries else ""
    dump = parsed.model_dump() if hasattr(parsed, "model_dump") else parsed
    block = [
        "",
        "=" * 80,
        f"ROLE: {role.name}",
        "-" * 80,
        "INPUT:",
        str(inp),
        "-" * 80,
        "RAW LLM OUTPUT:",
        raw.strip(),
        "-" * 80,
        "PARSED (model_dump):",
        json.dumps(dump, indent=2, default=str),
        "=" * 80,
        "",
    ]
    print("\n".join(block), flush=True)


@pytest.mark.requires_llm
@pytest.mark.asyncio
@pytest.mark.parametrize("role,inp,ctx", _execute_role_cases())
async def test_execute_role_all_roles_parse_and_follow_prompt(wemg_config, role, inp, ctx):
    """Each role returns at least one valid output; optional checks enforce prompt intent."""
    client = _make_client(wemg_config)
    try:
        results, log_data = await execute_role(
            client=client,
            role=role,
            input_data=inp,
            n=1,
        )
        assert isinstance(results, list)
        assert len(results) >= 1, f"{role.name}: LLM output did not parse to schema"
        parsed = results[0]
        assert role.name in log_data
        assert log_data[role.name][0][0] == str(inp)
        assert log_data[role.name][0][1]
        _print_role_inspection(role, inp, log_data, parsed)
        _interactive_pause_after_print(
            role=role,
            inp=inp,
            parsed=parsed,
            log_data=log_data,
            results=results,
        )
        _assert_prompt_follows(role, parsed, ctx)
    finally:
        client.close()


@pytest.mark.parametrize("role,inp,ctx", _execute_role_cases())
def test_format_messages_includes_system_prompt(role, inp, ctx):
    """Sanity: every role exposes a non-empty system prompt and user body."""
    _ = ctx
    msgs = format_messages(role, inp)
    assert msgs[0]["role"] == "system"
    assert len(msgs[0]["content"].strip()) > 20
    assert msgs[-1]["role"] == "user"
    assert msgs[-1]["content"] == str(inp)
