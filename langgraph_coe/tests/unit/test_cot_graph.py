"""Phase 2 §7 — CoTGraph target specs.

``CoTGraph`` replaces the legacy ``cot_search`` while-loop with explicit
LangGraph nodes:

  1. ``gen_subq``
  2. conditional route to final answer or retrieval fan-out
  3. KG/Web ``Send`` per subquestion + joined corpus search
  4. rerank
  5. subanswer generation
  6. ``MemoryUpdateGraph``
  7. increment/clear scratch and loop

These tests are target specs: they skip cleanly only while the Phase 2 graph
module itself is absent, then fail on contract drift once implemented.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence
from unittest.mock import MagicMock

import networkx as nx
import pytest

from langgraph_coe import roles as roles_mod


def _import_module():
    """Phase 2 introduces ``langgraph_coe.graphs.cot``."""
    try:
        from langgraph_coe.graphs import cot as cot_mod  # type: ignore
    except ImportError:
        pytest.skip("Phase 2 §7 introduces langgraph_coe.graphs.cot")
    if not hasattr(cot_mod, "build_cot_graph"):
        pytest.skip("build_cot_graph not implemented yet (§7 target)")
    return cot_mod


def _config(*, web_enabled: bool = False):
    from langgraph_coe.config import LangGraphCoeConfig

    cfg = LangGraphCoeConfig.from_yaml()
    cfg.reranker.enabled = False
    cfg.reranker.top_k = 3
    cfg.web_search.enabled = web_enabled
    return cfg


def _registry():
    from langgraph_coe.llm import RoleModelRegistry

    registry = RoleModelRegistry(_config().llm)
    registry.get_model = lambda _role_name: MagicMock()  # type: ignore[assignment]
    return registry


def _subq_out(
    *,
    answerable: bool,
    subquestions: Sequence[str] | None = None,
    needs_kg: Sequence[bool] | None = None,
) -> roles_mod.SubquestionGenerationOutput:
    return roles_mod.SubquestionGenerationOutput(
        is_answerable=answerable,
        subquestions=list(subquestions or []),
        needs_kg=list(needs_kg) if needs_kg is not None else None,
    )


def _answer_out(text: str) -> roles_mod.AnswerGenerationOutput:
    return roles_mod.AnswerGenerationOutput(
        answer=text,
        concise_answer=text,
        reasoning=f"Reasoning for {text}",
        confidence_level="high",
    )


def _final_out(text: str = "Final answer.") -> roles_mod.FinalAnswerSynthesisOutput:
    return roles_mod.FinalAnswerSynthesisOutput(
        final_answer=text,
        concise_answer=text,
        reasoning="Synthesized from working memory.",
        confidence_level="high",
    )


def _extraction_out(*facts: str) -> roles_mod.ExtractionOutput:
    return roles_mod.ExtractionOutput(relevant_information=list(facts))


_EXTRACTOR_BATCH_SEP = "\n\n---\n\n"


class RoleExecutorSpy:
    """Async stand-in for ``execute_role_lc`` with per-role canned outputs.

    The ``extractor`` branch acts as a passthrough by default: it splits the
    incoming ``raw_data`` on the CoT graph's batch separator and returns the
    pieces as ``relevant_information``. Tests that want a non-trivial extractor
    response can override via ``extractor_outputs``.
    """

    def __init__(
        self,
        *,
        subq_outputs: Sequence[roles_mod.SubquestionGenerationOutput],
        answers: Sequence[str] = (),
        final_answer: str = "Final answer.",
        extractor_outputs: Sequence[Sequence[str]] | None = None,
    ) -> None:
        self.calls: List[Dict[str, Any]] = []
        self._subq_outputs = list(subq_outputs)
        self._answers = list(answers)
        self._final_answer = final_answer
        self._extractor_outputs = (
            [list(facts) for facts in extractor_outputs]
            if extractor_outputs is not None
            else None
        )

    async def __call__(
        self,
        registry: Any,
        role: roles_mod.Role,
        input_data: Any,
        n: int = 1,
        tier_override: str | None = None,
    ) -> tuple[Any, Dict[str, Any]]:
        self.calls.append(
            {
                "role": role.name,
                "input": input_data,
                "n": n,
                "tier_override": tier_override,
            }
        )

        if role.name == "subquestion_generator":
            if not self._subq_outputs:
                return _subq_out(answerable=True), {}
            return self._subq_outputs.pop(0), {}

        if role.name == "answer_generator":
            inputs = input_data if isinstance(input_data, list) else [input_data]
            outputs = []
            for i, _item in enumerate(inputs):
                text = self._answers.pop(0) if self._answers else f"Subanswer {i + 1}"
                outputs.append(_answer_out(text))
            return (outputs if isinstance(input_data, list) else outputs[0]), {}

        if role.name == "final_answer_synthesizer":
            return _final_out(self._final_answer), {}

        if role.name == "extractor":
            if self._extractor_outputs is not None:
                facts = (
                    self._extractor_outputs.pop(0) if self._extractor_outputs else []
                )
                return _extraction_out(*facts), {}
            raw = str(getattr(input_data, "raw_data", "") or "")
            pieces = [
                chunk.strip()
                for chunk in raw.split(_EXTRACTOR_BATCH_SEP)
                if chunk.strip()
            ]
            return _extraction_out(*pieces), {}

        raise AssertionError(f"Unexpected role call: {role.name}")

    def role_inputs(self, role_name: str) -> List[Any]:
        return [c["input"] for c in self.calls if c["role"] == role_name]


class CompiledGraphSpy:
    """Small compiled-graph-like object recording ``ainvoke`` payloads."""

    def __init__(self, output: Dict[str, Any]) -> None:
        self.output = output
        self.calls: List[Dict[str, Any]] = []

    async def ainvoke(
        self, state: Dict[str, Any], *args: Any, **kwargs: Any
    ) -> Dict[str, Any]:
        self.calls.append(state)
        return dict(self.output)


class CorpusSearchSpy:
    name = "corpus_search"

    def __init__(self, results: Sequence[str]) -> None:
        self.results = list(results)
        self.invocations: List[Any] = []

    async def ainvoke(self, inp: Any, *args: Any, **kwargs: Any) -> List[str]:
        self.invocations.append(inp)
        return list(self.results)


def _state(
    *,
    max_depth: int = 1,
    depth: int = 0,
    text_memory: Sequence[str] = ("Existing text memory.",),
    graph_memory: nx.DiGraph | None = None,
    entity_dict: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    return {
        "question": "What is the capital of France?",
        "max_depth": max_depth,
        "depth": depth,
        "is_answerable": False,
        "subquestions": [],
        "retrieved_raw_context": [],
        "retrieved_raw_triples": [],
        "reranked_context": [],
        "current_subanswers": [],
        "iteration_history": [],
        "text_memory": list(text_memory),
        "graph_memory": graph_memory or nx.DiGraph(),
        "entity_dict": dict(entity_dict or {}),
        "final_answer": "",
    }


def _input_text(value: Any) -> str:
    if isinstance(value, list):
        return "\n".join(_input_text(v) for v in value)
    if hasattr(value, "context"):
        return str(value.context)
    if hasattr(value, "question"):
        return str(value.question)
    if hasattr(value, "input_text"):
        return str(value.input_text)
    if hasattr(value, "candidate_answers"):
        return "\n".join(map(str, value.candidate_answers))
    return str(value)


def _install_graph_spies(
    monkeypatch: pytest.MonkeyPatch,
    cot_mod: Any,
    *,
    executor: RoleExecutorSpy,
    kg_graph: CompiledGraphSpy | None = None,
    web_graph: CompiledGraphSpy | None = None,
    memory_graph: CompiledGraphSpy | None = None,
    corpus_search: CorpusSearchSpy | None = None,
) -> tuple[CompiledGraphSpy, CompiledGraphSpy, CompiledGraphSpy, CorpusSearchSpy]:
    kg_graph = kg_graph or CompiledGraphSpy(
        {"kg_articles": ["KG article"], "triples": ["France | capital | Paris"]}
    )
    web_graph = web_graph or CompiledGraphSpy(
        {
            "results": [
                {
                    "title": "Web",
                    "url": "https://example.com",
                    "snippet": "Web ctx",
                    "full_text": "",
                }
            ]
        }
    )
    memory_graph = memory_graph or CompiledGraphSpy(
        {
            "updated_text_memory": ["Updated memory"],
            "updated_graph": nx.DiGraph(),
            "updated_entity_dict": {"Q142": "France"},
        }
    )
    corpus_search = corpus_search or CorpusSearchSpy(["Corpus ctx"])

    monkeypatch.setattr(cot_mod, "execute_role_lc", executor, raising=False)
    monkeypatch.setattr(
        cot_mod, "build_kg_search_graph", lambda *_a, **_kw: kg_graph, raising=False
    )
    monkeypatch.setattr(
        cot_mod, "build_web_research_graph", lambda *_a, **_kw: web_graph, raising=False
    )
    monkeypatch.setattr(
        cot_mod,
        "build_memory_update_graph",
        lambda *_a, **_kw: memory_graph,
        raising=False,
    )
    monkeypatch.setattr(cot_mod, "corpus_search", corpus_search, raising=False)
    return kg_graph, web_graph, memory_graph, corpus_search


def _build_graph(cot_mod: Any, cfg: Any | None = None):
    cfg = cfg if cfg is not None else _config()
    try:
        return cot_mod.build_cot_graph(_registry(), cfg)
    except TypeError:
        return cot_mod.build_cot_graph(_registry())


def test_clear_reducer_uses_typed_sentinel_not_string():
    """§7.1 requires a ``Clear`` sentinel and ``append_or_clear`` reducer."""
    cot_mod = _import_module()
    assert hasattr(cot_mod, "Clear"), "CoTGraph must export Clear (§7.1)"
    assert hasattr(cot_mod, "append_or_clear"), (
        "CoTGraph must export append_or_clear (§7.1)"
    )

    assert cot_mod.append_or_clear(["a"], ["b"]) == ["a", "b"]
    assert cot_mod.append_or_clear(None, ["b"]) == ["b"]
    assert cot_mod.append_or_clear(["a"], cot_mod.Clear()) == []
    assert cot_mod.append_or_clear(["a"], ["CLEAR"]) == ["a", "CLEAR"], (
        "Clear must be a typed sentinel; string 'CLEAR' is data, not control"
    )


async def test_answerable_route_skips_retrieval_and_memory_update(monkeypatch):
    """When ``gen_subq`` marks the question answerable, route directly to final synthesis."""
    cot_mod = _import_module()
    executor = RoleExecutorSpy(
        subq_outputs=[_subq_out(answerable=True)],
        final_answer="Paris.",
    )
    kg_graph, web_graph, memory_graph, corpus = _install_graph_spies(
        monkeypatch,
        cot_mod,
        executor=executor,
    )

    graph = _build_graph(cot_mod)
    final = await graph.ainvoke(_state(max_depth=3))

    assert final["final_answer"] == "Paris."
    assert executor.role_inputs("final_answer_synthesizer"), (
        "answerable route must invoke final_answer_synthesizer"
    )
    assert not kg_graph.calls
    assert not web_graph.calls
    assert not memory_graph.calls
    assert not corpus.invocations


async def test_gen_subq_uses_text_and_graph_memory_context(monkeypatch):
    """``gen_subq`` should reason over text memory plus textualized graph memory (§7.3)."""
    cot_mod = _import_module()
    graph_memory = nx.DiGraph()
    graph_memory.add_edge("France", "Paris", relation="capital")
    executor = RoleExecutorSpy(
        subq_outputs=[_subq_out(answerable=True)],
        final_answer="Paris.",
    )
    _install_graph_spies(monkeypatch, cot_mod, executor=executor)

    graph = _build_graph(cot_mod)
    await graph.ainvoke(
        _state(
            graph_memory=graph_memory,
            text_memory=["France is a country in Europe."],
        )
    )

    subq_input = executor.role_inputs("subquestion_generator")[0]
    subq_text = _input_text(subq_input)
    assert "France is a country in Europe." in subq_text
    assert "France" in subq_text and "Paris" in subq_text and "capital" in subq_text


async def test_retrieval_fanout_invokes_kg_web_and_corpus_per_subquestion(monkeypatch):
    """§7.3 fan-out is symmetric across *active* surfaces: one invocation per subq.

    The earlier draft routed corpus through a ``query_generator`` LLM call that
    rewrote subquestions into queries. That step turned out to be largely
    duplicative of what ``subquestion_generator`` already enforces (atomic,
    self-contained, rank-aware, retrieval-ready); dropping it saves one LLM
    call per iteration with no measurable recall loss on a phrasing-robust
    dense embedder. Corpus now fans out per subquestion like KG / web.

    Web is gated off by default (§1a, paper parity), so this test enables it and
    tags both subquestions ``needs_kg=True`` to exercise the full three-surface
    symmetric fan-out.
    """
    cot_mod = _import_module()
    subquestions = ["Where is France?", "What is France's capital?"]
    executor = RoleExecutorSpy(
        subq_outputs=[
            _subq_out(
                answerable=False, subquestions=subquestions, needs_kg=[True, True]
            ),
            _subq_out(answerable=True),
        ],
        answers=["France is in Europe.", "Paris is the capital."],
        final_answer="Paris.",
    )
    kg_graph, web_graph, memory_graph, corpus = _install_graph_spies(
        monkeypatch,
        cot_mod,
        executor=executor,
    )

    graph = _build_graph(cot_mod, _config(web_enabled=True))
    final = await graph.ainvoke(_state(max_depth=2))

    assert len(kg_graph.calls) == len(subquestions)
    assert len(web_graph.calls) == len(subquestions)
    assert len(corpus.invocations) == len(subquestions), (
        "corpus_join must issue one corpus_search per subquestion"
    )
    invoked_corpus_queries = [str(inv) for inv in corpus.invocations]
    for sq in subquestions:
        assert any(sq in inv for inv in invoked_corpus_queries), (
            f"subquestion {sq!r} never reached corpus_search; saw {invoked_corpus_queries!r}"
        )
    assert [c.get("subquery") for c in kg_graph.calls] == subquestions
    assert [c.get("subquery") for c in web_graph.calls] == subquestions
    assert memory_graph.calls, "non-answerable iteration must invoke MemoryUpdateGraph"
    assert final["final_answer"] == "Paris."


async def test_web_fanout_disabled_by_default(monkeypatch):
    """§1a: web fan-out is gated off by default (paper parity).

    With the default config (``web_search.enabled is False``), the web branch
    never fires, while KG (default-tagged) and corpus still fan out per subq.
    """
    cot_mod = _import_module()
    subquestions = ["Where is France?", "What is France's capital?"]
    executor = RoleExecutorSpy(
        subq_outputs=[
            _subq_out(
                answerable=False, subquestions=subquestions
            ),  # needs_kg absent → default KG-on
            _subq_out(answerable=True),
        ],
        answers=["France is in Europe.", "Paris is the capital."],
        final_answer="Paris.",
    )
    kg_graph, web_graph, _memory_graph, corpus = _install_graph_spies(
        monkeypatch, cot_mod, executor=executor
    )

    graph = _build_graph(cot_mod)  # default config: web disabled
    final = await graph.ainvoke(_state(max_depth=2))

    assert web_graph.calls == [], (
        "web fan-out must be skipped when web_search.enabled is False"
    )
    # Absent needs_kg tags default to KG-on, so KG + corpus still fan out fully.
    assert len(kg_graph.calls) == len(subquestions)
    assert len(corpus.invocations) == len(subquestions)
    assert final["final_answer"] == "Paris."


async def test_needs_kg_false_skips_kg_for_that_subquestion(monkeypatch):
    """§1a: a subquestion tagged ``needs_kg=False`` skips the KG branch.

    Corpus still covers it (the unconditional recall floor), so no evidence
    surface is fully starved.
    """
    cot_mod = _import_module()
    subquestions = [
        "What is the capital of France?",
        "What is the mechanism of photosynthesis?",
    ]
    executor = RoleExecutorSpy(
        subq_outputs=[
            _subq_out(
                answerable=False, subquestions=subquestions, needs_kg=[True, False]
            ),
            _subq_out(answerable=True),
        ],
        answers=["Paris.", "Light-driven carbon fixation."],
        final_answer="Done.",
    )
    kg_graph, web_graph, _memory_graph, corpus = _install_graph_spies(
        monkeypatch, cot_mod, executor=executor
    )

    graph = _build_graph(cot_mod)
    await graph.ainvoke(_state(max_depth=2))

    kg_queries = [c.get("subquery") for c in kg_graph.calls]
    assert kg_queries == ["What is the capital of France?"], (
        f"KG must fire only for the entity-centric subquestion; saw {kg_queries!r}"
    )
    assert web_graph.calls == []
    # The non-KG subquestion still reaches corpus.
    assert len(corpus.invocations) == len(subquestions)


async def test_known_entity_overrides_needs_kg_false(monkeypatch):
    """§1a override: a subquestion mentioning an already-linked entity fires KG.

    Even when the generator tags ``needs_kg=False``, holding a resolved QID for
    an entity named in the subquestion forces the KG branch on (highest-yield
    multi-hop case).
    """
    cot_mod = _import_module()
    subquestions = ["Where is France located?"]
    executor = RoleExecutorSpy(
        subq_outputs=[
            _subq_out(answerable=False, subquestions=subquestions, needs_kg=[False]),
            _subq_out(answerable=True),
        ],
        answers=["Western Europe."],
        final_answer="Done.",
    )
    kg_graph, web_graph, _memory_graph, _corpus = _install_graph_spies(
        monkeypatch, cot_mod, executor=executor
    )

    entity_dict = {
        "Q142": roles_mod.WikidataEntity(
            qid="Q142", label="France", description="country"
        ),
    }
    graph = _build_graph(cot_mod)
    await graph.ainvoke(_state(max_depth=2, entity_dict=entity_dict))

    kg_queries = [c.get("subquery") for c in kg_graph.calls]
    assert kg_queries == ["Where is France located?"], (
        f"known-entity override must force KG on despite needs_kg=False; saw {kg_queries!r}"
    )
    assert web_graph.calls == []


async def test_answer_generation_receives_reranked_joined_context(monkeypatch):
    """All retrieval branches append into context, then answer generation uses reranked context."""
    cot_mod = _import_module()
    kg_graph = CompiledGraphSpy(
        {"kg_articles": ["KG article ctx"], "triples": ["France | capital | Paris"]}
    )
    web_graph = CompiledGraphSpy(
        {
            "results": [
                {
                    "title": "Web",
                    "url": "u",
                    "snippet": "Web snippet ctx",
                    "full_text": "Web full ctx",
                }
            ]
        }
    )
    corpus = CorpusSearchSpy(["Corpus ctx A", "Corpus ctx B"])
    executor = RoleExecutorSpy(
        subq_outputs=[
            _subq_out(answerable=False, subquestions=["What is France's capital?"]),
            _subq_out(answerable=True),
        ],
        answers=["Paris is the capital."],
    )
    _install_graph_spies(
        monkeypatch,
        cot_mod,
        executor=executor,
        kg_graph=kg_graph,
        web_graph=web_graph,
        corpus_search=corpus,
    )

    async def _rerank(
        query: Any, contexts: Sequence[str], top_k: int | None = None, **_kw: Any
    ):
        assert any("KG article ctx" in c for c in contexts)
        assert any("Web snippet ctx" in c or "Web full ctx" in c for c in contexts)
        assert any("Corpus ctx A" in c for c in contexts)
        return ["KG article ctx", "Corpus ctx A"]

    monkeypatch.setattr(cot_mod, "rerank_context", _rerank, raising=False)
    monkeypatch.setattr(cot_mod, "_rerank_context", _rerank, raising=False)

    # Web enabled so all three retrieval branches contribute context here.
    graph = _build_graph(cot_mod, _config(web_enabled=True))
    await graph.ainvoke(_state(max_depth=2))

    answer_inputs = executor.role_inputs("answer_generator")
    assert len(answer_inputs) == 1
    answer_text = _input_text(answer_inputs[0])
    assert "KG article ctx" in answer_text
    assert "Corpus ctx A" in answer_text
    assert "Corpus ctx B" not in answer_text, (
        "answer_generator must receive the reranked/top-k context, not the raw full list"
    )
    # NB: ``reranked_context`` is per-iteration scratch cleared by ``increment``
    # (see ``test_increment_records_iteration_history_and_clears_scratch``); the
    # behaviour we care about — the rerank output flowed into ``answer_generator``
    # — is verified above.


async def test_rerank_context_uses_sglang_reranker_when_enabled(monkeypatch):
    """Enabled reranker config should call SGLang and apply returned ranking."""
    cot_mod = _import_module()

    from langgraph_coe.config import RerankerConfig

    cfg = RerankerConfig(enabled=True, url="http://reranker.local/v1", top_k=2)
    calls: List[Dict[str, Any]] = []

    async def _call_sglang(query: str, texts: Sequence[str], got_cfg: Any):
        calls.append({"query": query, "texts": list(texts), "cfg": got_cfg})
        return [(2, 0.91), (0, 0.42), (1, 0.05)]

    monkeypatch.setattr(cot_mod, "call_sglang_reranker", _call_sglang, raising=False)

    ranked = await cot_mod.rerank_context(
        "capital of France",
        ["alpha", "", "beta", "gamma"],
        top_k=2,
        cfg=cfg,
    )

    assert ranked == ["gamma", "alpha"]
    assert calls == [
        {
            "query": "capital of France",
            "texts": ["alpha", "beta", "gamma"],
            "cfg": cfg,
        }
    ]


async def test_rerank_node_forwards_configured_reranker(monkeypatch):
    """CoTGraph rerank node must pass builder config into ``rerank_context``."""
    cot_mod = _import_module()
    executor = RoleExecutorSpy(
        subq_outputs=[
            _subq_out(answerable=False, subquestions=["What is France's capital?"]),
            _subq_out(answerable=True),
        ],
        answers=["Paris is the capital."],
    )
    _install_graph_spies(monkeypatch, cot_mod, executor=executor)

    from langgraph_coe.config import LangGraphCoeConfig
    from langgraph_coe.llm import RoleModelRegistry

    cfg = LangGraphCoeConfig.from_yaml()
    cfg.reranker.enabled = True
    cfg.reranker.top_k = 2
    registry = RoleModelRegistry(cfg.llm)
    registry.get_model = lambda _role_name: MagicMock()  # type: ignore[assignment]

    seen: List[Dict[str, Any]] = []

    async def _rerank(
        query: str,
        contexts: Sequence[str],
        top_k: int | None = None,
        cfg: Any = None,
        **_kw: Any,
    ):
        seen.append(
            {
                "query": query,
                "contexts": list(contexts),
                "top_k": top_k,
                "cfg": cfg,
            }
        )
        return list(contexts)[:1]

    monkeypatch.setattr(cot_mod, "rerank_context", _rerank, raising=False)

    graph = cot_mod.build_cot_graph(registry, cfg)
    await graph.ainvoke(_state(max_depth=2))

    assert seen, "rerank node did not call rerank_context"
    assert seen[0]["cfg"] is cfg.reranker
    assert seen[0]["top_k"] == 2
    assert "What is France's capital?" in seen[0]["query"]


async def test_memory_update_receives_subanswers_and_kg_triples(monkeypatch):
    """``mem_update`` maps subanswers/triples into ``MemoryUpdateState`` (§7.3)."""
    cot_mod = _import_module()
    memory_graph = CompiledGraphSpy(
        {
            "updated_text_memory": ["Memory after subanswers"],
            "updated_graph": nx.DiGraph(),
            "updated_entity_dict": {"Q90": "Paris"},
        }
    )
    executor = RoleExecutorSpy(
        subq_outputs=[
            _subq_out(answerable=False, subquestions=["What is France's capital?"]),
            _subq_out(answerable=True),
        ],
        answers=["Paris is the capital of France."],
        final_answer="Paris.",
    )
    _install_graph_spies(
        monkeypatch,
        cot_mod,
        executor=executor,
        memory_graph=memory_graph,
    )

    graph = _build_graph(cot_mod)
    final = await graph.ainvoke(_state(max_depth=2))

    assert len(memory_graph.calls) == 1
    payload = memory_graph.calls[0]
    assert payload["new_text_items"] == ["Paris is the capital of France."]
    assert payload["new_raw_triples"] == ["France | capital | Paris"]
    assert final["text_memory"] == ["Memory after subanswers"]
    assert final["entity_dict"] == {"Q90": "Paris"}


async def test_increment_records_iteration_history_and_clears_scratch(monkeypatch):
    """``increment`` records trajectory before clearing per-iteration scratch (§7.3)."""
    cot_mod = _import_module()
    executor = RoleExecutorSpy(
        subq_outputs=[
            _subq_out(answerable=False, subquestions=["What is France's capital?"]),
            _subq_out(answerable=True),
        ],
        answers=["Paris is the capital."],
    )
    _install_graph_spies(monkeypatch, cot_mod, executor=executor)

    graph = _build_graph(cot_mod)
    final = await graph.ainvoke(_state(max_depth=2))

    assert final["depth"] == 1
    assert final["iteration_history"] == [
        {
            "depth": 0,
            "subquestions": ["What is France's capital?"],
            "subanswers": ["Paris is the capital."],
        }
    ]
    assert final["subquestions"] == []
    assert final["retrieved_raw_context"] == []
    assert final["retrieved_raw_triples"] == []
    assert final["reranked_context"] == []
    assert final["extracted_facts"] == []
    assert final["current_subanswers"] == []


async def test_depth_limit_routes_to_final_without_retrieval(monkeypatch):
    """``depth >= max_depth`` routes to final synthesis even if subquestions exist."""
    cot_mod = _import_module()
    executor = RoleExecutorSpy(
        subq_outputs=[
            _subq_out(answerable=False, subquestions=["Unasked subquestion"])
        ],
        final_answer="Depth-limited final answer.",
    )
    kg_graph, web_graph, memory_graph, corpus = _install_graph_spies(
        monkeypatch,
        cot_mod,
        executor=executor,
    )

    graph = _build_graph(cot_mod)
    final = await graph.ainvoke(_state(max_depth=0, depth=0))

    assert final["final_answer"] == "Depth-limited final answer."
    assert not kg_graph.calls
    assert not web_graph.calls
    assert not memory_graph.calls
    assert not corpus.invocations


# ──────────────────────────────────────────────────────────────────────────────
# §7.3 EXTRACTOR step — distills reranked passages into atomic facts
# ──────────────────────────────────────────────────────────────────────────────


async def test_extractor_runs_between_rerank_and_subanswers(monkeypatch):
    """The reranked top-k flows through EXTRACTOR before reaching answer_generator.

    answer_generator must see the extractor's atomic facts, not the raw
    reranked passages. The extractor is the self-containment layer: anaphora
    resolution + per-claim atomization for downstream subanswer grounding.
    """
    cot_mod = _import_module()
    extracted_facts = [
        "Paris is the capital of France.",
        "The Eiffel Tower is located in Paris.",
    ]
    executor = RoleExecutorSpy(
        subq_outputs=[
            _subq_out(answerable=False, subquestions=["What is France's capital?"]),
            _subq_out(answerable=True),
        ],
        answers=["Paris is the capital."],
        extractor_outputs=[extracted_facts],
    )
    _install_graph_spies(monkeypatch, cot_mod, executor=executor)

    graph = _build_graph(cot_mod)
    await graph.ainvoke(_state(max_depth=2))

    extractor_calls = [c for c in executor.calls if c["role"] == "extractor"]
    assert extractor_calls, (
        "extractor must be invoked between rerank and gen_subanswers"
    )

    # The extractor's input includes the original question + current subquestions
    # so its relevance lens covers both the global intent and the iteration's gaps.
    ext_input = extractor_calls[0]["input"]
    assert "What is France's capital?" in str(ext_input.question), (
        "extractor must see the current subquestion(s) as part of the relevance lens"
    )

    answer_inputs = executor.role_inputs("answer_generator")
    assert len(answer_inputs) == 1
    answer_text = _input_text(answer_inputs[0])
    for fact in extracted_facts:
        assert fact in answer_text, (
            f"answer_generator missed extracted fact {fact!r}; saw {answer_text!r}"
        )


async def test_extractor_splits_oversized_contexts_into_multiple_batches(monkeypatch):
    """When joined contexts exceed ``memory.extractor_max_input_chars``, the
    extractor receives multiple batched calls (parallel), and its outputs are
    merged + deduped before reaching ``answer_generator``.
    """
    cot_mod = _import_module()

    # Two huge contexts forcing a 2-batch split.
    big_ctx_a = "A" * 5_000 + " — fact alpha about France"
    big_ctx_b = "B" * 5_000 + " — fact beta about Paris"

    executor = RoleExecutorSpy(
        subq_outputs=[
            _subq_out(answerable=False, subquestions=["What is France's capital?"]),
            _subq_out(answerable=True),
        ],
        answers=["Paris is the capital."],
        extractor_outputs=[
            ["Fact alpha: France is in Europe."],
            ["Fact beta: Paris is the capital of France."],
        ],
    )
    kg_graph = CompiledGraphSpy({"kg_articles": [big_ctx_a, big_ctx_b], "triples": []})
    _install_graph_spies(
        monkeypatch,
        cot_mod,
        executor=executor,
        kg_graph=kg_graph,
    )

    # Tighten the char budget so the two passages must split across batches.
    from langgraph_coe.config import LangGraphCoeConfig

    cfg = LangGraphCoeConfig.from_yaml()
    cfg.reranker.enabled = False
    cfg.reranker.top_k = 5
    cfg.memory.extractor_max_input_chars = 6_000  # forces 1 ctx per batch

    from langgraph_coe.llm import RoleModelRegistry
    from unittest.mock import MagicMock

    registry = RoleModelRegistry(cfg.llm)
    registry.get_model = lambda _role_name: MagicMock()  # type: ignore[assignment]
    graph = cot_mod.build_cot_graph(registry, cfg)

    await graph.ainvoke(_state(max_depth=2))

    extractor_calls = [c for c in executor.calls if c["role"] == "extractor"]
    assert len(extractor_calls) == 2, (
        f"oversized contexts must split into >1 extractor batch; saw {len(extractor_calls)}"
    )

    # Each batch input must respect the char budget.
    for call in extractor_calls:
        raw_data = str(getattr(call["input"], "raw_data", "") or "")
        assert len(raw_data) <= 6_000, (
            f"extractor batch exceeded max_input_chars: len={len(raw_data)}"
        )

    # Both batches' facts must flow through to answer_generator (merged + deduped).
    answer_inputs = executor.role_inputs("answer_generator")
    answer_text = _input_text(answer_inputs[0])
    assert "Fact alpha: France is in Europe." in answer_text
    assert "Fact beta: Paris is the capital of France." in answer_text


async def test_extractor_falls_back_to_reranked_when_empty(monkeypatch):
    """If EXTRACTOR returns no facts, the reranked passages must still reach
    ``answer_generator`` — silent evidence loss is unacceptable.
    """
    cot_mod = _import_module()
    executor = RoleExecutorSpy(
        subq_outputs=[
            _subq_out(answerable=False, subquestions=["What is France's capital?"]),
            _subq_out(answerable=True),
        ],
        answers=["Paris is the capital."],
        extractor_outputs=[[]],  # extractor yields nothing
    )
    kg_graph = CompiledGraphSpy(
        {"kg_articles": ["Paris is the capital of France."], "triples": []}
    )
    _install_graph_spies(
        monkeypatch,
        cot_mod,
        executor=executor,
        kg_graph=kg_graph,
    )

    graph = _build_graph(cot_mod)
    await graph.ainvoke(_state(max_depth=2))

    answer_text = _input_text(executor.role_inputs("answer_generator")[0])
    assert "Paris is the capital of France." in answer_text, (
        "answer_generator must still receive the raw reranked passage when "
        "extractor produces no facts"
    )
