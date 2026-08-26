"""Tests for the Stage-A lexical prefilter and batched Stage-B pruning."""
import asyncio
import pytest
from langgraph_coe.tools import wikidata as wd


class T:
    """Minimal triple stand-in: _lexical_prefilter only ever calls str() on it."""
    def __init__(self, s): self.s = s
    def __str__(self): return self.s
    def __hash__(self): return hash(self.s)
    def __eq__(self, o): return isinstance(o, T) and o.s == self.s


def test_top_k_is_honoured_without_a_reranker():
    """The defect: `pruning_top_k` was dead in the configuration the project runs in.

    `_stage_a_prune` opened with `if not triples or not reranker_url: return triples`, so
    with `reranker_url: null` every triple reached Stage B — which charges one LLM call per
    16. Measured on a 12-row run: 97.7 `triple_pruner` calls per question, **82% of all LLM
    completions**, against a configured `pruning_top_k` of 64 that allows 4.
    """
    triples = [T(f"subject{i} -- relation -- object{i}") for i in range(200)]
    kept = asyncio.run(wd._stage_a_prune("q", triples, None, None, top_k=64))
    assert len(kept) == 64


def test_the_prefilter_ranks_by_overlap_rather_than_truncating():
    """Ordering is what makes the cap safe.

    Arbitrary truncation drops the answer at random. The relevant triple here sits at the
    END of the list, so a `[:top_k]` slice would lose it.
    """
    noise = [T(f"unrelated{i} -- foo -- bar{i}") for i in range(50)]
    needle = T("Danyang-Kunshan Grand Bridge -- opening date -- 2011")
    kept = asyncio.run(
        wd._stage_a_prune("When did the Danyang-Kunshan Grand Bridge open?",
                          noise + [needle], None, None, top_k=5)
    )
    assert needle in kept, "the overlapping triple must survive the cap"


def test_the_prefilter_is_a_no_op_when_it_need_not_act():
    triples = [T(f"a{i} -- r -- b{i}") for i in range(10)]
    assert asyncio.run(wd._stage_a_prune("q", triples, None, None, top_k=64)) == triples
    # A query of pure stopwords has no content words to rank by; still capped, not dropped.
    many = [T(f"a{i} -- r -- b{i}") for i in range(100)]
    assert len(asyncio.run(wd._stage_a_prune("what is the", many, None, None, top_k=7))) == 7


def test_a_reranker_outage_falls_back_rather_than_passing_everything_through():
    """A transient outage must not multiply a question's LLM bill by an order of magnitude.

    The old exception path returned every triple, so losing the reranker silently moved the
    run into the same 97-calls-per-question regime as having none configured.
    """
    triples = [T(f"a{i} -- r -- b{i}") for i in range(200)]
    kept = asyncio.run(
        wd._stage_a_prune("q", triples, "http://127.0.0.1:9/unreachable", "m", top_k=64)
    )
    assert len(kept) == 64


def test_stage_b_prunes_all_chunks_in_one_gathered_call():
    """Stage B looped a chunk at a time, making pruning strictly serial.

    Same token count either way, but N sequential round-trips instead of one gathered
    batch. `memory_update` already used the batched form for this identical role.
    """
    calls = []

    class Out:
        def __init__(self, keep): self.keep_indices = keep

    async def fake_execute(registry, role, inp, **kw):
        calls.append(inp)
        assert isinstance(inp, list), "all chunks must go in one call"
        return [Out([0]) for _ in inp], {}

    import langgraph_coe.llm as llm_mod
    orig = llm_mod.execute_role_lc
    llm_mod.execute_role_lc = fake_execute
    try:
        triples = [T(f"a{i} -- r -- b{i}") for i in range(48)]  # 3 chunks of 16
        kept = asyncio.run(wd._stage_b_prune("q", triples, registry=object()))
    finally:
        llm_mod.execute_role_lc = orig
    assert len(calls) == 1, f"expected one gathered call, got {len(calls)}"
    assert len(calls[0]) == 3, "one input per chunk"
    assert len(kept) == 3, "index 0 kept from each chunk"
