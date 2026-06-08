"""Contract tests for rate limiting and 429/Retry-After handling.

Uses the ``mock_monotonic`` fixture to virtualize time:
  - ``time.monotonic`` returns a deterministic counter
  - ``asyncio.sleep`` instantly advances that counter
This lets us assert exact wait intervals without real sleeping.
"""

from __future__ import annotations

import pytest

from ._fixtures import (
    QID_BERLIN,
    QID_FRANCE,
    QID_GERMANY,
    QID_PARIS,
)
from .contracts import WikidataRateLimitError


# ---------------- min-interval RPS ----------------


async def test_backend_calls_respect_min_interval_at_configured_rps(
    slow_client, mock_monotonic
):
    """At 2 RPS, two sequential SPARQL calls are spaced by at least 0.5s."""
    await slow_client.get_k_hop_triples(
        QID_BERLIN, k=1, bidirectional=False, enrich=False
    )
    t1 = mock_monotonic.now()
    await slow_client.get_k_hop_triples(
        QID_GERMANY, k=1, bidirectional=False, enrich=False
    )
    t2 = mock_monotonic.now()
    assert (t2 - t1) >= 0.5 - 1e-6, (
        f"second SPARQL call only {t2 - t1:.4f}s after first (min 0.5s at 2 RPS)"
    )


async def test_sparql_and_wikipedia_rates_independent(mini_graph, mock_monotonic):
    """SPARQL rate limit must not block Wikipedia calls (and vice-versa)."""
    from langgraph_coe.tools.wikidata_client import WikidataClient
    c = WikidataClient(
        backend=mini_graph,
        max_sparql_rps=2.0,
        max_wikipedia_rps=1000.0,
    )
    # Do one SPARQL then one Wikipedia — the Wikipedia call must not pay the
    # SPARQL spacing cost.
    t0 = mock_monotonic.now()
    await c.get_k_hop_triples(
        QID_BERLIN, k=1, bidirectional=False, enrich=False
    )
    t1 = mock_monotonic.now()
    await c.enrich_entities([QID_BERLIN], get_details=True)
    t2 = mock_monotonic.now()
    # Wikipedia after SPARQL should not require 0.5s wait
    assert (t2 - t1) < 0.5, (
        f"Wikipedia call paid SPARQL spacing ({t2 - t1:.4f}s); rates are not independent"
    )


async def test_burst_then_spaced(slow_client, mock_monotonic):
    """First call after a quiet period is immediate; subsequent calls are spaced."""
    t0 = mock_monotonic.now()
    await slow_client.get_k_hop_triples(
        QID_BERLIN, k=1, bidirectional=False, enrich=False
    )
    t1 = mock_monotonic.now()
    # First call should be (effectively) immediate
    assert (t1 - t0) < 0.5
    await slow_client.get_k_hop_triples(
        QID_GERMANY, k=1, bidirectional=False, enrich=False
    )
    t2 = mock_monotonic.now()
    assert (t2 - t1) >= 0.5 - 1e-6


# ---------------- 429 / Retry-After ----------------


async def test_backend_429_retry_after_respected(client, mini_graph, mock_monotonic):
    """A single 429 with Retry-After=0.25s sleeps exactly 0.25s, then succeeds."""
    mini_graph.inject_error(
        "fetch_outgoing", WikidataRateLimitError(retry_after=0.25)
    )
    result = await client.get_k_hop_triples(
        QID_BERLIN, k=1, bidirectional=False, enrich=False
    )
    assert result, "client should retry and produce triples"
    assert 0.25 in mock_monotonic.sleep_log, (
        f"expected a 0.25s sleep for Retry-After, got {mock_monotonic.sleep_log}"
    )


async def test_backend_429_retry_after_missing_uses_default_backoff(
    client, mini_graph, mock_monotonic
):
    """If Retry-After is missing, the client uses its own bounded backoff."""
    mini_graph.inject_error(
        "fetch_outgoing", WikidataRateLimitError(retry_after=None)
    )
    result = await client.get_k_hop_triples(
        QID_BERLIN, k=1, bidirectional=False, enrich=False
    )
    assert result, "client should retry with default backoff and succeed"
    # Some non-trivial sleep must have happened
    assert any(s > 0 for s in mock_monotonic.sleep_log), (
        "expected a default-backoff sleep when Retry-After is missing"
    )


async def test_backend_429_max_retries_then_raises(client, mini_graph, mock_monotonic):
    """After exhausting retries the client raises the original rate-limit error."""
    mini_graph.inject_error(
        "fetch_outgoing",
        WikidataRateLimitError(retry_after=0.01),
        times=20,
    )
    with pytest.raises(WikidataRateLimitError):
        await client.get_k_hop_triples(
            QID_BERLIN, k=1, bidirectional=False, enrich=False
        )


async def test_wikipedia_429_retry_after_respected(client, mini_graph, mock_monotonic):
    """Wikipedia-side 429 also honors Retry-After."""
    mini_graph.inject_error(
        "get_wikipedia_contents", WikidataRateLimitError(retry_after=0.10)
    )
    result = await client.enrich_entities([QID_BERLIN], get_details=True)
    assert result and result[0].wikipedia_content is not None
    assert 0.10 in mock_monotonic.sleep_log
