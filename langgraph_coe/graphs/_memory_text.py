"""Shared rendering helpers for the reasoning graphs' working memory.

``cot.py`` and ``mcts.py`` both turn the in-state ``graph_memory`` (an
``nx.DiGraph``) into a flat triple listing for prompt context. The single
implementation lives here so the two graphs cannot drift apart.
"""

from __future__ import annotations

from typing import List, Optional

import networkx as nx


def textualize_graph(graph: Optional[nx.DiGraph]) -> str:
    """Render a memory ``DiGraph`` as newline-joined ``subj — rel — obj`` lines.

    Multi-valued ``relation`` edge data (set/list/tuple) expands to one line per
    relation; an edge with no relation falls back to ``related_to``. Returns the
    empty string for an empty or ``None`` graph.
    """
    if graph is None or graph.number_of_edges() == 0:
        return ""
    lines: List[str] = []
    for u, v, data in graph.edges(data=True):
        rel = data.get("relation")
        if isinstance(rel, (set, list, tuple)):
            for r in rel:
                lines.append(f"{u} — {r} — {v}")
        elif rel:
            lines.append(f"{u} — {rel} — {v}")
        else:
            lines.append(f"{u} — related_to — {v}")
    return "\n".join(lines)
