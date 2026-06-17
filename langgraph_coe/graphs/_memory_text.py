"""Shared rendering helpers for the reasoning graphs' working memory.

``cot.py`` and ``mcts.py`` both turn the in-state ``graph_memory`` (an
``nx.DiGraph``) into a flat triple listing for prompt context. The single
implementation lives here so the two graphs cannot drift apart.
"""

from __future__ import annotations

from typing import List, Optional

import networkx as nx

_TRIPLE_SEP = " — "


def format_triple_line(subject: str, relation: str, obj: str) -> str:
    """Render one triple as ``subj — rel — obj`` (canonical LGC memory format)."""
    rel = (relation or "").strip() or "related_to"
    return f"{subject}{_TRIPLE_SEP}{rel}{_TRIPLE_SEP}{obj}"


def textualize_graph(graph: Optional[nx.DiGraph]) -> str:
    """Render a memory ``DiGraph`` as newline-joined ``subj — rel — obj`` lines.

    Multi-valued ``relation`` edge data (set/list/tuple) expands to one line per
    relation; an edge with no relation falls back to ``related_to``. Returns the
    empty string for an empty or ``None`` graph.
    """
    if graph is None or graph.number_of_edges() == 0:
        return ""

    def _label(node_id: str) -> str:
        # QID-linked nodes are keyed by QID; prompts need the human-readable
        # ``name`` attribute or the line is opaque to the model.
        name = graph.nodes[node_id].get("name") if node_id in graph else None
        return str(name) if name else str(node_id)

    lines: List[str] = []
    for u, v, data in graph.edges(data=True):
        su, sv = _label(u), _label(v)
        rel = data.get("relation")
        if isinstance(rel, (set, list, tuple)):
            for r in rel:
                lines.append(format_triple_line(su, str(r), sv))
        elif rel:
            lines.append(format_triple_line(su, str(rel), sv))
        else:
            lines.append(format_triple_line(su, "related_to", sv))
    return "\n".join(lines)
