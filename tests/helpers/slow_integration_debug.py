"""Pretty-print return values for @pytest.mark.slow_integration tests.

Run with ``pytest -s`` (or ``--capture=no``) to see stdout.
"""

from __future__ import annotations

import dataclasses
from pprint import pprint
from typing import Any

from pydantic import BaseModel


def _format_for_print(obj: Any) -> str:
    """Prefer ``str(obj)`` so types with a readable ``__str__`` stay compact."""
    return str(obj)


def to_debug_dict(obj: Any) -> Any:
    """Recursively convert payloads for printing (compact where models define ``__str__``)."""
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if isinstance(obj, BaseModel):
        # model_dump() can be huge (e.g. WikidataEntity.wikipedia_content); str() uses concise __str__.
        return _format_for_print(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {
            f.name: to_debug_dict(getattr(obj, f.name))
            for f in dataclasses.fields(obj)
        }
    if isinstance(obj, dict):
        return {str(k): to_debug_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_debug_dict(x) for x in obj]
    return _format_for_print(obj)


def _format_reasoning_tree(root: Any) -> str:
    if root is None:
        return ""
    try:
        from anytree import RenderTree

        from wemg.reasoning.nodes import ReasoningNode
    except ImportError:
        return _format_for_print(root)
    if not isinstance(root, ReasoningNode):
        return _format_for_print(root)
    r = root
    while getattr(r, "parent", None) is not None:
        r = r.parent
    lines: list[str] = []
    for pre, _, node in RenderTree(r):
        lines.append(f"{pre}{node._stable_id()} {node._summary()}")
    return "\n".join(lines)


def working_memory_snapshot(wm: Any) -> Any:
    if wm is None:
        return None
    try:
        import networkx as nx

        combined_entity_dict = dict(wm.entity_dict)

        entity_graph = getattr(wm, "_entity_graph", None)
        eg_nodes = entity_graph.number_of_nodes() if entity_graph is not None else 0
        eg_edges = entity_graph.number_of_edges() if entity_graph is not None else 0

        return {
            "textual_memory": wm.format_textual_memory(),
            "entity_dict_keys": sorted(combined_entity_dict.keys()),
            "entity_graph_nodes": eg_nodes,
            "entity_graph_edges": eg_edges,
        }
    except Exception as exc:  # noqa: BLE001 — best-effort debug dump only
        return {"error_serializing_working_memory": repr(exc)}


def _pprint_debug_payload(value: Any) -> None:
    """Pretty-print converted payload; expand multiline string lists for readability."""
    conv = to_debug_dict(value)
    if (
        isinstance(conv, list)
        and conv
        and all(isinstance(x, str) for x in conv)
        and any("\n" in x for x in conv)
    ):
        for i, item in enumerate(conv):
            print(f"[{i}]")
            print(item)
        return
    pprint(conv, width=120, sort_dicts=False)


def answer_result_snapshot(ar: Any) -> dict[str, Any]:
    from wemg.system import AnswerResult

    if not isinstance(ar, AnswerResult):
        return to_debug_dict(ar)
    return {
        "question": ar.question,
        "answer": ar.answer,
        "concise_answer": ar.concise_answer,
        "reasoning": ar.reasoning,
        "metadata": dict(ar.metadata or {}),
        "search_tree_render": _format_reasoning_tree(ar.search_tree),
        "working_memory": working_memory_snapshot(ar.working_memory),
    }


def _print_section(name: str, value: Any) -> None:
    print(f"\n--- {name} ---")
    if isinstance(value, str):
        print(value)
        return
    if (
        isinstance(value, dict)
        and "search_tree_render" in value
        and isinstance(value["search_tree_render"], str)
    ):
        rest = {k: v for k, v in value.items() if k != "search_tree_render"}
        _pprint_debug_payload(rest)
        print("\n(search_tree_render)")
        print(value["search_tree_render"])
        return
    _pprint_debug_payload(value)


def print_slow_integration_output(title: str, **sections: Any) -> None:
    """Print labeled sections for manual inspection (use ``pytest -s``)."""
    from wemg.reasoning.working_memory import WorkingMemory
    from wemg.system import AnswerResult

    bar = "=" * 72
    print(f"\n{bar}\n[slow_integration] {title}\n{bar}")
    for name, value in sections.items():
        if isinstance(value, AnswerResult):
            _print_section(name, answer_result_snapshot(value))
        elif (
            isinstance(value, list)
            and value
            and all(isinstance(x, AnswerResult) for x in value)
        ):
            _print_section(name, [answer_result_snapshot(x) for x in value])
        elif isinstance(value, WorkingMemory):
            _print_section(name, working_memory_snapshot(value))
        else:
            _print_section(name, value)
    print(bar)
