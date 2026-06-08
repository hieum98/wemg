"""Freebase subgraph preprocessing script.

Converts per-question Freebase subgraphs from rmanluo/RoG-{webqsp|cwq} into a
split cache directory consumed by GraphRetrieverClient at evaluation time.

Usage (with COEConfig-style CLI overrides):
    conda run -n coe python -m coe.retrieval.freebase_preprocess \
        dataset=webqsp_sub \
        output=./cache/webqsp_sub_graph_cache

Optional overrides mirror the COEConfig schema, e.g.:
    retriever.corpus.embedder.url=http://n0385:4000/v1
    retriever.corpus.embedder.model_name=Qwen3-Embedding-4B

Output layout:
    <output>/index.jsonl
    <output>/entries/q_<id>_<question_hash>.json

Resume-safe: questions whose id already appears in index.jsonl are skipped.
"""

import asyncio
import hashlib
import json
import logging
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Freebase helpers
# ---------------------------------------------------------------------------

_MID_RE = re.compile(r"^[mg]\.[0-9a-z_]+$")
_SCHEMA_PREFIXES = ("rdf-schema#", "type.type.", "freebase.type_profile.")


def _is_mid(node: str) -> bool:
    return bool(_MID_RE.match(node))


def _is_schema_relation(rel: str) -> bool:
    return any(rel.startswith(p) for p in _SCHEMA_PREFIXES)


def _humanize(freebase_path: str) -> str:
    """Extract human-readable label from a Freebase relation path.

    Examples:
        "baseball.baseball_team.roster"               → "roster"
        "baseball.baseball_team.team_stats"            → "team stats"
        "baseball.baseball_team.team_stats::year"      → "team stats year"
          (compound flattened relation stored as R1::R2)
    """
    if "::" in freebase_path:
        r1, r2 = freebase_path.split("::", 1)
        return f"{_humanize(r1)} {_humanize(r2)}"
    last = freebase_path.split(".")[-1]
    return last.replace("_", " ")


def _question_hash(question: str) -> str:
    return hashlib.sha1(question.encode("utf-8")).hexdigest()[:12]


# ---------------------------------------------------------------------------
# Per-sample processing
# ---------------------------------------------------------------------------

def _process_sample(sample: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one dataset sample into a cache entry (no embedding step).

    Returns a dict with all fields except 'embeddings' (filled later).
    """
    raw_triples: List[List[str]] = sample.get("graph", [])
    q_entity_texts: List[str] = sample.get("q_entity") or []

    # ------------------------------------------------------------------
    # Step 1 — filter schema triples
    # ------------------------------------------------------------------
    triples = [t for t in raw_triples if len(t) == 3 and not _is_schema_relation(t[1])]

    # ------------------------------------------------------------------
    # Step 2 — identify CVT nodes and build maps
    # ------------------------------------------------------------------
    all_nodes: Set[str] = set()
    for h, _, t in triples:
        all_nodes.add(h)
        all_nodes.add(t)

    cvt_nodes: Set[str] = {n for n in all_nodes if _is_mid(n)}

    # parent_map[M]    = [(entity, relation)] — incoming edges to CVT M
    # attribute_map[M] = [(attr_rel, value)]  — scalar outgoing edges
    # nested_map[M]    = [(attr_rel, cvt_child)] — CVT-valued outgoing edges
    parent_map: Dict[str, List[Tuple[str, str]]] = {}
    attribute_map: Dict[str, List[Tuple[str, str]]] = {}
    nested_map: Dict[str, List[Tuple[str, str]]] = {}

    for cvt in cvt_nodes:
        parent_map[cvt] = []
        attribute_map[cvt] = []
        nested_map[cvt] = []

    for h, r, t in triples:
        if t in cvt_nodes:
            parent_map[t].append((h, r))
        if h in cvt_nodes:
            if t in cvt_nodes:
                nested_map[h].append((r, t))
            else:
                attribute_map[h].append((r, t))

    # ------------------------------------------------------------------
    # Step 3 — process CVTs bottom-up
    # ------------------------------------------------------------------
    # Topological order: CVTs with no CVT-valued attributes first
    processed_order = _topological_cvt_order(cvt_nodes, nested_map)

    synthesized: Dict[str, str] = {}  # cvt -> synthesized label
    new_triples: List[Tuple[str, str, str]] = []   # (subj, rel, obj) — post-CVT
    drop_subjects: Set[str] = set()                # edges to drop (CVT-as-obj)
    drop_heads: Set[str] = set()                   # CVT head edges to drop

    for cvt in processed_order:
        parents = parent_map.get(cvt, [])
        attrs = attribute_map.get(cvt, [])
        nested = nested_map.get(cvt, [])

        if not parents:
            continue  # dangling CVT — skip

        # Part A: CVT string synthesis
        parts: List[str] = []
        for attr_rel, nested_cvt in nested:
            if nested_cvt in synthesized:
                parts.append(f"{_humanize(attr_rel)}: {synthesized[nested_cvt]}")
        for attr_rel, val in attrs:
            parts.append(f"{_humanize(attr_rel)}: {val}")
        synthesized_str = "; ".join(parts) if parts else ""

        # Include parent entities in the synthesized string for cross-entity linking
        parent_parts = [f"{_humanize(parent_rel)}: {parent_ent}" for parent_ent, parent_rel in parents]
        if parent_parts:
            synthesized_str = "; ".join(parent_parts + (parts if parts else []))

        synthesized[cvt] = synthesized_str

        # Emit synthetic triples: (parent_entity, parent_rel, synthesized_string)
        for parent_ent, parent_rel in parents:
            new_triples.append((parent_ent, parent_rel, synthesized_str))

        # Part B: CVT flattening — Rule 1 (parent + attribute)
        for parent_ent, parent_rel in parents:
            for attr_rel, val in attrs:
                compound_rel = f"{parent_rel}::{attr_rel}"
                new_triples.append((parent_ent, compound_rel, val))
            for attr_rel, nested_cvt in nested:
                compound_rel = f"{parent_rel}::{attr_rel}"
                nested_val = synthesized.get(nested_cvt, nested_cvt)
                new_triples.append((parent_ent, compound_rel, nested_val))

        # Part B: CVT flattening — Rule 2 (cross-entity: parent + parent)
        for i, (ent_i, rel_i) in enumerate(parents):
            for j, (ent_j, rel_j) in enumerate(parents):
                if i >= j:
                    continue
                new_triples.append((ent_i, rel_j, ent_j))
                new_triples.append((ent_j, rel_i, ent_i))

        drop_subjects.add(cvt)
        drop_heads.add(cvt)

    # ------------------------------------------------------------------
    # Merge: keep non-CVT original triples + all new_triples
    # ------------------------------------------------------------------
    merged: List[Tuple[str, str, str]] = []

    for h, r, t in triples:
        if t in drop_subjects:
            continue  # was (entity, rel, CVT) — replaced by synthesis/flattening
        if h in drop_heads:
            continue  # was (CVT, attr, value) — replaced
        merged.append((h, r, t))

    # Add synthesized and flattened triples (skip if obj is still a CVT without synthesis)
    for h, r, t in new_triples:
        if _is_mid(t) and t not in synthesized:
            continue  # unresolvable CVT reference
        merged.append((h, r, t))

    # Deduplicate
    seen_triples: Set[Tuple[str, str, str]] = set()
    dedup_triples: List[Tuple[str, str, str]] = []
    for triple in merged:
        if triple not in seen_triples:
            seen_triples.add(triple)
            dedup_triples.append(triple)

    # ------------------------------------------------------------------
    # Step 4 — collect readable entity nodes and relation paths
    # ------------------------------------------------------------------
    readable_nodes: Set[str] = set()
    relation_paths: Set[str] = set()

    for h, r, t in dedup_triples:
        if not _is_mid(h):
            readable_nodes.add(h)
        relation_paths.add(r)
        if not _is_mid(t):
            readable_nodes.add(t)

    # Synthesized CVT strings show up as object values — they are string literals,
    # not entity nodes.  So readable_nodes only contains real entity labels.

    # Separate CVT synthesis strings from real entity nodes
    # (synthesis strings always start with humanized relation parts; they look long and
    # contain "; " — real entity names rarely do in Freebase)
    # A safer heuristic: check if the string appears as a synthesized value
    cvt_synthesized_values: Set[str] = set(synthesized.values())

    entity_nodes = sorted(readable_nodes - cvt_synthesized_values)

    # ------------------------------------------------------------------
    # Step 5 / 6 — assign synthetic IDs; build entity_dict / property_dict
    # ------------------------------------------------------------------
    entity_to_qid: Dict[str, str] = {}
    entity_dict: Dict[str, Dict] = {}

    for idx, label in enumerate(entity_nodes):
        qid = f"Q{idx}"
        entity_to_qid[label] = qid
        entity_dict[qid] = {
            "qid": qid,
            "label": label,
            "description": "",
            "is_cvt_record": False,
        }

    # CVT synthesis strings get QCVT IDs
    cvt_str_to_qid: Dict[str, str] = {}
    cvt_counter = 0
    for cvt_str in sorted(cvt_synthesized_values):
        if not cvt_str:
            continue
        qid = f"QCVT{cvt_counter}"
        cvt_counter += 1
        cvt_str_to_qid[cvt_str] = qid
        entity_dict[qid] = {
            "qid": qid,
            "label": cvt_str,
            "description": "",
            "is_cvt_record": True,
        }

    # Relations
    sorted_rels = sorted(relation_paths)
    rel_to_pid: Dict[str, str] = {}
    property_dict: Dict[str, Dict] = {}
    for idx, rel in enumerate(sorted_rels):
        pid = f"P{idx}"
        rel_to_pid[rel] = pid
        property_dict[pid] = {
            "pid": pid,
            "label": _humanize(rel),
            "description": "",
        }

    # ------------------------------------------------------------------
    # Step 7 — Build serialisable triple list (qid, pid, qid_or_str)
    # ------------------------------------------------------------------
    def _resolve_node(node: str) -> Optional[str]:
        if node in entity_to_qid:
            return entity_to_qid[node]
        if node in cvt_str_to_qid:
            return cvt_str_to_qid[node]
        # String literal value that is not an entity node
        return node

    serialised_triples: List[List[str]] = []
    for h, r, t in dedup_triples:
        subj_id = _resolve_node(h)
        prop_id = rel_to_pid.get(r, r)
        obj_id = _resolve_node(t)
        if subj_id is None or obj_id is None:
            continue
        serialised_triples.append([subj_id, prop_id, obj_id])

    # ------------------------------------------------------------------
    # Step 8 — map q_entity texts to Q-IDs
    # ------------------------------------------------------------------
    q_entity_qids: List[str] = []
    for text in q_entity_texts:
        qid = entity_to_qid.get(text)
        if qid is not None:
            q_entity_qids.append(qid)
        else:
            # Try case-insensitive fallback
            text_lower = text.lower()
            for label, qid_cand in entity_to_qid.items():
                if label.lower() == text_lower:
                    logger.warning("Case-insensitive entity lookup fallback used for %r -> %s.", text, qid_cand)
                    q_entity_qids.append(qid_cand)
                    break

    return {
        "id": sample.get("id", ""),
        "question": sample.get("question", ""),
        "q_entity_qids": q_entity_qids,
        "entity_dict": entity_dict,
        "property_dict": property_dict,
        "triples": serialised_triples,
        # embeddings filled by caller after embedding API call
    }


def _topological_cvt_order(
    cvt_nodes: Set[str],
    nested_map: Dict[str, List[Tuple[str, str]]],
) -> List[str]:
    """Return CVT nodes in processing order (leaves first, deepest nesting last)."""
    in_degree: Dict[str, int] = {c: 0 for c in cvt_nodes}
    dependents: Dict[str, List[str]] = {c: [] for c in cvt_nodes}

    for parent_cvt, nested_list in nested_map.items():
        for _, child_cvt in nested_list:
            if child_cvt in cvt_nodes:
                in_degree[parent_cvt] = in_degree.get(parent_cvt, 0) + 1
                dependents[child_cvt].append(parent_cvt)

    queue = [c for c, d in in_degree.items() if d == 0]
    order: List[str] = []
    while queue:
        node = queue.pop()
        order.append(node)
        for dep in dependents.get(node, []):
            in_degree[dep] -= 1
            if in_degree[dep] == 0:
                queue.append(dep)

    # Any remaining (cycle) appended in arbitrary order
    remaining = [c for c in cvt_nodes if c not in set(order)]
    return order + remaining


# ---------------------------------------------------------------------------
# Embedding step (async, batched)
# ---------------------------------------------------------------------------

async def _embed_samples_async(
    processed_samples: List[Dict[str, Any]],
    embed_client,
    batch_size: int = 512,
) -> List[Dict[str, Any]]:
    """Fill 'embeddings' field for all processed samples using embed_client."""
    if embed_client is None:
        for s in processed_samples:
            s["embeddings"] = {}
        return processed_samples

    for s in processed_samples:
        s["embeddings"] = {}

    # Collect (sample_idx, qid, label) for non-CVT entities
    tasks: List[Tuple[int, str, str]] = []
    for sidx, s in enumerate(processed_samples):
        for qid, info in s["entity_dict"].items():
            if not info.get("is_cvt_record") and info.get("label"):
                tasks.append((sidx, qid, info["label"]))

    # Process in batches with progress visibility for long embedding runs.
    for start in tqdm(
        range(0, len(tasks), batch_size),
        desc="Computing embeddings",
        unit="batch",
    ):
        batch = tasks[start : start + batch_size]
        labels = [t[2] for t in batch]
        try:
            embeddings = await asyncio.to_thread(embed_client.get_embeddings, labels)
            if embeddings is not None:
                for (sidx, qid, _), vec in zip(batch, embeddings):
                    processed_samples[sidx]["embeddings"][qid] = vec
        except Exception as e:
            logger.warning("Embedding batch failed (start=%d): %s", start, e)

    return processed_samples


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

async def _run(
    dataset_name: str,
    output_path: str,
    config_overrides: Optional[List[str]] = None,
) -> None:
    from coe.config import COEConfig
    from coe.evaluation.datasets import load_dataset_any

    cfg = COEConfig.from_yaml(overrides=config_overrides)

    output_dir = Path(output_path)
    if output_dir.suffix == ".jsonl":
        raise ValueError(
            "output must be a directory for split cache format (remove .jsonl suffix), "
            f"got: {output_dir}"
        )
    index_file = output_dir / "index.jsonl"
    entries_dir = output_dir / "entries"
    output_dir.mkdir(parents=True, exist_ok=True)
    entries_dir.mkdir(parents=True, exist_ok=True)

    # Load already-processed IDs for resume support
    done_ids: Set[str] = set()
    if index_file.exists():
        with open(index_file, encoding="utf-8") as f:
            for line in f:
                try:
                    entry = json.loads(line)
                    done_ids.add(str(entry.get("id", "")))
                except json.JSONDecodeError:
                    pass
        logger.info("Resume: %d samples already processed.", len(done_ids))

    # Load dataset (needs graph + q_entity columns)
    dataset = load_dataset_any(dataset_name)
    if "graph" not in dataset.column_names or "q_entity" not in dataset.column_names:
        raise ValueError(
            f"Dataset '{dataset_name}' must have 'graph' and 'q_entity' columns. "
            f"Got: {dataset.column_names}"
        )

    # Build embedding client from corpus embedder config
    embed_client = None
    try:
        from coe.llm.client import LLMClient
        ec = cfg.retriever.corpus.embedder
        if ec.url and cfg.llm.api_key:
            embed_client = LLMClient(
                model_name=ec.model_name,
                url=ec.url,
                api_key=cfg.llm.api_key,
                is_embedding=True,
            )
            logger.info("Embedding client: %s @ %s", ec.model_name, ec.url)
        else:
            logger.warning(
                "No embedding client configured (need retriever.corpus.embedder.url and api_key). "
                "Embeddings will be empty — asearch_entities will fall back to string matching."
            )
    except Exception as e:
        logger.warning("Could not create embedding client: %s", e)

    pending_samples: List[Dict[str, Any]] = []
    total = len(dataset)

    # Process graph structure (CPU-bound, no IO)
    for i, sample in enumerate(dataset):
        sample_id = str(sample.get("id", i))
        if sample_id in done_ids:
            continue

        processed = _process_sample(sample)
        processed["id"] = sample_id
        pending_samples.append(processed)

        if (i + 1) % 100 == 0 or i + 1 == total:
            logger.info("Graph processing: %d / %d", i + 1, total)

    logger.info("Graph processing done. %d samples to embed + write.", len(pending_samples))

    if not pending_samples:
        logger.info("Nothing new to process. Output: %s", output_dir)
        return

    # Embed labels
    pending_samples = await _embed_samples_async(pending_samples, embed_client)

    # Write split cache layout (append mode for index resume safety)
    written = 0
    with open(index_file, "a", encoding="utf-8") as f:
        for s in pending_samples:
            sample_id = str(s.get("id", written))
            question = str(s.get("question", ""))
            filename = f"q_{sample_id}_{_question_hash(question)}.json"
            rel_path = Path("entries") / filename
            entry_path = output_dir / rel_path

            with open(entry_path, "w", encoding="utf-8") as ef:
                json.dump(s, ef, ensure_ascii=False)

            index_row = {
                "question": question,
                "id": sample_id,
                "entry_path": str(rel_path),
            }
            f.write(json.dumps(index_row, ensure_ascii=False) + "\n")
            written += 1

    logger.info("Done. Wrote %d split-cache entries to %s", written, output_dir)


def main(argv: Optional[List[str]] = None) -> None:
    """CLI entry point.

    Positional-style CLI: first unknown arg that contains '=' is treated as an
    override; the special keys ``dataset`` and ``output`` are consumed before
    passing the rest to COEConfig.
    """
    import argparse

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(
        description="Preprocess Freebase subgraph dataset for COE evaluation."
    )
    parser.add_argument("overrides", nargs="*", help="key=value config overrides")
    args = parser.parse_args(argv)

    dataset_name: Optional[str] = None
    output_path: Optional[str] = None
    remaining_overrides: List[str] = []

    for override in args.overrides:
        if "=" not in override:
            logger.warning("Ignoring non-override arg: %s", override)
            continue
        key, _, value = override.partition("=")
        if key == "dataset":
            dataset_name = value
        elif key == "output":
            output_path = value
        else:
            remaining_overrides.append(override)

    if dataset_name is None:
        parser.error("Must provide dataset=<name>")
    if output_path is None:
        parser.error("Must provide output=<path>")

    asyncio.run(_run(dataset_name, output_path, remaining_overrides))


if __name__ == "__main__":
    main()
