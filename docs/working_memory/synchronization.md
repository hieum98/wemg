# Synchronization

WEMG maintains two related representations for each reasoning path:

- **Text** (`textual_memory`) -- flat list of tagged fact strings.
- **Graph** (`graph_memory`) -- `networkx.DiGraph` of entity-grounded triples.

The redesign keeps them consistent without expensive circular conversion.

## Old design (circular, removed)

```mermaid
flowchart LR
    G1["graph"] -->|"textualize"| T1["text"]
    T1 -->|"LLM consolidate"| T2["consolidated text"]
    T2 -->|"entity link +\nrelation extract"| G2["new graph"]
    G2 -->|"LLM consolidate\ngraph"| G3["consolidated graph"]
    style G1 fill:#fdd
    style G3 fill:#fdd
```

Problems: 2 full lossy round-trips per sync, ~5 LLM calls, information loss.

## New design (one-directional)

```mermaid
flowchart TD
    subgraph input [New discoveries]
        Ret["Retrieval results"]
        Reas["Reasoning outputs"]
    end

    subgraph textTier [Text tier]
        TM["textual_memory"]
        TC["LLM consolidate\n(threshold-gated)"]
    end

    subgraph graphTier [Graph tier]
        GM["graph_memory"]
        GD["deduplicate_graph()\nstructural QID merge\n(threshold-gated)"]
    end

    subgraph query [Query time]
        FC["format_textual_memory()\nformat_graph_memory()\nmerge global + local on the fly"]
    end

    Ret --> TM
    Reas --> TM
    Ret --> GM
    Reas --> GM
    TM -->|"when items >= threshold"| TC --> TM
    TM -->|"unprocessed items only\n(one-directional)"| GM
    GM -->|"when nodes >= threshold"| GD --> GM
    TM --> FC
    GM --> FC
```

Key rule: **graph never feeds back into text**.  Text is only extracted into the
graph once (tracked by `_text_item_ids_processed`).

## Conditional / incremental gating

Synchronization is skipped unless enough new data has accumulated.

| Tracking field | Set by | Checked by |
|---------------|--------|------------|
| `_dirty` | `add_textual_memory()` | `should_synchronize()` |
| `_items_since_last_sync` | `add_textual_memory()` | `should_synchronize()` and text consolidation gate |
| `_text_item_ids_processed` | `asynchronize_memory()` | unprocessed-items filter |

Decision logic in `should_synchronize()`:

```
dirty AND items_since_last_sync >= sync_text_threshold
```

Graph dedup triggers when:

```
graph_memory.number_of_nodes() > sync_graph_node_threshold
```

Both thresholds are configurable in `wemg/config.py` via `WorkingMemoryConfig`.

## Async pipeline

`asynchronize_memory()` is the async implementation.  It parallelises the two
most expensive I/O-bound steps on the same text batch:

```mermaid
sequenceDiagram
    participant Sync as asynchronize_memory()
    participant EL as entity linking
    participant RE as relation extraction

    Sync->>Sync: consolidate text (if threshold)
    Sync->>Sync: collect unprocessed text items
    par asyncio.gather
        Sync->>EL: _link_entities_async(new_text, ...)
        Sync->>RE: parse_graph_from_text(new_text, ...)
    end
    EL-->>Sync: linked entities
    RE-->>Sync: triples
    Sync->>Sync: enhance triples + add to graph
    Sync->>Sync: deduplicate graph (if threshold)
    Sync->>Sync: reset dirty flags
```

The sync wrapper `synchronize_memory()` calls `asyncio.run(asynchronize_memory(...))`.

Internally, text consolidation uses `_aconsolidate_textual_memory()` (an async
version that awaits `_run_consolidation` directly) to avoid nesting
`asyncio.run()` inside a running event loop.
