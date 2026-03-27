# Architecture

## Class diagram

```mermaid
classDiagram
    class MemoryDelta {
        +List~str~ new_textual_items
        +List~WikiTriple~ new_triples
        +Dict~str,WikidataEntity~ new_entities
    }

    class GlobalKnowledge {
        +List~str~ confirmed_facts
        +DiGraph graph
        +Dict~str,WikidataEntity~ entity_dict
        +absorb(delta, reward, min_reward)
        +consolidate_if_needed(client, question, ...)
        +deduplicate_graph()
        +format_textual_memory() str
        +format_graph_memory() str
    }

    class WorkingMemory {
        +List~str~ textual_memory
        +DiGraph graph_memory
        +Dict~str,WikidataEntity~ entity_dict
        +GlobalKnowledge global_knowledge
        +snapshot() WorkingMemory
        +get_delta() MemoryDelta
        +should_synchronize() bool
        +synchronize_memory(client, question, ...)
        +asynchronize_memory(client, question, ...)
        +add_textual_memory(text, source)
        +add_edge_to_graph_memory(triple, ...)
        +format_textual_memory() str
        +format_graph_memory() str
        +deduplicate_graph()
    }

    WorkingMemory --> GlobalKnowledge : "optional ref\n(MCTS mode)"
    WorkingMemory ..> MemoryDelta : "produces via get_delta()"
    GlobalKnowledge ..> MemoryDelta : "consumes via absorb()"
```

## `WorkingMemory` (local / per-path)

Defined in `wemg/reasoning/working_memory.py`.

Holds the **local** reasoning state for one path or step:

| Attribute | Type | Purpose |
|-----------|------|---------|
| `textual_memory` | `List[str]` | Deduplicated text facts with provenance tags |
| `graph_memory` | `nx.DiGraph` | Extracted triples (nodes carry `Entity` data, edges carry `relation` sets and optional `provenance`) |
| `entity_dict` | `Dict[str, WikidataEntity]` | Wikidata entity cache |
| `global_knowledge` | `GlobalKnowledge or None` | Shared store reference (set for MCTS, `None` for CoT) |

When `global_knowledge` is set, the formatters merge both views:

- `format_textual_memory()` prepends `global_knowledge.confirmed_facts` before local items.
- `format_graph_memory()` composes `global_knowledge.graph` with `graph_memory` using `nx.compose`.

When `global_knowledge` is `None`, the instance is standalone (CoT, unit tests).

## `GlobalKnowledge` (shared across MCTS tree)

Also in `wemg/reasoning/working_memory.py`.

| Attribute | Type | Purpose |
|-----------|------|---------|
| `confirmed_facts` | `List[str]` | Text facts promoted from successful branches |
| `graph` | `nx.DiGraph` | Structured triples extracted from confirmed facts |
| `entity_dict` | `Dict[str, WikidataEntity]` | Entities discovered across all branches |

Updates arrive through two paths:

1. **`absorb(delta, reward, min_reward)`** -- reward-gated promotion of branch
   discoveries.  Only text facts and entities are absorbed directly; the graph
   is updated from those facts during the next consolidation.
2. **`consolidate_if_needed(client, question, ...)`** -- periodic maintenance:
   - LLM text consolidation when `confirmed_facts` count exceeds threshold
   - One-directional extraction of unprocessed text into the global graph
   - Structural QID-based graph dedup when node count exceeds threshold

## `MemoryDelta`

A lightweight dataclass (`@dataclass`) returned by `WorkingMemory.get_delta()`:

```python
@dataclass
class MemoryDelta:
    new_textual_items: List[str]
    new_triples: List[WikiTriple]
    new_entities: Dict[str, WikidataEntity]
```

## Snapshot / delta lifecycle

```mermaid
sequenceDiagram
    participant MCTS as mcts_search loop
    participant Base as WorkingMemory (base)
    participant Snap as WorkingMemory (snapshot)
    participant GK as GlobalKnowledge

    loop each iteration
        MCTS->>Base: snapshot()
        Base-->>Snap: copy local state, share GK ref
        MCTS->>Snap: expand / evaluate
        Note over Snap: branch mutations stay local
        Snap-->>MCTS: reward
        MCTS->>Snap: get_delta()
        Snap-->>MCTS: MemoryDelta
        MCTS->>GK: absorb(delta, reward, min_reward)
        Note over GK: only if reward >= min_reward
        opt every N iterations
            MCTS->>GK: consolidate_if_needed(...)
        end
        MCTS->>Base: restore generator.working_memory
    end
```

Key invariants:

- Mutations on a snapshot never propagate back to the base or to other snapshots.
- `global_knowledge` is shared by reference, so all snapshots read the same
  confirmed facts (but never write to it directly -- only `absorb` does).

## CoT vs MCTS wiring

```mermaid
flowchart LR
    subgraph cotPath [CoT path]
        SysCot["system.py\n_answer_with_cot()"]
        SysCot -->|"WorkingMemory(\nglobal_knowledge=None)"| CotSearch["cot_search()"]
        CotSearch -->|"synchronize_memory()\nper step"| CotSearch
    end

    subgraph mctsPath [MCTS path]
        SysMcts["system.py\n_answer_with_mcts()"]
        SysMcts -->|"GlobalKnowledge()"| GK2["GK"]
        SysMcts -->|"WorkingMemory(\nglobal_knowledge=GK)"| MctsSearch["mcts_search()"]
        MctsSearch -->|"snapshot() per iteration\nabsorb() after eval"| GK2
    end
```

- `wemg/system.py` creates `GlobalKnowledge` only for the `mcts` strategy.
- `wemg/config.py` exposes thresholds via `WorkingMemoryConfig`:
  `sync_text_threshold`, `sync_graph_node_threshold`, `absorption_min_reward`.
