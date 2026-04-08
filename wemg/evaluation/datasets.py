"""Dataset loading and preprocessing for evaluation."""

import logging
import os
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def load_dataset_any(name_or_path: str, max_examples: Optional[int] = None, shuffle: bool = False):
    """Load a dataset from HuggingFace, local disk, or JSON file.

    Known dataset names use the same loaders as ``main`` branch
    ``wemg/evaluation/evaluate.py`` (``preprocess_dataset``).

    Returns a HuggingFace Dataset object.
    """
    from datasets import load_dataset, load_from_disk

    path = Path(name_or_path)

    known = _get_known_dataset(name_or_path)
    if known is not None:
        ds = known
    elif path.is_dir() and _is_hf_dataset_dir(path):
        ds = load_from_disk(str(path))
    elif path.is_file() and path.suffix.lower() in (".json", ".jsonl"):
        ds = load_dataset("json", data_files=str(path), split="train")
    else:
        ds = load_dataset(name_or_path, split="train")

    if shuffle:
        ds = ds.shuffle(seed=42)
    if max_examples is not None and len(ds) > max_examples:
        ds = ds.select(range(max_examples))

    return ds


def _is_hf_dataset_dir(path: Path) -> bool:
    return (path / "dataset_info.json").exists() or (path / "state.json").exists()


def _get_known_dataset(name: str):
    """Load and preprocess known evaluation datasets (aligned with main ``evaluate.py``)."""
    from datasets import load_dataset

    key = name.lower().replace("-", "_")

    # Graph QA datasets
    if key == "cwq":
        data = load_dataset("Hieuman/cwq-train-eval", split="validation")
        to_remove_columns = set(data.column_names) - {"question", "answers"}
        data = data.remove_columns(list(to_remove_columns))

        def preprocess_example(example):
            answers = example["answers"]
            all_answers = list(answers.get("answer", []))
            aliases = answers.get("aliases", [])
            if aliases and isinstance(aliases[0], list):
                aliases = sum(aliases, [])
            all_answers = all_answers + list(aliases)
            return {"question": example["question"], "answer": list(set(all_answers))}

        return data.map(
            preprocess_example,
            num_proc=os.cpu_count(),
            remove_columns=["answers"],
        )

    if key == "webqsp":
        data = load_dataset("ml1996/webqsp", split="test")
        to_remove_columns = set(data.column_names) - {"question", "answer"}
        return data.remove_columns(list(to_remove_columns))

    if key in ("webqsp_sub", "cwq_sub"):
        rog_name = "webqsp" if "webqsp" in key else "cwq"
        data = load_dataset(f"rmanluo/RoG-{rog_name}", split="test")

        def _a_entity_in_graph(example):
            graph_nodes = set()
            for triple in example["graph"]:
                graph_nodes.add(triple[0])
                graph_nodes.add(triple[2])
            return any(e in graph_nodes for e in (example["a_entity"] or []))

        filtered = data.filter(_a_entity_in_graph, num_proc=os.cpu_count())
        to_remove = set(filtered.column_names) - {"question", "answer", "graph", "q_entity"}
        return filtered.remove_columns(list(to_remove))

    if key == "qald_10":
        return load_dataset("Hieuman/qald_10_en", split="train")

    if key == "hotpotqa_adv":
        return load_dataset("Hieuman/Hotpotqa-adv", split="train")

    if key == "grail_qa":
        data = load_dataset("Hieuman/grail_qa", split="validation")
        data = data.map(
            lambda x: {
                "question": x["question"],
                "answer": x["answer"].get("entity_name", []),
            }
        )
        to_remove_columns = set(data.column_names) - {"question", "answer"}
        return data.remove_columns(list(to_remove_columns))

    # Text QA datasets
    if key == "2wiki":
        data = load_dataset(
            "RUC-NLPIR/FlashRAG_datasets", "2wikimultihopqa", split="dev"
        )
        data = data.rename_column("golden_answers", "answer")
        to_remove_columns = set(data.column_names) - {"question", "answer"}
        return data.remove_columns(list(to_remove_columns))

    if key == "hotpotqa":
        data = load_dataset("RUC-NLPIR/FlashRAG_datasets", "hotpotqa", split="dev")
        data = data.rename_column("golden_answers", "answer")
        to_remove_columns = set(data.column_names) - {"question", "answer"}
        return data.remove_columns(list(to_remove_columns))

    if key == "musique":
        data = load_dataset("RUC-NLPIR/FlashRAG_datasets", "musique", split="dev")
        data = data.rename_column("golden_answers", "answer")
        to_remove_columns = set(data.column_names) - {"question", "answer"}
        return data.remove_columns(list(to_remove_columns))

    if key == "bamboogle":
        data = load_dataset("RUC-NLPIR/FlashRAG_datasets", "bamboogle", split="test")
        data = data.rename_column("golden_answers", "answer")
        to_remove_columns = set(data.column_names) - {"question", "answer"}
        return data.remove_columns(list(to_remove_columns))

    if key == "frames":
        data = load_dataset("google/frames-benchmark", split="test")
        return data.map(
            lambda x: {
                "question": x["Prompt"],
                "answer": [x["Answer"]],
            },
            remove_columns=data.column_names,
        )

    return None
