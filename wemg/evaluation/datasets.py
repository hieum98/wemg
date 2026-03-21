"""Dataset loading and preprocessing for evaluation."""

import logging
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


def load_dataset_any(name_or_path: str, max_examples: Optional[int] = None, shuffle: bool = False):
    """Load a dataset from HuggingFace, local disk, or JSON file.
    
    Returns a HuggingFace Dataset object.
    """
    from datasets import load_dataset, load_from_disk, Dataset
    import json
    
    path = Path(name_or_path)
    
    # Try known dataset names first
    known = _get_known_dataset(name_or_path)
    if known is not None:
        ds = known
    elif path.is_dir() and _is_hf_dataset_dir(path):
        ds = load_from_disk(str(path))
    elif path.is_file() and path.suffix in (".json", ".jsonl"):
        with open(path) as f:
            if path.suffix == ".jsonl":
                data = [json.loads(line) for line in f]
            else:
                data = json.load(f)
        ds = Dataset.from_list(data)
    else:
        ds = load_dataset(name_or_path, split="train")
    
    if shuffle:
        ds = ds.shuffle(seed=42)
    if max_examples and len(ds) > max_examples:
        ds = ds.select(range(max_examples))
    
    return ds


def _is_hf_dataset_dir(path: Path) -> bool:
    return (path / "dataset_info.json").exists() or (path / "state.json").exists()


def _get_known_dataset(name: str):
    """Load and preprocess known evaluation datasets."""
    from datasets import load_dataset
    
    name_lower = name.lower()
    
    # Graph QA datasets
    if name_lower == "cwq":
        ds = load_dataset("rewenthy/ComplexWebQuestions", split="test")
        return ds.rename_columns({"question": "question", "answers": "answer"})
    
    elif name_lower == "webqsp":
        ds = load_dataset("rmanluo/RoG-webqsp", split="test")
        return ds.map(lambda x: {
            "question": x["question"],
            "answer": [a["answer"] for a in x.get("answer", [])] if isinstance(x.get("answer"), list) else x.get("answer", ""),
        })
    
    elif name_lower == "qald_10":
        ds = load_dataset("KGQA/qald_10-en", split="test")
        return ds.rename_columns({"question": "question", "answer": "answer"})
    
    elif name_lower in ("hotpotqa_adv", "hotpotqa-adv"):
        ds = load_dataset("rmanluo/RoG-hotpotqa", split="test")
        return ds.rename_columns({"question": "question", "answer": "answer"})
    
    elif name_lower == "grail_qa":
        ds = load_dataset("grail_qa", split="validation")
        return ds.rename_columns({"question": "question", "answer": "answer"})
    
    # Text QA datasets
    elif name_lower == "2wiki":
        ds = load_dataset("xanhho/2WikiMultihopQA", split="test")
        return ds.rename_columns({"question": "question", "answer": "answer"})
    
    elif name_lower == "hotpotqa":
        ds = load_dataset("hotpotqa/hotpot_qa", "fullwiki", split="validation")
        return ds.rename_columns({"question": "question", "answer": "answer"})
    
    elif name_lower == "musique":
        ds = load_dataset("drt/musique", split="validation")
        return ds.rename_columns({"question": "question", "answer": "answer"})
    
    elif name_lower == "bamboogle":
        ds = load_dataset("m-ric/bamboogle", split="test")
        return ds.rename_columns({"question": "question", "answer": "answer"})
    
    elif name_lower == "frames":
        ds = load_dataset("google/frames-benchmark", split="test")
        return ds.rename_columns({"Prompt": "question", "Answer": "answer"})
    
    return None
