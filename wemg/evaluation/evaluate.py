import logging
import os
from pathlib import Path
import datasets
import hydra
from omegaconf import DictConfig, OmegaConf
from typing import Optional

from wemg.main import WEMGSystem
from wemg.evaluation.dataset_evaluator import DatasetEvaluator

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("LOGGING_LEVEL", "INFO"))


def preprocess_dataset(dataset_name_or_path: str, max_examples: Optional[int] = 1000) -> datasets.Dataset:
    # ===== Graph-based QA datasets =====
    if dataset_name_or_path == 'cwq':
        data = datasets.load_dataset('Hieuman/cwq-train-eval', split='validation')
        to_remove_columns = set(data.column_names) - set(['question', 'answers'])
        data = data.remove_columns(list(to_remove_columns))
        # Preprocess the data
        def preprocess_example(example):
            answers = example['answers']
            all_answers = answers.get('answer', [])
            aliases = answers.get('aliases', [])
            aliases = sum(aliases, []) if isinstance(aliases[0], list) else aliases
            all_answers = all_answers + aliases
            return {
                'question': example['question'],
                'answer': list(set(all_answers))
            }
        data = data.map(preprocess_example, batched=True, batch_size=1000, num_proc=os.cpu_count(), remove_columns=['answers'])
    elif dataset_name_or_path == 'webqsp':
        data = datasets.load_dataset('ml1996/webqsp', split='test')
        to_remove_columns = set(data.column_names) - set(['question', 'answer'])
        data = data.remove_columns(list(to_remove_columns))
    elif dataset_name_or_path == 'qald_10':
        data = datasets.load_dataset('Hieuman/qald_10', split='train')
    elif dataset_name_or_path == 'hotpotqa_adv':
        data = datasets.load_dataset('Hieuman/Hotpotqa-adv', split='train')
    elif dataset_name_or_path == 'grail_qa':
        data = datasets.load_dataset('Hieuman/grail_qa', split='validation')
        data = data.map(lambda x: {'question': x['question'], 'answer': x['answer'].get('entity_name', [])})
        to_remove_columns = set(data.column_names) - set(['question', 'answer'])
        data = data.remove_columns(list(to_remove_columns))
    # ===== Text-based QA datasets =====
    elif dataset_name_or_path == '2wiki':
        data = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", "2wikimultihopqa", split='dev')
        data = data.rename_column('golden_answers', 'answer')
        to_remove_columns = set(data.column_names) - set(['question', 'answer'])
        data = data.remove_columns(list(to_remove_columns))
    elif dataset_name_or_path == 'hotpotqa':
        data = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", "hotpotqa", split='dev')
        data = data.rename_column('golden_answers', 'answer')
        to_remove_columns = set(data.column_names) - set(['question', 'answer'])
        data = data.remove_columns(list(to_remove_columns))
    elif dataset_name_or_path == 'musique':
        data = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", "musique", split='dev')
        data = data.rename_column('golden_answers', 'answer')
        to_remove_columns = set(data.column_names) - set(['question', 'answer'])
        data = data.remove_columns(list(to_remove_columns))
    elif dataset_name_or_path == 'bamboogle':
        data = datasets.load_dataset("RUC-NLPIR/FlashRAG_datasets", "bamboogle", split='test')
        data = data.rename_column('golden_answers', 'answer')
        to_remove_columns = set(data.column_names) - set(['question', 'answer'])
        data = data.remove_columns(list(to_remove_columns))
    elif dataset_name_or_path == 'frames':
        data = datasets.load_dataset("google/frames-benchmark", split='test')
        data = data.map(
            lambda x: {
                "question": x["Prompt"],
                "answer": [x["Answer"]],
            },
            remove_columns=data.column_names,
        )
    else:
        raise ValueError(f"Unknown dataset: {dataset_name_or_path}. Supported datasets: cwq, webqsp, qald_10, hotpotqa_adv, grail_qa, 2wiki, hotpotqa, musique, bamboogle, frames")
    
    max_examples = max_examples if max_examples is not None else 1000
    data = data.shuffle(seed=42)
    if len(data) > max_examples:
        data = data.select(range(max_examples))
    return data


def load_dataset_any(
    dataset_name_or_path: str,
    max_examples: Optional[int] = 1000,
    shuffle: bool = True,
) -> datasets.Dataset:
    """Load a dataset from either:
    - known dataset name handled by `preprocess_dataset`
    - a local HuggingFace dataset directory (`datasets.load_from_disk`)
    - a local JSON/JSONL file (`datasets.load_dataset("json")`)
    """
    p = Path(dataset_name_or_path)
    if p.exists():
        if p.is_dir():
            data = datasets.load_from_disk(str(p))
        else:
            suffix = p.suffix.lower()
            if suffix in {".jsonl", ".json"}:
                data = datasets.load_dataset("json", data_files=str(p), split="train")
            else:
                raise ValueError(
                    f"Unsupported local dataset file type: '{suffix}'. "
                    f"Supported: .jsonl, .json, or a HF dataset directory."
                )
    else:
        data = preprocess_dataset(dataset_name_or_path, max_examples=max_examples)

    max_examples = max_examples if max_examples is not None else 1000
    if shuffle:
        data = data.shuffle(seed=42)
    if len(data) > max_examples:
        data = data.select(range(max_examples))
    return data

@hydra.main(version_base=None, config_path="..", config_name="config")
def main(cfg: DictConfig) -> None:
    """Main entry point with Hydra configuration.
    
    Run with:
        python -m wemg.evaluation.evaluate +dataset_name_or_path=path/to/dataset +output_path=path/to/output
        python -m wemg.evaluation.evaluate +dataset_name_or_path=path/to/dataset llm.model_name=gpt-4o
        python -m wemg.evaluation.evaluate +dataset_name_or_path=path/to/dataset search.strategy=mcts
    """
    # Remove the 'question' key from config before passing to WEMGSystem
    # since it's not a system configuration parameter
    system_cfg = OmegaConf.to_container(cfg, resolve=True)
    system_cfg.pop("question", None)
    system_cfg.pop("hydra", None)  # Remove hydra internal config
    dataset_name_or_path: Optional[str] = system_cfg.pop("dataset_name_or_path", None)  # get and remove
    if dataset_name_or_path is None:
        print("No dataset name or path provided. Please provide a dataset name or path.")
        print("Usage: python -m wemg.evaluation.evaluate +dataset_name_or_path=path/to/dataset")
        return
    else:
        print(f"Evaluating dataset: {dataset_name_or_path}")
    resume: bool = system_cfg.pop("resume", True)
    mode: str = str(system_cfg.pop("mode", "eval")).lower()  # "eval" or "score"
    max_examples: Optional[int] = system_cfg.pop("max_examples", 1000)
    output_path_cfg: Optional[str] = system_cfg.pop("output_path", None)

    # Scoring options (only used in mode=score)
    predicted_answer_column: str = str(system_cfg.pop("predicted_answer_column", "predicted_answer"))
    concise_answer_column: Optional[str] = system_cfg.pop("concise_answer_column", "concise_answer")
    compute_acc_scores: bool = bool(system_cfg.pop("compute_acc_scores", True))
    overwrite_existing_scores: bool = bool(system_cfg.pop("overwrite_existing_scores", False))

    # Resolve output path default
    if output_path_cfg:
        output_path = output_path_cfg
    else:
        p = Path(dataset_name_or_path)
        if p.exists() and p.is_dir():
            output_path = str(p)
        elif p.exists() and p.is_file():
            output_path = f"results/{p.stem}"
        else:
            output_path = f"results/{dataset_name_or_path}"

    # Load dataset
    # - In eval mode: shuffle by default (existing behavior)
    # - In score mode: do NOT shuffle (preserve alignment with existing predictions/logs)
    data = load_dataset_any(
        dataset_name_or_path,
        max_examples=max_examples,
        shuffle=(mode != "score"),
    )
    wemg_system = WEMGSystem(config_dict=system_cfg)
    evaluator = DatasetEvaluator(wemg_system)
    
    if resume:
        # If the user explicitly points to a local predictions file, do not
        # override it by loading from `output_path` when resuming.
        if os.path.exists(output_path) and not (Path(dataset_name_or_path).exists() and Path(dataset_name_or_path).is_file()):
            data = datasets.load_from_disk(output_path)
            print(f"Loaded {len(data)} questions for resuming evaluation from {output_path}")
        else:
            print(f"No existing results found for {dataset_name_or_path}. Starting fresh evaluation.")
            resume = False

    if mode == "score":
        # Compute metrics from existing predictions (no generation)
        result_dataset = evaluator.score_dataset_from_predictions(
            dataset=data,
            output_path=output_path,
            resume=resume,
            question_column="question",
            answer_column="answer",
            predicted_answer_column=predicted_answer_column,
            concise_answer_column=concise_answer_column,
            batch_size=8,
            max_workers=8,
            compute_acc_scores=compute_acc_scores,
            overwrite_existing_scores=overwrite_existing_scores,
        )
    else:
        # Run evaluation (generation + metrics)
        result_dataset = evaluator.evaluate(
            dataset=data,
            output_path=output_path,
            resume=resume,
            batch_size=8,
            max_workers=8
        )
    # Compute metrics
    metrics = evaluator.compute_aggregate_metrics(result_dataset)
    print(f"Metrics for {dataset_name_or_path}: {metrics}")
    # Save metrics
    evaluator.save_metrics(metrics, f"{output_path}/metrics.json")
    print(f"Metrics saved to {output_path}/metrics.json")


if __name__ == "__main__":
    main()