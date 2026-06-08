"""CLI entry point: python -m coe"""

import argparse
import sys

from coe.config import COEConfig, get_default_config_path
from coe.system import COESystem


def main():
    parser = argparse.ArgumentParser(description="COE - Question Answering with Graph-Enhanced Retrieval")
    parser.add_argument("question", nargs="?", help="Question to answer")
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML")
    parser.add_argument("overrides", nargs="*", help="Config overrides (e.g., llm.model_name=gpt-4o)")
    
    args = parser.parse_args()
    
    if not args.question:
        parser.print_help()
        print("\nExamples:")
        print('  python -m coe "What is the capital of France?"')
        print('  python -m coe "Who directed Inception?" search.strategy=mcts')
        sys.exit(0)
    
    overrides = args.overrides or []
    
    system = COESystem(config_path=args.config, config_overrides=overrides)
    
    try:
        print(f"\nQuestion: {args.question}")
        print(f"Strategy: {system.cfg.search.strategy}")
        print(f"Model: {system.cfg.llm.model_name}")
        print("=" * 60)
        
        result = system.answer(args.question)
        
        print(f"\nAnswer: {result.answer}\n")
        if result.concise_answer and system.cfg.output.include_concise_answer:
            print(f"Concise: {result.concise_answer}\n")
    finally:
        system.close()


if __name__ == "__main__":
    main()
