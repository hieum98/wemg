"""Example usage of WEMG Question Answering System.

This file demonstrates various ways to use WEMG for answering questions.
"""

import os
from pathlib import Path

# Make sure to set your API keys before running
# export OPENAI_API_KEY="your-api-key"
# export SERPER_API_KEY="your-serper-api-key"  # Optional, for web search


def example_basic_usage():
    """Basic usage with default configuration."""
    from wemg import answer_question
    
    question = "What is the capital of France?"
    answer = answer_question(question)
    print(f"Question: {question}")
    print(f"Answer: {answer}")


def example_with_system_class():
    """Using WEMGSystem class for more control."""
    from wemg import WEMGSystem
    
    # Create system with default config
    system = WEMGSystem()
    
    # Ask multiple questions with the same system instance
    questions = [
        "What is machine learning?",
        "Who invented the telephone?",
        "What is the speed of light?",
    ]
    
    try:
        for question in questions:
            result = system.answer(question)
            print(f"\nQ: {question}")
            print(f"A: {result.concise_answer}")
            if result.metadata:
                print(f"Strategy: {result.metadata.get('strategy')}")
    finally:
        system.close()


def example_with_config_overrides():
    """Using configuration overrides."""
    from wemg import WEMGSystem
    
    # Override specific configuration values
    overrides = [
        "llm.model_name=gpt-4o",
        "llm.generation.temperature=0.5",
        "search.strategy=cot",
        "search.cot.max_depth=5",
    ]
    
    system = WEMGSystem(config_overrides=overrides)
    
    try:
        result = system.answer("Explain quantum entanglement in simple terms.")
        print(f"Answer: {result.answer}")
    finally:
        system.close()


def example_with_dict_config():
    """Using dictionary configuration."""
    from wemg import WEMGSystem
    
    # Create configuration from dictionary
    config = {
        "llm": {
            "model_name": "gpt-4o-mini",
            "generation": {
                "temperature": 0.7,
                "max_tokens": 4096,
            }
        },
        "search": {
            "strategy": "cot",
            "cot": {
                "max_depth": 8
            }
        },
        "output": {
            "include_reasoning": True,
            "verbose": True,
        }
    }
    
    system = WEMGSystem(config_dict=config)
    
    try:
        result = system.answer("What are the main causes of climate change?")
        print(f"Answer: {result.answer}")
        print(f"\nConcise: {result.concise_answer}")
    finally:
        system.close()


def example_with_mcts():
    """Using MCTS search strategy."""
    from wemg import WEMGSystem
    
    config = {
        "search": {
            "strategy": "mcts",
            "mcts": {
                "num_iterations": 5,
                "max_tree_depth": 6,
                "exploration_weight": 1.0,
            }
        }
    }
    
    system = WEMGSystem(config_dict=config)
    
    try:
        result = system.answer("What is the relationship between Einstein and Newton's work?")
        print(f"Answer: {result.answer}")
        print(f"\nMetadata: {result.metadata}")
    finally:
        system.close()


def example_batch_processing():
    """Process multiple questions efficiently."""
    from wemg import answer_questions_batch
    
    questions = [
        "What is artificial intelligence?",
        "What is the capital of Japan?",
        "Who wrote 'Romeo and Juliet'?",
    ]
    
    results = answer_questions_batch(questions)
    
    for result in results:
        print(f"\nQ: {result.question}")
        print(f"A: {result.concise_answer}")


def example_custom_config_file():
    """Using a custom configuration file."""
    from wemg import WEMGSystem
    
    # Create a custom config file path
    custom_config_path = Path(__file__).parent / "custom_config.yaml"
    
    if custom_config_path.exists():
        system = WEMGSystem(config_path=custom_config_path)
        
        try:
            result = system.answer("What is deep learning?")
            print(f"Answer: {result.answer}")
        finally:
            system.close()
    else:
        print(f"Custom config not found at: {custom_config_path}")
        print("Please create a custom_config.yaml file based on wemg/config.yaml")


def example_with_corpus_retriever():
    """Using corpus-based retriever (requires corpus setup)."""
    from wemg import WEMGSystem
    
    config = {
        "retriever": {
            "type": "corpus",
            "corpus": {
                "embedder": {
                    "model_name": "text-embedding-3-small",
                    "url": "https://api.openai.com/v1",
                    "embedder_type": "openai",
                },
                "corpus_path": "/path/to/your/corpus",
                "index_path": "/path/to/your/index.faiss",
            }
        }
    }
    
    # Only run if corpus paths are properly configured
    if config["retriever"]["corpus"]["corpus_path"] != "/path/to/your/corpus":
        system = WEMGSystem(config_dict=config)
        try:
            result = system.answer("Your domain-specific question")
            print(f"Answer: {result.answer}")
        finally:
            system.close()
    else:
        print("Please configure corpus paths before running this example")


def main():
    """Run all examples."""
    print("=" * 60)
    print("WEMG Usage Examples")
    print("=" * 60)
    
    # Check for API key
    if not os.environ.get("OPENAI_API_KEY"):
        print("\nWarning: OPENAI_API_KEY environment variable not set.")
        print("Set it with: export OPENAI_API_KEY='your-api-key'")
        print("\nShowing example code only (not executing)...")
        
        # Print example code without executing
        examples = [
            ("Basic Usage", example_basic_usage),
            ("WEMGSystem Class", example_with_system_class),
            ("Config Overrides", example_with_config_overrides),
            ("Dictionary Config", example_with_dict_config),
            ("MCTS Strategy", example_with_mcts),
            ("Batch Processing", example_batch_processing),
        ]
        
        for name, func in examples:
            print(f"\n{'='*60}")
            print(f"Example: {name}")
            print("=" * 60)
            print(func.__doc__)
        
        return
    
    # Run examples
    examples = [
        ("Basic Usage", example_basic_usage),
        ("WEMGSystem Class", example_with_system_class),
        ("Config Overrides", example_with_config_overrides),
        ("Dictionary Config", example_with_dict_config),
        ("MCTS Strategy", example_with_mcts),
        ("Batch Processing", example_batch_processing),
    ]
    
    for name, func in examples:
        print(f"\n{'='*60}")
        print(f"Running: {name}")
        print("=" * 60)
        try:
            func()
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()
