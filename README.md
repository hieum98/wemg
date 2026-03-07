# WEMG - When Embedding Model Meet Graph RAG

WEMG is a question-answering system that combines graph-based retrieval from knowledge graphs (Wikidata) with text-based retrieval from web search or corpus embeddings. It supports both Monte Carlo Tree Search (MCTS) and Chain-of-Thought (CoT) reasoning strategies.

## Features

- **Dual Retrieval**: Combines knowledge graph (Wikidata) and text-based (web/corpus) retrieval
- **Multiple Reasoning Strategies**: MCTS and CoT search strategies
- **LLM-based Agents**: Specialized roles (generator, evaluator, extractor, etc.)
- **Memory Systems**: Working memory and interaction memory for context management
- **Comprehensive Evaluation**: Supports Acc, Sub-EM, and Pass@k metrics
- **Resume Capability**: Evaluation can resume from logs/datasets
- **Async Support**: Efficient async/await for I/O-bound operations
- **Caching**: Redis caching for LLM calls and embedding cache

## Installation

### Prerequisites

- Python >= 3.8 (3.10+ recommended)
- Redis server (for caching, optional but recommended)

### Install from Source

```bash
# Clone the repository
git clone https://github.com/hieum98/wemg.git
cd wemg

# Install the package
pip install -e .

# Or install with development dependencies
pip install -e ".[dev]"
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## Quick Start

### Basic Usage

```python
from wemg.main import WEMGSystem

# Initialize system with default config
system = WEMGSystem()

# Answer a question
result = system.answer("What is the capital of France?")
print(result.answer)
print(result.concise_answer)
```

### Using Hydra CLI

```bash
# Answer a single question
python -m wemg.main question="What is the capital of France?"

# Override configuration
python -m wemg.main question="..." llm.model_name=gpt-4o search.strategy=mcts
```

### Evaluate a Dataset

```bash
# Evaluate a dataset
python -m wemg.evaluation.evaluate \
    +dataset_name_or_path=bamboogle \
    +output_path=results/bamboogle \
    llm.model_name=Qwen3-8B \
    search.strategy=mcts

# Resume from previous evaluation
python -m wemg.evaluation.evaluate \
    +dataset_name_or_path=bamboogle \
    +output_path=results/bamboogle \
    resume=true
```

## Configuration

WEMG uses Hydra and OmegaConf for flexible configuration management. The main configuration file is `wemg/config.yaml`.

### Key Configuration Sections

#### LLM Configuration
```yaml
llm:
  model_name: "gpt-4o"  # or "Qwen3-Next-80B-A3B-Thinking-FP8", etc.
  url: "https://api.openai.com/v1"
  api_key: null  # Set via API_KEY environment variable
  concurrency: 8  # Number of concurrent LLM calls
  generation:
    temperature: 0.8
    max_tokens: 32768
```

#### Search Strategy
```yaml
search:
  strategy: "mcts"  # or "cot"
  mcts:
    num_iterations: 30
    max_tree_depth: 10
    exploration_weight: 2.5
```

#### Retriever
```yaml
retriever:
  type: "web_search"  # or "corpus"
  web_search:
    api_key: null  # Set via SERPER_API_KEY environment variable
  corpus:
    corpus_path: "path/to/corpus"
    index_path: "path/to/index.faiss"
```

### Environment Variables

- `API_KEY`: LLM API key (required)
- `SERPER_API_KEY`: Serper API key for web search (optional)
- `REDIS_PASSWORD`: Redis password for caching (optional)

## Architecture

### Core Components

1. **WEMGSystem**: Central orchestrator for question answering
2. **Search Strategies**: MCTS and CoT reasoning
3. **Agents & Roles**: LLM agents with specialized roles
4. **Memory Systems**: Working memory and interaction memory
5. **Retrievers**: Web search and corpus-based retrieval

### Data Flow

```
Question → WEMGSystem.answer()
  ↓
[Strategy: MCTS or CoT]
  ↓
NodeGenerator.generate_answer()
  ↓
explore() → [Web Search + KB Retrieval]
  ↓
Extractor → WorkingMemory
  ↓
AnswerGenerator → Final Answer
```

## Supported Datasets

WEMG supports evaluation on multiple datasets:

- **Graph-based QA**: CWQ, WebQSP, QALD-10, HotpotQA-Adv, GrailQA
- **Text-based QA**: 2Wiki, HotpotQA, Musique, Bamboogle, Frames

## Evaluation Metrics

- **Acc**: Accuracy score (0-1) using Evaluator role
- **Sub-EM**: Substring Exact Match (case-insensitive)
- **Pass@k**: Pass rate at k attempts

## Development

### Running Tests

```bash
# Run all tests
pytest wemg/test/

# Run only fast tests (skip slow ones)
pytest wemg/test/ -m "not slow"

# Run only unit tests
pytest wemg/test/ -m "unit"

# Run integration tests
pytest wemg/test/ -m "integration"
```

## Performance

### Current Performance

- MCTS iteration: ~185s (5 iterations)
- LLM calls: ~20 per iteration
- Default concurrency: 8

### Optimization Tips

1. **Increase Concurrency**: Set `llm.concurrency` to 8-16 for better parallelization
2. **Enable Caching**: Ensure Redis cache is enabled in config
3. **Reduce Simulation Depth**: Lower `search.mcts.max_simulation_depth` for faster iterations
4. **Use Dataset-level Memory**: Set `memory.interaction_memory.scope=dataset` for shared memory


## Project Structure

```
wemg/
├── agents/          # LLM agents and roles
│   ├── roles/       # Specialized agent roles
│   └── tools/       # External tools (web search, Wikidata)
├── runners/         # Search strategies and memory
├── evaluation/      # Dataset evaluation
├── config.py        # Configuration management
├── main.py          # Main system interface
└── config.yaml      # Default configuration
```

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass
6. Submit a pull request

## License

MIT License - see LICENSE file for details

## Authors

- Hieu Man (hieum@uoregon.edu)

## Citation

If you use WEMG in your research, please cite:

```bibtex
@software{wemg2024,
  title={WEMG: When Embedding Model Meet Graph RAG},
  author={Man, Hieu},
  year={2024}
}
```

## Acknowledgments

- Built with [Hydra](https://hydra.cc/) for configuration management
- Uses [LiteLLM](https://github.com/BerriAI/litellm) for LLM abstraction
- Integrates with [Wikidata](https://www.wikidata.org/) for knowledge graph retrieval

## Support

For issues, questions, or contributions, please open an issue on GitHub.

