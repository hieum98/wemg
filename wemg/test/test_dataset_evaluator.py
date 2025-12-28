"""
Comprehensive tests for the dataset_evaluator module.

Tests cover:
- Helper functions (check_answer_correctness, compute_sub_em)
- DatasetEvaluator class with real WEMGSystem
- Evaluation with sequential and parallel processing
- Log-based resume functionality
- Metrics computation

These are integration tests that use real LLM servers.
"""
import os
import pytest
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

import datasets
from wemg.evaluation.dataset_evaluator import (
    check_answer_correctness,
    compute_sub_em,
    compute_acc,
    compute_acc_batch,
    DatasetEvaluator
)
from wemg.main import WEMGSystem, AnswerResult
from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents.retriever_agent import RetrieverAgent


# ============================================================================
# Test Configuration
# ============================================================================

TEST_LLM_API_BASE = os.getenv("TEST_LLM_API_BASE", "http://n0999:4000/v1")
TEST_LLM_API_KEY = os.getenv("TEST_LLM_API_KEY", "sk-your-very-secure-master-key-here")
TEST_LLM_MODEL = os.getenv("TEST_LLM_MODEL", "Qwen3-Next-80B-A3B-Thinking-FP8")

TEST_EMBEDDING_API_BASE = os.getenv("TEST_EMBEDDING_API_BASE", "http://n0999:4000/v1")
TEST_EMBEDDING_MODEL = os.getenv("TEST_EMBEDDING_MODEL", "Qwen3-Embedding-4B")

# Wiki corpus configuration for RetrieverAgent tests
WIKI_CORPUS_HF = os.getenv("WIKI_CORPUS_HF", "Hieuman/wiki23-processed")
WIKI_INDEX_PATH = Path(os.getenv("WIKI_INDEX_PATH", "retriever_corpora/Qwen3-4B-Emb-index.faiss"))


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture
def llm_agent():
    """Create a BaseLLMAgent for testing."""
    return BaseLLMAgent(
        model_name=TEST_LLM_MODEL,
        url=TEST_LLM_API_BASE,
        api_key=TEST_LLM_API_KEY,
        temperature=0.7,
        max_tokens=16384,
        concurrency=32,
        max_retries=3
    )


@pytest.fixture
def retriever_agent_embedder_config():
    """Create embedder configuration for RetrieverAgent."""
    return {
        'model_name': TEST_EMBEDDING_MODEL,
        'url': TEST_EMBEDDING_API_BASE,
        'api_key': TEST_LLM_API_KEY,
        'is_embedding': True,
        'timeout': 60,
    }


@pytest.fixture
def retriever_agent(retriever_agent_embedder_config):
    """Create a RetrieverAgent instance with wiki corpus and pre-indexed FAISS."""
    # Check if index exists
    if not WIKI_INDEX_PATH.exists():
        pytest.skip(f"Wiki index not found at {WIKI_INDEX_PATH}. Please ensure the index is available.")
    
    # Try to load corpus from local directory first (if it was previously saved)
    # Otherwise, load from HuggingFace
    local_corpus_path = WIKI_INDEX_PATH.parent / "without_embeddings"
    if local_corpus_path.exists():
        corpus_path = local_corpus_path
    else:
        # Load from HuggingFace - will be saved locally after first load
        corpus_path = Path(WIKI_CORPUS_HF)
    
    agent = RetrieverAgent(
        embedder_config=retriever_agent_embedder_config,
        corpus_path=corpus_path,
        index_path=WIKI_INDEX_PATH,
        embedder_type='openai'
    )
    
    return agent


@pytest.fixture
def wemg_system():
    """Create a WEMGSystem instance for testing with corpus retriever."""
    # Check if index exists, skip if not
    if not WIKI_INDEX_PATH.exists():
        pytest.skip(f"Wiki index not found at {WIKI_INDEX_PATH}. Please ensure the index is available.")
    
    # Determine corpus path
    local_corpus_path = WIKI_INDEX_PATH.parent / "without_embeddings"
    if local_corpus_path.exists():
        corpus_path = str(local_corpus_path)
    else:
        corpus_path = WIKI_CORPUS_HF
    
    config_dict = {
        'llm': {
            'model_name': TEST_LLM_MODEL,
            'url': TEST_LLM_API_BASE,
            'api_key': TEST_LLM_API_KEY,
            'temperature': 0.7,
            'max_tokens': 4096,
            'concurrency': 4
        },
        'search': {
            'strategy': 'cot'
        },
        'retriever': {
            'type': 'corpus',
            'corpus': {
                'embedder': {
                    'model_name': TEST_EMBEDDING_MODEL,
                    'url': TEST_EMBEDDING_API_BASE,
                    'api_key': TEST_LLM_API_KEY,
                    'embedder_type': 'openai'
                },
                'corpus_path': corpus_path,
                'index_path': str(WIKI_INDEX_PATH)
            }
        }
    }
    return WEMGSystem(config_dict=config_dict)


@pytest.fixture
def simple_question():
    """Simple factual question."""
    return "What is the capital of France?"


@pytest.fixture
def multi_hop_question():
    """Multi-hop question requiring multiple reasoning steps."""
    return "Who was the president of the United States when the first iPhone was released?"


@pytest.fixture
def comparison_question():
    """Question requiring comparison."""
    return "Which magazine was started first: Arthur's Magazine or First for Women?"


@pytest.fixture
def temporal_question():
    """Question requiring temporal reasoning."""
    return "What was the capital of the country that won the most Olympic gold medals in 2016?"


@pytest.fixture
def scientific_question():
    """Question requiring scientific knowledge."""
    return "What was the name of the scientist who discovered the structure of DNA, and in which year did they receive the Nobel Prize?"


@pytest.fixture
def historical_question():
    """Question requiring historical knowledge."""
    return "What was the name of the battle that took place in 1066 and changed the course of English history?"


@pytest.fixture
def geographical_question():
    """Question requiring geographical knowledge."""
    return "What is the longest river in Africa?"


@pytest.fixture
def sample_dataset():
    """Create a sample HuggingFace dataset with various question types."""
    return datasets.Dataset.from_dict({
        'id': ['q1', 'q2', 'q3'],
        'question': [
            'What is the capital of France?',
            'What is the capital of UK?',
            'What is the capital of Germany?'
        ],
        'answer': ['Paris', 'London', 'Berlin']
    })


@pytest.fixture
def diverse_dataset(simple_question, multi_hop_question, comparison_question, temporal_question):
    """Create a diverse dataset with various question types."""
    return datasets.Dataset.from_dict({
        'id': ['q1', 'q2', 'q3', 'q4'],
        'question': [
            simple_question,
            multi_hop_question,
            comparison_question,
            temporal_question
        ],
        'answer': [
            'Paris',
            'George W. Bush',
            "Arthur's Magazine",
            'Washington, D.C.'
        ]
    })


@pytest.fixture
def small_dataset():
    """Create a small dataset for quick tests."""
    return datasets.Dataset.from_dict({
        'id': ['q1'],
        'question': ['What is the capital of France?'],
        'answer': ['Paris']
    })


# ============================================================================
# Test Helper Functions (Unit Tests - No Real Systems Needed)
# ============================================================================

class TestHelperFunctions:
    """Test helper functions for answer checking and metrics."""
    
    def test_check_answer_correctness_single_string_match(self):
        """Test check_answer_correctness with single correct answer (match)."""
        assert check_answer_correctness("The capital is Paris", "Paris") is True
        assert check_answer_correctness("Paris is the capital", "Paris") is True
    
    def test_check_answer_correctness_single_string_no_match(self):
        """Test check_answer_correctness with single correct answer (no match)."""
        assert check_answer_correctness("The capital is London", "Paris") is False
        assert check_answer_correctness("", "Paris") is False
        assert check_answer_correctness("Paris", "") is False
        assert check_answer_correctness("", "") is False
    
    def test_check_answer_correctness_list_match(self):
        """Test check_answer_correctness with list of correct answers."""
        assert check_answer_correctness("The capital is Paris", ["Paris", "Lyon"]) is True
        assert check_answer_correctness("Lyon is a city", ["Paris", "Lyon"]) is True
        assert check_answer_correctness("Neither city", ["Paris", "Lyon"]) is False
    
    def test_check_answer_correctness_case_insensitive(self):
        """Test check_answer_correctness is case-insensitive."""
        assert check_answer_correctness("The capital is PARIS", "paris") is True
        assert check_answer_correctness("The capital is paris", "PARIS") is True
        assert check_answer_correctness("The capital is Paris", "pArIs") is True
    
    def test_compute_sub_em(self):
        """Test compute_sub_em function."""
        assert compute_sub_em("The capital is Paris", "Paris") is True
        assert compute_sub_em("The capital is London", "Paris") is False
        assert compute_sub_em("Paris or Lyon", ["Paris", "Lyon"]) is True


# ============================================================================
# Test DatasetEvaluator - Unit Tests (No Real Systems)
# ============================================================================

class TestDatasetEvaluatorUnit:
    """Unit tests for DatasetEvaluator class methods that don't require real systems."""
    
    @pytest.fixture
    def evaluator(self, wemg_system):
        """Create a DatasetEvaluator instance."""
        return DatasetEvaluator(wemg_system)
    
    def test_normalize_answer_field_string(self, evaluator):
        """Test _normalize_answer_field with string input."""
        assert evaluator._normalize_answer_field("Paris") == ["Paris"]
        assert evaluator._normalize_answer_field("") == [""]
    
    def test_normalize_answer_field_list(self, evaluator):
        """Test _normalize_answer_field with list input."""
        assert evaluator._normalize_answer_field(["Paris", "Lyon"]) == ["Paris", "Lyon"]
        assert evaluator._normalize_answer_field(["Paris", None, "Lyon"]) == ["Paris", "Lyon"]
        assert evaluator._normalize_answer_field([]) == []
    
    def test_normalize_answer_field_none(self, evaluator):
        """Test _normalize_answer_field with None input."""
        assert evaluator._normalize_answer_field(None) == []
    
    def test_normalize_answer_field_other(self, evaluator):
        """Test _normalize_answer_field with other types."""
        assert evaluator._normalize_answer_field(123) == ["123"]
        assert evaluator._normalize_answer_field(True) == ["True"]
    
    def test_create_empty_result(self, evaluator):
        """Test _create_empty_result method."""
        result = evaluator._create_empty_result()
        assert result == {
            'predicted_answer': None,
            'concise_answer': None,
            'acc_score': None,
            'sub_em': None,
            'pass_at_k': None,
            'error': None,
            'metadata': None
        }
    
    def test_get_log_path(self, evaluator):
        """Test _get_log_path method."""
        output_path = Path("/tmp/test_output")
        log_path = evaluator._get_log_path(output_path)
        assert log_path == Path("/tmp/test_output/evaluation_log.jsonl")
    
    def test_load_logged_results_empty(self, evaluator):
        """Test _load_logged_results with non-existent file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "nonexistent.jsonl"
            results = evaluator._load_logged_results(log_path)
            assert results == {}
    
    def test_load_logged_results_valid(self, evaluator):
        """Test _load_logged_results with valid log file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "evaluation_log.jsonl"
            log_path.write_text(json.dumps({
                'question_index': 0,
                'predicted_answer': 'Paris',
                'acc_score': 0.9
            }) + '\n')
            log_path.write_text(log_path.read_text() + json.dumps({
                'question_index': 1,
                'predicted_answer': 'London',
                'acc_score': 0.8
            }) + '\n')
            
            results = evaluator._load_logged_results(log_path)
            assert len(results) == 2
            assert results[0]['predicted_answer'] == 'Paris'
            assert results[1]['predicted_answer'] == 'London'
    
    def test_load_logged_results_duplicate(self, evaluator):
        """Test _load_logged_results handles duplicate entries (uses latest)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "evaluation_log.jsonl"
            log_path.write_text(json.dumps({
                'question_index': 0,
                'acc_score': 0.5
            }) + '\n')
            log_path.write_text(log_path.read_text() + json.dumps({
                'question_index': 0,
                'acc_score': 0.9  # Updated score
            }) + '\n')
            
            results = evaluator._load_logged_results(log_path)
            assert len(results) == 1
            assert results[0]['acc_score'] == 0.9  # Latest entry
    
    def test_log_result(self, evaluator):
        """Test _log_result method."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_path = Path(tmpdir) / "evaluation_log.jsonl"
            result = {'predicted_answer': 'Paris', 'acc_score': 0.9}
            
            evaluator._log_result(0, result, log_path)
            
            assert log_path.exists()
            logged = json.loads(log_path.read_text())
            assert logged['question_index'] == 0
            assert logged['predicted_answer'] == 'Paris'
            assert logged['acc_score'] == 0.9


# ============================================================================
# Test DatasetEvaluator - Integration Tests (Real Systems)
# ============================================================================

class TestDatasetEvaluatorIntegration:
    """Integration tests for DatasetEvaluator with real WEMGSystem."""
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_process_single_question_success(self, wemg_system):
        """Test _process_single_question with successful processing."""
        evaluator = DatasetEvaluator(wemg_system)
        
        result = evaluator._process_single_question(
            question="What is the capital of France?",
            correct_answers="Paris",
            question_id="test_1",
            compute_acc_now=False  # Skip Acc for faster test
        )
        
        assert result['predicted_answer'] is not None
        assert result['concise_answer'] is not None
        assert result['error'] is None
        assert result['sub_em'] is True  # Should match "Paris"
        
        print(f"✓ process_single_question success")
        print(f"  Predicted: {result['predicted_answer'][:100]}")
        print(f"  Concise: {result['concise_answer']}")
        print(f"  Sub-EM: {result['sub_em']}")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_process_single_question_no_answers(self, wemg_system):
        """Test _process_single_question with no correct answers."""
        evaluator = DatasetEvaluator(wemg_system)
        
        result = evaluator._process_single_question(
            question="What is the capital?",
            correct_answers=None
        )
        
        assert result['error'] == "No correct answers provided"
        assert result['predicted_answer'] is None
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_evaluate_sequential(self, wemg_system, small_dataset):
        """Test evaluate method with sequential processing (batch_size=1)."""
        evaluator = DatasetEvaluator(wemg_system)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results"
            result_dataset = evaluator.evaluate(
                dataset=small_dataset,
                output_path=output_path,
                resume=False,
                batch_size=1
            )
            
            assert len(result_dataset) == 1
            assert result_dataset[0]['predicted_answer'] is not None
            assert result_dataset[0]['concise_answer'] is not None
            assert result_dataset[0]['sub_em'] is True  # Should match "Paris"
            
            # Check log file exists
            log_path = output_path / "evaluation_log.jsonl"
            assert log_path.exists()
            
            print(f"✓ evaluate sequential")
            print(f"  Predicted: {result_dataset[0]['predicted_answer'][:100]}")
            print(f"  Concise: {result_dataset[0]['concise_answer']}")
            print(f"  Sub-EM: {result_dataset[0]['sub_em']}")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_evaluate_parallel(self, wemg_system, sample_dataset):
        """Test evaluate method with parallel processing."""
        evaluator = DatasetEvaluator(wemg_system)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results"
            result_dataset = evaluator.evaluate(
                dataset=sample_dataset,
                output_path=output_path,
                resume=False,
                batch_size=2,
                max_workers=2
            )
            
            assert len(result_dataset) == 3
            assert all(r['predicted_answer'] is not None for r in result_dataset)
            assert all(r['concise_answer'] is not None for r in result_dataset)
            
            print(f"✓ evaluate parallel")
            for i, result in enumerate(result_dataset):
                print(f"  Q{i+1}: {result['concise_answer']} (Sub-EM: {result['sub_em']})")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_evaluate_diverse_questions(self, wemg_system, diverse_dataset):
        """Test evaluate method with diverse question types."""
        evaluator = DatasetEvaluator(wemg_system)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results"
            result_dataset = evaluator.evaluate(
                dataset=diverse_dataset,
                output_path=output_path,
                resume=False,
                batch_size=1
            )
            
            assert len(result_dataset) == 4
            assert all(r['predicted_answer'] is not None for r in result_dataset)
            
            print(f"✓ evaluate diverse questions")
            for i, result in enumerate(result_dataset):
                question = diverse_dataset[i]['question']
                print(f"  Q{i+1} ({question[:50]}...): {result['concise_answer']} (Sub-EM: {result['sub_em']})")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_process_simple_question(self, wemg_system, simple_question):
        """Test processing a simple factual question."""
        evaluator = DatasetEvaluator(wemg_system)
        
        result = evaluator._process_single_question(
            question=simple_question,
            correct_answers="Paris",
            question_id="test_simple",
            compute_acc_now=False
        )
        
        assert result['predicted_answer'] is not None
        assert result['sub_em'] is True
        assert result['error'] is None
        
        print(f"✓ process simple question")
        print(f"  Question: {simple_question}")
        print(f"  Answer: {result['concise_answer']}")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_process_multi_hop_question(self, wemg_system, multi_hop_question):
        """Test processing a multi-hop question."""
        evaluator = DatasetEvaluator(wemg_system)
        
        result = evaluator._process_single_question(
            question=multi_hop_question,
            correct_answers="George W. Bush",
            question_id="test_multi_hop",
            compute_acc_now=False
        )
        
        assert result['predicted_answer'] is not None
        assert result['error'] is None
        
        print(f"✓ process multi-hop question")
        print(f"  Question: {multi_hop_question}")
        print(f"  Answer: {result['concise_answer']}")
        print(f"  Sub-EM: {result['sub_em']}")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_process_comparison_question(self, wemg_system, comparison_question):
        """Test processing a comparison question."""
        evaluator = DatasetEvaluator(wemg_system)
        
        result = evaluator._process_single_question(
            question=comparison_question,
            correct_answers="Arthur's Magazine",
            question_id="test_comparison",
            compute_acc_now=False
        )
        
        assert result['predicted_answer'] is not None
        assert result['error'] is None
        
        print(f"✓ process comparison question")
        print(f"  Question: {comparison_question}")
        print(f"  Answer: {result['concise_answer']}")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_process_temporal_question(self, wemg_system, temporal_question):
        """Test processing a temporal reasoning question."""
        evaluator = DatasetEvaluator(wemg_system)
        
        result = evaluator._process_single_question(
            question=temporal_question,
            correct_answers=["Washington, D.C.", "Washington DC", "Washington"],
            question_id="test_temporal",
            compute_acc_now=False
        )
        
        assert result['predicted_answer'] is not None
        assert result['error'] is None
        
        print(f"✓ process temporal question")
        print(f"  Question: {temporal_question}")
        print(f"  Answer: {result['concise_answer']}")
        print(f"  Sub-EM: {result['sub_em']}")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_process_scientific_question(self, wemg_system, scientific_question):
        """Test processing a scientific question."""
        evaluator = DatasetEvaluator(wemg_system)
        
        result = evaluator._process_single_question(
            question=scientific_question,
            correct_answers=["Watson and Crick", "James Watson and Francis Crick"],
            question_id="test_scientific",
            compute_acc_now=False
        )
        
        assert result['predicted_answer'] is not None
        assert result['error'] is None
        
        print(f"✓ process scientific question")
        print(f"  Question: {scientific_question}")
        print(f"  Answer: {result['concise_answer']}")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_process_geographical_question(self, wemg_system, geographical_question):
        """Test processing a geographical question."""
        evaluator = DatasetEvaluator(wemg_system)
        
        result = evaluator._process_single_question(
            question=geographical_question,
            correct_answers=["Nile", "Nile River"],
            question_id="test_geographical",
            compute_acc_now=False
        )
        
        assert result['predicted_answer'] is not None
        assert result['error'] is None
        
        print(f"✓ process geographical question")
        print(f"  Question: {geographical_question}")
        print(f"  Answer: {result['concise_answer']}")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_evaluate_resume(self, wemg_system, sample_dataset):
        """Test evaluate method with resume functionality."""
        evaluator = DatasetEvaluator(wemg_system)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "results"
            output_path.mkdir()
            log_path = output_path / "evaluation_log.jsonl"
            
            # Create initial log file with first question completed
            log_path.write_text(json.dumps({
                'question_index': 0,
                'predicted_answer': 'The capital of France is Paris',
                'concise_answer': 'Paris',
                'acc_score': None,  # Not computed yet
                'sub_em': True,
                'pass_at_k': None,
                'error': None,
                'id': 'q1',
                'question': 'What is the capital of France?',
                'answer': 'Paris'
            }) + '\n')
            
            # Run evaluation with resume
            result_dataset = evaluator.evaluate(
                dataset=sample_dataset,
                output_path=output_path,
                resume=True,
                batch_size=1
            )
            
            # Should have processed all 3 questions
            assert len(result_dataset) == 3
            # First question should be from log
            assert result_dataset[0]['predicted_answer'] == 'The capital of France is Paris'
            # Remaining questions should be processed
            assert result_dataset[1]['predicted_answer'] is not None
            assert result_dataset[2]['predicted_answer'] is not None
            
            print(f"✓ evaluate resume")
            print(f"  Q1 (from log): {result_dataset[0]['concise_answer']}")
            print(f"  Q2 (processed): {result_dataset[1]['concise_answer']}")
            print(f"  Q3 (processed): {result_dataset[2]['concise_answer']}")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_evaluate_column_validation(self, wemg_system, sample_dataset):
        """Test evaluate method validates columns exist."""
        evaluator = DatasetEvaluator(wemg_system)
        
        with pytest.raises(ValueError, match="Column 'wrong_column' not found"):
            evaluator.evaluate(
                dataset=sample_dataset,
                question_column="wrong_column",
                resume=False
            )
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_compute_acc(self, wemg_system, llm_agent):
        """Test compute_acc function with real LLM."""
        score = compute_acc(
            question="What is the capital of France?",
            predicted_answer="The capital of France is Paris",
            correct_answers="Paris",
            llm_agent=llm_agent
        )
        
        assert 0.0 <= score <= 1.0
        
        print(f"✓ compute_acc")
        print(f"  Score: {score:.2f}")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_compute_acc_batch(self, wemg_system, llm_agent):
        """Test compute_acc_batch function with real LLM."""
        acc_tasks = [
            {
                'question': 'What is the capital of France?',
                'predicted_answer': 'The capital of France is Paris',
                'correct_answers': ['Paris']
            },
            {
                'question': 'What is the capital of UK?',
                'predicted_answer': 'The capital of UK is London',
                'correct_answers': ['London']
            }
        ]
        
        scores = compute_acc_batch(acc_tasks, llm_agent, max_workers=2)
        
        assert len(scores) == 2
        assert all(0.0 <= score <= 1.0 for score in scores)
        
        print(f"✓ compute_acc_batch")
        print(f"  Scores: {scores}")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_compute_aggregate_metrics(self, wemg_system):
        """Test compute_aggregate_metrics method."""
        result_dataset = datasets.Dataset.from_list([
            {
                'predicted_answer': 'Paris',
                'acc_score': 0.9,
                'sub_em': True,
                'pass_at_k': 1,
                'error': None
            },
            {
                'predicted_answer': 'London',
                'acc_score': 0.8,
                'sub_em': True,
                'pass_at_k': 2,
                'error': None
            },
            {
                'predicted_answer': None,
                'acc_score': None,
                'sub_em': None,
                'pass_at_k': None,
                'error': 'Test error'
            }
        ])
        
        evaluator = DatasetEvaluator(wemg_system)
        metrics = evaluator.compute_aggregate_metrics(result_dataset, max_k=5)
        
        assert metrics['mean_acc'] == 0.85  # (0.9 + 0.8) / 2
        assert metrics['mean_sub_em'] == 1.0  # Both valid results have sub_em=True
        assert metrics['total_questions'] == 3
        assert metrics['valid_questions'] == 2
        assert metrics['error_questions'] == 1
        assert 'pass_at_1' in metrics
        assert 'pass_at_2' in metrics
        assert metrics['pass_at_1'] == 0.5  # 1 out of 2 valid questions
        assert metrics['pass_at_2'] == 1.0  # 2 out of 2 valid questions
        
        print(f"✓ compute_aggregate_metrics")
        print(f"  Mean Acc: {metrics['mean_acc']:.2f}")
        print(f"  Mean Sub-EM: {metrics['mean_sub_em']:.2f}")
        print(f"  Valid: {metrics['valid_questions']}/{metrics['total_questions']}")
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_compute_aggregate_metrics_no_valid(self, wemg_system):
        """Test compute_aggregate_metrics with no valid results."""
        result_dataset = datasets.Dataset.from_list([
            {'predicted_answer': None, 'error': 'Error 1'},
            {'predicted_answer': None, 'error': 'Error 2'}
        ])
        
        evaluator = DatasetEvaluator(wemg_system)
        metrics = evaluator.compute_aggregate_metrics(result_dataset)
        
        assert metrics['mean_acc'] == 0.0
        assert metrics['mean_sub_em'] == 0.0
        assert metrics['valid_questions'] == 0
        assert metrics['error_questions'] == 2
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_save_metrics(self, wemg_system):
        """Test save_metrics method."""
        metrics = {
            'mean_acc': 0.85,
            'mean_sub_em': 0.9,
            'total_questions': 10,
            'valid_questions': 9,
            'error_questions': 1,
            'pass_at_1': 0.5,
            'pass_at_2': 0.7
        }
        
        evaluator = DatasetEvaluator(wemg_system)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metrics.json"
            evaluator.save_metrics(metrics, output_path, save_summary=True)
            
            assert output_path.exists()
            assert output_path.with_suffix('.txt').exists()
            
            # Check JSON content
            with open(output_path) as f:
                saved_metrics = json.load(f)
            assert saved_metrics == metrics
            
            # Check summary content
            summary_path = output_path.with_suffix('.txt')
            summary_text = summary_path.read_text()
            assert "Dataset Evaluation Metrics Summary" in summary_text
            assert "Mean Acc: 0.8500" in summary_text
            
            print(f"✓ save_metrics")
            print(f"  Saved to: {output_path}")


# ============================================================================
# Test Edge Cases
# ============================================================================

class TestEdgeCases:
    """Test edge cases and error handling."""
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_evaluate_empty_dataset(self, wemg_system):
        """Test evaluate with empty dataset."""
        empty_dataset = datasets.Dataset.from_dict({
            'question': [],
            'answer': []
        })
        
        evaluator = DatasetEvaluator(wemg_system)
        result_dataset = evaluator.evaluate(empty_dataset, resume=False)
        
        assert len(result_dataset) == 0
    
    @pytest.mark.slow
    @pytest.mark.integration
    def test_evaluate_missing_answer_column(self, wemg_system):
        """Test evaluate handles missing answer gracefully."""
        dataset = datasets.Dataset.from_dict({
            'question': ['What is the capital?'],
            'answer': [None]
        })
        
        evaluator = DatasetEvaluator(wemg_system)
        
        with tempfile.TemporaryDirectory() as tmpdir:
            result_dataset = evaluator.evaluate(
                dataset,
                output_path=Path(tmpdir) / "results",
                resume=False,
                batch_size=1
            )
            
            # Should handle None answer
            assert len(result_dataset) == 1
            assert result_dataset[0].get('error') is not None or result_dataset[0].get('predicted_answer') is not None
    
    def test_compute_acc_batch_empty(self, llm_agent):
        """Test compute_acc_batch with empty task list."""
        scores = compute_acc_batch([], llm_agent)
        assert scores == []
