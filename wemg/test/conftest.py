"""
Pytest configuration and shared fixtures for WEMG tests.

Configure test environment variables before running:
- TEST_LLM_API_BASE: LLM API endpoint (default: http://n0142:4000/v1)
- TEST_LLM_API_KEY: API key for LLM
- TEST_LLM_MODEL: Model name (default: Qwen3-32B)
- TEST_EMBEDDING_API_BASE: Embedding API endpoint
- TEST_EMBEDDING_MODEL: Embedding model name
"""
import os
import pytest
from typing import Generator

from wemg.agents.base_llm_agent import BaseLLMAgent
from wemg.agents.tools.web_search import WebSearchTool, DDGSAPIWrapper
from wemg.runners.working_memory import WorkingMemory
from wemg.runners.interaction_memory import InteractionMemory


# ============================================================================
# Test Configuration
# ============================================================================

TEST_CONFIG = {
    'llm_api_base': os.getenv("TEST_LLM_API_BASE", "http://n0142:4000/v1"),
    'llm_api_key': os.getenv("TEST_LLM_API_KEY", "sk-your-very-secure-master-key-here"),
    'llm_model': os.getenv("TEST_LLM_MODEL", "Qwen3-32B"),
    'embedding_api_base': os.getenv("TEST_EMBEDDING_API_BASE", "http://n0372:4000/v1"),
    'embedding_model': os.getenv("TEST_EMBEDDING_MODEL", "Qwen3-Embedding-4B"),
}


# ============================================================================
# Pytest Configuration
# ============================================================================

def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )


def pytest_collection_modifyitems(config, items):
    """Add markers based on test location."""
    for item in items:
        # Mark integration tests
        if "integration" in item.nodeid.lower() or "Integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        
        # Auto-mark tests that call external services as slow
        if any(keyword in item.nodeid for keyword in ["mcts_search", "cot_search", "ainvoke"]):
            item.add_marker(pytest.mark.slow)


# ============================================================================
# Shared Fixtures
# ============================================================================

@pytest.fixture(scope="session")
def test_config():
    """Provide test configuration."""
    return TEST_CONFIG


@pytest.fixture(scope="session")
def llm_agent_session():
    """Create a session-scoped LLM agent (reused across tests)."""
    return BaseLLMAgent(
        model_name=TEST_CONFIG['llm_model'],
        url=TEST_CONFIG['llm_api_base'],
        api_key=TEST_CONFIG['llm_api_key'],
        temperature=0.7,
        max_tokens=4096,
        concurrency=4,
        max_retries=3
    )


@pytest.fixture
def llm_agent():
    """Create a function-scoped LLM agent."""
    return BaseLLMAgent(
        model_name=TEST_CONFIG['llm_model'],
        url=TEST_CONFIG['llm_api_base'],
        api_key=TEST_CONFIG['llm_api_key'],
        temperature=0.7,
        max_tokens=4096,
        concurrency=2,
        max_retries=3
    )


@pytest.fixture
def web_search_tool():
    """Create a WebSearchTool with DuckDuckGo backend."""
    return WebSearchTool(
        api_wrapper=DDGSAPIWrapper(),
        max_tokens=8192
    )


@pytest.fixture
def working_memory():
    """Create an empty WorkingMemory."""
    return WorkingMemory()


@pytest.fixture
def interaction_memory():
    """Create an InteractionMemory instance."""
    return InteractionMemory()


@pytest.fixture
def sample_question():
    """Provide a sample question for testing."""
    return "What is the capital of France?"


@pytest.fixture
def sample_context():
    """Provide sample context for testing."""
    return """France is a country in Western Europe. Paris is the capital and 
    most populous city of France. The city has a population of about 2.1 million 
    in the city proper and over 12 million in the metropolitan area."""


@pytest.fixture
def multi_hop_question():
    """Provide a multi-hop question for testing."""
    return "Who was the president of the United States when the first iPhone was released?"


# ============================================================================
# Test Utilities
# ============================================================================

class TestHelpers:
    """Helper methods for tests."""
    
    @staticmethod
    def assert_valid_answer(answer: str, min_length: int = 10):
        """Assert that an answer is valid."""
        assert answer is not None
        assert len(answer) >= min_length
        assert answer.strip() != ""
    
    @staticmethod
    def assert_node_valid(node):
        """Assert that a reasoning node is valid."""
        assert node is not None
        assert node.node_state is not None
        assert node.node_state.node_type is not None
    
    @staticmethod
    def print_reasoning_path(path):
        """Print a reasoning path for debugging."""
        print("\n=== Reasoning Path ===")
        for i, node in enumerate(path):
            node_type = node.node_state.node_type.name
            content = node.node_state.content
            print(f"Step {i}: {node_type}")
            if 'sub_question' in content:
                print(f"  Q: {content['sub_question']}")
            if 'sub_answer' in content:
                print(f"  A: {content['sub_answer'][:100]}...")
            if 'final_answer' in content:
                print(f"  Final: {content['final_answer'][:100]}...")
        print("=" * 50)


@pytest.fixture
def test_helpers():
    """Provide test helper methods."""
    return TestHelpers()


# ============================================================================
# Skip Conditions
# ============================================================================

def skip_if_no_llm():
    """Skip test if LLM is not available."""
    import socket
    try:
        host = TEST_CONFIG['llm_api_base'].replace('http://', '').replace('https://', '').split(':')[0]
        port = int(TEST_CONFIG['llm_api_base'].split(':')[-1].replace('/v1', ''))
        socket.create_connection((host, port), timeout=2)
        return False
    except (socket.timeout, socket.error, OSError):
        return True


skip_no_llm = pytest.mark.skipif(
    skip_if_no_llm(),
    reason="LLM server not available"
)
