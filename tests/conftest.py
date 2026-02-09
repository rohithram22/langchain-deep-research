"""
Pytest fixtures and configuration for tests.
"""

import os
import pytest
from unittest.mock import MagicMock, patch
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()


# Check for API keys
HAS_OPENAI_KEY = bool(os.getenv("OPENAI_API_KEY"))
HAS_TAVILY_KEY = bool(os.getenv("TAVILY_API_KEY"))
HAS_ALL_KEYS = HAS_OPENAI_KEY and HAS_TAVILY_KEY


@pytest.fixture
def sample_query():
    """Sample research query for testing."""
    return "What are the benefits of meditation?"


@pytest.fixture
def sample_search_results():
    """Mock search results for testing."""
    return [
        {
            "title": "Benefits of Meditation - Healthline",
            "url": "https://www.healthline.com/meditation-benefits",
            "content": "Meditation has been shown to reduce stress, improve focus, and enhance emotional well-being. Studies indicate regular practice can lower cortisol levels and improve sleep quality."
        },
        {
            "title": "Scientific Research on Meditation - NIH",
            "url": "https://www.nih.gov/meditation-research",
            "content": "Research demonstrates that meditation can help with anxiety, depression, and chronic pain management. Brain imaging studies show changes in neural pathways associated with attention and emotion regulation."
        },
        {
            "title": "How Meditation Affects the Brain - Psychology Today",
            "url": "https://www.psychologytoday.com/meditation-brain",
            "content": "Meditation increases gray matter density in areas associated with learning, memory, and emotional regulation. It also reduces activity in the amygdala, the brain's stress center."
        },
        {
            "title": "Meditation for Beginners - Mayo Clinic",
            "url": "https://www.mayoclinic.org/meditation-guide",
            "content": "Meditation is a simple practice that can be done anywhere. Start with just 5 minutes a day and gradually increase. Focus on your breath and let thoughts pass without judgment."
        },
        {
            "title": "Types of Meditation - Verywell Mind",
            "url": "https://www.verywellmind.com/meditation-types",
            "content": "There are many types of meditation including mindfulness, transcendental, loving-kindness, and body scan meditation. Each type offers unique benefits and suits different preferences."
        }
    ]


@pytest.fixture
def mock_llm_response():
    """Mock LLM response object."""
    mock = MagicMock()
    mock.content = "meditation health benefits research"
    return mock


@pytest.fixture
def mock_tavily_response(sample_search_results):
    """Mock Tavily API response."""
    return {"results": sample_search_results}


@pytest.fixture
def agent_config():
    """Test configuration with minimal iterations."""
    from src.agent.config import AgentConfig
    return AgentConfig(
        model_name="gpt-4o-mini",
        max_iterations=2,
        max_search_results=3,
        temperature=0
    )


@pytest.fixture
def sample_state(sample_query):
    """Sample initial research state."""
    return {
        "messages": [HumanMessage(content=sample_query)],
        "topic": "",
        "running_summary": "",
        "sources": [],
        "search_results": [],
        "current_query": "",
        "iteration": 0,
        "max_iterations": 3
    }


@pytest.fixture
def sample_state_with_topic(sample_query):
    """Sample state with topic already extracted."""
    return {
        "messages": [HumanMessage(content=sample_query)],
        "topic": sample_query,
        "running_summary": "",
        "sources": [],
        "search_results": [],
        "current_query": "",
        "iteration": 0,
        "max_iterations": 3
    }


@pytest.fixture
def sample_state_with_results(sample_query, sample_search_results):
    """Sample state with search results ready for summarization."""
    return {
        "messages": [HumanMessage(content=sample_query)],
        "topic": sample_query,
        "running_summary": "",
        "sources": sample_search_results,
        "search_results": sample_search_results,
        "current_query": "meditation health benefits",
        "iteration": 0,
        "max_iterations": 3
    }


@pytest.fixture
def sample_state_with_summary(sample_query, sample_search_results):
    """Sample state with existing research summary."""
    return {
        "messages": [HumanMessage(content=sample_query)],
        "topic": sample_query,
        "running_summary": """
        Meditation has been extensively studied and shown to provide numerous health benefits.
        
        Key findings include:
        - Reduced stress and anxiety levels through lower cortisol production
        - Improved focus and concentration abilities
        - Enhanced emotional regulation and well-being
        - Better sleep quality and reduced insomnia
        - Changes in brain structure, including increased gray matter density
        
        Research from NIH and other institutions confirms these benefits through brain imaging studies.
        [Source: https://www.nih.gov/meditation-research]
        [Source: https://www.healthline.com/meditation-benefits]
        """,
        "sources": sample_search_results[:3],
        "search_results": [],
        "current_query": "meditation mental health research",
        "iteration": 2,
        "max_iterations": 5
    }


@pytest.fixture
def mock_openai_client():
    """Mock OpenAI ChatCompletion client."""
    mock = MagicMock()
    mock_response = MagicMock()
    mock_response.content = "Test response"
    mock.invoke.return_value = mock_response
    return mock


@pytest.fixture
def mock_tavily_client(mock_tavily_response):
    """Mock Tavily client."""
    mock = MagicMock()
    mock.search.return_value = mock_tavily_response
    return mock