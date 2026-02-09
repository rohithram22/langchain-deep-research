"""
Integration tests for the Deep Research Agent.

These tests require valid API keys and make real API calls.
Run with: pytest tests/test_integration.py -v

To skip these tests: pytest -m "not integration"
"""

import os
import pytest
from dotenv import load_dotenv

load_dotenv()

# Check if API keys are available
HAS_OPENAI_KEY = bool(os.getenv("OPENAI_API_KEY"))
HAS_TAVILY_KEY = bool(os.getenv("TAVILY_API_KEY"))
HAS_ALL_KEYS = HAS_OPENAI_KEY and HAS_TAVILY_KEY

# Skip reason message
SKIP_REASON = "API keys not available (OPENAI_API_KEY and TAVILY_API_KEY required)"


@pytest.mark.integration
@pytest.mark.skipif(not HAS_ALL_KEYS, reason=SKIP_REASON)
class TestEndToEndResearch:
    """End-to-end tests with real API calls."""
    
    def test_simple_query_produces_report(self):
        """Test that a simple query produces a report."""
        from src.agent import run_research, get_report, AgentConfig
        
        config = AgentConfig(
            max_iterations=2,
            max_search_results=3
        )
        
        result = run_research("What is machine learning?", config)
        report = get_report(result)
        
        assert len(report) > 200, "Report should be substantial"
        assert "machine learning" in report.lower() or "ML" in report
    
    def test_report_mentions_sources(self):
        """Test that the report includes source references."""
        from src.agent import run_research, get_report, AgentConfig
        
        config = AgentConfig(
            max_iterations=2,
            max_search_results=3
        )
        
        result = run_research("Benefits of regular exercise", config)
        report = get_report(result)
        
        # Report should reference sources somehow
        has_sources = (
            "source" in report.lower() or 
            "http" in report.lower() or
            "[" in report  # Citation format
        )
        assert has_sources, "Report should mention sources"
    
    def test_state_contains_gathered_sources(self):
        """Test that final state contains gathered sources."""
        from src.agent import run_research, AgentConfig
        
        config = AgentConfig(
            max_iterations=2,
            max_search_results=3
        )
        
        result = run_research("Climate change effects", config)
        
        assert "sources" in result
        assert len(result["sources"]) > 0, "Should have gathered at least one source"
        
        # Verify source structure
        for source in result["sources"]:
            assert "title" in source
            assert "url" in source
            assert "content" in source
    
    def test_iteration_count_respects_max(self):
        """Test that iterations don't exceed max_iterations."""
        from src.agent import run_research, AgentConfig
        
        config = AgentConfig(
            max_iterations=3,
            max_search_results=2
        )
        
        result = run_research("History of the internet", config)
        
        assert result["iteration"] <= config.max_iterations
    
    def test_messages_contains_query_and_report(self):
        """Test that messages field has both input query and output report."""
        from src.agent import run_research, AgentConfig
        from langchain_core.messages import HumanMessage, AIMessage
        
        config = AgentConfig(
            max_iterations=2,
            max_search_results=2
        )
        
        query = "What is blockchain technology?"
        result = run_research(query, config)
        
        messages = result["messages"]
        
        # Should have at least 2 messages
        assert len(messages) >= 2, "Should have query and report"
        
        # First should be HumanMessage
        assert isinstance(messages[0], HumanMessage)
        assert query.lower() in messages[0].content.lower()
        
        # Last should be AIMessage (the report)
        assert isinstance(messages[-1], AIMessage)
        assert len(messages[-1].content) > 100


@pytest.mark.integration
@pytest.mark.skipif(not HAS_ALL_KEYS, reason=SKIP_REASON)
class TestReportQuality:
    """Tests for report quality and structure."""
    
    def test_report_has_structure(self):
        """Test that report has headers or sections."""
        from src.agent import run_research, get_report, AgentConfig
        
        config = AgentConfig(max_iterations=3)
        
        result = run_research("Solar energy advantages and disadvantages", config)
        report = get_report(result)
        
        # Report should have some structure
        has_structure = (
            "#" in report or  # Markdown headers
            "Introduction" in report or
            "Conclusion" in report or
            "1." in report  # Numbered sections
        )
        assert has_structure, "Report should have discernible structure"
    
    def test_report_addresses_topic(self):
        """Test that report actually addresses the research topic."""
        from src.agent import run_research, get_report, AgentConfig
        
        config = AgentConfig(max_iterations=2)
        
        topic = "artificial intelligence in healthcare"
        result = run_research(topic, config)
        report = get_report(result)
        
        report_lower = report.lower()
        
        # Should mention key terms from the topic
        mentions_ai = "ai" in report_lower or "artificial intelligence" in report_lower
        mentions_health = "health" in report_lower or "medical" in report_lower or "patient" in report_lower
        
        assert mentions_ai, "Report should mention AI"
        assert mentions_health, "Report should mention healthcare"
    
    def test_report_has_conclusion(self):
        """Test that report has some form of conclusion."""
        from src.agent import run_research, get_report, AgentConfig
        
        config = AgentConfig(max_iterations=3)
        
        result = run_research("Electric vehicles environmental impact", config)
        report = get_report(result)
        
        report_lower = report.lower()
        
        has_conclusion = (
            "conclusion" in report_lower or
            "in summary" in report_lower or
            "to summarize" in report_lower or
            "overall" in report_lower or
            "in conclusion" in report_lower
        )
        assert has_conclusion, "Report should have a conclusion"


@pytest.mark.integration
@pytest.mark.skipif(not HAS_ALL_KEYS, reason=SKIP_REASON)
class TestEdgeCases:
    """Tests for edge cases and unusual inputs."""
    
    def test_handles_short_query(self):
        """Test handling of very short queries."""
        from src.agent import run_research, get_report, AgentConfig
        
        config = AgentConfig(max_iterations=2)
        
        result = run_research("Python", config)
        report = get_report(result)
        
        assert len(report) > 100, "Should produce report even for short query"
    
    def test_handles_question_format(self):
        """Test handling of question-formatted queries."""
        from src.agent import run_research, get_report, AgentConfig
        
        config = AgentConfig(max_iterations=2)
        
        result = run_research("What are the benefits of meditation?", config)
        report = get_report(result)
        
        assert len(report) > 200
        assert "meditation" in report.lower()
    
    def test_handles_complex_query(self):
        """Test handling of complex multi-part queries."""
        from src.agent import run_research, get_report, AgentConfig
        
        config = AgentConfig(max_iterations=3)
        
        query = "Compare renewable and non-renewable energy sources in terms of cost and environmental impact"
        result = run_research(query, config)
        report = get_report(result)
        
        report_lower = report.lower()
        
        # Should address multiple aspects
        assert "renewable" in report_lower or "solar" in report_lower or "wind" in report_lower
        assert "energy" in report_lower


@pytest.mark.integration  
@pytest.mark.skipif(not HAS_ALL_KEYS, reason=SKIP_REASON)
class TestVerboseMode:
    """Tests for verbose mode functionality."""
    
    def test_verbose_mode_runs_without_error(self, capsys):
        """Test that verbose mode executes without errors."""
        from src.agent.graph import create_graph
        from src.agent.config import AgentConfig
        from langchain_core.messages import HumanMessage
        
        config = AgentConfig(max_iterations=1, max_search_results=2)
        
        # Create graph with verbose=True
        graph = create_graph(config, verbose=True)
        
        initial_state = {
            "messages": [HumanMessage(content="What is Python?")],
            "topic": "",
            "running_summary": "",
            "sources": [],
            "search_results": [],
            "current_query": "",
            "iteration": 0,
            "max_iterations": 1
        }
        
        # Should complete without error
        result = graph.invoke(initial_state)
        
        assert "messages" in result
        
        # Check that verbose output was printed
        captured = capsys.readouterr()
        assert "[NODE]" in captured.out or "LLM Call" in captured.out