"""
Tests for node functions.
"""

import pytest
from unittest.mock import MagicMock, patch
from langchain_core.messages import HumanMessage, AIMessage


class TestInitializeState:
    """Tests for initialize_state node."""
    
    def test_extracts_topic_from_human_message(self, sample_state, agent_config):
        """Test that topic is extracted from HumanMessage."""
        from src.agent.nodes import initialize_state, init_agent
        
        with patch('src.agent.nodes.TavilyClient'):
            init_agent(agent_config, verbose=False)
        
        result = initialize_state(sample_state)
        
        assert "topic" in result
        assert result["topic"] == "What are the benefits of meditation?"
    
    def test_initializes_empty_summary(self, sample_state, agent_config):
        """Test that running_summary starts empty."""
        from src.agent.nodes import initialize_state, init_agent
        
        with patch('src.agent.nodes.TavilyClient'):
            init_agent(agent_config, verbose=False)
        
        result = initialize_state(sample_state)
        
        assert result["running_summary"] == ""
    
    def test_initializes_empty_sources(self, sample_state, agent_config):
        """Test that sources list starts empty."""
        from src.agent.nodes import initialize_state, init_agent
        
        with patch('src.agent.nodes.TavilyClient'):
            init_agent(agent_config, verbose=False)
        
        result = initialize_state(sample_state)
        
        assert result["sources"] == []
    
    def test_initializes_iteration_to_zero(self, sample_state, agent_config):
        """Test that iteration counter starts at zero."""
        from src.agent.nodes import initialize_state, init_agent
        
        with patch('src.agent.nodes.TavilyClient'):
            init_agent(agent_config, verbose=False)
        
        result = initialize_state(sample_state)
        
        assert result["iteration"] == 0
    
    def test_sets_max_iterations_from_config(self, sample_state, agent_config):
        """Test that max_iterations comes from config."""
        from src.agent.nodes import initialize_state, init_agent
        
        with patch('src.agent.nodes.TavilyClient'):
            init_agent(agent_config, verbose=False)
        
        result = initialize_state(sample_state)
        
        assert result["max_iterations"] == agent_config.max_iterations


class TestGenerateQuery:
    """Tests for generate_query node."""
    
    def test_generates_query_string(self, sample_state_with_topic, agent_config):
        """Test that a query string is generated."""
        from src.agent.nodes import generate_query, init_agent
        
        mock_response = MagicMock()
        mock_response.content = "meditation health benefits"
        
        with patch('src.agent.nodes.TavilyClient'):
            init_agent(agent_config, verbose=False)
        
        with patch('src.agent.nodes.llm') as mock_llm:
            mock_llm.invoke.return_value = mock_response
            result = generate_query(sample_state_with_topic)
        
        assert "current_query" in result
        assert isinstance(result["current_query"], str)
        assert len(result["current_query"]) > 0
    
    def test_strips_quotes_from_query(self, sample_state_with_topic, agent_config):
        """Test that quotes are removed from generated query."""
        from src.agent.nodes import generate_query, init_agent
        
        mock_response = MagicMock()
        mock_response.content = '"meditation benefits research"'
        
        with patch('src.agent.nodes.TavilyClient'):
            init_agent(agent_config, verbose=False)
        
        with patch('src.agent.nodes.llm') as mock_llm:
            mock_llm.invoke.return_value = mock_response
            result = generate_query(sample_state_with_topic)
        
        assert result["current_query"] == "meditation benefits research"
        assert '"' not in result["current_query"]
    
    def test_strips_single_quotes(self, sample_state_with_topic, agent_config):
        """Test that single quotes are also removed."""
        from src.agent.nodes import generate_query, init_agent
        
        mock_response = MagicMock()
        mock_response.content = "'meditation mental health'"
        
        with patch('src.agent.nodes.TavilyClient'):
            init_agent(agent_config, verbose=False)
        
        with patch('src.agent.nodes.llm') as mock_llm:
            mock_llm.invoke.return_value = mock_response
            result = generate_query(sample_state_with_topic)
        
        assert "'" not in result["current_query"]


class TestSearch:
    """Tests for search node."""
    
    def test_returns_sources(self, sample_state_with_topic, agent_config, sample_search_results):
        """Test that search returns sources."""
        from src.agent.nodes import search, init_agent
        
        mock_tavily = MagicMock()
        mock_tavily.search.return_value = {"results": sample_search_results}
        
        with patch('src.agent.nodes.TavilyClient', return_value=mock_tavily):
            init_agent(agent_config, verbose=False)
        
        state = {**sample_state_with_topic, "current_query": "meditation benefits"}
        
        with patch('src.agent.nodes.search_client', mock_tavily):
            result = search(state)
        
        assert "sources" in result
        assert len(result["sources"]) > 0
    
    def test_returns_search_results_for_summarize(self, sample_state_with_topic, agent_config, sample_search_results):
        """Test that search_results is populated for summarize node."""
        from src.agent.nodes import search, init_agent
        
        mock_tavily = MagicMock()
        mock_tavily.search.return_value = {"results": sample_search_results}
        
        with patch('src.agent.nodes.TavilyClient', return_value=mock_tavily):
            init_agent(agent_config, verbose=False)
        
        state = {**sample_state_with_topic, "current_query": "meditation benefits"}
        
        with patch('src.agent.nodes.search_client', mock_tavily):
            result = search(state)
        
        assert "search_results" in result
        assert len(result["search_results"]) > 0
    
    def test_avoids_duplicate_urls(self, sample_state_with_topic, agent_config, sample_search_results):
        """Test that duplicate URLs are not added to sources."""
        from src.agent.nodes import search, init_agent
        
        mock_tavily = MagicMock()
        mock_tavily.search.return_value = {"results": sample_search_results}
        
        with patch('src.agent.nodes.TavilyClient', return_value=mock_tavily):
            init_agent(agent_config, verbose=False)
        
        # Pre-populate with one existing source
        existing_source = sample_search_results[0]
        state = {
            **sample_state_with_topic,
            "current_query": "meditation benefits",
            "sources": [existing_source]
        }
        
        with patch('src.agent.nodes.search_client', mock_tavily):
            result = search(state)
        
        urls = [s["url"] for s in result["sources"]]
        assert len(urls) == len(set(urls)), "Duplicate URLs found in sources"
    
    def test_handles_search_error_gracefully(self, sample_state_with_topic, agent_config):
        """Test that search errors are handled gracefully."""
        from src.agent.nodes import search, init_agent
        
        mock_tavily = MagicMock()
        mock_tavily.search.side_effect = Exception("API Error")
        
        with patch('src.agent.nodes.TavilyClient', return_value=mock_tavily):
            init_agent(agent_config, verbose=False)
        
        state = {**sample_state_with_topic, "current_query": "test query"}
        
        with patch('src.agent.nodes.search_client', mock_tavily):
            result = search(state)
        
        assert "sources" in result
        assert "search_results" in result
        assert result["search_results"] == []


class TestSummarize:
    """Tests for summarize node."""
    
    def test_updates_running_summary(self, sample_state_with_results, agent_config):
        """Test that running summary is updated with new information."""
        from src.agent.nodes import summarize, init_agent
        
        mock_response = MagicMock()
        mock_response.content = "Updated summary with meditation benefits including stress reduction and improved focus."
        
        with patch('src.agent.nodes.TavilyClient'):
            init_agent(agent_config, verbose=False)
        
        with patch('src.agent.nodes.llm') as mock_llm:
            mock_llm.invoke.return_value = mock_response
            result = summarize(sample_state_with_results)
        
        assert "running_summary" in result
        assert len(result["running_summary"]) > 0
    
    def test_increments_iteration(self, sample_state_with_results, agent_config):
        """Test that iteration counter is incremented."""
        from src.agent.nodes import summarize, init_agent
        
        mock_response = MagicMock()
        mock_response.content = "Summary content"
        
        with patch('src.agent.nodes.TavilyClient'):
            init_agent(agent_config, verbose=False)
        
        state = {**sample_state_with_results, "iteration": 2}
        
        with patch('src.agent.nodes.llm') as mock_llm:
            mock_llm.invoke.return_value = mock_response
            result = summarize(state)
        
        assert result["iteration"] == 3
    
    def test_skips_when_no_search_results(self, sample_state_with_topic, agent_config):
        """Test that summarize skips when there are no search results."""
        from src.agent.nodes import summarize, init_agent
        
        with patch('src.agent.nodes.TavilyClient'):
            init_agent(agent_config, verbose=False)
        
        state = {**sample_state_with_topic, "search_results": [], "iteration": 1}
        result = summarize(state)
        
        assert result["iteration"] == 2
        assert "running_summary" not in result


class TestShouldContinue:
    """Tests for should_continue routing function."""
    
    def test_returns_write_report_at_max_iterations(self, agent_config):
        """Test that write_report is returned at max iterations."""
        from src.agent.nodes import should_continue, init_agent
        
        with patch('src.agent.nodes.TavilyClient'):
            init_agent(agent_config, verbose=False)
        
        state = {
            "topic": "test",
            "running_summary": "A" * 300,
            "iteration": 5,
            "max_iterations": 5
        }
        
        result = should_continue(state)
        assert result == "write_report"
    
    def test_returns_generate_query_with_short_summary(self, agent_config):
        """Test that generate_query is returned when summary is too short."""
        from src.agent.nodes import should_continue, init_agent
        
        with patch('src.agent.nodes.TavilyClient'):
            init_agent(agent_config, verbose=False)
        
        state = {
            "topic": "test",
            "running_summary": "Short summary",  # Less than 200 chars
            "iteration": 1,
            "max_iterations": 5
        }
        
        result = should_continue(state)
        assert result == "generate_query"
    
    def test_consults_llm_with_long_summary(self, agent_config):
        """Test that LLM is consulted when summary is substantial."""
        from src.agent.nodes import should_continue, init_agent
        
        mock_response = MagicMock()
        mock_response.content = "SUFFICIENT"
        
        with patch('src.agent.nodes.TavilyClient'):
            init_agent(agent_config, verbose=False)
        
        state = {
            "topic": "test",
            "running_summary": "A" * 300,  # More than 200 chars
            "iteration": 2,
            "max_iterations": 5
        }
        
        with patch('src.agent.nodes.llm') as mock_llm:
            mock_llm.invoke.return_value = mock_response
            result = should_continue(state)
        
        assert result == "write_report"
        mock_llm.invoke.assert_called_once()
    
    def test_continues_when_llm_says_continue(self, agent_config):
        """Test that research continues when LLM says CONTINUE."""
        from src.agent.nodes import should_continue, init_agent
        
        mock_response = MagicMock()
        mock_response.content = "CONTINUE"
        
        with patch('src.agent.nodes.TavilyClient'):
            init_agent(agent_config, verbose=False)
        
        state = {
            "topic": "test",
            "running_summary": "A" * 300,
            "iteration": 2,
            "max_iterations": 5
        }
        
        with patch('src.agent.nodes.llm') as mock_llm:
            mock_llm.invoke.return_value = mock_response
            result = should_continue(state)
        
        assert result == "generate_query"


class TestWriteReport:
    """Tests for write_report node."""
    
    def test_returns_ai_message(self, sample_state_with_summary, agent_config):
        """Test that write_report returns an AIMessage."""
        from src.agent.nodes import write_report, init_agent
        
        mock_response = MagicMock()
        mock_response.content = "# Research Report\n\nThis is the final report content."
        
        with patch('src.agent.nodes.TavilyClient'):
            init_agent(agent_config, verbose=False)
        
        with patch('src.agent.nodes.llm') as mock_llm:
            mock_llm.invoke.return_value = mock_response
            result = write_report(sample_state_with_summary)
        
        assert "messages" in result
        assert len(result["messages"]) == 1
        assert isinstance(result["messages"][0], AIMessage)
    
    def test_report_contains_content(self, sample_state_with_summary, agent_config):
        """Test that generated report has substantial content."""
        from src.agent.nodes import write_report, init_agent
        
        mock_response = MagicMock()
        mock_response.content = "# Research Report\n\n## Introduction\n\nDetailed findings about meditation benefits..."
        
        with patch('src.agent.nodes.TavilyClient'):
            init_agent(agent_config, verbose=False)
        
        with patch('src.agent.nodes.llm') as mock_llm:
            mock_llm.invoke.return_value = mock_response
            result = write_report(sample_state_with_summary)
        
        report_content = result["messages"][0].content
        assert len(report_content) > 50
        assert "Research Report" in report_content