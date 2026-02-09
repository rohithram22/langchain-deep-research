"""
Tests for the ResearchState definition.
"""

import pytest
from langchain_core.messages import HumanMessage, AIMessage
from src.agent.state import ResearchState


class TestResearchState:
    """Tests for ResearchState TypedDict."""
    
    def test_state_has_required_fields(self):
        """Test that state can be created with all required fields."""
        state: ResearchState = {
            "messages": [HumanMessage(content="test query")],
            "topic": "test topic",
            "running_summary": "",
            "sources": [],
            "search_results": [],
            "current_query": "",
            "iteration": 0,
            "max_iterations": 5
        }
        
        assert "messages" in state
        assert "topic" in state
        assert "running_summary" in state
        assert "sources" in state
        assert "search_results" in state
        assert "current_query" in state
        assert "iteration" in state
        assert "max_iterations" in state
    
    def test_messages_field_accepts_human_message(self):
        """Test that messages field works with HumanMessage."""
        human_msg = HumanMessage(content="What is AI?")
        state: ResearchState = {
            "messages": [human_msg],
            "topic": "",
            "running_summary": "",
            "sources": [],
            "search_results": [],
            "current_query": "",
            "iteration": 0,
            "max_iterations": 5
        }
        
        assert len(state["messages"]) == 1
        assert state["messages"][0].content == "What is AI?"
    
    def test_messages_field_accepts_ai_message(self):
        """Test that messages field works with AIMessage."""
        ai_msg = AIMessage(content="Here is the report...")
        state: ResearchState = {
            "messages": [ai_msg],
            "topic": "",
            "running_summary": "",
            "sources": [],
            "search_results": [],
            "current_query": "",
            "iteration": 0,
            "max_iterations": 5
        }
        
        assert len(state["messages"]) == 1
        assert state["messages"][0].content == "Here is the report..."
    
    def test_messages_field_accepts_multiple_messages(self):
        """Test that messages can hold both human and AI messages."""
        state: ResearchState = {
            "messages": [
                HumanMessage(content="Research quantum computing"),
                AIMessage(content="Here is my report on quantum computing...")
            ],
            "topic": "quantum computing",
            "running_summary": "",
            "sources": [],
            "search_results": [],
            "current_query": "",
            "iteration": 0,
            "max_iterations": 5
        }
        
        assert len(state["messages"]) == 2
        assert isinstance(state["messages"][0], HumanMessage)
        assert isinstance(state["messages"][1], AIMessage)
    
    def test_sources_accepts_list_of_dicts(self):
        """Test that sources field accepts list of dictionaries."""
        sources = [
            {"title": "Source 1", "url": "https://example.com/1", "content": "Content 1"},
            {"title": "Source 2", "url": "https://example.com/2", "content": "Content 2"}
        ]
        
        state: ResearchState = {
            "messages": [],
            "topic": "",
            "running_summary": "",
            "sources": sources,
            "search_results": [],
            "current_query": "",
            "iteration": 0,
            "max_iterations": 5
        }
        
        assert len(state["sources"]) == 2
        assert state["sources"][0]["title"] == "Source 1"
        assert state["sources"][1]["url"] == "https://example.com/2"
    
    def test_search_results_separate_from_sources(self):
        """Test that search_results and sources are independent."""
        state: ResearchState = {
            "messages": [],
            "topic": "",
            "running_summary": "",
            "sources": [{"title": "Old", "url": "http://old.com", "content": "old"}],
            "search_results": [{"title": "New", "url": "http://new.com", "content": "new"}],
            "current_query": "",
            "iteration": 0,
            "max_iterations": 5
        }
        
        assert len(state["sources"]) == 1
        assert len(state["search_results"]) == 1
        assert state["sources"][0]["title"] == "Old"
        assert state["search_results"][0]["title"] == "New"
    
    def test_iteration_tracking(self):
        """Test iteration counter fields."""
        state: ResearchState = {
            "messages": [],
            "topic": "",
            "running_summary": "",
            "sources": [],
            "search_results": [],
            "current_query": "",
            "iteration": 3,
            "max_iterations": 5
        }
        
        assert state["iteration"] == 3
        assert state["max_iterations"] == 5
        assert state["iteration"] < state["max_iterations"]