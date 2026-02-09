"""
Tests for AgentConfig and prompts.
"""

import pytest
from src.agent.config import (
    AgentConfig,
    GENERATE_QUERY_PROMPT,
    SUMMARIZE_PROMPT,
    REFLECT_PROMPT,
    WRITE_REPORT_PROMPT
)


class TestAgentConfig:
    """Tests for AgentConfig dataclass."""
    
    def test_default_values(self):
        """Test that AgentConfig has correct default values."""
        config = AgentConfig()
        
        assert config.model_name == "gpt-4o-mini"
        assert config.temperature == 0
        assert config.max_iterations == 5
        assert config.max_search_results == 5
        assert config.search_depth == "advanced"
    
    def test_custom_model_name(self):
        """Test setting custom model name."""
        config = AgentConfig(model_name="gpt-4o")
        assert config.model_name == "gpt-4o"
    
    def test_custom_temperature(self):
        """Test setting custom temperature."""
        config = AgentConfig(temperature=0.7)
        assert config.temperature == 0.7
    
    def test_custom_max_iterations(self):
        """Test setting custom max iterations."""
        config = AgentConfig(max_iterations=10)
        assert config.max_iterations == 10
    
    def test_custom_max_search_results(self):
        """Test setting custom max search results."""
        config = AgentConfig(max_search_results=10)
        assert config.max_search_results == 10
    
    def test_custom_search_depth(self):
        """Test setting search depth to basic."""
        config = AgentConfig(search_depth="basic")
        assert config.search_depth == "basic"
    
    def test_api_keys_default_to_none(self):
        """Test that API keys are None by default."""
        config = AgentConfig()
        
        assert config.openai_api_key is None
        assert config.anthropic_api_key is None
        assert config.tavily_api_key is None
    
    def test_api_keys_can_be_set(self):
        """Test that API keys can be provided."""
        config = AgentConfig(
            openai_api_key="test-openai-key",
            tavily_api_key="test-tavily-key"
        )
        
        assert config.openai_api_key == "test-openai-key"
        assert config.tavily_api_key == "test-tavily-key"
    
    def test_multiple_custom_values(self):
        """Test setting multiple custom values at once."""
        config = AgentConfig(
            model_name="gpt-4o",
            temperature=0.5,
            max_iterations=8,
            max_search_results=7,
            search_depth="basic"
        )
        
        assert config.model_name == "gpt-4o"
        assert config.temperature == 0.5
        assert config.max_iterations == 8
        assert config.max_search_results == 7
        assert config.search_depth == "basic"


class TestGenerateQueryPrompt:
    """Tests for GENERATE_QUERY_PROMPT template."""
    
    def test_has_topic_placeholder(self):
        """Test that prompt has topic placeholder."""
        assert "{topic}" in GENERATE_QUERY_PROMPT
    
    def test_has_running_summary_placeholder(self):
        """Test that prompt has running_summary placeholder."""
        assert "{running_summary}" in GENERATE_QUERY_PROMPT
    
    def test_formats_correctly(self):
        """Test that prompt formats without errors."""
        formatted = GENERATE_QUERY_PROMPT.format(
            topic="quantum computing",
            running_summary="Some existing research notes"
        )
        
        assert "quantum computing" in formatted
        assert "Some existing research notes" in formatted
    
    def test_formats_with_empty_summary(self):
        """Test formatting with empty summary."""
        formatted = GENERATE_QUERY_PROMPT.format(
            topic="machine learning",
            running_summary="No research yet."
        )
        
        assert "machine learning" in formatted
        assert "No research yet." in formatted


class TestSummarizePrompt:
    """Tests for SUMMARIZE_PROMPT template."""
    
    def test_has_required_placeholders(self):
        """Test that prompt has all required placeholders."""
        assert "{topic}" in SUMMARIZE_PROMPT
        assert "{running_summary}" in SUMMARIZE_PROMPT
        assert "{search_results}" in SUMMARIZE_PROMPT
    
    def test_formats_correctly(self):
        """Test that prompt formats without errors."""
        formatted = SUMMARIZE_PROMPT.format(
            topic="artificial intelligence",
            running_summary="Previous research notes here",
            search_results="[1] Title: Test\nContent: Test content"
        )
        
        assert "artificial intelligence" in formatted
        assert "Previous research notes here" in formatted
        assert "Test content" in formatted


class TestReflectPrompt:
    """Tests for REFLECT_PROMPT template."""
    
    def test_has_required_placeholders(self):
        """Test that prompt has all required placeholders."""
        assert "{topic}" in REFLECT_PROMPT
        assert "{running_summary}" in REFLECT_PROMPT
        assert "{iteration}" in REFLECT_PROMPT
        assert "{max_iterations}" in REFLECT_PROMPT
    
    def test_formats_correctly(self):
        """Test that prompt formats without errors."""
        formatted = REFLECT_PROMPT.format(
            topic="climate change",
            running_summary="Research summary here",
            iteration=2,
            max_iterations=5
        )
        
        assert "climate change" in formatted
        assert "Research summary here" in formatted
        assert "2" in formatted
        assert "5" in formatted
    
    def test_mentions_sufficient_and_continue(self):
        """Test that prompt mentions expected response options."""
        assert "SUFFICIENT" in REFLECT_PROMPT
        assert "CONTINUE" in REFLECT_PROMPT


class TestWriteReportPrompt:
    """Tests for WRITE_REPORT_PROMPT template."""
    
    def test_has_required_placeholders(self):
        """Test that prompt has all required placeholders."""
        assert "{topic}" in WRITE_REPORT_PROMPT
        assert "{running_summary}" in WRITE_REPORT_PROMPT
        assert "{sources}" in WRITE_REPORT_PROMPT
    
    def test_formats_correctly(self):
        """Test that prompt formats without errors."""
        formatted = WRITE_REPORT_PROMPT.format(
            topic="renewable energy",
            running_summary="Comprehensive research findings",
            sources="[1] Source One: https://example.com"
        )
        
        assert "renewable energy" in formatted
        assert "Comprehensive research findings" in formatted
        assert "https://example.com" in formatted
    
    def test_mentions_report_structure(self):
        """Test that prompt mentions expected report structure."""
        prompt_lower = WRITE_REPORT_PROMPT.lower()
        assert "introduction" in prompt_lower
        assert "conclusion" in prompt_lower
        assert "source" in prompt_lower