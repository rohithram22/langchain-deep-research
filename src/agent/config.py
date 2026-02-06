"""
Configuration and prompts for the Deep Research Agent.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class AgentConfig:
    """Configuration for the research agent."""
    
    # LLM settings
    model_name: str = "gpt-4o-mini"
    temperature: float = 0
    
    # Search settings
    max_iterations: int = 5
    max_search_results: int = 5
    search_depth: str = "advanced"  # "basic" or "advanced"
    
    # API keys (optional, will use env vars if not provided)
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None


# ============================================================
# PROMPTS
# ============================================================

GENERATE_QUERY_PROMPT = """You are a research assistant helping to gather information on a topic.

TOPIC: {topic}

CURRENT SUMMARY OF RESEARCH:
{running_summary}

Based on the topic and what has been researched so far, generate the next search query to find NEW, RELEVANT information that we don't already have.

If the current summary is empty, generate a broad initial query about the topic.
If we already have some information, identify GAPS or MISSING ASPECTS and search for those.

Requirements:
- Keep the query concise (3-7 words work best)
- Focus on finding NEW information not already in the summary
- Be specific enough to get relevant results

Return ONLY the search query, nothing else."""


SUMMARIZE_PROMPT = """You are a research assistant. Your job is to update a running summary with new information.

TOPIC: {topic}

CURRENT SUMMARY:
{running_summary}

NEW SEARCH RESULTS:
{search_results}

Instructions:
1. Read the new search results carefully
2. Extract information that is RELEVANT to the topic
3. Add NEW information to the summary (don't repeat what's already there)
4. Keep the summary well-organized and coherent
5. Note the source URLs for important facts

If the new results don't contain useful new information, return the current summary unchanged.

Return the updated summary:"""


REFLECT_PROMPT = """You are evaluating whether we have enough research to write a comprehensive report.

TOPIC: {topic}

CURRENT RESEARCH SUMMARY:
{running_summary}

ITERATIONS COMPLETED: {iteration} / {max_iterations}

Evaluate the research so far:
1. Do we have enough information to thoroughly address the topic?
2. Are there critical gaps or missing perspectives?
3. Would more searching likely yield valuable new information?

If we have sufficient information OR we've reached max iterations, respond with: SUFFICIENT
If we need more research and have iterations remaining, respond with: CONTINUE

Respond with ONLY one word: either SUFFICIENT or CONTINUE"""


WRITE_REPORT_PROMPT = """You are an expert research report writer. Write a comprehensive report based on the research gathered.

TOPIC: {topic}

RESEARCH SUMMARY:
{running_summary}

SOURCES USED:
{sources}

Write a well-structured report that:
1. Has a clear introduction stating what the report covers
2. Is organized into logical sections with headers
3. Presents information clearly and objectively
4. Cites sources using [Source: URL] format where appropriate
5. Ends with a brief conclusion summarizing key findings

Write the report in a professional, informative tone.

REPORT:"""