"""
State definition for the Deep Research Agent.

The state tracks all information as the agent progresses through
the research loop: Topic → Generate Query → Search → Summarize → Reflect → ...
"""

from typing import Annotated, TypedDict, Optional
from langgraph.graph.message import add_messages


class ResearchState(TypedDict):
    """
    The state of our research agent.
    
    Following the Local Deep Researcher pattern:
    - Start with a topic
    - Generate search queries based on what we know so far
    - Search and gather sources
    - Update our running summary
    - Reflect to decide if we need more research
    - Finally write the report
    """
    
    # Required: messages field for user input and final output
    # User's query comes in as HumanMessage, final report as AIMessage
    messages: Annotated[list, add_messages]
    
    # The research topic extracted from user's query
    topic: str
    
    # Running summary that gets updated after each search
    # This is the key to the Local Deep Researcher pattern
    running_summary: str
    
    # All sources gathered (for citations)
    sources: list[dict]  # [{title, url, content}, ...]
    
    # The most recent search query used
    current_query: str
    
    # Loop control
    iteration: int
    max_iterations: int