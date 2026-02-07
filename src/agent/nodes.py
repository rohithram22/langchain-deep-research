"""
Node functions for the Deep Research Agent.

Each function represents a step in the research process:
1. initialize_state - Set up initial state from user query
2. generate_query - Create search query based on current knowledge
3. search - Execute web search
4. summarize - Update running summary with new findings
5. reflect - Decide whether to continue or stop
6. write_report - Generate final report
"""

import os
from typing import Literal
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage
from tavily import TavilyClient

from .state import ResearchState
from .config import (
    AgentConfig,
    GENERATE_QUERY_PROMPT,
    SUMMARIZE_PROMPT,
    REFLECT_PROMPT,
    WRITE_REPORT_PROMPT,
)


# Global instances (initialized in init_agent)
llm = None
search_client = None
config = None


def init_agent(agent_config: AgentConfig = None):
    """Initialize the LLM and search client."""
    global llm, search_client, config
    
    config = agent_config or AgentConfig()
    
    # Initialize LLM
    api_key = config.openai_api_key or os.getenv("OPENAI_API_KEY")
    llm = ChatOpenAI(
        model=config.model_name,
        temperature=config.temperature,
        api_key=api_key
    )
    
    # Initialize Tavily search
    tavily_key = config.tavily_api_key or os.getenv("TAVILY_API_KEY")
    if not tavily_key:
        raise ValueError("TAVILY_API_KEY is required")
    search_client = TavilyClient(api_key=tavily_key)


def initialize_state(state: ResearchState) -> dict:
    """
    Initialize the research state from the user's message.
    Extracts the topic and sets up initial values.
    """
    # Get the user's query from messages
    topic = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            topic = msg.content
            break
        elif hasattr(msg, "type") and msg.type == "human":
            topic = msg.content
            break
    
    return {
        "topic": topic,
        "running_summary": "",
        "sources": [],
        "current_query": "",
        "iteration": 0,
        "max_iterations": config.max_iterations if config else 5
    }


def generate_query(state: ResearchState) -> dict:
    """
    Generate the next search query based on the topic and current summary.
    
    This is the key to adaptive research - each query is informed by
    what we've already learned, helping fill gaps.
    """
    topic = state["topic"]
    running_summary = state.get("running_summary", "") or "No research yet."
    
    prompt = GENERATE_QUERY_PROMPT.format(
        topic=topic,
        running_summary=running_summary
    )
    
    response = llm.invoke(prompt)
    query = response.content.strip()
    
    # Clean up the query (remove quotes if present)
    query = query.strip('"\'')
    
    return {"current_query": query}


def search(state: ResearchState) -> dict:
    """
    Execute web search using the current query.
    Returns search results to be processed.
    """
    query = state["current_query"]
    
    try:
        response = search_client.search(
            query=query,
            max_results=config.max_search_results if config else 5,
            search_depth=config.search_depth if config else "advanced"
        )
        
        results = response.get("results", [])
        
        # Format results for storage
        new_sources = []
        for r in results:
            new_sources.append({
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "content": r.get("content", "")
            })
        
        # Add to existing sources (avoid duplicates by URL)
        existing_urls = {s["url"] for s in state.get("sources", [])}
        existing_sources = list(state.get("sources", []))
        
        for source in new_sources:
            if source["url"] not in existing_urls:
                existing_sources.append(source)
                existing_urls.add(source["url"])
        
        return {
            "sources": existing_sources,
            "_search_results": new_sources  # Temporary, for summarize node
        }
        
    except Exception as e:
        print(f"Search error: {e}")
        return {
            "sources": state.get("sources", []),
            "_search_results": []
        }


def summarize(state: ResearchState) -> dict:
    """
    Update the running summary with information from the latest search.
    
    This incrementally builds our knowledge, which then informs
    the next search query.
    """
    topic = state["topic"]
    running_summary = state.get("running_summary", "") or "No research yet."
    
    # Get the latest search results (stored temporarily)
    search_results = state.get("_search_results", [])
    
    if not search_results:
        return {"iteration": state["iteration"] + 1}
    
    # Format search results for the prompt
    results_text = ""
    for i, r in enumerate(search_results, 1):
        results_text += f"\n[{i}] {r['title']}\nURL: {r['url']}\n{r['content']}\n"
    
    prompt = SUMMARIZE_PROMPT.format(
        topic=topic,
        running_summary=running_summary,
        search_results=results_text
    )
    
    response = llm.invoke(prompt)
    updated_summary = response.content.strip()
    
    return {
        "running_summary": updated_summary,
        "iteration": state["iteration"] + 1
    }


def reflect(state: ResearchState) -> dict:
    """
    Reflect on current research and decide whether to continue.
    
    This node exists for the graph structure - the actual routing
    decision is made in should_continue().
    """
    return {}


def should_continue(state: ResearchState) -> Literal["generate_query", "write_report"]:
    """
    Routing function: decide whether to continue research or write the report.
    
    Uses LLM to evaluate if we have sufficient information.
    """
    iteration = state["iteration"]
    max_iterations = state.get("max_iterations", 5)
    
    # Hard stop at max iterations
    if iteration >= max_iterations:
        return "write_report"
    
    # Get current state
    topic = state["topic"]
    running_summary = state.get("running_summary", "")
    
    # If we have very little content, continue
    if len(running_summary) < 200:
        return "generate_query"
    
    # Ask LLM to evaluate
    prompt = REFLECT_PROMPT.format(
        topic=topic,
        running_summary=running_summary,
        iteration=iteration,
        max_iterations=max_iterations
    )
    
    response = llm.invoke(prompt)
    decision = response.content.strip().upper()
    
    if "SUFFICIENT" in decision:
        return "write_report"
    else:
        return "generate_query"


def write_report(state: ResearchState) -> dict:
    """
    Generate the final research report based on all gathered information.
    """
    topic = state["topic"]
    running_summary = state.get("running_summary", "")
    sources = state.get("sources", [])
    
    # Format sources for the prompt
    sources_text = ""
    unique_urls = []
    for s in sources:
        if s["url"] not in unique_urls:
            unique_urls.append(s["url"])
    
    for i, url in enumerate(unique_urls[:15], 1):  # Limit to 15 sources
        # Find the title for this URL
        title = next((s["title"] for s in sources if s["url"] == url), "Source")
        sources_text += f"[{i}] {title}: {url}\n"
    
    prompt = WRITE_REPORT_PROMPT.format(
        topic=topic,
        running_summary=running_summary,
        sources=sources_text
    )
    
    response = llm.invoke(prompt)
    report = response.content.strip()
    
    # Return as AIMessage in messages
    return {
        "messages": [AIMessage(content=report)]
    }