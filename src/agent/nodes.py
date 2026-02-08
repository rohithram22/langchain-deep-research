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
verbose_mode = False


def init_agent(agent_config: AgentConfig = None, verbose: bool = False):
    """Initialize the LLM and search client."""
    global llm, search_client, config, verbose_mode
    
    config = agent_config or AgentConfig()
    verbose_mode = verbose
    
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


def _log(message: str, indent: int = 0):
    """Print message if verbose mode is enabled."""
    if verbose_mode:
        prefix = "  " * indent
        print(f"{prefix}{message}")


def _log_section(title: str):
    """Print a section header if verbose mode is enabled."""
    if verbose_mode:
        print(f"\n{'='*60}")
        print(f"[NODE] {title}")
        print(f"{'='*60}")


def _truncate(text: str, max_len: int = 200) -> str:
    """Truncate text for display."""
    if not text:
        return "(empty)"
    text = " ".join(text.split())  # Normalize whitespace
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


def initialize_state(state: ResearchState) -> dict:
    """
    Initialize the research state from the user's message.
    Extracts the topic and sets up initial values.
    """
    _log_section("INITIALIZING RESEARCH STATE")
    
    # Get the user's query from messages
    topic = ""
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            topic = msg.content
            break
        elif hasattr(msg, "type") and msg.type == "human":
            topic = msg.content
            break
    
    _log("\nInput:", 1)
    _log("- Extracting topic from user message", 2)
    _log("\nOutput:", 1)
    _log(f"- Topic: \"{topic}\"", 2)
    _log(f"- Running summary: (empty)", 2)
    _log(f"- Sources: []", 2)
    _log(f"- Iteration: 0", 2)
    
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
    """
    _log_section("GENERATING SEARCH QUERY")
    
    topic = state["topic"]
    running_summary = state.get("running_summary", "") or "No research yet."
    
    prompt = GENERATE_QUERY_PROMPT.format(
        topic=topic,
        running_summary=running_summary
    )
    
    _log("\nLLM Call:", 1)
    _log("- Generating search query based on topic and current knowledge", 2)
    _log("\nPrompt Sent to LLM:", 1)
    _log(_truncate(prompt, 300), 2)
    
    response = llm.invoke(prompt)
    query = response.content.strip()
    
    # Clean up the query (remove quotes if present)
    query = query.strip('"\'')
    
    _log("\nLLM Response:", 1)
    _log(f"\"{response.content.strip()}\"", 2)
    _log("\nOutput:", 1)
    _log(f"- Search query: \"{query}\"", 2)
    
    return {"current_query": query}


def search(state: ResearchState) -> dict:
    """
    Execute web search using the current query.
    """
    _log_section("EXECUTING WEB SEARCH")
    
    query = state["current_query"]
    max_results = config.max_search_results if config else 5
    search_depth = config.search_depth if config else "advanced"
    
    _log("\nTool Call:", 1)
    _log(f"- Tool: Tavily Search", 2)
    _log(f"- Query: \"{query}\"", 2)
    _log(f"- Max Results: {max_results}", 2)
    _log(f"- Search Depth: {search_depth}", 2)
    
    try:
        response = search_client.search(
            query=query,
            max_results=max_results,
            search_depth=search_depth
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
        
        _log(f"\nResults Found ({len(new_sources)}):", 1)
        for i, r in enumerate(new_sources[:5], 1):
            _log(f"\n[{i}] {r['title'][:60]}", 2)
            _log(f"    URL: {r['url']}", 2)
            _log(f"    Content: {_truncate(r['content'], 100)}", 2)
        
        _log(f"\nOutput:", 1)
        _log(f"- New sources found: {len(new_sources)}", 2)
        _log(f"- Total sources collected: {len(existing_sources)}", 2)
        
        return {
            "sources": existing_sources,
            "search_results": new_sources  # Renamed: removed underscore so LangGraph persists it
        }
        
    except Exception as e:
        _log(f"\nError: {str(e)}", 1)
        return {
            "sources": state.get("sources", []),
            "search_results": []
        }


def summarize(state: ResearchState) -> dict:
    """
    Update the running summary with information from the latest search.
    """
    _log_section("SUMMARIZING RESULTS")
    
    topic = state["topic"]
    running_summary = state.get("running_summary", "") or "No research yet."
    search_results = state.get("search_results", [])  # Renamed: removed underscore
    
    if not search_results:
        _log("\nSkipped: No search results to summarize", 1)
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
    
    _log("\nLLM Call:", 1)
    _log("- Updating running summary with new information", 2)
    _log("\nPrompt Sent to LLM:", 1)
    _log(_truncate(prompt, 300), 2)
    
    response = llm.invoke(prompt)
    updated_summary = response.content.strip()
    
    _log("\nLLM Response (Updated Summary):", 1)
    _log(_truncate(updated_summary, 400), 2)
    _log("\nOutput:", 1)
    _log(f"- Summary length: {len(updated_summary)} characters", 2)
    _log(f"- Iteration: {state['iteration'] + 1}", 2)
    
    return {
        "running_summary": updated_summary,
        "iteration": state["iteration"] + 1
    }


def reflect(state: ResearchState) -> dict:
    """
    Reflect on current research and decide whether to continue.
    """
    _log_section("REFLECTING ON PROGRESS")
    
    iteration = state["iteration"]
    max_iterations = state.get("max_iterations", 5)
    running_summary = state.get("running_summary", "")
    
    _log("\nCurrent State:", 1)
    _log(f"- Iteration: {iteration}/{max_iterations}", 2)
    _log(f"- Summary length: {len(running_summary)} characters", 2)
    
    # Note: actual decision is made in should_continue()
    # This node just logs the state for visibility
    
    if iteration >= max_iterations:
        _log("\nDecision Preview:", 1)
        _log(f"- Will write report (reached max iterations)", 2)
    elif len(running_summary) < 200:
        _log("\nDecision Preview:", 1)
        _log(f"- Will continue (summary too short: {len(running_summary)} < 200 chars)", 2)
    else:
        _log("\nDecision Preview:", 1)
        _log(f"- Will consult LLM to decide", 2)
    
    return {}


def should_continue(state: ResearchState) -> Literal["generate_query", "write_report"]:
    """
    Routing function: decide whether to continue research or write the report.
    """
    iteration = state["iteration"]
    max_iterations = state.get("max_iterations", 5)
    
    # Hard stop at max iterations
    if iteration >= max_iterations:
        _log("\nRouting Decision: write_report", 1)
        _log(f"- Reason: Reached max iterations ({iteration}/{max_iterations})", 2)
        return "write_report"
    
    # Get current state
    topic = state["topic"]
    running_summary = state.get("running_summary", "")
    
    # If we have very little content, continue
    if len(running_summary) < 200:
        _log("\nRouting Decision: generate_query", 1)
        _log(f"- Reason: Summary too short ({len(running_summary)} chars < 200)", 2)
        return "generate_query"
    
    # Ask LLM to evaluate
    prompt = REFLECT_PROMPT.format(
        topic=topic,
        running_summary=running_summary,
        iteration=iteration,
        max_iterations=max_iterations
    )
    
    _log("\nLLM Call (Routing Decision):", 1)
    _log("- Asking LLM if research is sufficient", 2)
    
    response = llm.invoke(prompt)
    decision = response.content.strip().upper()
    
    _log(f"\nLLM Response: \"{response.content.strip()}\"", 1)
    
    if "SUFFICIENT" in decision:
        _log("\nRouting Decision: write_report", 1)
        _log("- Reason: LLM determined research is sufficient", 2)
        return "write_report"
    else:
        _log("\nRouting Decision: generate_query", 1)
        _log("- Reason: LLM determined more research needed", 2)
        return "generate_query"


def write_report(state: ResearchState) -> dict:
    """
    Generate the final research report based on all gathered information.
    """
    _log_section("WRITING FINAL REPORT")
    
    topic = state["topic"]
    running_summary = state.get("running_summary", "")
    sources = state.get("sources", [])
    
    # Format sources for the prompt
    sources_text = ""
    unique_urls = []
    for s in sources:
        if s["url"] not in unique_urls:
            unique_urls.append(s["url"])
    
    for i, url in enumerate(unique_urls[:15], 1):
        title = next((s["title"] for s in sources if s["url"] == url), "Source")
        sources_text += f"[{i}] {title}: {url}\n"
    
    prompt = WRITE_REPORT_PROMPT.format(
        topic=topic,
        running_summary=running_summary,
        sources=sources_text
    )
    
    _log("\nLLM Call:", 1)
    _log("- Generating final research report", 2)
    _log("\nPrompt Sent to LLM:", 1)
    _log(_truncate(prompt, 400), 2)
    _log(f"\nSources to cite: {len(unique_urls)}", 1)
    
    response = llm.invoke(prompt)
    report = response.content.strip()
    
    _log("\nOutput:", 1)
    _log(f"- Report length: {len(report)} characters", 2)
    _log("- Report generated successfully", 2)
    
    return {
        "messages": [AIMessage(content=report)]
    }