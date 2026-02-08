from langgraph.graph import StateGraph, START, END
from langchain_core.messages import HumanMessage

from .state import ResearchState
from .config import AgentConfig
from .nodes import (
    init_agent,
    initialize_state,
    generate_query,
    search,
    summarize,
    reflect,
    should_continue,
    write_report,
)


def create_graph(config: AgentConfig = None, verbose: bool = False) -> StateGraph:
    """
    Create and compile the research agent graph.
    
    Args:
        config: Optional configuration for the agent
        verbose: Whether to print verbose output
        
    Returns:
        Compiled StateGraph ready to invoke
    """
    # Initialize the agent components
    init_agent(config, verbose=verbose)
    
    # Create the graph
    graph = StateGraph(ResearchState)
    
    # Add all nodes
    graph.add_node("initialize", initialize_state)
    graph.add_node("generate_query", generate_query)
    graph.add_node("search", search)
    graph.add_node("summarize", summarize)
    graph.add_node("reflect", reflect)
    graph.add_node("write_report", write_report)
    
    # Define the flow
    graph.add_edge(START, "initialize")
    graph.add_edge("initialize", "generate_query")
    graph.add_edge("generate_query", "search")
    graph.add_edge("search", "summarize")
    graph.add_edge("summarize", "reflect")
    
    # Conditional edge: continue researching or write report
    graph.add_conditional_edges(
        "reflect",
        should_continue,
        {
            "generate_query": "generate_query",
            "write_report": "write_report"
        }
    )
    
    graph.add_edge("write_report", END)
    
    # Compile and return
    return graph.compile()


def run_research(
    query: str,
    config: AgentConfig = None
) -> dict:
    """
    Convenience function to run research on a query.
    
    Args:
        query: The research topic/question
        config: Optional configuration
        
    Returns:
        Final state including the report in messages
    """
    # Create the graph
    graph = create_graph(config)
    
    # Set up initial state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "topic": "",
        "running_summary": "",
        "sources": [],
        "search_results": [],
        "current_query": "",
        "iteration": 0,
        "max_iterations": config.max_iterations if config else 5
    }
    
    # Run the graph
    result = graph.invoke(initial_state)
    
    return result


def get_report(result: dict) -> str:
    """Extract the report from the result state."""
    messages = result.get("messages", [])
    for msg in reversed(messages):
        if hasattr(msg, "content"):
            return msg.content
    return "No report generated."