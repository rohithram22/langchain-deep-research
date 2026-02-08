"""
Main entry point for the Deep Research Agent.

Usage:
    python main.py "What are the benefits of meditation?"
    python main.py --interactive
    python main.py "Your query" --max-iterations 10 --model gpt-4o
    python main.py "Your query" --verbose  # See agent thinking
"""

import argparse
import os
from dotenv import load_dotenv

from src.agent import run_research, get_report, AgentConfig
from src.agent.graph import create_graph
from langchain_core.messages import HumanMessage


def main():
    # Load environment variables from .env file
    load_dotenv()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Deep Research Agent - Conduct comprehensive web research on any topic"
    )
    parser.add_argument(
        "query",
        type=str,
        nargs="?",
        help="The research topic or question"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="gpt-4o-mini",
        help="LLM model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--max-iterations", "-n",
        type=int,
        default=5,
        help="Maximum research iterations (default: 5)"
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show agent thinking/working process"
    )
    
    args = parser.parse_args()
    
    # Verify API keys
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not found in environment")
        print("Please set it in your .env file or environment")
        return
    
    if not os.getenv("TAVILY_API_KEY"):
        print("Error: TAVILY_API_KEY not found in environment")
        print("Please set it in your .env file or environment")
        return
    
    # Create configuration
    config = AgentConfig(
        model_name=args.model,
        max_iterations=args.max_iterations
    )
    
    if args.interactive:
        run_interactive(config, args.verbose)
    elif args.query:
        run_single_query(args.query, config, args.verbose)
    else:
        parser.print_help()


def run_single_query(query: str, config: AgentConfig, verbose: bool = False):
    """Run a single research query."""
    print(f"\n{'='*60}")
    print(f"Research Topic: {query}")
    print(f"Model: {config.model_name}")
    print(f"Max Iterations: {config.max_iterations}")
    if verbose:
        print(f"Verbose Mode: ON")
    print(f"{'='*60}\n")
    
    if verbose:
        run_with_streaming(query, config)
    else:
        print("Researching... This may take a minute or two.\n")
        try:
            result = run_research(query, config)
            report = get_report(result)
            
            print(f"\n{'='*60}")
            print("RESEARCH REPORT")
            print(f"{'='*60}\n")
            print(report)
            print(f"\n{'='*60}")
            
            # Print some stats
            sources = result.get("sources", [])
            iterations = result.get("iteration", 0)
            print(f"\nStats: {iterations} iterations, {len(sources)} sources gathered")
            
        except Exception as e:
            print(f"Error during research: {e}")
            raise


def run_with_streaming(query: str, config: AgentConfig):
    """Run research with verbose output showing each step."""
    
    # Create the graph with verbose mode enabled
    graph = create_graph(config, verbose=True)
    
    # Set up initial state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "topic": "",
        "running_summary": "",
        "sources": [],
        "search_results": [],
        "current_query": "",
        "iteration": 0,
        "max_iterations": config.max_iterations
    }
    
    print("Starting research...\n")
    
    try:
        # Simply invoke the graph - verbose logging happens inside nodes
        result = graph.invoke(initial_state)
        
        # Extract the report from messages
        report = ""
        if "messages" in result:
            messages = result["messages"]
            if isinstance(messages, list):
                for msg in reversed(messages):
                    if hasattr(msg, "content"):
                        report = msg.content
                        break
            elif hasattr(messages, "content"):
                report = messages.content
        
        print(f"\n{'='*60}")
        print("FINAL RESEARCH REPORT")
        print(f"{'='*60}\n")
        print(report if report else "No report generated.")
        print(f"\n{'='*60}")
        
        sources = result.get("sources", [])
        iterations = result.get("iteration", 0)
        print(f"\nStats: {iterations} iterations, {len(sources)} sources gathered")
        
    except Exception as e:
        print(f"Error during research: {e}")
        raise


def print_node_output(node_name: str, output: dict):
    """Pretty print the output of each node with detailed internal information."""
    
    # Handle None output
    if output is None:
        output = {}
    
    # Get verbose info if available
    verbose_info = output.get("_verbose", {})
    
    # Define descriptions for each node
    descriptions = {
        "initialize": "INITIALIZING RESEARCH STATE",
        "generate_query": "GENERATING SEARCH QUERY",
        "search": "EXECUTING WEB SEARCH",
        "summarize": "SUMMARIZING RESULTS",
        "reflect": "REFLECTING ON PROGRESS",
        "write_report": "WRITING FINAL REPORT"
    }
    
    description = descriptions.get(node_name, "PROCESSING")
    
    print(f"\n{'='*60}")
    print(f"[NODE] {description}")
    print(f"{'='*60}")
    
    if node_name == "initialize":
        # Get topic from verbose_info first, then output, then empty string
        topic = verbose_info.get("topic_extracted") or output.get("topic") or "(not set)"
        print(f"\n  Input:")
        print(f"    - User query from messages")
        print(f"\n  Output:")
        print(f"    - Topic: \"{topic}\"")
        print(f"    - Running summary: (empty)")
        print(f"    - Sources: []")
        print(f"    - Iteration: 0")
    
    elif node_name == "generate_query":
        # Get values with proper fallbacks
        prompt = verbose_info.get("prompt_sent") or "(prompt not captured)"
        llm_response = verbose_info.get("llm_response") or "(response not captured)"
        final_query = verbose_info.get("final_query") or output.get("current_query") or "(no query)"
        
        print(f"\n  LLM Call:")
        print(f"    - Model: Generating search query based on topic and current knowledge")
        print(f"\n  Prompt Sent to LLM:")
        print(f"    {_truncate_text(prompt, 300)}")
        print(f"\n  LLM Response:")
        print(f"    \"{llm_response}\"")
        print(f"\n  Output:")
        print(f"    - Search query: \"{final_query}\"")
    
    elif node_name == "search":
        # Get values with proper fallbacks
        tool_name = verbose_info.get("tool_name") or "Tavily Search"
        query = verbose_info.get("query") or output.get("current_query") or "(no query)"
        max_results = verbose_info.get("max_results") or 5
        search_depth = verbose_info.get("search_depth") or "advanced"
        results = verbose_info.get("results") or output.get("_search_results") or []
        error = verbose_info.get("error")
        total_sources = verbose_info.get("total_sources") or len(output.get("sources", []))
        
        print(f"\n  Tool Call:")
        print(f"    - Tool: {tool_name}")
        print(f"    - Query: \"{query}\"")
        print(f"    - Max Results: {max_results}")
        print(f"    - Search Depth: {search_depth}")
        
        if error:
            print(f"\n  Error:")
            print(f"    {error}")
        else:
            print(f"\n  Results Found ({len(results)}):")
            if results:
                for i, result in enumerate(results[:5], 1):  # Show up to 5
                    title = result.get("title", "No title")[:60]
                    url = result.get("url", "")
                    content_preview = _truncate_text(result.get("content", ""), 100)
                    print(f"\n    [{i}] {title}")
                    print(f"        URL: {url}")
                    print(f"        Content: {content_preview}")
            else:
                print(f"    (no results)")
            
            print(f"\n  Output:")
            print(f"    - Total sources collected: {total_sources}")
    
    elif node_name == "summarize":
        if verbose_info.get("skipped"):
            print(f"\n  Skipped: {verbose_info.get('reason') or 'No results'}")
        else:
            prompt = verbose_info.get("prompt_sent") or "(prompt not captured)"
            llm_response = verbose_info.get("llm_response") or output.get("running_summary") or "(no summary)"
            summary_length = verbose_info.get("summary_length") or len(output.get("running_summary", ""))
            iteration = verbose_info.get("iteration") or output.get("iteration") or 0
            
            print(f"\n  LLM Call:")
            print(f"    - Model: Updating running summary with new information")
            print(f"\n  Prompt Sent to LLM:")
            print(f"    {_truncate_text(prompt, 300)}")
            print(f"\n  LLM Response (Updated Summary):")
            print(f"    {_truncate_text(llm_response, 500)}")
            print(f"\n  Output:")
            print(f"    - Summary length: {summary_length} characters")
            print(f"    - Iteration: {iteration}")
    
    elif node_name == "reflect":
        iteration = verbose_info.get("iteration") or 0
        max_iterations = verbose_info.get("max_iterations") or 5
        summary_length = verbose_info.get("summary_length") or 0
        llm_consulted = verbose_info.get("llm_consulted", False)
        llm_response = verbose_info.get("llm_response") or ""
        decision = verbose_info.get("decision") or "(unknown)"
        reason = verbose_info.get("reason") or "(no reason captured)"
        
        print(f"\n  Current State:")
        print(f"    - Iteration: {iteration}/{max_iterations}")
        print(f"    - Summary length: {summary_length} characters")
        
        if llm_consulted:
            print(f"\n  LLM Call:")
            print(f"    - Model: Evaluating if research is sufficient")
            print(f"\n  LLM Response:")
            print(f"    \"{llm_response}\"")
        else:
            print(f"\n  LLM Call:")
            print(f"    - Skipped (decision made by rule)")
        
        print(f"\n  Decision:")
        print(f"    - Next node: {decision}")
        print(f"    - Reason: {reason}")
    
    elif node_name == "write_report":
        prompt = verbose_info.get("prompt_sent") or "(prompt not captured)"
        sources_count = verbose_info.get("sources_count") or 0
        report_length = verbose_info.get("report_length") or 0
        
        print(f"\n  LLM Call:")
        print(f"    - Model: Generating final research report")
        print(f"\n  Prompt Sent to LLM:")
        print(f"    {_truncate_text(prompt, 400)}")
        print(f"\n  Output:")
        print(f"    - Sources cited: {sources_count}")
        print(f"    - Report length: {report_length} characters")
        print(f"    - Report generated successfully")


def _truncate_text(text: str, max_length: int) -> str:
    """Truncate text and add ellipsis if needed."""
    if not text:
        return "(empty)"
    # Clean up whitespace
    text = " ".join(text.split())
    if len(text) <= max_length:
        return text
    return text[:max_length] + "..."


def run_interactive(config: AgentConfig, verbose: bool = False):
    """Run in interactive mode."""
    print(f"\n{'='*60}")
    print("Deep Research Agent - Interactive Mode")
    print(f"Model: {config.model_name}")
    print(f"Max Iterations: {config.max_iterations}")
    if verbose:
        print(f"Verbose Mode: ON")
    print("Type 'quit' or 'exit' to stop")
    print(f"{'='*60}\n")
    
    while True:
        try:
            query = input("\nEnter your research topic: ").strip()
            
            if query.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            
            if not query:
                continue
            
            if verbose:
                run_with_streaming(query, config)
            else:
                print("\nResearching... This may take a minute or two.\n")
                
                result = run_research(query, config)
                report = get_report(result)
                
                print(f"\n{'='*60}")
                print("RESEARCH REPORT")
                print(f"{'='*60}\n")
                print(report)
                print(f"\n{'='*60}")
                
                sources = result.get("sources", [])
                iterations = result.get("iteration", 0)
                print(f"\nStats: {iterations} iterations, {len(sources)} sources gathered")
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    main()