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
    
    # Create the graph
    graph = create_graph(config)
    
    # Set up initial state
    initial_state = {
        "messages": [HumanMessage(content=query)],
        "topic": "",
        "running_summary": "",
        "sources": [],
        "current_query": "",
        "iteration": 0,
        "max_iterations": config.max_iterations
    }
    
    print("Starting research...\n")
    
    try:
        # Stream through the graph and capture final state
        final_state = dict(initial_state)
        
        for event in graph.stream(initial_state, stream_mode="updates"):
            if event is None:
                continue
                
            for node_name, node_output in event.items():
                # Handle None or empty outputs
                if node_output is None:
                    node_output = {}
                
                print_node_output(node_name, node_output)
                
                # Update final_state with node outputs
                if isinstance(node_output, dict):
                    for key, value in node_output.items():
                        if not key.startswith("_"):  # Skip internal keys
                            final_state[key] = value
        
        # Extract the report from messages
        report = ""
        if "messages" in final_state:
            messages = final_state["messages"]
            # Handle both list and single message
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
        
        sources = final_state.get("sources", [])
        iterations = final_state.get("iteration", 0)
        print(f"\nStats: {iterations} iterations, {len(sources)} sources gathered")
        
    except Exception as e:
        print(f"Error during research: {e}")
        raise


def print_node_output(node_name: str, output: dict):
    """Pretty print the output of each node."""
    
    # Handle None output
    if output is None:
        output = {}
    
    # Define descriptions for each node
    descriptions = {
        "initialize": "Initializing Research State",
        "generate_query": "Generating Search Query",
        "search": "Searching the Web",
        "summarize": "Summarizing Results",
        "reflect": "Reflecting on Progress",
        "write_report": "Writing Final Report"
    }
    
    description = descriptions.get(node_name, "Processing")
    
    print(f"\n{'-'*50}")
    print(f"{description}")
    print(f"{'-'*50}")
    
    if node_name == "initialize":
        if "topic" in output:
            print(f"   Topic: {output['topic']}")
        print(f"   Initialized research state")
    
    elif node_name == "generate_query":
        if "current_query" in output:
            print(f"   Generated search query: \"{output['current_query']}\"")
    
    elif node_name == "search":
        if "sources" in output:
            num_sources = len(output.get("sources", []))
            print(f"   Found {num_sources} total sources")
            # Show the latest sources
            new_sources = output.get("_search_results", [])
            if new_sources:
                print(f"   New results from this search:")
                for s in new_sources[:3]:  # Show first 3
                    title = s.get("title", "No title")[:50]
                    print(f"      • {title}...")
    
    elif node_name == "summarize":
        if "running_summary" in output:
            summary = output["running_summary"]
            # Show a preview of the summary
            preview = summary[:200] + "..." if len(summary) > 200 else summary
            print(f"   Updated summary ({len(summary)} chars)")
            print(f"   Preview: {preview}")
        if "iteration" in output:
            print(f"   Iteration: {output['iteration']}")
    
    elif node_name == "reflect":
        print(f"   Evaluating if more research is needed...")
    
    elif node_name == "write_report":
        print(f"   Writing final report...")
        if "messages" in output:
            print(f"   Report generated successfully!")


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