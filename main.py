"""
Main entry point for the Deep Research Agent.

Usage:
    python main.py "What are the benefits of meditation?"
    python main.py --interactive
    python main.py "Your query" --max-iterations 10 --model gpt-4o
"""

import argparse
import os
from dotenv import load_dotenv

from src.agent import run_research, get_report, AgentConfig


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
        run_interactive(config)
    elif args.query:
        run_single_query(args.query, config)
    else:
        parser.print_help()


def run_single_query(query: str, config: AgentConfig):
    """Run a single research query."""
    print(f"\n{'='*60}")
    print(f"Research Topic: {query}")
    print(f"Model: {config.model_name}")
    print(f"Max Iterations: {config.max_iterations}")
    print(f"{'='*60}\n")
    
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


def run_interactive(config: AgentConfig):
    """Run in interactive mode."""
    print(f"\n{'='*60}")
    print("Deep Research Agent - Interactive Mode")
    print(f"Model: {config.model_name}")
    print(f"Max Iterations: {config.max_iterations}")
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