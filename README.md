# Deep Research Agent

A LangGraph-powered research agent that conducts iterative web research and generates comprehensive reports with citations.

## Overview

This project implements a deep research agent that:
1. Accepts a research query from the user
2. Iteratively searches the web to gather relevant information
3. Builds a running summary of findings
4. Generates a comprehensive report with citations

For details on the architecture and design decisions, see [DESIGN.md](DESIGN.md).

## Project Structure

```
langchain-deep-research/
├── src/
│   └── agent/
│       ├── config.py        # Configuration and prompts
│       ├── graph.py         # LangGraph definition
│       ├── nodes.py         # Node functions
│       └── state.py         # State definition
├── tests/                   # Test files
├── main.py                  # CLI entry point
├── requirements.txt
├── README.md
└── DESIGN.md                # Architecture and design decisions
```

## Setup

### Prerequisites

- Python 3.10 or higher
- OpenAI API key
- Tavily API key (free tier available at https://tavily.com)

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/langchain-deep-research.git
   cd langchain-deep-research
   ```

2. **Create and activate a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**
   ```bash
   cp .env.example .env
   ```
   
   Edit `.env` and add your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   TAVILY_API_KEY=your_tavily_api_key_here
   ```

## Usage

### Basic Usage

Run a research query:

```bash
python main.py "What is quantum computing?"
```

### Verbose Mode

See the agent's internal reasoning, LLM calls, and search results:

```bash
python main.py "What are the benefits of meditation?" --verbose
```

Or use the short flag:

```bash
python main.py "What are the benefits of meditation?" -v
```

### Interactive Mode

Run multiple research queries in a session:

```bash
python main.py --interactive
```

With verbose output:

```bash
python main.py --interactive --verbose
```

### Custom Configuration

Change the model:
```bash
python main.py "Your query" --model gpt-4o
```

Change max iterations:
```bash
python main.py "Your query" --max-iterations 10
```

Combine options:
```bash
python main.py "Climate change solutions" --model gpt-4o --max-iterations 8 --verbose
```

### Command Line Options

| Option | Short | Default | Description |
|--------|-------|---------|-------------|
| `query` | - | - | The research topic or question |
| `--model` | `-m` | gpt-4o-mini | LLM model to use |
| `--max-iterations` | `-n` | 5 | Maximum research iterations |
| `--verbose` | `-v` | False | Show agent's internal reasoning |
| `--interactive` | `-i` | False | Run in interactive mode |

## Running Tests

### Run all unit tests (no API keys required)

```bash
python -m pytest tests/test_state.py tests/test_config.py tests/test_nodes.py tests/test_graph.py -v
```

### Run integration tests (requires API keys)

```bash
python -m pytest tests/test_integration.py -v
```

### Run all tests

```bash
python -m pytest -v
```

### Run with coverage report

```bash
python -m pytest --cov=src --cov-report=html -v
```

## Example Output

**Query:** `python main.py "What is quantum computing?"`

```
============================================================
Research Topic: What is quantum computing?
Model: gpt-4o-mini
Max Iterations: 5
============================================================

Researching... This may take a minute or two.

============================================================
RESEARCH REPORT
============================================================

# Report on Quantum Computing

## Introduction

This report provides a comprehensive overview of quantum computing...

## Understanding Quantum Computing

Quantum computing diverges significantly from classical computing...

## Recent Advancements

As of 2024, significant progress has been made...
[Source: https://mitsloan.mit.edu/...]

## Conclusion

Quantum computing represents a transformative shift...

### References

1. [MIT Sloan Management Review](https://mitsloan.mit.edu/...)
2. [National Science Foundation](https://www.nsf.gov/...)

============================================================

Stats: 2 iterations, 10 sources gathered
```