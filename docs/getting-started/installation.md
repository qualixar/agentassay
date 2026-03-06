# Installation

## Requirements

- Python 3.10 or later
- pip (Python package manager)

## Install from PyPI

The simplest way to install AgentAssay:

```bash
pip install agentassay
```

This installs the core package with all required dependencies.

## Install with Framework Adapters

To use AgentAssay with a specific agent framework, install the corresponding extra:

```bash
# For LangGraph agents
pip install agentassay[langgraph]

# For CrewAI agents
pip install agentassay[crewai]

# For AutoGen agents
pip install agentassay[autogen]

# For OpenAI Agents SDK
pip install agentassay[openai]

# For smolagents
pip install agentassay[smolagents]

# Install all framework adapters
pip install agentassay[all]
```

## Install with Contract Integration

If you use [AgentAssert](https://github.com/qualixar/agentassert) for behavioral contracts, install the contracts extra:

```bash
pip install agentassay[contracts]
```

## Install from Source

For development or to get the latest unreleased changes:

```bash
# Clone the repository
git clone https://github.com/qualixar/agentassay.git
cd agentassay

# Create a virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install in editable mode with dev dependencies
pip install -e ".[dev]"
```

## Development Dependencies

The `[dev]` extra includes everything needed for contributing:

- Test runner and coverage tools
- Linter and formatter
- Type checker
- Property-based testing support

## Verify Installation

After installing, verify that everything is working:

```bash
# Check the CLI is available
agentassay --version

# Check the pytest plugin is registered
python -m pytest --co -q 2>&1 | head -5

# Run a quick Python import check
python -c "import agentassay; print(f'AgentAssay v{agentassay.__version__}')"
```

## Upgrading

```bash
pip install --upgrade agentassay
```

## Uninstalling

```bash
pip uninstall agentassay
```
