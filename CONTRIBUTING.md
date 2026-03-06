# Contributing to AgentAssay

Thank you for your interest in contributing to AgentAssay. This document describes the development setup, code standards, and contribution process.

## Development Setup

### Prerequisites

- Python 3.10 or later
- Git

### Installation

```bash
# Clone the repository
git clone https://github.com/qualixar/agentassay.git
cd agentassay

# Create a virtual environment
python -m venv .venv
source .venv/bin/activate  # macOS/Linux
# .venv\Scripts\activate   # Windows

# Install in development mode with all dev dependencies
pip install -e ".[dev]"
```

### Verify Installation

```bash
# Run the test suite
python -m pytest tests/ -v --tb=short

# Run linting
ruff check src/ tests/

# Run type checking
mypy src/agentassay/
```

## Code Style

### Formatting and Linting

AgentAssay uses **ruff** for linting and formatting.

```bash
# Lint
ruff check src/ tests/

# Auto-fix lint issues
ruff check --fix src/ tests/

# Format
ruff format src/ tests/
```

Configuration is in `pyproject.toml`:

- Target: Python 3.10
- Line length: 100 characters
- Enabled rules: E, F, I, N, W, UP

### Type Checking

AgentAssay uses **mypy** for static type analysis.

```bash
mypy src/agentassay/
```

All public functions must include type annotations. Use `from __future__ import annotations` at the top of every module.

### Docstrings

- Use NumPy-style docstrings for all public classes, methods, and functions.
- Include `Parameters`, `Returns`, and `Raises` sections where applicable.
- Provide at least one usage example for non-trivial functions.

## Testing

### Running Tests

```bash
# Full suite
python -m pytest tests/ -v --tb=short

# Specific test file
python -m pytest tests/test_core_models.py -v

# Run with coverage
python -m pytest tests/ --cov=agentassay --cov-report=term-missing

# Skip slow tests
python -m pytest tests/ -m "not slow"
```

### Writing Tests

- Every new feature or bug fix must include tests.
- Test files go in `tests/` and must be named `test_*.py`.
- Use descriptive test function names: `test_verdict_is_inconclusive_when_ci_straddles_threshold`.
- For statistical tests, use deterministic seeds or fixed data to ensure reproducibility.
- Aim for 100% branch coverage on new code.

### Test Requirements

Before submitting a PR, ensure:

1. All existing tests pass (`python -m pytest tests/ -q`)
2. No lint errors (`ruff check src/ tests/`)
3. No type errors (`mypy src/agentassay/`)
4. New code has corresponding tests

## Pull Request Process

### Before You Start

1. Check existing issues and PRs to avoid duplicate work.
2. For significant changes, open an issue first to discuss the approach.
3. Fork the repository and create a feature branch from `main`.

### Branch Naming

- `feat/description` -- New features
- `fix/description` -- Bug fixes
- `docs/description` -- Documentation changes
- `test/description` -- Test additions or improvements
- `refactor/description` -- Code restructuring without behavior changes

### Submitting a PR

1. Keep PRs focused. One logical change per PR.
2. Write a clear PR description explaining **what** changed and **why**.
3. Ensure all CI checks pass.
4. Update the CHANGELOG.md if your change affects users.
5. Request review from a maintainer.

### Review Process

- All PRs require at least one approval.
- Address review feedback promptly or explain why you disagree.
- Squash commits into logical units before merge.

## Reporting Issues

- Use the GitHub issue templates for bugs and feature requests.
- Include reproduction steps for bugs.
- Include the output of `agentassay --version` and `python --version`.

## License

By contributing to AgentAssay, you agree that your contributions will be licensed under the Apache-2.0 License.
