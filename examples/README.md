# AgentAssay Examples

Complete working examples demonstrating AgentAssay's framework adapters and testing capabilities.

## Quick Start

### Basic Examples (`basic/`)

| Example | What It Shows | Run |
|---------|--------------|-----|
| **`quickstart.py`** | Simplest possible usage with CustomAdapter | `python examples/basic/quickstart.py` |
| **`token_efficient.py`** | Core innovation: 5-20x cost reduction techniques | `python examples/basic/token_efficient.py` |

### Framework-Specific Examples (`framework-specific/`)

| Framework | Example | Prerequisites | Run |
|-----------|---------|--------------|-----|
| **LangGraph** | `langgraph_example.py` | `pip install agentassay[langgraph]` | `python examples/framework-specific/langgraph_example.py` |
| **CrewAI** | `crewai_example.py` | `pip install agentassay[crewai]` | `python examples/framework-specific/crewai_example.py` |
| **OpenAI Agents** | `openai_example.py` | `pip install agentassay[openai]` | `python examples/framework-specific/openai_example.py` |
| **AutoGen/AG2** | `autogen_example.py` | `pip install agentassay[autogen]` | `python examples/framework-specific/autogen_example.py` |
| **smolagents** | `smolagents_example.py` | `pip install agentassay[smolagents]` | `python examples/framework-specific/smolagents_example.py` |

### pytest Integration (`pytest-plugin/`)

| Example | What It Shows | Run |
|---------|--------------|-----|
| **`test_my_agent.py`** | Statistical testing with pytest markers | `pytest examples/pytest-plugin/test_my_agent.py -v` |

## Installation

```bash
# Install base package
pip install agentassay

# Install with specific framework support
pip install agentassay[langgraph]   # LangGraph
pip install agentassay[crewai]      # CrewAI
pip install agentassay[openai]      # OpenAI Agents SDK
pip install agentassay[autogen]     # AutoGen/AG2
pip install agentassay[smolagents]  # HuggingFace smolagents

# Install all frameworks
pip install agentassay[all]

# Install with development tools (includes pytest plugin)
pip install agentassay[dev]
```

## Example Structure

Each framework example follows this pattern:

1. **Create your agent** using the framework's API
2. **Wrap with AgentAssay adapter** (one line of code)
3. **Define test scenarios** with input data and expected properties
4. **Configure statistical testing** (trials, significance level, power)
5. **Run stochastic trials** and get a statistical verdict

## Key Concepts Demonstrated

### 1. CustomAdapter (quickstart.py)
Wrap ANY Python callable — no framework lock-in. Accepts:
- `ExecutionTrace` objects (zero overhead)
- `dict` with `output`/`steps`/`cost` keys
- `str` (wrapped as single-step trace)

### 2. Token-Efficient Testing (token_efficient.py)
**The core innovation of AgentAssay:**
- **Behavioral Fingerprinting**: Compact trace vectors + Hotelling's T² regression test (3-5x savings)
- **Adaptive Budget**: Variance-based minimum N calculation (2-4x fewer trials)
- **Trace-First Analysis**: Coverage + contracts on production traces (FREE!)

**Combined savings: 5-20x cost reduction at equivalent statistical power.**

### 3. Framework Adapters (framework-specific/)
All adapters provide:
- Automatic step extraction from framework-specific events
- Tool call capture with input/output
- LLM interaction logging
- Cost tracking (where available)
- Timing per step

### 4. pytest Integration (pytest-plugin/)
Use familiar pytest patterns:
```python
@pytest.mark.agentassay(trials=20, alpha=0.05, power=0.80)
def test_my_agent():
    # Test runs 20 times, statistical verdict computed automatically
    assert agent_passes_quality_threshold()
```

## Real-World Usage Patterns

### Pattern 1: CI/CD Gate
```python
# tests/test_agent_quality.py
@pytest.mark.agentassay(trials=30, alpha=0.05)
def test_agent_meets_sla():
    adapter = LangGraphAdapter(production_agent, model="gpt-4o")
    trace = adapter.run({"query": "production_query"})
    assert trace.success is True
```

Run in CI:
```bash
pytest tests/test_agent_quality.py --junit-xml=results.xml
```

### Pattern 2: Regression Testing
```python
from agentassay.efficiency.regression import behavioral_regression_test

baseline_fp = load_baseline_fingerprints()
current_fp = collect_current_fingerprints()

result = behavioral_regression_test(baseline_fp, current_fp, alpha=0.05)
if result.regressed:
    raise ValueError("Behavioral regression detected!")
```

### Pattern 3: Nightly Quality Checks
```python
# Run 100 trials overnight, check pass rate > 95%
assay_config = AssayConfig(num_trials=100, significance_level=0.05)
verdict = runner.run_trials(scenarios)

for scenario_id, result in verdict.items():
    if result.pass_rate < 0.95:
        alert_oncall(f"{scenario_id} quality dropped to {result.pass_rate:.1%}")
```

## Next Steps

- **Docs:** Read [docs/getting-started/quickstart.md](../docs/getting-started/quickstart.md)
- **Paper:** arXiv:2603.02601 (full technical details)
- **Adapters:** See [docs/guides/adapters/](../docs/guides/adapters/) for per-framework docs
- **CLI:** Try `agentassay --help` for command-line usage

## Need Help?

- GitHub Issues: https://github.com/qualixar/agentassay/issues
- Paper: https://arxiv.org/abs/2603.02601
- Website: https://qualixar.com/products/agentassay
