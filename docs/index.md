# AgentAssay Documentation

**Test More. Spend Less. Ship Confident.**

AgentAssay is the first agent testing framework that delivers statistical guarantees without burning your token budget. It combines behavioral fingerprinting, adaptive budget optimization, and trace-first offline analysis to achieve the same confidence at 5-20x lower cost.

---

## Why AgentAssay?

Traditional software testing is binary: a test passes or fails. But AI agents are inherently non-deterministic. The same agent, given the same input, may produce different outputs across runs. A single pass/fail tells you nothing reliable.

Worse, testing agents is **expensive**. Every trial means LLM calls, tool executions, and token consumption. Run 100 trials across 20 scenarios, and you have burned thousands of tokens before you even know if your change broke something.

AgentAssay solves both problems:

1. **Statistical rigor.** Run your agent N times and get three-valued verdicts (PASS / FAIL / INCONCLUSIVE) backed by confidence intervals. No guesswork.
2. **Token efficiency.** Behavioral fingerprinting detects regression from patterns, not raw text. Adaptive budgeting computes the exact minimum N. Trace-first analysis reuses existing traces for free. Same confidence, 83% less cost.

---

## Key Capabilities

| Capability | What It Does | Token Cost |
|-----------|--------------|:----------:|
| **Behavioral fingerprinting** | Detect regression from behavioral patterns, not raw text | Low (fewer samples needed) |
| **Adaptive budget optimization** | Calibrate and compute exact minimum trials per scenario | Minimal (5-10 calibration runs) |
| **Trace-first offline analysis** | Coverage, contracts, metamorphic checks on existing traces | Zero |
| **Multi-fidelity proxy testing** | Screen with cheap models, confirm with expensive | Reduced |
| **Warm-start sequential testing** | Incorporate prior results for faster convergence | Reduced |
| **Three-valued verdicts** | PASS, FAIL, or INCONCLUSIVE -- never misleading binary | -- |
| **5D coverage metrics** | Tool, path, state, boundary, model coverage | Runs on traces |
| **Mutation testing** | Perturb agents to validate test sensitivity | Requires runs |
| **Metamorphic testing** | Verify behavioral invariants | Mixed |
| **Deployment gates** | Block broken deployments in CI/CD | -- |
| **Framework adapters** | Works with popular agent frameworks | -- |
| **Interactive dashboard** | Real-time test monitoring, trend analysis, fingerprint visualization | -- |
| **pytest integration** | Familiar conventions with statistical assertions | -- |

---

## Documentation

### Getting Started

- [Installation](getting-started/installation.md) -- Install from PyPI or source
- [Quickstart](getting-started/quickstart.md) -- Test your first agent in 5 minutes, with token-efficient budget optimization

### Core Concepts

- [Token-Efficient Testing](concepts/token-efficient-testing.md) -- **START HERE** -- The core philosophy: fingerprinting, budgeting, trace-first
- [Stochastic Testing](concepts/stochastic-testing.md) -- Why agent testing needs statistics
- [Coverage Metrics](concepts/coverage.md) -- The five-dimensional coverage model
- [Mutation Testing](concepts/mutation-testing.md) -- Operators to stress-test your test suite

### Guides

- [pytest Plugin](guides/pytest-plugin.md) -- Use AgentAssay with pytest
- [CI/CD Integration](guides/ci-cd-integration.md) -- Deployment gates and automation
- [Framework Adapters](guides/framework-adapters.md) -- Connect to your agent framework
- [Dashboard](guides/dashboard.md) -- Visualize results, trends, and behavioral fingerprints

### Framework Quickstarts

- [LangGraph](guides/adapters/langgraph.md)
- [CrewAI](guides/adapters/crewai.md)
- [AutoGen](guides/adapters/autogen.md)
- [OpenAI Agents](guides/adapters/openai.md)
- [smolagents](guides/adapters/smolagents.md)
- [Semantic Kernel](guides/adapters/semantic-kernel.md)
- [AWS Bedrock Agents](guides/adapters/bedrock.md)
- [MCP](guides/adapters/mcp.md)
- [Vertex AI Agents](guides/adapters/vertex.md)
- [Custom Agents](guides/adapters/custom.md)

### Reference

- [CLI Reference](reference/cli.md) -- All six commands documented
- [Configuration](reference/configuration.md) -- YAML config and environment variables
- [Glossary](reference/glossary.md) -- Terminology and definitions

### Architecture

- [System Overview](architecture/overview.md) -- Six-layer architecture and data flow

### Operations

- [Troubleshooting](operations/troubleshooting.md) -- Common issues and solutions

---

## The Token-Efficiency Advantage

```
Traditional agent testing:
  20 scenarios x 100 trials = 2,000 invocations = ~$150/run

AgentAssay:
  5 scenarios offline (traces)      =     0 invocations
  10 scenarios adaptive (avg N=18)  =   180 invocations
  5 scenarios full (avg N=45)       =   225 invocations
                                    ─────────────────────
  Total                             =   405 invocations = ~$30/run

  Savings: 80% with identical statistical confidence.
```

Read [Token-Efficient Testing](concepts/token-efficient-testing.md) for the full explanation.

---

## Quick Links

- **Source code:** [github.com/qualixar/agentassay](https://github.com/qualixar/agentassay)
- **PyPI:** `pip install agentassay`
- **License:** Apache-2.0

---

## Research

AgentAssay is backed by a peer-reviewed research paper establishing formal foundations for stochastic agent regression testing. The paper includes 31 definitions, 5 theorems with proofs, and experimental validation.
