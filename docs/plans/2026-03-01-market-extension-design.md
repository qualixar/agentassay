# AgentAssay Market Extension — Design Document

**Date:** 2026-03-01
**Author:** Varun Pratap Bhardwaj + Partner (Jarvis Mode)
**Status:** APPROVED
**Scope:** Extend AgentAssay from 10-adapter testing framework to universal agent quality platform

---

## 1. Vision

**Before:** "pytest for AI agents" (10 frameworks, CLI only)
**After:** "The universal agent quality platform. Works with every framework. Costs 5-20x less."

One command: `pip install agentassay`
One CI line: `uses: agentassay/test-action@v1`
One protocol: OpenTelemetry GenAI traces in, quality verdicts out

---

## 2. Architecture: The 3-Layer Strategy

### Layer 1: OTel Universal Ingestion (The Viral Play)

Build ONE adapter that reads OpenTelemetry GenAI spans. Any framework that emits OTel traces gets AgentAssay support for FREE.

```
[Any Framework] → OTel GenAI Spans → AgentAssay OTel Ingester → ExecutionTrace → All Analyses
```

Already emitting OTel: AutoGen/AG2, LangChain (adding), Semantic Kernel, Google ADK
OTel-compatible backends: Datadog, Grafana, New Relic, Splunk, Honeycomb

**Innovation:** Propose `gen_ai.testing.*` OTel semantic conventions:
- `gen_ai.testing.run_id` — test run identifier
- `gen_ai.testing.verdict` — PASS/FAIL/INCONCLUSIVE
- `gen_ai.testing.statistical_power` — confidence of verdict
- `gen_ai.testing.token_cost` — cost of test execution
- `gen_ai.testing.fingerprint_distance` — behavioral drift metric
- `gen_ai.testing.regression_detected` — boolean flag

### Layer 2: Framework-Native Adapters (15 total)

Existing 10 + 5 new priority adapters:

| # | Adapter | Target Framework | Integration Pattern | Effort |
|---|---------|-----------------|---------------------|--------|
| 11 | PydanticAI | PydanticAI 1.63+ | Agent.run() hook, type-safe traces | 3 days |
| 12 | DSPy | DSPy 3.1+ | Module compilation traces, optimizer logs | 3 days |
| 13 | Haystack | Haystack 2.x | Pipeline component tracing | 3 days |
| 14 | LlamaIndex | LlamaIndex 0.14+ | Workflow event streaming, callback manager | 3 days |
| 15 | Google ADK | Google ADK 1.x | Agent.run() traces, Vertex AI grounding | 3 days |

Future (v1.x): Agno, Mastra (TypeScript), MetaGPT, CAMEL, Composio

### Layer 3: Distribution & Integration

#### 3a. GitHub Action (Marketplace)

```yaml
- uses: agentassay/test-action@v1
  with:
    config: agentassay.yaml
    gate-policy: strict
    report: html
  env:
    OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

Outputs: JUnit XML (native GitHub test summary), HTML report artifact, SARIF (optional), badge JSON

#### 3b. MCP Server

```bash
agentassay mcp-server --stdio
```

Tools exposed:
- `agentassay_run` — execute tests from IDE
- `agentassay_compare` — compare baselines
- `agentassay_fingerprint` — compute behavioral fingerprint
- `agentassay_gate` — deployment decision

Works in: Claude Code, Cursor, Windsurf, any MCP client

#### 3c. OTel Exporter

```python
from agentassay.telemetry import AgentAssayOTelExporter

exporter = AgentAssayOTelExporter(endpoint="http://localhost:4317")
runner = TrialRunner(otel_exporter=exporter)
```

Exports to: Datadog, Grafana/Prometheus, New Relic, Splunk, Honeycomb, Jaeger

#### 3d. Notifications

```yaml
# agentassay.yaml
notifications:
  slack:
    webhook_url: ${SLACK_WEBHOOK}
    on: [fail, regression]
  teams:
    webhook_url: ${TEAMS_WEBHOOK}
    on: [fail]
  pagerduty:
    routing_key: ${PD_KEY}
    on: [gate_rejected]
```

---

## 3. New IP Contributions (Paper Extension)

| # | Contribution | Novelty | Section |
|---|-------------|---------|---------|
| 11 | Universal Agent Test Protocol (OTel conventions) | 10/10 | New Section 7b |
| 12 | Cross-Framework Behavioral Fingerprinting | 9/10 | Section 7 extension |
| 13 | Protocol-Aware Testing (MCP/A2A/AG-UI compliance) | 9/10 | New Section 6b |

### Contribution 11: Universal Agent Test Protocol

Define the first-ever OpenTelemetry semantic conventions for agent testing. Submit as proposal to OTel GenAI SIG. If adopted, AgentAssay becomes reference implementation.

### Contribution 12: Cross-Framework Fingerprinting

Prove that behavioral fingerprints are framework-invariant — the same agent behavior produces similar fingerprints whether running on LangGraph, CrewAI, or OpenAI SDK. This enables cross-framework regression detection.

### Contribution 13: Protocol Compliance Testing

Test that MCP tool implementations, A2A message exchanges, and AG-UI streaming conform to protocol specifications. First formal testing framework for the 3 open standards.

---

## 4. Coverage Matrix (Before vs After)

| Category | v0.3.0 (Now) | v1.0.0 (After) |
|----------|:------------:|:--------------:|
| Agent Frameworks | 10 | 15 + OTel universal |
| Cloud Providers | 3 | 3 + protocol-level |
| CI/CD Platforms | 1 (docs) | GitHub Action + 4 via JUnit XML |
| Observability | 0 | ALL via OTel |
| IDE/Editor | 0 | 4 via MCP Server |
| Notifications | 0 | 3 (Slack, Teams, PagerDuty) |
| Protocol Testing | 0 | 3 (MCP, A2A, AG-UI) |
| Package Registries | 0 | PyPI (TS SDK in v2.0) |
| Paper Contributions | 10 | 13 |

---

## 5. Implementation Phases

### Phase A: Core IP (Week 1-2)
- [ ] OTel GenAI trace ingestion adapter
- [ ] `gen_ai.testing.*` semantic convention proposal draft
- [ ] Cross-framework fingerprint validation tests
- [ ] Protocol compliance test framework (MCP first)

### Phase B: Framework Blast (Week 2-3)
- [ ] PydanticAI adapter + tests
- [ ] DSPy adapter + tests
- [ ] Haystack adapter + tests
- [ ] LlamaIndex adapter + tests
- [ ] Google ADK adapter + tests
- [ ] OTel universal adapter + tests

### Phase C: Distribution (Week 3-4)
- [ ] GitHub Action (Docker-based, Marketplace listing)
- [ ] MCP Server (4 tools, stdio transport)
- [ ] PyPI publish (`pip install agentassay`)
- [ ] Slack webhook notifications
- [ ] Microsoft Teams webhook notifications
- [ ] PagerDuty Events API v2

### Phase D: Observability (Week 4-5)
- [ ] OTel SDK instrumentation (spans + metrics)
- [ ] OTLP exporter (gRPC + HTTP)
- [ ] Prometheus metrics endpoint (`/metrics`)
- [ ] 4 Grafana dashboard templates (JSON)
- [ ] W&B experiment logging integration

### Phase E: Paper Extension (Week 5-6)
- [ ] Section 7b: Universal Agent Test Protocol
- [ ] Section 6b: Protocol Compliance Testing
- [ ] Extended Section 7: Cross-Framework Fingerprinting
- [ ] arXiv v2 submission

---

## 6. File Structure (New Additions)

```
src/agentassay/
├── integrations/
│   ├── (existing 10 adapters)
│   ├── pydantic_ai.py        # NEW
│   ├── dspy_adapter.py        # NEW
│   ├── haystack.py            # NEW
│   ├── llama_index.py         # NEW
│   ├── google_adk.py          # NEW
│   └── otel_ingester.py       # NEW — universal OTel adapter
├── telemetry/
│   ├── __init__.py
│   ├── exporter.py            # OTel span/metric exporter
│   ├── conventions.py         # gen_ai.testing.* definitions
│   └── prometheus.py          # /metrics endpoint
├── protocols/
│   ├── __init__.py
│   ├── mcp_compliance.py      # MCP protocol testing
│   ├── a2a_compliance.py      # A2A protocol testing
│   └── agui_compliance.py     # AG-UI protocol testing
├── cicd/
│   ├── __init__.py
│   ├── github_action/         # Action definition + Dockerfile
│   ├── notifications.py       # Slack, Teams, PagerDuty
│   └── badges.py              # Dynamic badge generation
├── mcp_server/
│   ├── __init__.py
│   └── server.py              # MCP server for IDE integration
└── (existing modules unchanged)
```

---

## 7. Non-Goals (Explicitly Deferred)

- TypeScript/npm SDK (v2.0)
- Java/Maven SDK (v2.0)
- VS Code extension (v1.x — MCP Server covers IDE for now)
- JetBrains plugin (v2.0)
- SaaS offering (v1.5)
- React production dashboard (v0.4.0 — Streamlit sufficient)
- SARIF output (v1.x)

---

## 8. Success Metrics

| Metric | Target | Timeframe |
|--------|--------|-----------|
| PyPI downloads | 1,000+ first month | Week 4-8 |
| GitHub stars | 500+ first month | Week 4-8 |
| Framework coverage | 15+ native + OTel universal | Week 3 |
| CI/CD platforms | 5+ supported | Week 4 |
| Paper contributions | 13 total | Week 6 |
| Hacker News front page | 1 launch post | Week 4 |

---

## 9. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| OTel GenAI conventions change | Build abstraction layer, adapt to spec updates |
| Framework APIs break | Pin adapter versions, integration tests per framework |
| Token cost for testing extends | Core invention (3 pillars) directly addresses this |
| Competitor launches | Paper + 13 contributions = academic moat |
| Azure Anthropic API issues | Reference `azure-anthropic-integration.md` — lessons learned |

---

## 10. Decisions to Lock

| # | Decision | Rationale |
|---|----------|-----------|
| 36 | OTel GenAI as universal ingestion format | Build once, support all frameworks via traces |
| 37 | Propose gen_ai.testing.* OTel conventions | First-mover in defining the standard |
| 38 | MCP Server for IDE integration (not VS Code ext) | Covers 4+ IDEs with one build |
| 39 | GitHub Action as only custom CI integration | JUnit XML covers all others |
| 40 | 5 new adapters: PydanticAI, DSPy, Haystack, LlamaIndex, Google ADK | Highest coverage-per-effort ratio |
| 41 | Protocol compliance testing (MCP, A2A, AG-UI) | Tests the new "HTTP of agents" stack |
| 42 | No TypeScript SDK until v2.0 | Python-first, TS is 6-8 weeks of work |
