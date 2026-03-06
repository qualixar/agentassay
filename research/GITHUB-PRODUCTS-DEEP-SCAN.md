# AgentTest Deep Scan: GitHub, Products, & Package Registries

**Date:** February 26, 2026
**Researcher:** Partner (CTO Mode)
**For:** Varun Pratap Bhardwaj
**Methodology:** GitHub API search (30+ queries), PyPI/npm/crates.io registry lookups, Hacker News Algolia API, arXiv API, direct repo README analysis
**Zero-Hallucination Policy:** Every entry below was verified via API calls. Sources noted for each. Items marked [UNVERIFIED] indicate data that could not be independently confirmed.

---

## 1. EXECUTIVE SUMMARY

### The Bottom Line

**The space is NOT empty. Multiple tools now exist for AI agent testing, including some that do statistical/probabilistic testing.** However, there is still a significant white space in the intersection of formal mathematical foundations + stochastic regression testing + behavioral contract integration.

### What EXISTS (as of Feb 26, 2026):

| Category | Key Players | Maturity |
|----------|------------|----------|
| **LLM Evaluation Frameworks** | DeepEval (13.8K stars), Ragas, Evidently (7.2K stars) | Mature, well-funded |
| **Agent Observability** | Langfuse (22.3K stars), AgentOps (5.3K stars), Arize Phoenix (8.7K stars) | Mature, production-grade |
| **Prompt Testing/Red-teaming** | promptfoo (10.6K stars), Giskard (5.1K stars) | Mature |
| **Agent Regression Testing** | EvalView (45 stars), agentrial (14 stars), AgentCheck (PyPI) | EARLY STAGE, just emerging |
| **Statistical/Probabilistic Testing** | PUnit (0 stars, Java), agentrial (14 stars), ProbTest-pytest (8 stars, academic) | VERY EARLY, fragmented |
| **Agent QA Platforms** | Rocketship (185 stars), Agentic QE (206 stars) | Early, different focus (E2E QA) |
| **Unit Testing for AI** | Cobalt (45 stars, TypeScript), basalt-ai | Very new (Feb 2026) |

### What NOBODY Has Built (Confirmed Gaps):

1. **Formal mathematical framework for stochastic agent regression testing** -- agentrial uses Wilson CIs and Fisher tests but has NO formal theoretical foundations (no proofs, no theorems, no published paper)
2. **Integration of behavioral contracts (like ABC) with testing** -- nobody connects contract specifications to test generation/verification
3. **Composition-aware testing** -- testing multi-agent pipelines where contract satisfaction is compositional (C1-C4 conditions from ABC)
4. **Drift-aware regression testing** -- connecting drift metrics D(t) to test pass/fail with formal probabilistic satisfaction (p, delta, k)
5. **Coverage metrics for agent behavioral space** -- traditional code coverage adapted for non-deterministic agent behavioral coverage

### CRITICAL FINDING: agentrial IS A DIRECT COMPETITOR

**agentrial** (github.com/alepot55/agentrial, 14 stars, created Feb 5, 2026) is the closest existing tool to what we envisioned. It explicitly markets itself as "The pytest for AI agents" with statistical rigor. However, it is:
- Only 3 weeks old (created Feb 5, 2026)
- 14 stars, 940 PyPI downloads/month
- No academic paper
- No formal mathematical foundations
- Pure engineering tool with good statistical methods but no theoretical grounding
- No behavioral contract integration
- No compositional testing

**This changes the strategy.** We cannot claim "nobody has stochastic agent testing" anymore. We CAN claim "nobody has formally grounded stochastic agent testing with mathematical proofs and contract integration."

---

## 2. GITHUB REPOS -- Full Inventory

### Tier 1: Major Established Frameworks (1000+ stars)

| Repo | Stars | Language | Last Updated | What It Does | Stochastic Testing? | CI/CD? |
|------|-------|----------|--------------|--------------|-------------------|--------|
| confident-ai/deepeval | 13,808 | Python | 2026-02-25 | LLM evaluation framework (pytest-like). Metrics: G-Eval, DAG, faithfulness, hallucination, etc. | NO. Single-run evaluation, not multi-trial statistical testing. Uses LLM-as-judge. | YES (pytest integration) |
| langfuse/langfuse | 22,269 | TypeScript | 2026-02-25 | Open-source LLM engineering platform: observability, evals, prompt management | NO. Observability, not testing. | NO (monitoring platform) |
| promptfoo/promptfoo | 10,648 | TypeScript | 2026-02-25 | Prompt/agent testing, red teaming, vulnerability scanning | NO. Deterministic assertion-based testing. No statistical confidence. | YES (GitHub Action) |
| Arize-ai/phoenix | 8,656 | Python | 2026-02-25 | AI observability & evaluation | NO. Tracing/observability focus. | NO |
| evidentlyai/evidently | 7,230 | Python | 2026-02-25 | ML/LLM observability. Test and monitor AI systems. | PARTIAL. Has statistical tests for data/model drift detection, but NOT for agent behavioral testing specifically. | YES |
| Giskard-AI/giskard-oss | 5,134 | Python | 2026-02-25 | Evaluation & testing for LLM agents. Red teaming, vulnerability scanning. | NO. Vulnerability/quality scanning, not regression testing. | NO |
| AgentOps-AI/agentops | 5,311 | Python | 2026-02-25 | Agent monitoring, cost tracking, benchmarking | NO. Observability/monitoring. | NO |
| truera/trulens | 3,115 | Python | 2026-02-25 | Evaluation and tracking for LLM experiments and AI agents | NO. Feedback functions, not statistical testing. | NO |

### Tier 2: Emerging Agent Testing Tools (10-1000 stars)

| Repo | Stars | Language | Created | What It Does | Stochastic? | Gap vs Our Idea |
|------|-------|----------|---------|--------------|-------------|-----------------|
| SalesforceAIResearch/MCP-Universe | 561 | Python | 2026 | Developing, testing, benchmarking AI agents (MCP focus) | NO | Benchmarking, not regression testing |
| ServiceNow/AgentLab | 520 | Python | 2024 | Testing/benchmarking web agents on diverse tasks | NO | Web agent benchmarks, not general testing |
| awslabs/agent-evaluation | 341 | Python | 2024 | LLM-agent evaluator orchestrating conversations with target agent | NO | LLM-as-judge eval, not statistical |
| proffesor-for-testing/agentic-qe | 206 | TypeScript | 2025-09 | AI-powered QA/QE platform for coding agents (Claude Code focus) | NO | QA testing of software, not testing of agents themselves |
| LLAMATOR-Core/llamator | 197 | Python | 2025 | Framework for testing vulnerabilities of LLMs | NO | Security/vulnerability testing only |
| rocketship-ai/rocketship | 185 | Go | 2025-03 | QA testing framework for coding agents. YAML-driven E2E tests. | NO | Tests web app flows, not agent behavior. E2E QA, not behavioral regression. |
| saharmor/voice-lab | 162 | Python | 2025 | Testing/evaluation for voice agents | NO | Voice-specific only |
| artas728/spelltest | 136 | Python | 2023 | AI-to-AI simulation testing for LLM apps using synthetic personas | PARTIAL. Runs simulations with synthetic users, provides quality scores 0-1. | No statistical confidence intervals. No regression detection. No CI/CD. Cost-prohibitive. |
| braintrustdata/braintrust-sdk | 116 | TypeScript | 2023 | SDK for Braintrust eval platform | NO | Platform SDK, not testing framework |
| Addepto/contextcheck | 91 | Python | 2024 | MIT-licensed RAG/chatbot testing framework (YAML config) | NO | RAG/chatbot focus, not agent behavioral testing |
| strands-agents/evals | 77 | Python | 2026 | Evaluation framework for AI agents (AWS Strands) | NO | Evaluation, not regression testing. LLM-as-judge. |
| georgeguimaraes/tribunal | 61 | Elixir | 2025 | LLM evaluation framework for Elixir | NO | Elixir ecosystem only, evaluation focus |
| ServiceNow/DoomArena | 54 | Python | 2025 | Testing AI agents against security threats | NO | Security threat testing only |
| hidai25/eval-view | **45** | Python | **2025-11** | **Regression testing for AI agents. Golden baseline diffing, tool-call comparison, output drift detection.** | **PARTIAL. Has "Multi-Reference Goldens" for non-deterministic agents (match any of 5 variants). But NO statistical confidence intervals, NO multi-trial runs, NO probabilistic pass/fail.** | **Closest to regression testing. Deterministic baseline diffing, not statistical. No mathematical foundations.** |
| basalt-ai/cobalt | **45** | TypeScript | **2026-02** | **Unit testing for AI agents. LLM judges, custom evaluators, SQLite tracking.** | **NO. Deterministic scoring, LLM-as-judge. No multi-trial statistics.** | **Unit test paradigm, not statistical. No regression detection. TypeScript only.** |
| kweinmeister/agentitest | 44 | Python | 2025 | Browser tests in natural language using browser-use + pytest + Allure | NO | Browser E2E testing, not agent behavior |
| alepot55/agentrial | **14** | Python | **2026-02-05** | **"The pytest for AI agents." Runs agent N times, gives confidence intervals. Wilson CI, Fisher exact test, step-level failure attribution, CUSUM drift detection, trajectory flame graphs.** | **YES. This is the closest direct competitor. Wilson score intervals, Fisher exact test for regression, Mann-Whitney U for cost/latency, Benjamini-Hochberg correction.** | **Strong engineering but NO formal mathematical foundations. No published paper. No behavioral contract integration. No composition-aware testing. No connection to formal semantics.** |
| aviralgarg05/agentunit | 7 | Python | 2025 | Evaluating, monitoring, benchmarking multi-agent systems | NO | Multi-agent eval/monitoring, not regression testing |
| radoslaw-sz/maia | 2 | Python | 2025 | pytest-based framework for testing multi AI agent systems | UNVERIFIED | Very early, 2 stars |

### Tier 3: Niche / Very Early (< 10 stars, Relevant)

| Repo | Stars | What It Does | Stochastic? |
|------|-------|--------------|-------------|
| javai-org/punit | 0 | JUnit 5 extension for probabilistic testing of non-deterministic systems. Statistical thresholds, early termination, SLA-based pass/fail. | **YES. Probabilistic unit testing. But Java-only, zero adoption, not AI-agent specific.** |
| fakkad/driftwatch | 0 | Automated prompt regression detector using McNemar's test for CI gating | **YES (McNemar's test)**. But prompts only, not agents. 0 stars, created Feb 22, 2026. |
| jingbinw/llm-prompt-regression | 3 | LLM prompt regression testing framework | NO. Detects output drift but no statistical guarantees. |
| geminimir/promptproof | 3 | Deterministic LLM testing for CI. Record/Replay + policy-as-code. | PARTIAL. Record/replay determinism, not statistical. |
| lxgicstudios/ai-prompt-test | 0 | Unit testing for AI prompts | NO. Basic assertions. |
| sbroenne/pytest-aitest | 1 | Testing framework for skill engineering | Very early, minimal |
| eniz1806/Vigil-AI | 1 | Testing framework for AI agents -- assertions, snapshots, cost tracking | NO statistical testing |

### Tier 4: Academic / Research Tools

| Repo | Stars | What It Does | Stochastic? |
|------|-------|--------------|-------------|
| itu-square/probtest-pytest | 8 | **ProbTest: PyTest plugin for testing probabilistic programs with statistical guarantees.** Published at SEFM'25 (arXiv:2509.02012). | **YES. Formal academic paper with proofs. But for probabilistic PROGRAMS (RL, randomized data structures), NOT AI agents specifically.** |
| eth-sri/ToolFuzz | 37 | Fuzzing framework for LLM agent tools | NO. Fuzzing, not regression |
| ellydee/acceptance-bench | 85 | LLM evaluation measuring acceptance vs refusal | NO. Evaluation benchmark |

---

## 3. PACKAGE REGISTRIES

### PyPI Packages

| Package | Version | Monthly Downloads | What It Does | Stochastic Agent Testing? |
|---------|---------|-------------------|--------------|--------------------------|
| deepeval | 3.8.6 | **1,853,239** | LLM evaluation framework (pytest-like) | NO |
| ragas | 0.4.3 | **1,045,628** | RAG evaluation framework | NO |
| agentops | 0.4.21 | **526,272** | Agent monitoring/observability | NO |
| trulens | 2.7.0 | 42,835 | LLM evaluation & tracking | NO |
| giskard | 2.19.1 | 30,942 | ML/LLM testing & red teaming | NO |
| promptfoo | 0.1.3 | 9,011 | Python wrapper for promptfoo CLI | NO |
| evalview | 0.3.1 | **1,209** | Agent regression testing (golden baselines) | PARTIAL (multi-reference goldens) |
| agentrial | 0.2.0 | **940** | Statistical agent evaluation (CIs, Fisher) | **YES** |
| agent-eval | 0.1.44 | N/A | Agent evaluation toolkit (AWS) | NO |
| agenttest | 0.1.0 | N/A | Pytest-like testing for AI agents | NO statistical testing |
| agentunit | 0.7.0 | N/A | Multi-agent eval/benchmarking | NO |
| agentcheck | 0.1.0 | N/A | Trace/replay/test for agents | NO |
| spelltest | 0.0.4 | N/A | AI-to-AI simulation testing | PARTIAL |
| baserun | 2.0.9 | N/A | Testing/debugging LLM features | NO |
| honeyhive | 0.2.57 | N/A | HoneyHive SDK | NO |
| parea-ai | 0.2.220 | N/A | Parea AI SDK | NO |
| log10-io | 0.16.1 | N/A | LLM data management | NO |

### npm Packages

| Package | Version | What It Does | Stochastic? |
|---------|---------|--------------|-------------|
| promptfoo | 0.120.25 | LLM eval & testing toolkit | NO |
| @basalt-ai/cobalt | latest | Unit testing for AI agents | NO |
| agenttest | 1.0.0 | **PLACEHOLDER** -- "This is a placeholder for the axon AI agent development tool" | N/A (squatted) |

### crates.io (Rust)

| Package | Downloads | What It Does | Stochastic? |
|---------|-----------|--------------|-------------|
| agents-test-harness | 15 | Test harness for validating AI agent behavior against steering guides | NO |
| adk-eval | 283 | Agent evaluation framework for ADK-Rust | NO |
| llm-test-bench | 35 | CLI for testing/benchmarking LLM apps | NO |

**Notable:** The Rust ecosystem has almost nothing for agent testing. Wide open.

---

## 4. COMMERCIAL PRODUCTS -- Gap Analysis

### Category A: LLM Evaluation Platforms (Well-Funded)

| Product | Funding | What It Does | Agent Regression Testing? | Statistical Guarantees? | Gap vs Our Idea |
|---------|---------|-------------|--------------------------|------------------------|-----------------|
| **Confident AI / DeepEval** | YC W25 | Open-source LLM eval. Metrics: G-Eval, faithfulness, hallucination. Pytest integration. | NO. Single-run evaluation. Compares outputs, not statistical regression. | NO. | No multi-trial statistical testing. No behavioral contracts. |
| **Braintrust** | Series A [UNVERIFIED] | Eval platform with logging, scoring, experiments. | PARTIAL. Can compare experiment runs. | NO formal guarantees. | Comparison tool, not formal testing framework. |
| **Patronus AI** | $3M seed [UNVERIFIED] | AI safety evaluation. Automated red teaming. | NO. Safety/red-teaming focus. | NO. | Different problem domain entirely. |
| **Arize AI / Phoenix** | $38M Series B [UNVERIFIED] | AI observability. Tracing, evaluation. | NO. Observability/monitoring. | PARTIAL. Drift detection via statistical methods. | Monitoring, not testing. Post-deployment, not pre-deployment. |
| **Galileo AI** | [UNVERIFIED] | LLM debugging & evaluation. Hallucination detection. | NO. | NO. | Debugging tool. |
| **Weights & Biases (Weave)** | Well-funded | ML experiment tracking, now with LLM eval features. | NO. Experiment tracking. | NO. | Platform, not testing framework. |
| **LangSmith (LangChain)** | Well-funded | Tracing, evaluation, datasets for LangChain apps. | PARTIAL. Can compare runs. | NO formal guarantees. | Vendor lock-in (LangChain). Eval platform, not testing framework. |
| **Humanloop** | $2.6M seed [UNVERIFIED] | Prompt engineering, evaluation, monitoring. | NO. | NO. | Prompt management platform. |

### Category B: AI Testing Startups (Newer)

| Product | What It Does | Agent Regression? | Statistical? | Gap |
|---------|-------------|-------------------|-------------|-----|
| **Parea AI** | LLM eval/testing with experiments | NO formal regression | NO | Experiment comparison, not regression testing |
| **Okareo** | AI model testing platform | PARTIAL. Synthetic scenario generation. | NO | Test generation, not statistical framework |
| **Kolena** | ML testing platform | YES for traditional ML. | PARTIAL for ML. | Not adapted for AI agents/LLMs specifically |
| **Gentrace** | Prompt testing & monitoring | NO. | NO. | Basic prompt testing |
| **Baserun** | LLM debugging & testing | NO. | NO. | Debugging focus |
| **Log10** | LLM data management | NO. | NO. | Logging/management |
| **HoneyHive** | AI evaluation & monitoring | NO formal regression | NO | Monitoring platform |
| **Helicone** | LLM proxy & observability | NO. | NO. | Observability only |
| **Giskard** | LLM testing & red teaming | NO regression testing | NO | Vulnerability/quality scanning |
| **DeepChecks** | ML/LLM testing | PARTIAL for traditional ML | PARTIAL for ML | Not agent-specific |

### Category C: Agent QA/E2E Testing

| Product | Stars | What It Does | Statistical Agent Regression? |
|---------|-------|-------------|------------------------------|
| **Rocketship** | 185 | QA testing framework for coding agents (YAML E2E tests) | NO. Tests web app flows, not agent behavior itself. |
| **Agentic QE** | 206 | AI-powered QA platform for coding agents | NO. QA of software, not testing agents. |
| **Magnitude** | HN: 179 pts | AI-native test framework for web apps | NO. Web app testing with AI, not agent testing. |

---

## 5. DEVELOPER PAIN SIGNALS

### Hacker News Threads (Verified via hn.algolia.com API)

**[2026-02-20] "Cobalt -- Unit tests for AI agents, like Jest but for LLMs" (3 pts)**
- Shows demand exists but very early traction

**[2025-02-20] "Launch HN: Confident AI (YC W25) -- Open-source evaluation framework for LLM apps" (117 pts, 27 comments)**
- Significant interest in LLM evaluation. YC-backed.
- But evaluation != regression testing

**[2025-04-25] "Magnitude -- open-source, AI-native test framework for web apps" (179 pts, 44 comments)**
- High interest in AI-powered testing of web apps
- Different problem: testing web apps WITH AI, not testing AI agents

**[2026-01-05] "Ask HN: How are you developing and testing agents without burning tokens?"**
- Direct developer pain signal about the cost and difficulty of testing agents

**[2026-02-13] "Khaos -- Every AI agent I tested broke in under 30 seconds"**
- Explicit signal: agents are fragile, testing is critical

### Reddit Threads (from r/LangChain)

**"Is Software Testing Going to Become Irrelevant Due to the Boom of Agentic AI?" (3 pts, 11 comments)**
- Shows the community is actively discussing how testing paradigms need to change for AI agents

### Key Developer Pain Points Identified:

1. **"My agent passes Monday, fails Wednesday. Same prompt, same model."** (agentrial README, citing arXiv:2407.02100 showing 72% variance at temperature=0)
2. **Cost of testing** -- developers burning tokens to manually test agents, no way to batch test efficiently
3. **No CI/CD integration** -- agents ship without automated behavioral testing
4. **Non-determinism** -- traditional pass/fail doesn't work, developers need statistical confidence
5. **Regression detection** -- "Did my agent break?" is the fundamental question nobody could answer until very recently
6. **Multi-agent testing** -- testing pipelines of agents is virtually unsupported

---

## 6. NAME AVAILABILITY

### Comprehensive Check (GitHub, PyPI, npm)

| Name | PyPI | npm | GitHub (notable repos) | Available? |
|------|------|-----|----------------------|------------|
| **agenttest** | TAKEN (v0.1.0, basic framework) | TAKEN (placeholder "axon") | Various small repos | NO |
| **agentprobe** | TAKEN (v0.3.13, CLI tool testing) | TAKEN (v1.3.1, security awareness) | nibzard/agentprobe (21 stars) | NO |
| **agentspec** | TAKEN (v2.1.1, spec-driven dev toolkit) | TAKEN (v1.2.0) | - | NO |
| **agentguard** | AVAILABLE | AVAILABLE | GoPlusSecurity/agentguard (254 stars, security) | NO (GitHub conflict) |
| **agentcheck** | TAKEN (v0.1.0, trace/replay) | AVAILABLE | - | PARTIAL |
| **agentunit** | TAKEN (v0.7.0, multi-agent eval) | AVAILABLE | aviralgarg05/agentunit (7 stars) | NO |
| **agentbench** | TAKEN (v0.0.1) | AVAILABLE | - | NO |
| **testpilot** | TAKEN (v0.2.9) | TAKEN (v0.4.0) | githubnext/testpilot (562 stars) | NO |
| **drifttest** | AVAILABLE | AVAILABLE | mnavascues/drifttest (1 star, biology) | YES (biology repo, no conflict) |
| **stochatest** | AVAILABLE | AVAILABLE | Mieussep/Stochatests (0 stars, test repo) | YES |
| **probtest** | AVAILABLE | AVAILABLE | itu-square/probtest-pytest (8 stars, academic) | PARTIAL (academic conflict) |
| **fluxtest** | AVAILABLE | TAKEN (v1.0.5, demo) | jpchullo/fluxtest (2 stars) | PARTIAL |
| **spectest** | AVAILABLE | TAKEN (v2.0.3, API testing) | - | PARTIAL |
| **behaviortest** | AVAILABLE | AVAILABLE | ircmaxell/BehaviorTest (2 stars, old PHP) | YES |
| **agentassay** | AVAILABLE | AVAILABLE | CLEAR on GitHub | **YES -- FULLY CLEAR** |
| **agentverify** | AVAILABLE | AVAILABLE | AbraxasDaemon/agentverify (0 stars) | YES |
| **agenttrial** | AVAILABLE | AVAILABLE | sjaghub/agenttrial (0 stars) | YES |
| **agenttest-ai** | AVAILABLE | AVAILABLE | Minimal conflicts | YES |

### Recommended Names (Fully Available):

1. **agentassay** -- "Assay" means rigorous scientific testing. Unique, memorable, no conflicts anywhere. Evokes laboratory-grade precision.
2. **stochatest** -- Portmanteau of "stochastic" + "test". Immediately communicates the core differentiator.
3. **drifttest** -- Connects to drift detection, available on all registries.
4. **behaviortest** -- Direct, descriptive. Available everywhere.
5. **agentverify** -- Available but less distinctive.

---

## 7. COMPETITIVE MOAT ANALYSIS -- Where Is the White Space?

### The Landscape Map

```
                    FORMAL FOUNDATIONS
                         ↑
                         |
    ProbTest-pytest ●    |         ??? (OUR OPPORTUNITY)
    (academic, general)  |         Formal + Agent-specific +
                         |         Contract-integrated +
                         |         Composition-aware
                         |
  ──────────────────────|──────────────────────→ AGENT-SPECIFIC
                         |
    Cobalt ●  DeepEval ● |    agentrial ●
    (deterministic)       |    (statistical but
    EvalView ●            |     no formal foundations)
    (baseline diffing)    |
                         |
                    ENGINEERING-ONLY
```

### What Each Competitor Lacks

| Competitor | What They HAVE | What They LACK |
|-----------|---------------|----------------|
| **agentrial** | Wilson CIs, Fisher exact test, trajectory analysis, step-level attribution, CI/CD, CUSUM drift detection | NO formal mathematical framework. NO published paper. NO behavioral contracts. NO composition-aware testing. NO formal proofs of statistical guarantees. |
| **EvalView** | Golden baseline diffing, tool-call comparison, multi-reference goldens, CI/CD | NO multi-trial statistical testing. NO confidence intervals. NO probabilistic pass/fail. Deterministic only. |
| **DeepEval** | 13.8K stars, wide adoption, many metrics, pytest integration | NO agent regression testing. Single-run evaluation only. No statistical guarantees for agents. |
| **PUnit** | Probabilistic testing with JUnit5, statistical thresholds, SLA-based verdicts | Java-only. Zero adoption. NOT agent-specific. NO behavioral contracts. |
| **ProbTest-pytest** | Academic paper (SEFM'25), formal guarantees for probabilistic programs | NOT agent-specific. Designed for RL/randomized algorithms. No agent behavioral testing. |
| **Cobalt** | Clean UX, TypeScript, MCP server for AI assistants | NO statistical testing at all. Deterministic scoring only. |
| **promptfoo** | 10.6K stars, mature, red teaming | Prompt/output testing only. NOT agent behavioral testing. No statistical regression. |

### The 5-Layer Moat We Can Build

**Layer 1: Formal Mathematical Foundations (UNIQUE)**
- Formal definition of stochastic test passing via (p, delta, k)-satisfaction (from ABC)
- Theorems with proofs for statistical guarantees
- Published paper (like ProbTest-pytest has for probabilistic programs, but for AI AGENTS)
- Nobody has this for agents. agentrial has good engineering but zero theory.

**Layer 2: Behavioral Contract Integration (UNIQUE)**
- Connect AgentAssert's ContractSpec to test specifications
- Auto-generate test suites from behavioral contracts
- Verify contract satisfaction statistically
- Nobody connects contracts to testing.

**Layer 3: Composition-Aware Testing (UNIQUE)**
- Test multi-agent pipelines compositionally
- C1-C4 composition conditions from ABC applied to testing
- Pipeline test = composed from individual agent tests
- agentrial has basic multi-agent metrics but no compositional theory.

**Layer 4: Drift-Integrated Regression (DIFFERENTIATED)**
- Connect D(t) drift metric to test verdicts
- Lyapunov stability applied to test suite evolution
- agentrial has CUSUM but no formal drift theory.

**Layer 5: Production Engineering (TABLE STAKES)**
- CI/CD integration (GitHub Action)
- Framework adapters (LangGraph, CrewAI, OpenAI, etc.)
- CLI, YAML config, pytest plugin
- This is what agentrial already does well. We must match it.

---

## 8. WHAT NOBODY HAS BUILT -- Confirmed Gaps

### Gap 1: Formal Theory of Stochastic Agent Testing
**Status: CONFIRMED OPEN**
- ProbTest-pytest has formal theory for probabilistic programs (SEFM'25 paper) but NOT for AI agents
- agentrial has engineering implementation but NO formal theory
- Nobody has published a paper formalizing stochastic regression testing specifically for AI agents
- **Opportunity: First paper to formally define what "test passing" means for non-deterministic agents, with proofs**

### Gap 2: Contract-to-Test Generation
**Status: CONFIRMED OPEN**
- Nobody generates tests FROM behavioral contracts
- Nobody verifies contract satisfaction via statistical testing
- ABC + testing = unique integration point nobody else has
- **Opportunity: ContractSpec YAML --> automated stochastic test suite**

### Gap 3: Compositional Test Theory
**Status: CONFIRMED OPEN**
- agentrial has multi-agent metrics (delegation accuracy, handoff fidelity)
- But NO compositional theory: "if agents A and B pass individually with confidence p, what can we say about pipeline A-->B?"
- **Opportunity: Composition theorems for agent test suites**

### Gap 4: Behavioral Coverage Metrics
**Status: CONFIRMED OPEN**
- Traditional code coverage (line, branch) doesn't apply to agents
- Nobody has defined what "coverage" means for agent behavioral space
- **Opportunity: Define agent behavioral coverage formally**

### Gap 5: Formal Drift-to-Regression Connection
**Status: CONFIRMED OPEN**
- Evidently has data drift detection (for ML models)
- agentrial has CUSUM/Page-Hinkley for change-point detection
- But nobody has formally connected behavioral drift metrics to regression test verdicts
- **Opportunity: D(t) metric --> formal regression criteria with proofs**

---

## APPENDIX A: Full arXiv Paper Search

### Directly Relevant Papers Found

| Paper | arXiv | Date | Relevance |
|-------|-------|------|-----------|
| ProbTest: Unit Testing for Probabilistic Programs | 2509.02012 | Sep 2025 | **HIGHLY RELEVANT.** Formal theory of probabilistic testing with statistical guarantees. Published at SEFM'25. For general probabilistic programs, NOT agents. |
| LLMs show 72% variance at temperature=0 | 2407.02100 | Jul 2024 | Cited by agentrial. Motivates the problem. |

### No Papers Found For:
- "Stochastic regression testing for AI agents" --> **ZERO results on arXiv**
- "Agent CI/CD testing formal framework" --> **ZERO results on arXiv**
- "Behavioral contract testing for LLM agents" --> **ZERO results on arXiv**
- "Compositional testing for multi-agent systems" --> **ZERO results on arXiv** (in the LLM context)

**This confirms the academic gap is wide open.**

---

## APPENDIX B: Awesome List Analysis

**chaosync-org/awesome-ai-agent-testing** (27 stars, last updated Feb 15, 2026)
- Comprehensive curated list covering all categories of agent testing
- Lists frameworks, methodologies, benchmarks, simulation environments
- Categories include: chaos engineering, fault injection, resilience testing
- Notable: Lists behavioral testing, unit testing, integration testing, system testing for agents
- **Does NOT list any tool that does formal stochastic regression testing**
- Confirms our gap analysis

---

## APPENDIX C: Key Competitor Deep Dives

### agentrial (MOST IMPORTANT COMPETITOR)

- **Created:** February 5, 2026 (3 weeks old)
- **Stars:** 14
- **PyPI Downloads:** 940/month
- **Author:** alepot55
- **License:** MIT
- **Statistical Methods:**
  - Wilson score interval for pass rate CI
  - Bootstrap resampling for cost/latency CI
  - Fisher exact test for regression detection
  - Mann-Whitney U for cost/latency comparison
  - Benjamini-Hochberg FDR correction
  - CUSUM/Page-Hinkley for production drift detection
  - Kolmogorov-Smirnov for distribution shift
  - Krippendorff's alpha for LLM-as-judge reliability
- **Features:**
  - Step-level failure attribution
  - Trajectory flame graphs
  - Snapshot testing with regression detection
  - Agent Reliability Score (composite 0-100)
  - Production monitoring (CUSUM drift)
  - Multi-agent evaluation metrics
  - Framework adapters: LangGraph, CrewAI, AutoGen, Pydantic AI, OpenAI Agents, smolagents
  - VS Code extension
  - Benchmark registry
  - MCP security scanner
  - Prompt version control
  - Pareto frontier analysis
  - Eval packs (domain-specific)
- **What It LACKS:**
  - No published paper
  - No formal mathematical framework
  - No proofs of statistical guarantees
  - No behavioral contract integration
  - No compositional testing theory
  - No formal drift theory (uses CUSUM/PH which are standard change-point methods, no Lyapunov)
  - No behavioral coverage metrics
  - No contract-to-test generation

### EvalView

- **Created:** November 17, 2025
- **Stars:** 45
- **PyPI Downloads:** 1,209/month
- **Key Feature:** Golden baseline diffing for agent regression detection
- **Approach:** Deterministic -- save baseline, compare against it
- **Multi-Reference Goldens:** Up to 5 acceptable variants
- **Framework Support:** LangGraph, CrewAI, OpenAI, Claude, HTTP API
- **What It LACKS:**
  - No multi-trial statistical testing
  - No confidence intervals
  - No probabilistic pass/fail
  - Deterministic only -- if ANY variant matches, it passes

### PUnit (Java)

- **Created:** January 3, 2026
- **Stars:** 0
- **Language:** Java only
- **Key Feature:** JUnit 5 extension for probabilistic testing
- **Approach:** Run N samples, statistical threshold for pass/fail
- **Modes:** EXPLORE, OPTIMIZE, MEASURE
- **Statistical:** Budget control, early termination, SLA-based thresholds
- **What It LACKS:**
  - Java only, zero adoption
  - Not agent-specific at all
  - No behavioral contracts
  - No agent framework integration

---

## APPENDIX D: Download/Adoption Comparison

| Tool | Monthly PyPI Downloads | GitHub Stars | Created |
|------|----------------------|--------------|---------|
| deepeval | 1,853,239 | 13,808 | Aug 2023 |
| ragas | 1,045,628 | N/A | 2023 |
| agentops | 526,272 | 5,311 | Aug 2023 |
| trulens | 42,835 | 3,115 | 2023 |
| giskard | 30,942 | 5,134 | 2022 |
| promptfoo | 9,011 | 10,648 | 2023 |
| evalview | 1,209 | 45 | Nov 2025 |
| agentrial | 940 | 14 | Feb 2026 |

**Key Insight:** The massive download numbers for deepeval/ragas/agentops show enormous developer demand for AI testing/eval tools. But the agent REGRESSION testing space (evalview, agentrial) is brand new with tiny adoption -- confirming market timing is NOW.
