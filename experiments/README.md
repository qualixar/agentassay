# AgentAssay Experiments

Experiment suite for validating AgentAssay's formal regression testing framework across 4 enterprise agent scenarios, 7 LLM models, and 6 research questions.

## Scenarios

| ID | Scenario | Test Cases | Tools | Domain |
|----|----------|-----------|-------|--------|
| ecommerce | Enterprise E-Commerce Agent | 12 | 6 | Retail (product search, inventory, cart, checkout, shipping) |
| customer_support | Customer Support Ticket Agent | 12 | 6 | CRM (classification, KB lookup, SLA, escalation) |
| code_generation | AI Code Generation Agent | 10 | 4 | DevTools (write code, test, iterate, debug) |
| financial_compliance | Financial Compliance Agent | 12 | 6 | FinTech (AML/KYC, sanctions, risk scoring, audit) |

Total: **46 test cases**, **22 tools**, **16 regression injection points**

## Experiments

| ID | Name | Research Question | Estimated Cost | API Calls |
|----|------|-------------------|---------------|-----------|
| E1 | Verdict Soundness & Power | Does the verdict framework maintain Type-I error <= alpha with sufficient detection power? | $38 | 2,500 |
| E2 | Coverage Validation | Do coverage metrics correlate with regression detection ability? | $22 | 1,200 |
| E3 | Mutation Effectiveness | Which mutation operators most effectively detect behavioral changes? | $30 | 12,000 |
| E4 | SPRT Efficiency | How much cost does sequential testing save while preserving guarantees? | $20 | 2,250 |
| E5 | Contract Integration | Can behavioral contracts serve as effective regression test oracles? | $14 | 1,500 |
| E6 | CI/CD Gate | How effective is the deployment gate at blocking regressions in CI/CD? | $20 | 7,500 |

**Total estimated cost: ~$144** (well within $675 remaining budget)

## Models

| Model | Provider | Tier |
|-------|----------|------|
| gpt-4o | OpenAI | 1 |
| gpt-4o-mini | OpenAI | 1 |
| claude-sonnet-4-6 | Anthropic | 3 |
| claude-haiku-3.5 | Anthropic | 3 |
| gemini-2.0-flash | Google | 2 |
| llama-3.3-70b | Together AI | 2 |
| deepseek-v3 | DeepSeek | 1 |

## How to Run

### Prerequisites

```bash
cd 01-agenttest
pip install -e ".[dev]"
```

### Running Individual Experiments

```bash
# E1: Verdict soundness and power
agentassay run --config experiments/configs/e1_verdict.yaml

# E2: Coverage validation
agentassay run --config experiments/configs/e2_coverage.yaml

# E3: Mutation effectiveness
agentassay run --config experiments/configs/e3_mutation.yaml

# E4: SPRT efficiency
agentassay run --config experiments/configs/e4_sprt.yaml

# E5: Contract integration
agentassay run --config experiments/configs/e5_contracts.yaml

# E6: CI/CD deployment gate
agentassay run --config experiments/configs/e6_cicd.yaml
```

### Running All Experiments

```bash
agentassay run --config experiments/configs/e1_verdict.yaml \
               --config experiments/configs/e2_coverage.yaml \
               --config experiments/configs/e3_mutation.yaml \
               --config experiments/configs/e4_sprt.yaml \
               --config experiments/configs/e5_contracts.yaml \
               --config experiments/configs/e6_cicd.yaml
```

### Running with Mock Agents (No API Calls)

Each scenario provides a mock agent function for testing the infrastructure without spending API credits:

```python
from experiments.scenarios.ecommerce import run_ecommerce_agent, TEST_CASES
from agentassay.core.models import AssayConfig
from agentassay.core.runner import TrialRunner, AgentConfig

# Create runner with mock agent
runner = TrialRunner(
    agent_callable=run_ecommerce_agent,
    config=AssayConfig(num_trials=30),
    agent_config=AgentConfig(
        agent_id="ecom-mock",
        name="E-Commerce Mock Agent",
        framework="custom",
        model="mock",
    ),
)

# Run trials
results = runner.run_trials(TEST_CASES[0])
print(f"Pass rate: {sum(r.passed for r in results) / len(results):.2%}")
```

## Results Structure

```
experiments/results/
├── e1/
│   ├── e1_report.json          # Full experiment report
│   ├── e1_type1_rates.csv      # Type-I error rates per alpha
│   └── e1_power_curves.csv     # Detection power per effect size
├── e2/
│   ├── e2_report.json
│   └── e2_coverage_correlation.csv
├── e3/
│   ├── e3_report.json
│   └── e3_mutation_scores.csv
├── e4/
│   ├── e4_report.json
│   └── e4_sprt_savings.csv
├── e5/
│   ├── e5_report.json
│   └── e5_contract_agreement.csv
└── e6/
    ├── e6_report.json
    └── e6_gate_decisions.csv

experiments/figures/
├── e1/
│   ├── type1_error_rates.pdf
│   └── power_curves.pdf
├── e2/
│   └── coverage_correlation.pdf
├── e3/
│   └── mutation_heatmap.pdf
├── e4/
│   └── sprt_savings.pdf
├── e5/
│   └── contract_agreement.pdf
└── e6/
    └── gate_roc.pdf
```

## Scenario Details

### E-Commerce (HERO SCENARIO)

The e-commerce scenario models a real-world enterprise retail assistant with:
- 12 products across 4 categories (footwear, apparel, electronics, accessories)
- 3-warehouse inventory system with realistic stock levels
- 6-tier customer loyalty pricing (standard to employee)
- Cart management with tax calculation
- Checkout validation with payment method verification
- Shipping estimation with carrier-specific delivery windows

**Regression injections:**
1. Remove inventory check instruction (recommends OOS products)
2. Remove pricing tool access (cannot apply tier discounts)
3. Inject misleading availability context (hallucination trigger)
4. Increase temperature 0.3 -> 0.9 (inconsistent recommendations)

### Customer Support

Tier-1 support workflow with:
- 8-article knowledge base
- Order history lookup
- Ticket classification (8 categories, 4 priority levels)
- SLA enforcement (15min to 8hr response windows)
- Escalation routing to 10 departments

### Code Generation

AI coding assistant with:
- 10 coding problems (FizzBuzz to retry decorators)
- Virtual file system for code execution
- Simulated test runner with pass/fail results
- Write-test-execute-debug iteration loop

### Financial Compliance

AML/KYC compliance agent with:
- Risk scoring (amount, geography, customer profile, velocity)
- Sanctions screening (OFAC, EU, UN, UK lists)
- KYC identity verification database
- Regulatory rules by jurisdiction (US BSA, EU AMLD6, UK MLR)
- Mandatory audit trail logging
- PEP (Politically Exposed Person) handling
