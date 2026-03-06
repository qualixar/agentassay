# Token-Efficient Testing

## The Token Cost Problem

Testing AI agents is expensive. Every trial requires invoking your agent, which means LLM API calls, tool executions, and token consumption. A straightforward testing strategy looks like this:

> Run every scenario 100 times. Compare pass rates. Decide.

For a test suite with 20 scenarios at 100 trials each, that is 2,000 agent invocations. At $0.01-$0.10 per invocation (depending on model and tool use), a single regression test run costs $20-$200. Run that in CI on every pull request, and you are looking at thousands of dollars per month -- just for testing.

Most teams respond in one of three ways:

1. **Over-test.** Run a fixed, large N for every scenario. Waste budget on stable scenarios that would show clear results in 10 runs.
2. **Under-test.** Cut N to save money. Lose statistical power. Miss real regressions. Ship broken agents.
3. **Skip testing.** Decide the cost is not worth it. Hope for the best. Get burned in production.

All three are symptoms of the same root cause: **traditional testing frameworks treat every scenario as equally expensive to validate, and they waste tokens on information that could be obtained for free.**

AgentAssay eliminates this waste through three techniques that, combined, deliver the same statistical confidence at 5-20x lower cost.

---

## Technique 1: Behavioral Fingerprinting

### The Insight

When you test for regression, you are not really asking "did the agent produce the same text?" You are asking "did the agent *behave* the same way?" An agent that produces different words but follows the same tool sequence, visits the same states, and reaches the same outcome has not regressed.

Traditional approaches compare raw outputs -- high-dimensional, noisy, and sensitive to irrelevant variation. This means you need many samples to distinguish signal from noise.

Behavioral fingerprinting compresses an agent's execution into a compact representation:

- **Tool sequence:** Which tools were called, in what order
- **State transitions:** What states the agent visited during execution
- **Decision points:** Where the agent chose between alternatives, and what it chose
- **Outcome properties:** Success/failure, cost, latency, output structure

This fingerprint is low-dimensional. Detecting change in a low-dimensional space requires far fewer samples than detecting change in the raw output space.

### How It Works

1. **Extract:** From each execution trace, extract the behavioral fingerprint -- a structured summary of what the agent did (not what it said).
2. **Aggregate:** Compute a distribution over fingerprints from a set of runs.
3. **Compare:** Use a statistical distance measure to compare the baseline fingerprint distribution to the current version.
4. **Decide:** If the fingerprint distance exceeds a calibrated threshold, flag a regression.

### Why It Saves Tokens

Because the fingerprint captures the essential behavior in fewer dimensions, the statistical tests converge faster. Where raw output comparison might need 100 samples to detect a behavior change, fingerprint comparison often needs 15-25.

Think of it this way: if you are comparing two photographs pixel by pixel, you need many samples to be sure a change is real (individual pixels are noisy). But if you extract key features -- "is there a person?", "are they smiling?", "what color is the background?" -- you can detect differences with far fewer examples.

### When to Use

- Regression testing between agent versions (the primary use case)
- Comparing agent behavior across different models
- Monitoring production agent behavior over time
- Any scenario where *what the agent does* matters more than *what it says*

---

## Technique 2: Adaptive Budget Optimization

### The Insight

Not all scenarios are equally variable. A deterministic lookup agent might produce identical results 99% of the time -- you need very few trials to confirm it works. A creative writing agent might produce wildly different outputs every run -- you need many trials to estimate its pass rate.

Fixed-N testing ignores this completely. It runs 100 trials for the lookup agent (wasting 90 of them) and 100 trials for the writing agent (which might actually need 150 to reach a conclusive verdict).

Adaptive budget optimization computes the **exact minimum number of trials** each scenario needs, based on its actual behavioral variance.

### The Calibration Process

1. **Calibration phase:** Run a small set of trials (5-10 runs) for the scenario. This is the only mandatory token expenditure.
2. **Variance estimation:** From the calibration trials, estimate the behavioral variance -- how much does the agent's behavior fluctuate across runs?
3. **Sample size computation:** Given the estimated variance, the desired confidence level (alpha), and the minimum detectable effect size (delta), compute the exact number of additional trials needed.
4. **Budget report:** Output the recommended N, estimated cost, and savings compared to a fixed budget.

### The Math (Simplified)

The core equation for sample size determination:

```
N = f(variance, alpha, beta, delta)
```

Where:
- **variance** is measured from calibration
- **alpha** is the false positive rate you accept (e.g., 0.05)
- **beta** is the false negative rate you accept (e.g., 0.10)
- **delta** is the minimum effect size you want to detect (e.g., 10% pass rate drop)

High variance = more trials needed. Low variance = fewer trials needed. The formula accounts for all four parameters to give you the tightest possible bound.

### Cost Breakdown Example

| Scenario | Variance | Fixed-100 Cost | Adaptive N | Adaptive Cost | Savings |
|----------|----------|----------------|------------|---------------|---------|
| Lookup agent | Low | $1.00 | 12 | $0.12 | 88% |
| Calculator agent | Low | $1.50 | 15 | $0.23 | 85% |
| Research agent | Medium | $5.00 | 38 | $1.90 | 62% |
| Creative writer | High | $8.00 | 85 | $6.80 | 15% |
| **Total** | | **$15.50** | | **$9.05** | **42%** |

The savings are largest for low-variance scenarios (where fixed-N wastes the most) and smallest for high-variance scenarios (where you genuinely need many runs). Across a mixed test suite, typical savings are 40-83%.

### When to Use

- Every testing scenario benefits from adaptive budgeting. There is no downside.
- Particularly valuable for large test suites with heterogeneous scenario complexity.
- Essential for CI/CD pipelines where cost per run accumulates rapidly.

---

## Technique 3: Trace-First Offline Analysis

### The Insight

Many agent testing analyses do not require running the agent at all. They can be performed on traces that already exist:

- **Coverage analysis:** Which tools did the agent use? Which states did it visit? Which paths did it take? This information is in the traces.
- **Contract checking:** Does the agent's behavior satisfy its behavioral contracts? Check the traces.
- **Metamorphic relation verification:** Do invariant properties hold across related inputs? If you have traces for related inputs, check them offline.
- **Mutation score estimation:** How sensitive is your test suite to agent perturbations? Historical traces can inform this.

Running these analyses on existing production traces costs zero additional tokens. You already paid for those runs. The information is sitting in your trace store, waiting to be analyzed.

### What Can Be Tested Offline

| Analysis | Requires New Runs? | Offline Source |
|----------|:-------------------:|----------------|
| Coverage metrics (5D) | No | Production traces |
| Contract compliance | No | Production traces |
| Metamorphic relations | Depends | Pairs of related traces |
| Regression detection | Partially | New version needs runs; baseline is from traces |
| Mutation testing | Yes | Must run mutated agents |
| Pass rate estimation | Yes | Must run current version |

The key insight is the split: some analyses are **purely offline** (coverage, contracts), some are **hybrid** (regression needs new-version runs but reuses baseline traces), and some are **online-only** (mutation testing). AgentAssay handles all three modes and routes each analysis to the most cost-efficient path.

### Trace Sources

AgentAssay can ingest traces from multiple sources:

- **Production monitoring:** Real user interactions with your agent (highest fidelity)
- **Staging runs:** Pre-deployment test executions
- **Historical test results:** Previous AgentAssay test runs
- **Framework-native logs:** Traces exported from your agent framework

### Cost Impact

For a typical test suite with 20 scenarios:

| Analysis | Traditional Cost | Trace-First Cost |
|----------|:----------------:|:----------------:|
| Coverage report | $20-$40 (full re-run) | $0 (offline) |
| Contract audit | $20-$40 (full re-run) | $0 (offline) |
| Regression test | $40-$80 (baseline + current) | $20-$40 (current only) |
| Mutation testing | $80-$200 (multiple mutations) | $80-$200 (cannot skip) |
| **Total** | **$160-$360** | **$100-$240** |

That is a 30-40% reduction just from trace reuse -- and this stacks with the other two techniques.

---

## Technique 4: Multi-Fidelity Proxy Testing

### The Insight

Not every test needs to run against your most expensive model. A smaller, cheaper model that approximates your production model's behavior can serve as a screening layer:

1. **Screen:** Run scenarios against a cheaper proxy model. Scenarios that clearly pass or clearly fail are resolved immediately.
2. **Confirm:** Only scenarios in the uncertain range (INCONCLUSIVE) are escalated to the full production model.

This is analogous to a funnel: the cheap layer filters out the easy cases, and the expensive layer handles only the hard ones.

### Savings Profile

If 60-70% of scenarios produce clear verdicts on the cheap model, you only pay full price for 30-40% of your test suite. Combined with adaptive budgeting, this reduces total cost by an additional 30-50%.

### When to Use

- When your production model is expensive (e.g., large frontier models)
- When you have access to a cheaper model with similar behavioral characteristics
- When your test suite has a mix of easy and hard scenarios
- NOT appropriate when model-specific behavior is what you are testing

---

## Technique 5: Warm-Start Sequential Testing

### The Insight

Sequential testing (SPRT) already allows early stopping -- if the evidence is clear after 15 runs, stop there instead of running all 100. Warm-start takes this further by incorporating **prior information** from previous test runs.

If your agent passed at 95% last week, and you made a minor prompt change, the prior distribution starts near 95%. The sequential test only needs enough evidence to confirm or refute that prior. For minor changes, this often means fewer than 10 additional runs.

### How It Works

1. **Load prior:** Retrieve the most recent test results for this scenario from the trace store.
2. **Update prior:** Incorporate the prior pass rate into a Bayesian framework.
3. **Sequential testing:** Run trials one at a time, updating the posterior after each.
4. **Early stop:** Stop as soon as the posterior probability of regression (or no regression) exceeds the decision threshold.

### Savings Profile

- For stable agents with minor changes: 50-80% reduction in trials
- For agents with major changes: minimal savings (the prior is quickly overwhelmed)
- Best combined with adaptive budgeting as a fallback

---

## Combined Savings

The five techniques are not mutually exclusive. They stack:

```
                        Full-Price Testing: $200/run
                                |
                    Trace-First (-30-40%)
                                |
                           $120-$140
                                |
                 Adaptive Budget (-40-83%)
                                |
                            $20-$85
                                |
              Behavioral Fingerprinting (-30-50%)
                                |
                            $10-$60
                                |
              Multi-Fidelity Proxy (-30-50%)
                                |
                            $5-$40
                                |
              Warm-Start Sequential (-10-50%)
                                |
                            $4-$35
                                |
              ================================
              Final: $4-$35 vs $200 original
              Savings: 5-50x depending on suite
```

The exact savings depend on your test suite composition, model costs, and scenario variance. For a typical mixed suite (20 scenarios, mix of simple and complex agents), our experiments show **5-20x cost reduction** with no loss in statistical confidence.

---

## Practical Examples

### Example 1: CI/CD Pipeline

**Before AgentAssay:**
- 20 scenarios x 100 trials = 2,000 invocations per PR
- Cost: ~$150 per PR at $0.075/invocation
- 10 PRs/week = $1,500/week = $6,000/month

**After AgentAssay (all techniques):**
- 5 scenarios resolved offline (coverage + contracts): 0 invocations
- 10 scenarios resolved by adaptive budget (avg N=18): 180 invocations
- 5 scenarios need full runs (avg N=45): 225 invocations
- Total: 405 invocations per PR
- Cost: ~$30 per PR
- 10 PRs/week = $300/week = $1,200/month
- **Savings: $4,800/month (80%)**

### Example 2: Nightly Regression Suite

**Before:**
- 50 scenarios x 100 trials = 5,000 invocations per night
- Cost: ~$375/night = $11,250/month

**After:**
- 15 scenarios offline: 0 invocations
- 25 scenarios adaptive (avg N=22): 550 invocations
- 10 scenarios full (avg N=60): 600 invocations
- Warm-start reduces totals by 20%: 920 invocations
- Cost: ~$69/night = $2,070/month
- **Savings: $9,180/month (82%)**

---

## When Each Technique Applies

| Technique | Best For | Not For |
|-----------|----------|---------|
| Behavioral fingerprinting | Regression testing, monitoring | Evaluating output quality |
| Adaptive budget | All scenarios | None (always beneficial) |
| Trace-first offline | Coverage, contracts, metamorphic | Pass rate estimation, mutation |
| Multi-fidelity proxy | Mixed suites, expensive models | Model-specific behavior testing |
| Warm-start sequential | Minor changes, frequent testing | Major rewrites, new scenarios |

---

## Getting Started

The fastest path to token-efficient testing:

```python
from agentassay.efficiency import AdaptiveBudgetOptimizer

# 1. Run a small calibration (10 trials)
optimizer = AdaptiveBudgetOptimizer(alpha=0.05, beta=0.10)
estimate = optimizer.calibrate(calibration_traces)

# 2. See the savings
print(f"Recommended trials: {estimate.recommended_n}")
print(f"Estimated cost: ${estimate.estimated_cost_usd:.2f}")
print(f"Savings vs fixed-100: {estimate.savings_vs_fixed_100:.0%}")

# 3. Run only what you need
results = runner.run_trials(scenario, n=estimate.recommended_n)
```

For a complete walkthrough, see the [Quickstart](../getting-started/quickstart.md).
