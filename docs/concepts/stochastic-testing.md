# Stochastic Testing

## The Problem with Binary Testing

Traditional software tests are deterministic: given the same input, a function produces the same output. You run the test once, and the result is definitive.

AI agents are different. An agent powered by a large language model may:

- Produce different outputs for the same input due to sampling (temperature > 0)
- Choose different tools or reasoning paths across runs
- Encounter different results from external API calls
- Succeed 90% of the time but fail 10% -- and a single test run might happen to catch that 10%

If you run an agent test once and it passes, you know almost nothing. If you run it once and it fails, it might have been bad luck. **A single binary verdict is unreliable for non-deterministic systems.**

## The AgentAssay Approach

AgentAssay redefines what a "test" means for AI agents:

1. **Run the agent N times** (e.g., 30 trials) instead of once
2. **Compute statistical estimates** of the true pass rate
3. **Issue a three-valued verdict** backed by confidence intervals

This is the difference between "it passed" and "with 95% confidence, the true pass rate is between 82% and 96%."

## Three-Valued Verdicts

Every AgentAssay test produces one of three verdicts:

### PASS

The confidence interval lower bound is at or above the threshold.

**Meaning:** We have enough statistical evidence that the agent meets the required pass rate. The entire confidence interval is above the acceptable threshold.

**Example:** Threshold = 80%. Observed = 28/30 (93%). 95% CI = [79.3%, 98.5%]. CI lower bound >= 80%. Verdict: PASS.

### FAIL

The confidence interval upper bound is below the threshold.

**Meaning:** We have enough statistical evidence that the agent does NOT meet the required pass rate. Even the most optimistic estimate of the true pass rate falls below the threshold.

**Example:** Threshold = 80%. Observed = 15/30 (50%). 95% CI = [33.2%, 66.8%]. CI upper bound < 80%. Verdict: FAIL.

### INCONCLUSIVE

The confidence interval straddles the threshold.

**Meaning:** We do not have enough data to make a definitive determination. The true pass rate MIGHT be above or below the threshold -- we need more trials to narrow the confidence interval.

**Example:** Threshold = 80%. Observed = 23/30 (77%). 95% CI = [59.9%, 88.9%]. The interval spans the 80% threshold. Verdict: INCONCLUSIVE.

**What to do:** Increase the number of trials (N). More data = tighter confidence intervals = definitive verdicts.

## Why Not Just Use a Simple Percentage?

Suppose you run 10 trials and 8 pass (80%). Is the agent reliable?

Without a confidence interval, you cannot tell. The 95% confidence interval for 8/10 is [49.0%, 94.3%]. The true pass rate could be anywhere from 49% to 94%. That is useless for a deployment decision.

With 100 trials and 80 passes, the CI narrows to [71.1%, 86.9%]. Now you have actionable information.

AgentAssay enforces this discipline. It never lets you make deployment decisions based on point estimates alone.

## Regression Detection

Beyond single-version testing, AgentAssay detects regressions between versions using hypothesis testing:

- **Null hypothesis (H0):** The current version's pass rate is equal to or better than the baseline.
- **Alternative hypothesis (H1):** The current version has regressed (lower pass rate).
- **Test:** A one-sided exact test compares the two pass rate distributions.

The regression verdict also has three values:

| Outcome | Meaning |
|---------|---------|
| **FAIL** | Statistically significant regression detected (p < alpha) |
| **PASS** | No regression, and the test had sufficient statistical power |
| **INCONCLUSIVE** | No regression detected, but the test lacked power to confidently rule it out |

The INCONCLUSIVE verdict for regression is critical: it catches cases where you fail to find a regression simply because you ran too few trials, not because there is no regression.

## Adaptive Sampling (SPRT)

Running many trials is expensive when you use real LLM APIs. AgentAssay includes a Sequential Probability Ratio Test (SPRT) option that can stop sampling early:

- If the evidence strongly supports PASS after 15 trials, stop at 15 instead of running all 50.
- If the evidence strongly supports FAIL after 10 trials, stop early and save cost.
- If the evidence is ambiguous, continue to the maximum trial count.

SPRT maintains the same statistical guarantees (alpha and beta error rates) while reducing the average number of trials needed.

## Statistical Parameters

Every AgentAssay test is parameterized by:

| Parameter | Symbol | Default | Meaning |
|-----------|--------|---------|---------|
| Significance level | alpha | 0.05 | Maximum false positive rate (calling regression when there is none) |
| Power | 1 - beta | 0.80 | Probability of detecting a true regression |
| Number of trials | N | 30 | How many times to run the agent per scenario |
| Effect size threshold | delta | 0.10 | Minimum pass rate drop to flag as regression |

These are configurable per test. Stricter values (lower alpha, higher power) require more trials but provide stronger guarantees.

## Comparison with Evaluation Frameworks

| Aspect | Evaluation frameworks | AgentAssay |
|--------|----------------------|------------|
| Runs per test | 1 | N (configurable) |
| Verdict | Pass / Fail | Pass / Fail / Inconclusive |
| Confidence measure | None | Wilson confidence interval |
| Regression detection | Manual comparison | Automated hypothesis test |
| Power analysis | None | Built-in |
| Early stopping | None | SPRT adaptive sampling |
| Cost control | None | Budget limits + SPRT |
