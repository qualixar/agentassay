# Troubleshooting

Common issues and their solutions.

---

## Installation Issues

### `ModuleNotFoundError: No module named 'agentassay'`

**Cause:** AgentAssay is not installed in the active Python environment.

**Fix:**
```bash
pip install agentassay

# Or if working from source:
pip install -e ".[dev]"
```

Make sure you are using the same Python environment where you installed it:
```bash
which python
python -c "import agentassay; print(agentassay.__version__)"
```

### `ImportError` when using a framework adapter

**Cause:** The agent framework package is not installed.

**Fix:** Install the framework extra:
```bash
pip install agentassay[langgraph]   # for LangGraph
pip install agentassay[crewai]      # for CrewAI
pip install agentassay[all]         # for all frameworks
```

Framework adapters use lazy imports. The core package does not depend on any agent framework.

---

## Test Issues

### All tests return INCONCLUSIVE

**Cause:** Not enough trials. The confidence interval is too wide to make a definitive determination.

**Fix:** Increase the number of trials:
```python
@pytest.mark.agentassay(n=100)  # instead of n=10
def test_my_agent(trial_runner):
    ...
```

**Rule of thumb:** For a threshold of 80%, you typically need at least 30 trials for a definitive verdict. For stricter thresholds (90%+), you may need 50-100 trials.

### Tests pass locally but fail in CI

**Possible causes:**

1. **Different model behavior.** If the CI environment uses a different API key or model version, the agent may behave differently. Set a `seed` in your config for reproducibility where possible.

2. **Timing differences.** CI environments may be slower, causing timeouts. Increase `timeout_seconds` in your AssayConfig.

3. **Cost limits.** If the CI budget is lower, trials may be cut short. Check `max_cost_usd`.

4. **Randomness.** Stochastic tests have inherent variance. A test at 81% pass rate with a threshold of 80% may occasionally dip below. Increase trials to narrow the confidence interval.

### `assert_no_regression` fails with a small drop

**Cause:** The drop is statistically significant, even if the absolute difference seems small. With enough trials, even a 2% drop can be significant.

**Options:**
1. Increase `alpha` (e.g., from 0.05 to 0.10) to make the test less sensitive.
2. Increase `effect_size_threshold` to ignore small drops.
3. Check whether the drop is a real regression or an expected change.

### `assert_pass_rate` fails even though the pass rate looks good

**Cause:** The CI lower bound is below the threshold, not just the point estimate. For example, 25/30 (83%) has a Wilson CI of approximately [66.5%, 93.2%]. If your threshold is 80%, the lower bound (66.5%) is below it.

**Fix:** Run more trials to narrow the interval. With 85/100 (85%), the CI narrows to approximately [76.7%, 90.8%].

---

## CLI Issues

### `agentassay: command not found`

**Cause:** The CLI entry point is not on your PATH.

**Fix:**
```bash
# Ensure agentassay is installed
pip install agentassay

# Or run via Python
python -m agentassay --version

# If using a virtual environment, make sure it is activated
source .venv/bin/activate
```

### `agentassay compare` exits with code 1

**Expected behavior.** Exit code 1 means a regression was detected. This is designed for CI integration -- the non-zero exit code will fail your CI pipeline, which is the intended behavior.

To see the details, check the console output. The comparison table shows pass rates, p-values, and effect sizes.

### Empty coverage report

**Cause:** The results JSON file does not contain execution traces. Coverage analysis requires `ExecutionTrace` data inside each trial result.

**Fix:** Ensure your results JSON includes trace data:
```json
[
  {
    "passed": true,
    "trace": {
      "scenario_id": "s1",
      "steps": [...],
      "model": "gpt-4o",
      "framework": "custom",
      ...
    }
  }
]
```

If you are using the CLI `run` command for config validation (not full execution), traces will not be present. Use the Python API (`TrialRunner`) to generate results with full traces.

---

## Statistical Issues

### Warning: `num_trials=X is very low for stochastic testing`

**Cause:** You configured fewer than 10 trials. With very few trials, confidence intervals are extremely wide and verdicts are almost always INCONCLUSIVE.

**Fix:** Use at least 30 trials for meaningful results. For high-confidence production decisions, use 50-100.

### SPRT never stops early

**Possible causes:**
1. The true pass rate is close to the threshold, so evidence accumulates slowly.
2. Alpha and beta are very strict (e.g., 0.01), requiring more evidence.

**Fix:** SPRT is most beneficial when the true pass rate is clearly above or below the threshold. If the rate is borderline, SPRT will run to the maximum trial count, which is expected behavior.

### Effect size is "negligible" but regression is detected

**Possible cause:** Very large sample size. With enough trials, even a tiny difference (e.g., 92% vs. 90%) can be statistically significant. The effect size tells you whether the difference is practically meaningful.

**Recommendation:** Consider both the p-value and the effect size. A statistically significant but negligible effect size may not warrant blocking a deployment.

---

## Performance Issues

### Tests are slow

**Possible causes:**
1. Each trial invokes a real LLM API. Network latency and token generation time add up.
2. Running many trials sequentially.

**Fixes:**
1. Enable SPRT (`use_sprt=True`) to stop early when the answer is clear.
2. Increase parallelism (`parallel_trials=4`) to run trials concurrently.
3. For development, reduce trial count (`n=10`) and use full counts in CI.
4. Set cost and timeout limits to prevent runaway trials.

### High API costs

**Fixes:**
1. Set `max_cost_usd` to cap total spending per test run.
2. Enable SPRT to reduce average trial count.
3. Use cheaper models for development testing and production models for pre-deploy tests.
4. Save and reuse baseline results instead of re-running them on every test.

---

## Getting Help

If your issue is not covered here:

1. Check the [GitHub Issues](https://github.com/qualixar/agentassay/issues) for similar problems.
2. Open a new issue with:
   - AgentAssay version (`agentassay --version`)
   - Python version (`python --version`)
   - Minimal reproduction steps
   - Full error output
