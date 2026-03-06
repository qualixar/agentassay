# Glossary

Key terms used throughout AgentAssay documentation and the codebase.

---

### Alpha (Significance Level)

The maximum acceptable probability of a Type I error -- concluding that a regression exists when it does not. Default: 0.05 (5%). Lower alpha values require stronger evidence to declare a regression.

### Assay

A complete test run: N stochastic trials of an agent on a given scenario, producing a stochastic verdict. The term comes from laboratory science, where an assay is a procedure to measure the quality or composition of a substance.

### Beta (Type II Error Rate)

The probability of failing to detect a true regression. Related to power by: power = 1 - beta. Default beta: 0.20, so default power: 0.80.

### Boundary Coverage

One of the five coverage dimensions. Measures how well the test suite exercises edge cases: timeouts, cost limits, maximum step counts, empty inputs, and error conditions.

### Confidence Interval (CI)

A range of values that, with a specified probability (e.g., 95%), contains the true pass rate. AgentAssay uses Wilson score intervals by default because they have good coverage properties for small sample sizes.

### Coverage Tuple

The five-dimensional vector C = (C_tool, C_path, C_state, C_boundary, C_model) that measures how thoroughly a test suite exercises agent behavior. See [Coverage Metrics](../concepts/coverage.md).

### Deployment Gate

A decision point in a CI/CD pipeline that allows or blocks a deployment based on stochastic test verdicts. Gates aggregate multiple test results and apply a policy (strict, moderate, or permissive) to produce a binary ALLOW/BLOCK decision.

### Effect Size

A standardized measure of how large the difference between two groups is. AgentAssay uses Cohen's h for comparing proportions (pass rates). Small effect: 0.2, Medium: 0.5, Large: 0.8.

### Execution Trace

A complete record of one agent invocation: the ordered sequence of steps, the input, the output, timing, cost, and success/failure status. Execution traces are immutable after creation.

### Fisher's Exact Test

A statistical test for comparing two proportions (pass rates). Used as the default regression test because it is exact (not an approximation) and works well with small sample sizes.

### Gate Decision

The output of a deployment gate: ALLOW (proceed with deployment) or BLOCK (stop deployment). Includes the reason and supporting statistical evidence.

### Inconclusive

One of the three possible verdicts. Indicates that there is not enough statistical evidence to determine PASS or FAIL. The confidence interval straddles the threshold, or the test lacks sufficient statistical power. Remedy: increase the number of trials.

### Metamorphic Relation

A property that should hold when the input is transformed in a specific way. For example, "paraphrasing the input should not change the agent's conclusion." AgentAssay includes seven metamorphic relations organized into four families.

### Model Coverage

One of the five coverage dimensions. Measures the fraction of model variants (e.g., GPT-4o, Claude Opus) that the test suite validates.

### Mutation Operator

A function that introduces a specific perturbation to the agent's configuration (prompt, tools, model, or context). If the test suite still passes despite the mutation, the mutant "survives," indicating a blind spot. AgentAssay includes twelve operators across four categories.

### Mutation Score

The fraction of mutants killed (detected) by the test suite: killed / total. A high mutation score means the test suite is sensitive to changes. A low score means tests would pass even if the agent was broken.

### Pass Rate

The proportion of trials where the agent passed the evaluation criteria. Always expressed as a fraction or percentage.

### Path Coverage

One of the five coverage dimensions. Measures the fraction of distinct behavioral paths (action sequences) observed during testing.

### Power (Statistical Power)

The probability of detecting a true regression when one exists. Calculated as 1 - beta. Higher power requires more trials but reduces the chance of missing a real regression. Default: 0.80.

### Scenario

A test case definition that specifies the input, expected properties, and evaluation criteria for an agent test. A scenario is a template -- it does not contain results.

### SPRT (Sequential Probability Ratio Test)

An adaptive sampling method that can stop testing early when the statistical evidence is strong enough. Saves cost by not running unnecessary trials when the answer (PASS or FAIL) is already clear. Maintains the same error guarantees as fixed-sample testing.

### State Coverage

One of the five coverage dimensions. Measures the fraction of distinct intermediate states the agent visited during testing.

### Step Trace

A single atomic action in an agent's execution: a tool call, an LLM generation, a decision branch, or any custom action type. Steps are ordered within their parent execution trace.

### Stochastic Verdict

The output of an AgentAssay test: one of three values (PASS, FAIL, INCONCLUSIVE) backed by confidence intervals and hypothesis tests. Unlike binary verdicts, stochastic verdicts account for the inherent non-determinism of AI agents.

### Tool Coverage

One of the five coverage dimensions. Measures the fraction of available tools that were actually invoked during testing.

### Trial

A single invocation of the agent-under-test on a given input. An assay consists of N trials. Each trial produces a TrialResult containing the execution trace and the per-trial pass/fail verdict.

### Trial Result

The outcome of a single trial: the execution trace, a boolean pass/fail for that individual trial, a continuous quality score in [0, 1], and evaluation details.

### Trial Runner

The execution engine that runs N independent trials of an agent, captures execution traces, applies evaluators, and collects results.

### Verdict Function

The formal (alpha, beta, n)-test triple that maps trial outcomes to stochastic verdicts. Encapsulates the statistical parameters governing how results become verdicts. See [Stochastic Testing](../concepts/stochastic-testing.md).

### Wilson Score Interval

A method for computing confidence intervals for proportions. Recommended for small sample sizes because it has better coverage properties than the normal approximation. Named after Edwin B. Wilson (1927).
