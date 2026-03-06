# Mutation Testing

## What Is Mutation Testing?

Mutation testing answers a simple question: **How sensitive are your tests?**

If you change the agent slightly -- perturb a prompt, remove a tool, swap a model -- do your tests catch the change? If they do, the "mutant" is **killed**. If the tests still pass despite the mutation, the mutant **survives**, revealing a blind spot in your test suite.

The **mutation score** is the fraction of mutants killed:

```
Mutation Score = killed mutants / total mutants
```

A high mutation score means your tests are sensitive to changes. A low score means your tests would pass even if the agent was broken.

## Why It Matters for Agents

Consider this scenario: you have 30 tests, all passing at 90%+. You feel confident. But then you accidentally delete a critical tool from the agent's configuration, and... all 30 tests still pass. Your test suite was testing the happy path but never verified that the agent actually uses that tool.

Mutation testing would have caught this. The "tool removal" mutant would survive, telling you that no test depends on that tool being present.

## The Twelve Mutation Operators

AgentAssay includes twelve mutation operators organized into four categories.

### Category 1: Prompt Mutations

These operators modify the agent's system prompt or instructions.

| Operator | What it does |
|----------|-------------|
| **Synonym Substitution** | Replaces key terms in the prompt with synonyms (e.g., "analyze" becomes "examine"). Tests whether the agent's behavior depends on exact wording. |
| **Instruction Order** | Shuffles the order of instructions in the system prompt. Tests whether the agent is sensitive to instruction sequence. |
| **Noise Injection** | Adds irrelevant or distracting text to the prompt. Tests robustness to noisy inputs. |
| **Instruction Drop** | Removes one instruction from the prompt. Tests whether each instruction actually contributes to correct behavior. |

### Category 2: Tool Mutations

These operators modify the agent's tool configuration.

| Operator | What it does |
|----------|-------------|
| **Tool Removal** | Removes one tool from the agent's available tool set. Tests whether your test suite verifies that each tool is necessary. |
| **Tool Reorder** | Changes the order of tools in the tool list. Tests whether the agent (or tests) depend on tool ordering. |
| **Tool Noise** | Alters tool descriptions or parameter schemas. Tests whether the agent handles imprecise tool definitions gracefully. |

### Category 3: Model Mutations

These operators modify the model configuration.

| Operator | What it does |
|----------|-------------|
| **Model Swap** | Replaces the agent's model with a different one (e.g., swap a large model for a smaller one). Tests whether your test suite detects quality differences across models. |
| **Model Version** | Changes the model version (e.g., from a stable version to a preview version). Tests sensitivity to model version changes. |

### Category 4: Context Mutations

These operators modify the context or memory provided to the agent.

| Operator | What it does |
|----------|-------------|
| **Context Truncation** | Truncates the conversation history or context window. Tests whether the agent handles incomplete context. |
| **Context Noise** | Injects irrelevant information into the context. Tests robustness to noisy context. |
| **Context Permutation** | Reorders items in the context (e.g., shuffles conversation history). Tests whether the agent depends on context order. |

## Interpreting the Mutation Score

| Score | Interpretation |
|-------|---------------|
| >= 0.80 | **Strong.** Your test suite detects most perturbations. High confidence that real regressions will be caught. |
| 0.50 - 0.79 | **Moderate.** Some blind spots exist. Investigate surviving mutants to identify what your tests miss. |
| < 0.50 | **Weak.** Your tests pass regardless of significant changes to the agent. The test suite provides a false sense of security. |

## Analyzing Surviving Mutants

When a mutant survives, it tells you something specific:

- **Synonym substitution survives:** Your tests check outputs but not the reasoning process.
- **Tool removal survives:** No test depends on that tool. Add a scenario that requires it.
- **Model swap survives:** Your tests are not sensitive to quality differences between models.
- **Context truncation survives:** Your tests use short contexts that fit within any truncation.

Each surviving mutant is an actionable finding: it tells you exactly what kind of test to add.

## Running Mutation Testing

### From the CLI

```bash
# Run all mutation operators
agentassay mutate --config agent.yaml --scenario qa.yaml

# Run only specific categories
agentassay mutate -c agent.yaml -s qa.yaml --operators prompt,tool
```

### From Python

```python
from agentassay.mutation import MutationRunner, DEFAULT_OPERATORS

runner = MutationRunner(
    agent_callable=my_agent,
    operators=DEFAULT_OPERATORS,
    trials_per_mutant=30,
)

result = runner.run_suite(scenario)

print(f"Mutation score: {result.mutation_score:.2%}")
print(f"Killed: {result.killed} / {result.total}")

# Inspect surviving mutants
for mutant in result.survivors:
    print(f"  SURVIVED: {mutant.operator_name} - {mutant.description}")
```

## Mutation Testing + Stochastic Verdicts

Since agents are non-deterministic, killing a mutant is also a statistical question. A mutant is "killed" if the mutated agent's test results show a statistically significant regression compared to the original. This uses the same hypothesis testing framework as regression detection.

This avoids false kills from random variation and false survivals from underpowered tests.
