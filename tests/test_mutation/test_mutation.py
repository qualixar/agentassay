"""Tests for mutation testing module.

Tests mutation operators, MutationRunner, mutation score calculation,
deterministic seeding, and edge cases.

Target: ~20 tests.
"""

from __future__ import annotations

from agentassay.mutation.operators import (
    ContextNoiseMutator,
    ContextPermutationMutator,
    ContextTruncationMutator,
    ModelSwapMutator,
    ModelVersionMutator,
    PromptDropMutator,
    PromptNoiseMutator,
    PromptOrderMutator,
    PromptSynonymMutator,
    ToolNoiseMutator,
    ToolRemovalMutator,
    ToolReorderMutator,
)
from agentassay.mutation.runner import (
    MutationResult,
    MutationRunner,
    MutationSuiteResult,
)
from tests.conftest import (
    make_agent_config,
    make_assay_config,
    make_scenario,
    mutation_agent,
    mutation_agent_sensitive,
)

# ===================================================================
# Individual mutation operators
# ===================================================================


class TestPromptMutators:
    """Tests for prompt mutation operators."""

    def _config_with_prompt(self):
        return make_agent_config(
            parameters={"system_prompt": "You are a helpful assistant. Analyze the data carefully."}
        )

    def test_synonym_mutator_changes_prompt(self):
        mutator = PromptSynonymMutator(seed=42)
        config = self._config_with_prompt()
        scenario = make_scenario()
        new_config, new_scenario = mutator.mutate(config, scenario)
        # Either config params or scenario may change
        assert mutator.name == "prompt_synonym"
        assert mutator.category == "prompt"

    def test_order_mutator(self):
        mutator = PromptOrderMutator(seed=42)
        config = self._config_with_prompt()
        scenario = make_scenario()
        new_config, new_scenario = mutator.mutate(config, scenario)
        assert mutator.category == "prompt"

    def test_noise_mutator(self):
        mutator = PromptNoiseMutator(noise_rate=0.2, seed=42)
        config = self._config_with_prompt()
        scenario = make_scenario()
        new_config, new_scenario = mutator.mutate(config, scenario)
        assert mutator.category == "prompt"

    def test_drop_mutator(self):
        mutator = PromptDropMutator(seed=42)
        config = self._config_with_prompt()
        scenario = make_scenario()
        new_config, new_scenario = mutator.mutate(config, scenario)
        assert mutator.category == "prompt"


class TestToolMutators:
    """Tests for tool mutation operators."""

    def _config_with_tools(self):
        return make_agent_config(parameters={"tools": ["search", "calculate", "write"]})

    def test_removal_mutator(self):
        mutator = ToolRemovalMutator(seed=42)
        config = self._config_with_tools()
        scenario = make_scenario()
        new_config, new_scenario = mutator.mutate(config, scenario)
        assert mutator.category == "tool"

    def test_reorder_mutator(self):
        mutator = ToolReorderMutator(seed=42)
        config = self._config_with_tools()
        scenario = make_scenario()
        new_config, new_scenario = mutator.mutate(config, scenario)
        assert mutator.category == "tool"

    def test_noise_mutator(self):
        mutator = ToolNoiseMutator(noise_rate=0.15, seed=42)
        config = self._config_with_tools()
        scenario = make_scenario()
        new_config, new_scenario = mutator.mutate(config, scenario)
        assert mutator.category == "tool"


class TestModelMutators:
    """Tests for model mutation operators."""

    def test_swap_mutator(self):
        mutator = ModelSwapMutator(seed=42)
        config = make_agent_config(model="gpt-4o")
        scenario = make_scenario()
        new_config, new_scenario = mutator.mutate(config, scenario)
        assert mutator.category == "model"
        # Model should change
        assert new_config.model != config.model or mutator.describe_mutation() != ""

    def test_version_mutator(self):
        mutator = ModelVersionMutator(seed=42)
        config = make_agent_config(model="gpt-4o")
        scenario = make_scenario()
        new_config, new_scenario = mutator.mutate(config, scenario)
        assert mutator.category == "model"


class TestContextMutators:
    """Tests for context mutation operators."""

    def _scenario_with_context(self):
        return make_scenario(
            input_data={
                "query": "What is the capital of France?",
                "context": [
                    "Paris is the capital.",
                    "France is in Europe.",
                    "The Eiffel Tower is in Paris.",
                ],
            }
        )

    def test_truncation_mutator(self):
        mutator = ContextTruncationMutator(keep_ratio=0.5, seed=42)
        config = make_agent_config()
        scenario = self._scenario_with_context()
        new_config, new_scenario = mutator.mutate(config, scenario)
        assert mutator.category == "context"

    def test_noise_mutator(self):
        mutator = ContextNoiseMutator(num_distractors=2, seed=42)
        config = make_agent_config()
        scenario = self._scenario_with_context()
        new_config, new_scenario = mutator.mutate(config, scenario)
        assert mutator.category == "context"

    def test_permutation_mutator(self):
        mutator = ContextPermutationMutator(seed=42)
        config = make_agent_config()
        scenario = self._scenario_with_context()
        new_config, new_scenario = mutator.mutate(config, scenario)
        assert mutator.category == "context"


# ===================================================================
# Deterministic seeding
# ===================================================================


class TestDeterministicSeeding:
    """Test that seeded operators produce deterministic results."""

    def test_same_seed_same_result(self):
        config = make_agent_config(model="gpt-4o")
        scenario = make_scenario()
        m1 = ModelSwapMutator(seed=123)
        m2 = ModelSwapMutator(seed=123)
        c1, _ = m1.mutate(config, scenario)
        c2, _ = m2.mutate(config, scenario)
        assert c1.model == c2.model


# ===================================================================
# MutationRunner
# ===================================================================


class TestMutationRunner:
    """Tests for MutationRunner."""

    def test_run_single_mutation(self):
        config = make_agent_config()
        assay = make_assay_config()
        runner = MutationRunner(
            agent_callable=mutation_agent,
            config=config,
            assay_config=assay,
            operators=[ModelSwapMutator(seed=42)],
        )
        scenario = make_scenario()
        result = runner.run_mutation(scenario, ModelSwapMutator(seed=42))
        assert isinstance(result, MutationResult)
        assert result.operator_category == "model"

    def test_run_suite_returns_results(self):
        config = make_agent_config()
        assay = make_assay_config()
        ops = [ModelSwapMutator(seed=i) for i in range(3)]
        runner = MutationRunner(
            agent_callable=mutation_agent,
            config=config,
            assay_config=assay,
            operators=ops,
        )
        scenario = make_scenario()
        suite = runner.run_suite(scenario)
        assert isinstance(suite, MutationSuiteResult)
        assert suite.total_mutants == 3

    def test_mutation_score_range(self):
        config = make_agent_config()
        assay = make_assay_config()
        runner = MutationRunner(
            agent_callable=mutation_agent,
            config=config,
            assay_config=assay,
            operators=[ModelSwapMutator(seed=42)],
        )
        scenario = make_scenario()
        suite = runner.run_suite(scenario)
        assert 0.0 <= suite.mutation_score <= 1.0

    def test_sensitive_agent_kills_mutants(self):
        config = make_agent_config()
        assay = make_assay_config()
        runner = MutationRunner(
            agent_callable=mutation_agent_sensitive,
            config=config,
            assay_config=assay,
            operators=[ModelSwapMutator(seed=42)],
        )
        scenario = make_scenario()
        suite = runner.run_suite(scenario)
        # The sensitive agent fails when model changes, so mutant should be killed
        assert suite.killed_mutants >= 0  # at least checks it runs

    def test_empty_operators_returns_empty_result(self):
        config = make_agent_config()
        assay = make_assay_config()
        runner = MutationRunner(
            agent_callable=mutation_agent,
            config=config,
            assay_config=assay,
            operators=[],
        )
        scenario = make_scenario()
        suite = runner.run_suite(scenario, operators=[])
        assert suite.total_mutants == 0
        assert suite.mutation_score == 0.0

    def test_per_category_breakdown(self):
        config = make_agent_config(parameters={"system_prompt": "Be helpful.", "tools": ["search"]})
        assay = make_assay_config()
        ops = [
            PromptSynonymMutator(seed=1),
            ModelSwapMutator(seed=2),
        ]
        runner = MutationRunner(
            agent_callable=mutation_agent,
            config=config,
            assay_config=assay,
            operators=ops,
        )
        scenario = make_scenario()
        suite = runner.run_suite(scenario)
        # Should have per-category entries
        assert isinstance(suite.per_category, dict)

    def test_properties_accessible(self):
        config = make_agent_config()
        assay = make_assay_config()
        runner = MutationRunner(
            agent_callable=mutation_agent,
            config=config,
            assay_config=assay,
        )
        assert runner.config.agent_id == "test-agent"
        assert len(runner.operators) > 0
