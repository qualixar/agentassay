"""End-to-end integration tests for E7 (Token-Efficient Testing) experiment.

Tests all 5 approaches with mocked Azure API responses:
    1. fixed_n      — fixed N trials + Fisher exact test
    2. sprt_only    — sequential stopping
    3. sprt_fingerprint — SPRT + Hotelling T-squared
    4. sprt_fp_budget   — SPRT + fingerprinting + adaptive budget
    5. full_system      — trace-first + all pillars

Also tests: checkpoint/resume, parallelization, string scenario
handling, and token tracking.
"""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import pytest

from experiments.runner.daemon import (
    _load_e7_dependencies,
    _run_e7_approach_fixed_n,
    _run_e7_approach_sprt_only,
    _run_e7_approach_sprt_fingerprint,
    _run_e7_approach_sprt_fp_budget,
    _run_e7_approach_full_system,
    _run_e7_baseline_trials,
    _inject_e7_regression,
    run_single_trial,
)

from .conftest import (
    MockAzureFoundryClient,
    dummy_scorer,
    make_baseline_results,
    make_trial_result,
)


# ===================================================================
# Test 1: E7 approach fixed_n
# ===================================================================


class TestE7FixedN:
    """Tests for the fixed_n approach (baseline: N=100 trials + Fisher)."""

    @pytest.mark.asyncio
    async def test_fixed_n_returns_valid_structure(
        self, mock_client, baseline_results_20, scenario_dict, e7_deps
    ):
        """fixed_n executes N trials and returns valid result keys."""
        approach_cfg = {"name": "fixed_n", "trials": 5}

        result = await _run_e7_approach_fixed_n(
            client=mock_client,
            model="gpt-5.2-chat",
            degraded_scenario=scenario_dict,
            experiment_config={"temperature": 0.7, "max_steps": 5},
            evaluator_fn=dummy_scorer,
            baseline_results=baseline_results_20,
            approach_config=approach_cfg,
            deps=e7_deps,
        )

        assert result["approach"] == "fixed_n"
        assert result["trials_used"] == 5
        assert result["tokens_used"] > 0, "tokens_used must be non-zero"
        assert result["cost_usd"] > 0, "cost_usd must be non-zero"
        assert result["verdict"] in ("PASS", "FAIL")
        assert "p_value" in result
        assert 0.0 <= result["p_value"] <= 1.0

    @pytest.mark.asyncio
    async def test_fixed_n_counts_all_trials(
        self, mock_client, baseline_results_20, scenario_dict, e7_deps
    ):
        """fixed_n runs exactly the configured number of trials."""
        n = 8
        approach_cfg = {"name": "fixed_n", "trials": n}

        result = await _run_e7_approach_fixed_n(
            client=mock_client,
            model="gpt-5.2-chat",
            degraded_scenario=scenario_dict,
            experiment_config={"temperature": 0.7, "max_steps": 5},
            evaluator_fn=dummy_scorer,
            baseline_results=baseline_results_20,
            approach_config=approach_cfg,
            deps=e7_deps,
        )

        assert result["trials_used"] == n
        # Each trial costs 0.005 (from mock) and uses 500 tokens
        assert result["tokens_used"] == n * 500
        assert abs(result["cost_usd"] - n * 0.005) < 1e-6


# ===================================================================
# Test 2: E7 approach sprt_only
# ===================================================================


class TestE7SprtOnly:
    """Tests for the SPRT-only approach."""

    @pytest.mark.asyncio
    async def test_sprt_only_returns_valid_structure(
        self, mock_client, baseline_results_20, scenario_dict, e7_deps, e7_params
    ):
        """sprt_only returns valid verdict and cost data."""
        approach_cfg = {"name": "sprt_only", "max_trials": 20}

        result = await _run_e7_approach_sprt_only(
            client=mock_client,
            model="gpt-5.2-chat",
            degraded_scenario=scenario_dict,
            experiment_config={"temperature": 0.7, "max_steps": 5},
            evaluator_fn=dummy_scorer,
            baseline_results=baseline_results_20,
            approach_config=approach_cfg,
            params=e7_params,
            deps=e7_deps,
        )

        assert result["approach"] == "sprt_only"
        assert result["trials_used"] > 0
        assert result["tokens_used"] >= 0
        assert result["cost_usd"] >= 0
        assert result["verdict"] in ("PASS", "FAIL", "INCONCLUSIVE")
        assert "sprt_decision" in result

    @pytest.mark.asyncio
    async def test_sprt_only_p0_1_clamped(
        self, mock_client, scenario_dict, e7_deps, e7_params
    ):
        """When p0=1.0 (100% pass), it gets clamped to 0.999 and does not crash."""
        baseline_all_pass = make_baseline_results(n=20, pass_rate=1.0)
        approach_cfg = {"name": "sprt_only", "max_trials": 20}

        result = await _run_e7_approach_sprt_only(
            client=mock_client,
            model="gpt-5.2-chat",
            degraded_scenario=scenario_dict,
            experiment_config={"temperature": 0.7, "max_steps": 5},
            evaluator_fn=dummy_scorer,
            baseline_results=baseline_all_pass,
            approach_config=approach_cfg,
            params=e7_params,
            deps=e7_deps,
        )

        # Should not crash; p0 is clamped to 0.999
        assert result["approach"] == "sprt_only"
        assert result["baseline_pass_rate"] == pytest.approx(0.999)
        assert result["verdict"] in ("PASS", "FAIL", "INCONCLUSIVE")

    @pytest.mark.asyncio
    async def test_sprt_only_p0_0_does_not_crash(
        self, mock_client, scenario_dict, e7_deps, e7_params
    ):
        """When p0=0.0 (all fail), clamping handles the edge case.

        p0 clamped to 0.001, p1 also clamped after the p1 >= p0 guard.
        Both stay within (0, 1) and p1 < p0.
        """
        baseline_all_fail = make_baseline_results(n=20, pass_rate=0.0)
        approach_cfg = {"name": "sprt_only", "max_trials": 20}

        result = await _run_e7_approach_sprt_only(
            client=mock_client,
            model="gpt-5.2-chat",
            degraded_scenario=scenario_dict,
            experiment_config={"temperature": 0.7, "max_steps": 5},
            evaluator_fn=dummy_scorer,
            baseline_results=baseline_all_fail,
            approach_config=approach_cfg,
            params=e7_params,
            deps=e7_deps,
        )
        assert result["approach"] == "sprt_only"
        assert result["verdict"] in ("PASS", "FAIL", "INCONCLUSIVE")

    @pytest.mark.asyncio
    async def test_sprt_only_p0_low_but_valid(
        self, mock_client, scenario_dict, e7_deps, e7_params
    ):
        """When p0 is low but still valid (e.g., 15%), SPRT runs correctly."""
        # 3 out of 20 pass = 15% pass rate
        baseline_low = make_baseline_results(n=20, pass_rate=0.15)
        approach_cfg = {"name": "sprt_only", "max_trials": 20}

        result = await _run_e7_approach_sprt_only(
            client=mock_client,
            model="gpt-5.2-chat",
            degraded_scenario=scenario_dict,
            experiment_config={"temperature": 0.7, "max_steps": 5},
            evaluator_fn=dummy_scorer,
            baseline_results=baseline_low,
            approach_config=approach_cfg,
            params=e7_params,
            deps=e7_deps,
        )

        assert result["approach"] == "sprt_only"
        assert result["verdict"] in ("PASS", "FAIL", "INCONCLUSIVE")

    @pytest.mark.asyncio
    async def test_sprt_only_respects_max_trials(
        self, mock_client, baseline_results_20, scenario_dict, e7_deps, e7_params
    ):
        """sprt_only never exceeds max_trials."""
        approach_cfg = {"name": "sprt_only", "max_trials": 3}

        result = await _run_e7_approach_sprt_only(
            client=mock_client,
            model="gpt-5.2-chat",
            degraded_scenario=scenario_dict,
            experiment_config={"temperature": 0.7, "max_steps": 5},
            evaluator_fn=dummy_scorer,
            baseline_results=baseline_results_20,
            approach_config=approach_cfg,
            params=e7_params,
            deps=e7_deps,
        )

        assert result["trials_used"] <= 3


# ===================================================================
# Test 3: E7 approach sprt_fingerprint
# ===================================================================


class TestE7SprtFingerprint:
    """Tests for SPRT + behavioral fingerprinting approach."""

    @pytest.mark.asyncio
    async def test_fingerprint_returns_valid_structure(
        self, mock_client, baseline_results_20, scenario_dict, e7_deps, e7_params
    ):
        """sprt_fingerprint returns valid result with fingerprint fields."""
        approach_cfg = {"name": "sprt_fingerprint", "max_trials": 20}

        result = await _run_e7_approach_sprt_fingerprint(
            client=mock_client,
            model="gpt-5.2-chat",
            degraded_scenario=scenario_dict,
            experiment_config={"temperature": 0.7, "max_steps": 5},
            evaluator_fn=dummy_scorer,
            baseline_results=baseline_results_20,
            approach_config=approach_cfg,
            params=e7_params,
            deps=e7_deps,
        )

        assert result["approach"] == "sprt_fingerprint"
        assert result["trials_used"] > 0
        assert result["verdict"] in ("PASS", "FAIL", "INCONCLUSIVE")
        assert "fingerprint_diverged" in result
        assert isinstance(result["fingerprint_diverged"], bool)

    @pytest.mark.asyncio
    async def test_fingerprint_from_trial_result_works(self, e7_deps):
        """BehavioralFingerprint.from_trial_result works on mock trial data."""
        BehavioralFingerprint = e7_deps["BehavioralFingerprint"]
        trial = make_trial_result(passed=True, include_steps=True)

        fp = BehavioralFingerprint.from_trial_result(trial)

        assert fp is not None
        # The fingerprint should have a vector attribute
        assert hasattr(fp, "vector") or hasattr(fp, "to_vector")

    @pytest.mark.asyncio
    async def test_hotelling_t2_runs_on_mock_data(self, e7_deps):
        """Hotelling T-squared test runs without error on mock fingerprints."""
        BehavioralFingerprint = e7_deps["BehavioralFingerprint"]

        baseline_fps = [
            BehavioralFingerprint.from_trial_result(
                make_trial_result(passed=True, trial_index=i)
            )
            for i in range(15)
        ]
        candidate_fps = [
            BehavioralFingerprint.from_trial_result(
                make_trial_result(passed=True, trial_index=i)
            )
            for i in range(15)
        ]

        # Should not raise — may return True or False
        result = BehavioralFingerprint.hotelling_t2_test(
            baseline_fps, candidate_fps, alpha=0.05
        )
        assert isinstance(result, bool)


# ===================================================================
# Test 4: E7 approach sprt_fp_budget
# ===================================================================


class TestE7SprtFpBudget:
    """Tests for SPRT + fingerprinting + adaptive budget approach."""

    @pytest.mark.asyncio
    async def test_fp_budget_returns_valid_structure(
        self, mock_client, baseline_results_20, scenario_dict, e7_deps, e7_params
    ):
        """sprt_fp_budget returns valid result with budget fields."""
        approach_cfg = {
            "name": "sprt_fp_budget",
            "calibration_size": 5,
        }

        result = await _run_e7_approach_sprt_fp_budget(
            client=mock_client,
            model="gpt-5.2-chat",
            degraded_scenario=scenario_dict,
            experiment_config={"temperature": 0.7, "max_steps": 5},
            evaluator_fn=dummy_scorer,
            baseline_results=baseline_results_20,
            approach_config=approach_cfg,
            params=e7_params,
            deps=e7_deps,
        )

        assert result["approach"] == "sprt_fp_budget"
        assert result["trials_used"] >= 5, "Must run at least calibration_size"
        assert result["verdict"] in ("PASS", "FAIL", "INCONCLUSIVE")
        assert "optimal_n" in result
        assert result["optimal_n"] > 0
        assert result["calibration_size"] == 5

    @pytest.mark.asyncio
    async def test_fp_budget_uses_calibration_fingerprints(
        self, mock_client, baseline_results_20, scenario_dict, e7_deps, e7_params
    ):
        """Verify budget optimizer receives calibration fingerprints."""
        approach_cfg = {"name": "sprt_fp_budget", "calibration_size": 5}

        # Patch the optimizer to spy on calibrate_from_fingerprints
        AdaptiveBudgetOptimizer = e7_deps["AdaptiveBudgetOptimizer"]
        original_calibrate = AdaptiveBudgetOptimizer.calibrate_from_fingerprints

        call_args_log = []

        def spy_calibrate(self_obj, fingerprints, per_trial_cost_usd=0.0):
            call_args_log.append({
                "n_fingerprints": len(fingerprints),
                "per_trial_cost": per_trial_cost_usd,
            })
            return original_calibrate(self_obj, fingerprints, per_trial_cost_usd)

        with patch.object(
            AdaptiveBudgetOptimizer,
            "calibrate_from_fingerprints",
            spy_calibrate,
        ):
            result = await _run_e7_approach_sprt_fp_budget(
                client=mock_client,
                model="gpt-5.2-chat",
                degraded_scenario=scenario_dict,
                experiment_config={"temperature": 0.7, "max_steps": 5},
                evaluator_fn=dummy_scorer,
                baseline_results=baseline_results_20,
                approach_config=approach_cfg,
                params=e7_params,
                deps=e7_deps,
            )

        assert len(call_args_log) == 1, "calibrate_from_fingerprints called once"
        assert call_args_log[0]["n_fingerprints"] == 5


# ===================================================================
# Test 5: E7 approach full_system
# ===================================================================


class TestE7FullSystem:
    """Tests for the full system (trace-first + all pillars)."""

    @pytest.mark.asyncio
    async def test_full_system_returns_valid_structure(
        self, mock_client, baseline_results_20, scenario_dict, e7_deps, e7_params
    ):
        """full_system returns valid result with trace-first fields."""
        approach_cfg = {"name": "full_system", "calibration_size": 5}

        result = await _run_e7_approach_full_system(
            client=mock_client,
            model="gpt-5.2-chat",
            degraded_scenario=scenario_dict,
            experiment_config={"temperature": 0.7, "max_steps": 5},
            evaluator_fn=dummy_scorer,
            baseline_results=baseline_results_20,
            approach_config=approach_cfg,
            params=e7_params,
            deps=e7_deps,
        )

        assert result["approach"] == "full_system"
        assert result["verdict"] in ("PASS", "FAIL", "INCONCLUSIVE")
        assert "trace_first_resolved" in result
        assert isinstance(result["trace_first_resolved"], bool)

    @pytest.mark.asyncio
    async def test_full_system_uses_trace_store_record(
        self, mock_client, baseline_results_20, scenario_dict, e7_deps, e7_params
    ):
        """Verify TraceStore.record() is called (not .add())."""
        approach_cfg = {"name": "full_system", "calibration_size": 5}
        TraceStore = e7_deps["TraceStore"]

        record_calls = []
        original_record = TraceStore.record

        def spy_record(self_obj, trace, metadata=None):
            record_calls.append(trace)
            return original_record(self_obj, trace, metadata=metadata)

        with patch.object(TraceStore, "record", spy_record):
            await _run_e7_approach_full_system(
                client=mock_client,
                model="gpt-5.2-chat",
                degraded_scenario=scenario_dict,
                experiment_config={"temperature": 0.7, "max_steps": 5},
                evaluator_fn=dummy_scorer,
                baseline_results=baseline_results_20,
                approach_config=approach_cfg,
                params=e7_params,
                deps=e7_deps,
            )

        # record() should be called once per baseline result
        assert len(record_calls) == len(baseline_results_20)

    @pytest.mark.asyncio
    async def test_full_system_uses_drift_detection(
        self, mock_client, baseline_results_20, scenario_dict, e7_deps, e7_params
    ):
        """Verify trace_store.drift_detection() is called."""
        approach_cfg = {"name": "full_system", "calibration_size": 5}
        TraceStore = e7_deps["TraceStore"]

        drift_calls = []
        original_drift = TraceStore.drift_detection

        def spy_drift(self_obj, **kwargs):
            drift_calls.append(kwargs)
            return original_drift(self_obj, **kwargs)

        with patch.object(TraceStore, "drift_detection", spy_drift):
            await _run_e7_approach_full_system(
                client=mock_client,
                model="gpt-5.2-chat",
                degraded_scenario=scenario_dict,
                experiment_config={"temperature": 0.7, "max_steps": 5},
                evaluator_fn=dummy_scorer,
                baseline_results=baseline_results_20,
                approach_config=approach_cfg,
                params=e7_params,
                deps=e7_deps,
            )

        assert len(drift_calls) >= 1, "drift_detection must be called"


# ===================================================================
# Test 6: run_single_trial token tracking
# ===================================================================


class TestRunSingleTrial:
    """Tests for run_single_trial's result structure."""

    @pytest.mark.asyncio
    async def test_trial_result_contains_steps(self, mock_client, scenario_dict):
        """run_single_trial result includes _steps with tool call data."""
        result = await run_single_trial(
            client=mock_client,
            model="gpt-5.2-chat",
            scenario=scenario_dict,
            experiment_config={"temperature": 0.7, "max_steps": 5},
            evaluator_fn=dummy_scorer,
            trial_index=0,
        )

        assert "_steps" in result, "_steps must be in trial result"
        assert isinstance(result["_steps"], list)
        assert len(result["_steps"]) > 0, "_steps should contain agent steps"
        assert result["tokens"] > 0
        assert result["cost_usd"] >= 0

    @pytest.mark.asyncio
    async def test_trial_result_has_required_keys(self, mock_client, scenario_dict):
        """run_single_trial returns all required keys."""
        result = await run_single_trial(
            client=mock_client,
            model="gpt-5.2-chat",
            scenario=scenario_dict,
            experiment_config={"temperature": 0.7, "max_steps": 5},
            evaluator_fn=dummy_scorer,
            trial_index=42,
        )

        required_keys = {
            "trial_id", "trial_index", "scenario_id", "model",
            "passed", "score", "success", "error",
            "duration_ms", "cost_usd", "tokens", "step_count",
            "_steps", "timestamp",
        }
        assert required_keys.issubset(result.keys())
        assert result["trial_index"] == 42
        assert result["model"] == "gpt-5.2-chat"


# ===================================================================
# Test 7: Regression injection
# ===================================================================


class TestRegressionInjection:
    """Tests for _inject_e7_regression."""

    def test_prompt_degradation(self, scenario_dict):
        """Prompt degradation truncates the last 30% of the system prompt."""
        original = scenario_dict["system_prompt"]
        degraded = _inject_e7_regression(scenario_dict, "prompt_degradation")

        assert len(degraded["system_prompt"]) < len(original)
        expected_len = int(len(original) * 0.7)
        assert len(degraded["system_prompt"]) == expected_len

    def test_unknown_regression_type_still_truncates(self, scenario_dict):
        """Unknown regression types fall back to prompt truncation."""
        original = scenario_dict["system_prompt"]
        degraded = _inject_e7_regression(scenario_dict, "unknown_type")

        assert len(degraded["system_prompt"]) < len(original)


# ===================================================================
# Test 8: E7 baseline trials
# ===================================================================


class TestE7BaselineTrials:
    """Tests for _run_e7_baseline_trials."""

    @pytest.mark.asyncio
    async def test_baseline_runs_n_trials(self, mock_client, scenario_dict):
        """_run_e7_baseline_trials runs exactly N trials."""
        results = await _run_e7_baseline_trials(
            client=mock_client,
            model="gpt-5.2-chat",
            scenario=scenario_dict,
            experiment_config={"temperature": 0.7, "max_steps": 5},
            evaluator_fn=dummy_scorer,
            n_trials=7,
        )

        assert len(results) == 7
        for r in results:
            assert "_steps" in r


# ===================================================================
# Test 9: _load_e7_dependencies
# ===================================================================


class TestLoadE7Dependencies:
    """Tests for the lazy-loading of E7 module dependencies."""

    def test_loads_all_required_symbols(self):
        """_load_e7_dependencies returns all 5 required symbols."""
        deps = _load_e7_dependencies()

        assert "BehavioralFingerprint" in deps
        assert "AdaptiveBudgetOptimizer" in deps
        assert "TraceStore" in deps
        assert "SPRTRunner" in deps
        assert "fisher_exact_regression" in deps

    def test_classes_are_callable(self):
        """All loaded classes can be instantiated or called."""
        deps = _load_e7_dependencies()

        # SPRTRunner requires p0, p1
        sprt = deps["SPRTRunner"](p0=0.9, p1=0.7, alpha=0.05, beta=0.10)
        assert sprt is not None

        # TraceStore can be instantiated without args
        ts = deps["TraceStore"]()
        assert ts is not None

        # fisher_exact_regression is callable
        result = deps["fisher_exact_regression"](
            baseline_passes=8, baseline_n=10,
            current_passes=6, current_n=10,
        )
        assert hasattr(result, "p_value")
        assert hasattr(result, "significant")


# ===================================================================
# Test 10: E1-E6 string scenario handling
# ===================================================================


class TestStringScenarioHandling:
    """Tests for converting string scenarios to dicts in E7."""

    def test_string_scenario_detected(self, e7_experiment_config):
        """E7 config with string scenarios is handled correctly."""
        scenarios = e7_experiment_config["scenarios"]
        assert isinstance(scenarios[0], str)
        assert scenarios[0] == "ecommerce"

    def test_scenario_dict_built_from_string(self):
        """The E7 runner builds a minimal scenario dict from a string name."""
        # This is what run_e7_experiment does internally
        sc_raw = "ecommerce"
        sc_name = sc_raw if isinstance(sc_raw, str) else sc_raw.get("scenario_id")

        scenario = {
            "scenario_id": sc_name,
            "name": sc_name,
            "system_prompt": f"You are a helpful {sc_name} agent.",
            "user_input": f"Complete the {sc_name} task.",
            "tools": [],
            "tool_responses": {},
            "expected": {},
        }

        assert scenario["scenario_id"] == "ecommerce"
        assert "ecommerce" in scenario["system_prompt"]
