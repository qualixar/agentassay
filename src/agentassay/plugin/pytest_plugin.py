"""AgentAssay pytest plugin — stochastic regression testing for AI agents.

Integrates AgentAssay's formal testing framework with pytest, enabling
developers to write probabilistic agent tests using familiar pytest
conventions plus the ``@pytest.mark.agentassay`` marker.

Registration:
    Registered via entry point in pyproject.toml::

        [project.entry-points."pytest11"]
        agentassay = "agentassay.plugin.pytest_plugin"

Marker:
    ``@pytest.mark.agentassay(n=30, alpha=0.05, threshold=0.80)``

    Parameters:
        n : int
            Number of stochastic trials to run per scenario. Default 30.
        alpha : float
            Significance level (Type I error rate). Default 0.05.
        threshold : float
            Minimum acceptable pass rate. Default 0.80.
        confidence_method : str
            CI method: 'wilson', 'clopper-pearson', 'normal'. Default 'wilson'.
        regression_test : str
            Hypothesis test: 'fisher', 'chi2', 'ks', 'mann-whitney'. Default 'fisher'.
        use_sprt : bool
            Enable SPRT adaptive stopping. Default False.

Fixtures:
    assay_config
        Provides an ``AssayConfig`` built from marker arguments (or defaults).
    trial_runner
        Factory function: ``create_runner(agent_callable, agent_config=None)``
        that returns a fully configured ``TrialRunner``.

Custom Assertions:
    assert_no_regression(baseline_results, current_results, alpha=0.05)
        Raises ``AssertionError`` with detailed statistics if Fisher's exact
        test detects a statistically significant regression.

    assert_pass_rate(results, threshold=0.80, confidence=0.95)
        Raises ``AssertionError`` if the Wilson CI lower bound for the
        observed pass rate falls below the threshold.

Example:
    >>> import pytest
    >>> from agentassay.plugin.pytest_plugin import assert_pass_rate
    >>>
    >>> @pytest.mark.agentassay(n=50, threshold=0.85)
    ... def test_my_agent(trial_runner, assay_config):
    ...     runner = trial_runner(my_agent_function)
    ...     scenario = TestScenario(
    ...         scenario_id="s1", name="basic",
    ...         description="test", input_data={"prompt": "hello"},
    ...     )
    ...     results = runner.run_trials(scenario)
    ...     passed = [r.passed for r in results]
    ...     assert_pass_rate(passed, threshold=0.85)

References:
    - Wilson, E.B. (1927). Probable Inference. JASA 22(158): 209-212.
    - Fisher, R.A. (1922). Chi-Squared from Contingency Tables. JRSS 85(1): 87-94.
    - pytest plugin documentation: https://docs.pytest.org/en/stable/how-to/writing_plugins.html
"""

from __future__ import annotations

import logging
import textwrap
import time
from collections.abc import Callable
from typing import Any

import pytest

from agentassay.core.models import (
    AgentConfig,
    AssayConfig,
    ExecutionTrace,
    TestScenario,
    TrialResult,
)
from agentassay.core.runner import TrialRunner
from agentassay.statistics.confidence import ConfidenceInterval, wilson_interval
from agentassay.statistics.hypothesis import (
    RegressionTestResult,
    fisher_exact_regression,
)
from agentassay.verdicts.verdict import VerdictFunction, VerdictStatus

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Marker name constant
# ---------------------------------------------------------------------------

MARKER_NAME: str = "agentassay"
"""The pytest marker name for agent assay tests."""

# ---------------------------------------------------------------------------
# Plugin-level state (accumulated during the test session)
# ---------------------------------------------------------------------------

_assay_results: list[dict[str, Any]] = []
"""Accumulated results across all agentassay-marked tests for terminal summary."""


# ===================================================================
# Pytest hooks
# ===================================================================


def pytest_configure(config: pytest.Config) -> None:
    """Register the ``agentassay`` marker with pytest.

    Called once at the start of the test session. This prevents pytest
    from emitting ``PytestUnknownMarkWarning`` for our custom marker.
    """
    config.addinivalue_line(
        "markers",
        (
            f"{MARKER_NAME}(n, alpha, threshold, confidence_method, "
            f"regression_test, use_sprt): "
            "Mark a test as an AgentAssay stochastic agent test. "
            "Parameters control statistical rigor of the test run."
        ),
    )
    # Reset session-level state
    _assay_results.clear()


def pytest_collection_modifyitems(
    session: pytest.Session,
    config: pytest.Config,
    items: list[pytest.Item],
) -> None:
    """Process agentassay-marked tests during collection.

    Tags marked tests with a recognizable ``agentassay`` keyword so they
    can be selected with ``-m agentassay`` and logs how many were found.
    """
    assay_count = 0
    for item in items:
        marker = item.get_closest_marker(MARKER_NAME)
        if marker is not None:
            assay_count += 1
            # Ensure the item is tagged for -m filtering
            item.add_marker(pytest.mark.agentassay)

    if assay_count > 0:
        logger.info(
            "AgentAssay: collected %d stochastic agent test(s)", assay_count
        )


def pytest_terminal_summary(
    terminalreporter: Any,
    exitstatus: int,
    config: pytest.Config,
) -> None:
    """Add an AgentAssay summary section to pytest terminal output.

    Prints a table of all agentassay-marked tests that ran, showing
    pass rate, confidence interval, verdict, and timing for each.
    """
    if not _assay_results:
        return

    terminalreporter.section("AgentAssay Summary")

    for entry in _assay_results:
        name = entry.get("test_name", "unknown")
        pass_rate = entry.get("pass_rate")
        ci = entry.get("ci")
        verdict = entry.get("verdict", "N/A")
        n_trials = entry.get("n_trials", 0)
        n_passed = entry.get("n_passed", 0)
        duration_s = entry.get("duration_s", 0.0)

        if pass_rate is not None and ci is not None:
            line = (
                f"  {name}: "
                f"rate={pass_rate:.2%} "
                f"[{ci[0]:.2%}, {ci[1]:.2%}] "
                f"({n_passed}/{n_trials} trials) "
                f"verdict={verdict} "
                f"({duration_s:.1f}s)"
            )
        else:
            line = f"  {name}: {verdict} ({n_trials} trials, {duration_s:.1f}s)"

        terminalreporter.write_line(line)

    # Summary counts
    total = len(_assay_results)
    passed = sum(1 for r in _assay_results if r.get("verdict") == "PASS")
    failed = sum(1 for r in _assay_results if r.get("verdict") == "FAIL")
    inconclusive = sum(1 for r in _assay_results if r.get("verdict") == "INCONCLUSIVE")

    terminalreporter.write_line("")
    terminalreporter.write_line(
        f"  AgentAssay totals: {total} tests — "
        f"{passed} PASS, {failed} FAIL, {inconclusive} INCONCLUSIVE"
    )


# ===================================================================
# Fixtures
# ===================================================================


@pytest.fixture
def assay_config(request: pytest.FixtureRequest) -> AssayConfig:
    """Provide an ``AssayConfig`` from marker arguments or defaults.

    Reads parameters from the ``@pytest.mark.agentassay(...)`` marker
    on the current test and constructs an ``AssayConfig``. Any parameter
    not specified in the marker uses the ``AssayConfig`` default.

    Returns
    -------
    AssayConfig
        Configured for the current test's statistical requirements.
    """
    marker = request.node.get_closest_marker(MARKER_NAME)

    if marker is None:
        return AssayConfig()

    kwargs: dict[str, Any] = {}

    # Positional-or-keyword from marker
    if marker.args:
        # First positional arg is n
        if len(marker.args) >= 1:
            kwargs["num_trials"] = int(marker.args[0])
        if len(marker.args) >= 2:
            kwargs["significance_level"] = float(marker.args[1])
        if len(marker.args) >= 3:
            # threshold is not in AssayConfig — store in metadata
            kwargs.setdefault("metadata", {})["threshold"] = float(marker.args[2])

    # Keyword arguments from marker
    if "n" in marker.kwargs:
        kwargs["num_trials"] = int(marker.kwargs["n"])
    if "alpha" in marker.kwargs:
        kwargs["significance_level"] = float(marker.kwargs["alpha"])
    if "threshold" in marker.kwargs:
        kwargs.setdefault("metadata", {})["threshold"] = float(
            marker.kwargs["threshold"]
        )
    if "confidence_method" in marker.kwargs:
        kwargs["confidence_method"] = str(marker.kwargs["confidence_method"])
    if "regression_test" in marker.kwargs:
        kwargs["regression_test"] = str(marker.kwargs["regression_test"])
    if "use_sprt" in marker.kwargs:
        kwargs["use_sprt"] = bool(marker.kwargs["use_sprt"])
    if "power" in marker.kwargs:
        kwargs["power"] = float(marker.kwargs["power"])
    if "seed" in marker.kwargs:
        kwargs["seed"] = int(marker.kwargs["seed"])
    if "parallel" in marker.kwargs:
        kwargs["parallel_trials"] = int(marker.kwargs["parallel"])

    return AssayConfig(**kwargs)


@pytest.fixture
def trial_runner(
    assay_config: AssayConfig,
) -> Callable[..., TrialRunner]:
    """Factory fixture that creates a ``TrialRunner`` for the current test.

    Returns a factory function with signature::

        create_runner(
            agent_callable: Callable[[dict], ExecutionTrace],
            agent_config: AgentConfig | None = None,
        ) -> TrialRunner

    If ``agent_config`` is not provided, a minimal default config is used
    with framework='custom' and model='unknown'.

    Parameters
    ----------
    assay_config : AssayConfig
        Injected automatically from the ``assay_config`` fixture.

    Returns
    -------
    Callable[..., TrialRunner]
        A factory function for creating configured trial runners.

    Example
    -------
    >>> def test_agent(trial_runner):
    ...     runner = trial_runner(my_agent_func)
    ...     results = runner.run_trials(scenario)
    """

    def create_runner(
        agent_callable: Callable[[dict[str, Any]], ExecutionTrace],
        agent_config: AgentConfig | None = None,
    ) -> TrialRunner:
        if agent_config is None:
            agent_config = AgentConfig(
                agent_id="test-agent",
                name="test-agent",
                framework="custom",
                model="unknown",
            )
        return TrialRunner(
            agent_callable=agent_callable,
            config=assay_config,
            agent_config=agent_config,
        )

    return create_runner


# ===================================================================
# Custom assertion helpers
# ===================================================================


def assert_no_regression(
    baseline_results: list[bool],
    current_results: list[bool],
    alpha: float = 0.05,
) -> RegressionTestResult:
    """Assert that no statistically significant regression occurred.

    Compares two sets of binary trial outcomes using Fisher's exact test.
    Raises ``AssertionError`` with detailed statistical information if a
    regression is detected at the specified significance level.

    Parameters
    ----------
    baseline_results : list[bool]
        Pass/fail outcomes from the baseline version.
    current_results : list[bool]
        Pass/fail outcomes from the current version.
    alpha : float
        Significance level (Type I error rate). Default 0.05.

    Returns
    -------
    RegressionTestResult
        The full test result object (only returned if no regression detected).

    Raises
    ------
    AssertionError
        If regression is detected (p < alpha and current rate < baseline rate).
    ValueError
        If either result list is empty or alpha is out of range.

    Example
    -------
    >>> baseline = [True] * 90 + [False] * 10  # 90% pass rate
    >>> current = [True] * 88 + [False] * 12   # 88% pass rate
    >>> result = assert_no_regression(baseline, current)
    >>> # No assertion error — 2% drop is not significant at alpha=0.05
    """
    if not baseline_results:
        raise ValueError("baseline_results must not be empty")
    if not current_results:
        raise ValueError("current_results must not be empty")
    if not 0.0 < alpha < 1.0:
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")

    baseline_n = len(baseline_results)
    current_n = len(current_results)
    baseline_passes = sum(baseline_results)
    current_passes = sum(current_results)

    result = fisher_exact_regression(
        baseline_passes=baseline_passes,
        baseline_n=baseline_n,
        current_passes=current_passes,
        current_n=current_n,
        alpha=alpha,
    )

    if result.significant:
        # Record in session-level results for terminal summary
        _assay_results.append({
            "test_name": "assert_no_regression",
            "pass_rate": result.current_rate,
            "ci": None,
            "verdict": "FAIL",
            "n_trials": current_n,
            "n_passed": current_passes,
            "duration_s": 0.0,
        })

        msg = textwrap.dedent(f"""\
            AgentAssay regression detected!

            Baseline: {baseline_passes}/{baseline_n} ({result.baseline_rate:.2%})
            Current:  {current_passes}/{current_n} ({result.current_rate:.2%})
            Drop:     {result.baseline_rate - result.current_rate:+.2%}

            Statistical test: {result.test_name}
            p-value:          {result.p_value:.6f} (< alpha={alpha})
            Effect size:      {result.effect_size_name} = {result.effect_size:+.4f}

            {result.interpretation}
        """)
        raise AssertionError(msg)

    return result


def assert_pass_rate(
    results: list[bool],
    threshold: float = 0.80,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    """Assert that the observed pass rate meets a minimum threshold.

    Computes a Wilson confidence interval for the true pass rate and
    verifies that the lower bound of the CI meets or exceeds the
    threshold. This provides a statistical guarantee — not just that
    the sample pass rate is above the threshold, but that the *true*
    pass rate is above it with the specified confidence.

    Parameters
    ----------
    results : list[bool]
        Pass/fail outcomes from agent trials.
    threshold : float
        Minimum acceptable pass rate in [0, 1]. Default 0.80.
    confidence : float
        Confidence level for the interval in (0, 1). Default 0.95.

    Returns
    -------
    ConfidenceInterval
        The computed Wilson interval (only returned if assertion passes).

    Raises
    ------
    AssertionError
        If the CI lower bound is below the threshold.
    ValueError
        If results is empty or parameters are out of range.

    Example
    -------
    >>> results = [True] * 27 + [False] * 3  # 90% pass rate
    >>> ci = assert_pass_rate(results, threshold=0.80)
    >>> print(f"Pass rate: {ci.point_estimate:.0%}")
    Pass rate: 90%
    """
    if not results:
        raise ValueError("results must not be empty")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError(f"threshold must be in [0, 1], got {threshold}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    n = len(results)
    k = sum(results)

    ci = wilson_interval(successes=k, n=n, confidence=confidence)

    if ci.lower < threshold:
        # Determine verdict for summary reporting
        if ci.upper < threshold:
            verdict_label = "FAIL"
        else:
            verdict_label = "INCONCLUSIVE"

        _assay_results.append({
            "test_name": "assert_pass_rate",
            "pass_rate": ci.point_estimate,
            "ci": (ci.lower, ci.upper),
            "verdict": verdict_label,
            "n_trials": n,
            "n_passed": k,
            "duration_s": 0.0,
        })

        msg = textwrap.dedent(f"""\
            AgentAssay pass rate below threshold!

            Observed:   {k}/{n} ({ci.point_estimate:.2%})
            Threshold:  {threshold:.2%}
            CI ({confidence:.0%}):  [{ci.lower:.4f}, {ci.upper:.4f}]
            CI lower:   {ci.lower:.4f} < threshold {threshold:.4f}

            The {confidence:.0%} confidence interval lower bound ({ci.lower:.4f})
            is below the required threshold ({threshold:.4f}).

            This means we cannot confirm with {confidence:.0%} confidence that
            the true pass rate meets the {threshold:.0%} requirement.

            Suggestions:
              - If the pass rate is close, increase n (more trials) to narrow the CI.
              - If the pass rate is clearly below, the agent has regressed.
        """)
        raise AssertionError(msg)

    # Success — record in session results
    _assay_results.append({
        "test_name": "assert_pass_rate",
        "pass_rate": ci.point_estimate,
        "ci": (ci.lower, ci.upper),
        "verdict": "PASS",
        "n_trials": n,
        "n_passed": k,
        "duration_s": 0.0,
    })

    return ci


def assert_verdict_passes(
    results: list[bool],
    threshold: float = 0.80,
    alpha: float = 0.05,
    min_trials: int = 30,
) -> VerdictFunction:
    """Assert that the stochastic verdict is PASS (not FAIL or INCONCLUSIVE).

    Uses the full ``VerdictFunction`` from the verdicts module to produce
    a three-valued verdict (PASS/FAIL/INCONCLUSIVE). Raises on FAIL or
    INCONCLUSIVE.

    This is a higher-level assertion than ``assert_pass_rate`` — it
    integrates the formal (alpha, beta, n)-test semantics from the paper.

    Parameters
    ----------
    results : list[bool]
        Pass/fail outcomes from agent trials.
    threshold : float
        Minimum acceptable pass rate. Default 0.80.
    alpha : float
        Significance level. Default 0.05.
    min_trials : int
        Minimum trials for a definitive verdict. Default 30.

    Returns
    -------
    VerdictFunction
        The verdict function used (for inspection).

    Raises
    ------
    AssertionError
        If verdict is FAIL or INCONCLUSIVE.
    """
    if not results:
        raise ValueError("results must not be empty")

    vf = VerdictFunction(alpha=alpha, min_trials=min_trials)
    verdict = vf.evaluate_single(results, threshold=threshold)

    if verdict.status == VerdictStatus.FAIL:
        _assay_results.append({
            "test_name": "assert_verdict_passes",
            "pass_rate": verdict.pass_rate,
            "ci": verdict.pass_rate_ci,
            "verdict": "FAIL",
            "n_trials": verdict.num_trials,
            "n_passed": verdict.num_passed,
            "duration_s": 0.0,
        })
        msg = textwrap.dedent(f"""\
            AgentAssay verdict: FAIL

            Pass rate:  {verdict.pass_rate:.2%} ({verdict.num_passed}/{verdict.num_trials})
            Threshold:  {threshold:.2%}
            CI:         [{verdict.pass_rate_ci[0]:.4f}, {verdict.pass_rate_ci[1]:.4f}]
            Reason:     {verdict.details.get('reason', 'N/A')}

            The entire confidence interval is below the threshold.
        """)
        raise AssertionError(msg)

    if verdict.status == VerdictStatus.INCONCLUSIVE:
        _assay_results.append({
            "test_name": "assert_verdict_passes",
            "pass_rate": verdict.pass_rate,
            "ci": verdict.pass_rate_ci,
            "verdict": "INCONCLUSIVE",
            "n_trials": verdict.num_trials,
            "n_passed": verdict.num_passed,
            "duration_s": 0.0,
        })
        msg = textwrap.dedent(f"""\
            AgentAssay verdict: INCONCLUSIVE

            Pass rate:  {verdict.pass_rate:.2%} ({verdict.num_passed}/{verdict.num_trials})
            Threshold:  {threshold:.2%}
            CI:         [{verdict.pass_rate_ci[0]:.4f}, {verdict.pass_rate_ci[1]:.4f}]
            Reason:     {verdict.details.get('reason', 'N/A')}

            Not enough statistical evidence to confirm or deny.
            Try increasing the number of trials (n).
        """)
        raise AssertionError(msg)

    # PASS
    _assay_results.append({
        "test_name": "assert_verdict_passes",
        "pass_rate": verdict.pass_rate,
        "ci": verdict.pass_rate_ci,
        "verdict": "PASS",
        "n_trials": verdict.num_trials,
        "n_passed": verdict.num_passed,
        "duration_s": 0.0,
    })

    return vf
