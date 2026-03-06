# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""AgentAssay pytest plugin — public API exports.

Provides assertion helpers for use in pytest tests without needing
to import from the internal ``pytest_plugin`` module.

Usage:
    >>> from agentassay.plugin import assert_no_regression, assert_pass_rate
    >>>
    >>> def test_my_agent():
    ...     results = [True] * 28 + [False] * 2
    ...     assert_pass_rate(results, threshold=0.80)
"""

from agentassay.plugin.pytest_plugin import (
    assert_no_regression,
    assert_pass_rate,
    assert_verdict_passes,
)

__all__ = [
    "assert_no_regression",
    "assert_pass_rate",
    "assert_verdict_passes",
]
