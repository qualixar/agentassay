"""Tests for the Semantic Kernel framework adapter.

Validates adapter creation, run execution, step classification,
error handling, timing capture, output extraction, cost estimation,
filter registration, and ``to_callable()`` / ``get_config()`` helpers.

All tests use mock objects --- ``semantic_kernel`` does NOT need to be
installed.  The lazy-import guard is patched to allow testing the
adapter logic in isolation.

Target: 20+ tests.
"""

from __future__ import annotations

import time
from typing import Any
from unittest.mock import MagicMock, PropertyMock, patch

import pytest

from agentassay.core.models import AgentConfig, ExecutionTrace, StepTrace
from agentassay.integrations.base import FrameworkNotInstalledError
from agentassay.integrations.semantic_kernel_adapter import (
    SemanticKernelAdapter,
    _check_semantic_kernel_installed,
    _INSTALL_HINT,
)


# ===================================================================
# Helpers
# ===================================================================


def _make_mock_kernel(
    invoke_return: Any = None,
    has_add_filter: bool = False,
) -> MagicMock:
    """Create a mock Semantic Kernel ``Kernel`` object.

    Parameters
    ----------
    invoke_return
        Value returned by ``kernel.invoke()``.
    has_add_filter
        Whether the kernel exposes ``add_filter`` / ``remove_filter``.
    """
    kernel = MagicMock()
    kernel.invoke.return_value = invoke_return

    if not has_add_filter:
        # Remove add_filter so hasattr returns False
        del kernel.add_filter
        del kernel.remove_filter

    return kernel


def _make_function_result(
    value: Any = "Paris",
    metadata: dict[str, Any] | None = None,
) -> MagicMock:
    """Create a mock ``FunctionResult`` with a ``.value`` attribute."""
    result = MagicMock()
    result.value = value
    result.result = None
    result.metadata = metadata or {}
    return result


def _patch_sk_imports():
    """Return a combined patcher for semantic_kernel and KernelArguments.

    Usage::

        with _patch_sk_imports():
            ...
    """
    # Patch the install check to be a no-op
    p1 = patch(
        "agentassay.integrations.semantic_kernel_adapter"
        "._check_semantic_kernel_installed"
    )
    # Patch the KernelArguments import inside _build_kernel_arguments
    mock_ka = MagicMock()
    mock_ka.side_effect = lambda **kw: kw  # Just return kwargs as dict
    p2 = patch.dict(
        "sys.modules",
        {"semantic_kernel": MagicMock(KernelArguments=mock_ka)},
    )
    return p1, p2


# ===================================================================
# Test: Install check
# ===================================================================


class TestInstallCheck:
    """Tests for ``_check_semantic_kernel_installed``."""

    def test_raises_framework_not_installed_error(self):
        """Import check raises FrameworkNotInstalledError when SK missing."""
        with patch.dict("sys.modules", {"semantic_kernel": None}):
            with pytest.raises(FrameworkNotInstalledError, match="semantic-kernel"):
                _check_semantic_kernel_installed()

    def test_error_contains_install_hint(self):
        """Error message includes pip install instructions."""
        with patch.dict("sys.modules", {"semantic_kernel": None}):
            with pytest.raises(FrameworkNotInstalledError) as exc_info:
                _check_semantic_kernel_installed()
            assert "pip install" in str(exc_info.value)

    def test_no_error_when_installed(self):
        """No error raised when semantic_kernel is importable."""
        mock_sk = MagicMock()
        with patch.dict("sys.modules", {"semantic_kernel": mock_sk}):
            _check_semantic_kernel_installed()  # Should not raise


# ===================================================================
# Test: Adapter creation
# ===================================================================


class TestAdapterCreation:
    """Tests for SemanticKernelAdapter construction."""

    def test_basic_creation(self):
        """Adapter can be created with a mock kernel."""
        kernel = _make_mock_kernel()
        adapter = SemanticKernelAdapter(kernel=kernel, model="gpt-4o")
        assert adapter.model == "gpt-4o"
        assert adapter.framework == "semantic_kernel"

    def test_default_model(self):
        """Model defaults to 'unknown' when not specified."""
        kernel = _make_mock_kernel()
        adapter = SemanticKernelAdapter(kernel=kernel)
        assert adapter.model == "unknown"

    def test_default_agent_name(self):
        """Agent name defaults to 'semantic_kernel-agent'."""
        kernel = _make_mock_kernel()
        adapter = SemanticKernelAdapter(kernel=kernel)
        assert adapter.agent_name == "semantic_kernel-agent"

    def test_custom_agent_name(self):
        """Custom agent name is respected."""
        kernel = _make_mock_kernel()
        adapter = SemanticKernelAdapter(
            kernel=kernel, agent_name="my-sk-agent"
        )
        assert adapter.agent_name == "my-sk-agent"

    def test_plugin_and_function_stored(self):
        """Plugin and function names are stored on the adapter."""
        kernel = _make_mock_kernel()
        adapter = SemanticKernelAdapter(
            kernel=kernel,
            plugin_name="ChatPlugin",
            function_name="Respond",
        )
        assert adapter._plugin_name == "ChatPlugin"
        assert adapter._function_name == "Respond"

    def test_metadata_stored(self):
        """Metadata is stored and accessible."""
        kernel = _make_mock_kernel()
        adapter = SemanticKernelAdapter(
            kernel=kernel, metadata={"env": "test"}
        )
        assert adapter._metadata == {"env": "test"}

    def test_repr(self):
        """repr includes class name, framework, model, and agent name."""
        kernel = _make_mock_kernel()
        adapter = SemanticKernelAdapter(kernel=kernel, model="gpt-4o")
        r = repr(adapter)
        assert "SemanticKernelAdapter" in r
        assert "semantic_kernel" in r
        assert "gpt-4o" in r


# ===================================================================
# Test: run() execution
# ===================================================================


class TestRunExecution:
    """Tests for ``SemanticKernelAdapter.run()``."""

    def test_successful_run_returns_execution_trace(self):
        """A successful kernel.invoke() produces a valid ExecutionTrace."""
        result = _make_function_result(value="The capital is Paris")
        kernel = _make_mock_kernel(invoke_return=result)

        adapter = SemanticKernelAdapter(
            kernel=kernel,
            model="gpt-4o",
            plugin_name="chat",
            function_name="respond",
        )

        p1, p2 = _patch_sk_imports()
        with p1, p2:
            trace = adapter.run({"query": "What is the capital of France?"})

        assert isinstance(trace, ExecutionTrace)
        assert trace.success is True
        assert trace.error is None
        assert trace.framework == "semantic_kernel"
        assert trace.model == "gpt-4o"
        assert trace.output_data == "The capital is Paris"

    def test_scenario_id_extracted_from_input(self):
        """scenario_id is extracted from input_data if present."""
        result = _make_function_result()
        kernel = _make_mock_kernel(invoke_return=result)
        adapter = SemanticKernelAdapter(kernel=kernel, model="gpt-4o")

        p1, p2 = _patch_sk_imports()
        with p1, p2:
            trace = adapter.run(
                {"scenario_id": "test-scenario-42", "query": "Hi"}
            )

        assert trace.scenario_id == "test-scenario-42"

    def test_default_scenario_id(self):
        """scenario_id defaults to 'default' when absent from input."""
        result = _make_function_result()
        kernel = _make_mock_kernel(invoke_return=result)
        adapter = SemanticKernelAdapter(kernel=kernel, model="gpt-4o")

        p1, p2 = _patch_sk_imports()
        with p1, p2:
            trace = adapter.run({"query": "Hello"})

        assert trace.scenario_id == "default"

    def test_error_returns_failed_trace(self):
        """An exception during invoke produces success=False trace."""
        kernel = _make_mock_kernel()
        kernel.invoke.side_effect = RuntimeError("Kernel crashed")

        adapter = SemanticKernelAdapter(kernel=kernel, model="gpt-4o")

        p1, p2 = _patch_sk_imports()
        with p1, p2:
            trace = adapter.run({"query": "boom"})

        assert isinstance(trace, ExecutionTrace)
        assert trace.success is False
        assert "RuntimeError" in trace.error
        assert "Kernel crashed" in trace.error
        assert trace.steps == []
        assert trace.output_data is None

    def test_framework_not_installed_reraises(self):
        """FrameworkNotInstalledError is NOT swallowed by run()."""
        kernel = _make_mock_kernel()
        adapter = SemanticKernelAdapter(kernel=kernel)

        with patch(
            "agentassay.integrations.semantic_kernel_adapter"
            "._check_semantic_kernel_installed",
            side_effect=FrameworkNotInstalledError(_INSTALL_HINT),
        ):
            with pytest.raises(FrameworkNotInstalledError):
                adapter.run({"query": "test"})

    def test_timing_captured(self):
        """total_duration_ms is positive after a run."""
        result = _make_function_result()
        kernel = _make_mock_kernel(invoke_return=result)
        adapter = SemanticKernelAdapter(kernel=kernel, model="gpt-4o")

        p1, p2 = _patch_sk_imports()
        with p1, p2:
            trace = adapter.run({"query": "test"})

        assert trace.total_duration_ms > 0.0

    def test_input_data_preserved(self):
        """The original input_data is preserved in the trace."""
        result = _make_function_result()
        kernel = _make_mock_kernel(invoke_return=result)
        adapter = SemanticKernelAdapter(kernel=kernel, model="gpt-4o")

        input_data = {"query": "test", "scenario_id": "s1", "extra": 42}
        p1, p2 = _patch_sk_imports()
        with p1, p2:
            trace = adapter.run(input_data)

        assert trace.input_data == input_data

    def test_metadata_in_trace(self):
        """Adapter-level metadata appears in the trace."""
        result = _make_function_result()
        kernel = _make_mock_kernel(invoke_return=result)
        adapter = SemanticKernelAdapter(
            kernel=kernel, model="gpt-4o", metadata={"team": "alpha"}
        )

        p1, p2 = _patch_sk_imports()
        with p1, p2:
            trace = adapter.run({"query": "test"})

        assert trace.metadata == {"team": "alpha"}


# ===================================================================
# Test: Fallback single-step trace
# ===================================================================


class TestFallbackSteps:
    """Tests for fallback (no filter) step generation."""

    def test_fallback_produces_single_step(self):
        """Without filter support, run produces exactly one step."""
        result = _make_function_result(value="Result text")
        kernel = _make_mock_kernel(invoke_return=result, has_add_filter=False)
        adapter = SemanticKernelAdapter(
            kernel=kernel,
            model="gpt-4o",
            function_name="respond",
        )

        p1, p2 = _patch_sk_imports()
        with p1, p2:
            trace = adapter.run({"query": "test"})

        assert len(trace.steps) == 1
        step = trace.steps[0]
        assert step.step_index == 0
        assert step.action == "llm_response"
        assert step.llm_output == "Result text"
        assert step.model == "gpt-4o"

    def test_fallback_step_metadata_has_mode(self):
        """Fallback step metadata includes mode='fallback'."""
        result = _make_function_result(value="ok")
        kernel = _make_mock_kernel(invoke_return=result, has_add_filter=False)
        adapter = SemanticKernelAdapter(kernel=kernel, model="gpt-4o")

        p1, p2 = _patch_sk_imports()
        with p1, p2:
            trace = adapter.run({"query": "test"})

        assert trace.steps[0].metadata.get("mode") == "fallback"

    def test_empty_execution_on_none_result(self):
        """A None result still produces a valid trace with a step."""
        kernel = _make_mock_kernel(invoke_return=None, has_add_filter=False)
        adapter = SemanticKernelAdapter(kernel=kernel, model="gpt-4o")

        p1, p2 = _patch_sk_imports()
        with p1, p2:
            trace = adapter.run({"query": "test"})

        assert trace.success is True
        assert trace.output_data is None
        assert len(trace.steps) == 1
        assert trace.steps[0].llm_output is None


# ===================================================================
# Test: Step classification
# ===================================================================


class TestStepClassification:
    """Tests for ``_classify_function`` heuristics."""

    def test_tool_classification_by_name(self):
        """Functions with 'tool' in the name are classified as tool_call."""
        action, extra = SemanticKernelAdapter._classify_function(
            func_name="SearchTool",
            plugin_name=None,
            result=_make_function_result("found it"),
            arguments={"query": "test"},
        )
        assert action == "tool_call"
        assert extra["tool_name"] == "SearchTool"

    def test_tool_classification_by_plugin(self):
        """Functions in a plugin with 'http' are classified as tool_call."""
        action, extra = SemanticKernelAdapter._classify_function(
            func_name="get",
            plugin_name="HttpPlugin",
            result=_make_function_result("response"),
            arguments=None,
        )
        assert action == "tool_call"
        assert extra["tool_name"] == "HttpPlugin.get"

    def test_llm_classification_by_chat(self):
        """Functions with 'chat' in the name are classified as llm_response."""
        action, extra = SemanticKernelAdapter._classify_function(
            func_name="chat_completion",
            plugin_name=None,
            result=_make_function_result("Hello!"),
            arguments={"input": "Say hi"},
        )
        assert action == "llm_response"
        assert extra["llm_output"] == "Hello!"
        assert extra["llm_input"] == "Say hi"

    def test_llm_classification_by_semantic(self):
        """Functions with 'semantic' in the name are classified as llm_response."""
        action, extra = SemanticKernelAdapter._classify_function(
            func_name="SemanticFunction",
            plugin_name=None,
            result=_make_function_result("answer"),
            arguments=None,
        )
        assert action == "llm_response"

    def test_observation_for_unknown_function(self):
        """Functions with no matching heuristic are classified as observation."""
        action, extra = SemanticKernelAdapter._classify_function(
            func_name="transform_data",
            plugin_name="MyCustomModule",
            result=_make_function_result("done"),
            arguments=None,
        )
        assert action == "observation"
        assert extra["llm_output"] == "done"

    def test_tool_input_extracted_from_dict_arguments(self):
        """Dict arguments are passed through as tool_input."""
        action, extra = SemanticKernelAdapter._classify_function(
            func_name="WebSearch",
            plugin_name=None,
            result=None,
            arguments={"query": "climate change"},
        )
        assert action == "tool_call"
        assert extra["tool_input"] == {"query": "climate change"}

    def test_none_result_produces_none_output(self):
        """A None result sets tool_output / llm_output to None."""
        action, extra = SemanticKernelAdapter._classify_function(
            func_name="DatabaseLookup",
            plugin_name=None,
            result=None,
            arguments=None,
        )
        assert action == "tool_call"
        assert extra["tool_output"] is None

    def test_file_plugin_classified_as_tool(self):
        """Plugin named 'FilePlugin' triggers tool_call classification."""
        action, extra = SemanticKernelAdapter._classify_function(
            func_name="ReadAll",
            plugin_name="FileIOPlugin",
            result=_make_function_result("contents"),
            arguments={"path": "/tmp/a.txt"},
        )
        assert action == "tool_call"
        assert "file" in extra["tool_name"].lower()


# ===================================================================
# Test: to_callable
# ===================================================================


class TestToCallable:
    """Tests for ``to_callable()``."""

    def test_returns_callable(self):
        """to_callable returns a callable object."""
        kernel = _make_mock_kernel()
        adapter = SemanticKernelAdapter(kernel=kernel, model="gpt-4o")

        with patch(
            "agentassay.integrations.semantic_kernel_adapter"
            "._check_semantic_kernel_installed"
        ):
            fn = adapter.to_callable()

        assert callable(fn)

    def test_callable_is_run_method(self):
        """The returned callable is the adapter's run method."""
        kernel = _make_mock_kernel()
        adapter = SemanticKernelAdapter(kernel=kernel, model="gpt-4o")

        with patch(
            "agentassay.integrations.semantic_kernel_adapter"
            "._check_semantic_kernel_installed"
        ):
            fn = adapter.to_callable()

        assert fn == adapter.run

    def test_to_callable_checks_install(self):
        """to_callable raises FrameworkNotInstalledError if SK not installed."""
        kernel = _make_mock_kernel()
        adapter = SemanticKernelAdapter(kernel=kernel)

        with patch(
            "agentassay.integrations.semantic_kernel_adapter"
            "._check_semantic_kernel_installed",
            side_effect=FrameworkNotInstalledError(_INSTALL_HINT),
        ):
            with pytest.raises(FrameworkNotInstalledError):
                adapter.to_callable()


# ===================================================================
# Test: get_config
# ===================================================================


class TestGetConfig:
    """Tests for ``get_config()``."""

    def test_returns_agent_config(self):
        """get_config produces a valid AgentConfig."""
        kernel = _make_mock_kernel()
        adapter = SemanticKernelAdapter(kernel=kernel, model="gpt-4o")
        config = adapter.get_config()

        assert isinstance(config, AgentConfig)
        assert config.model == "gpt-4o"
        assert config.framework == "semantic_kernel"

    def test_config_includes_plugin_metadata(self):
        """Plugin and function names are stored in config metadata."""
        kernel = _make_mock_kernel()
        adapter = SemanticKernelAdapter(
            kernel=kernel,
            model="gpt-4o",
            plugin_name="ChatPlugin",
            function_name="Respond",
        )
        config = adapter.get_config()

        assert config.metadata["plugin_name"] == "ChatPlugin"
        assert config.metadata["function_name"] == "Respond"

    def test_config_agent_name(self):
        """Agent name is reflected in the config."""
        kernel = _make_mock_kernel()
        adapter = SemanticKernelAdapter(
            kernel=kernel, model="gpt-4o", agent_name="my-bot"
        )
        config = adapter.get_config()
        assert config.name == "my-bot"

    def test_config_has_unique_agent_id(self):
        """Each get_config call produces a unique agent_id."""
        kernel = _make_mock_kernel()
        adapter = SemanticKernelAdapter(kernel=kernel, model="gpt-4o")
        c1 = adapter.get_config()
        c2 = adapter.get_config()
        assert c1.agent_id != c2.agent_id


# ===================================================================
# Test: Output and cost extraction
# ===================================================================


class TestExtraction:
    """Tests for ``_extract_output`` and ``_extract_cost``."""

    def test_extract_output_from_value(self):
        """_extract_output returns .value when available."""
        result = _make_function_result(value="The answer")
        output = SemanticKernelAdapter._extract_output(result)
        assert output == "The answer"

    def test_extract_output_from_result_attr(self):
        """_extract_output falls back to .result when .value is None."""
        result = MagicMock()
        result.value = None
        result.result = "fallback answer"
        output = SemanticKernelAdapter._extract_output(result)
        assert output == "fallback answer"

    def test_extract_output_none(self):
        """_extract_output returns None for None input."""
        assert SemanticKernelAdapter._extract_output(None) is None

    def test_extract_output_stringifies(self):
        """_extract_output stringifies when no .value or .result."""
        result = MagicMock(spec=[])  # No .value, no .result
        output = SemanticKernelAdapter._extract_output(result)
        assert isinstance(output, str)

    def test_extract_cost_from_metadata(self):
        """_extract_cost picks up cost from metadata dict."""
        result = _make_function_result(
            metadata={"cost": 0.0015}
        )
        cost = SemanticKernelAdapter._extract_cost(result)
        assert cost == 0.0015

    def test_extract_cost_from_token_usage_dict(self):
        """_extract_cost estimates from token usage dict."""
        result = _make_function_result(
            metadata={"usage": {"total_tokens": 1000}}
        )
        cost = SemanticKernelAdapter._extract_cost(result)
        assert cost == pytest.approx(0.01, abs=1e-6)

    def test_extract_cost_returns_zero_when_unavailable(self):
        """_extract_cost returns 0.0 when no cost data is found."""
        result = _make_function_result(metadata={})
        cost = SemanticKernelAdapter._extract_cost(result)
        assert cost == 0.0

    def test_extract_cost_none_result(self):
        """_extract_cost returns 0.0 for None result."""
        assert SemanticKernelAdapter._extract_cost(None) == 0.0


# ===================================================================
# Test: Filter-based step capture
# ===================================================================


class TestFilterCapture:
    """Tests for FunctionInvocationFilter registration."""

    def test_filter_not_registered_without_add_filter(self):
        """Filter is not registered when kernel lacks add_filter."""
        kernel = _make_mock_kernel(has_add_filter=False)
        adapter = SemanticKernelAdapter(kernel=kernel, model="gpt-4o")

        captured: list[dict[str, Any]] = []
        result = adapter._try_register_filter(captured)
        assert result is False

    def test_filter_registered_with_add_filter(self):
        """Filter is registered when kernel has add_filter."""
        kernel = _make_mock_kernel(has_add_filter=True)
        adapter = SemanticKernelAdapter(kernel=kernel, model="gpt-4o")

        captured: list[dict[str, Any]] = []
        result = adapter._try_register_filter(captured)
        assert result is True
        kernel.add_filter.assert_called_once()

    def test_filter_registration_failure_returns_false(self):
        """If add_filter raises, registration returns False gracefully."""
        kernel = MagicMock()
        kernel.add_filter.side_effect = TypeError("unsupported filter")
        adapter = SemanticKernelAdapter(kernel=kernel, model="gpt-4o")

        captured: list[dict[str, Any]] = []
        result = adapter._try_register_filter(captured)
        assert result is False

    def test_build_steps_from_captures(self):
        """_build_steps_from_captures creates correct StepTrace objects."""
        kernel = _make_mock_kernel()
        adapter = SemanticKernelAdapter(kernel=kernel, model="gpt-4o")

        captures = [
            {
                "function_name": "WebSearch",
                "plugin_name": "SearchPlugin",
                "duration_ms": 150.0,
                "result": _make_function_result("search results"),
                "arguments": {"query": "test"},
            },
            {
                "function_name": "chat_completion",
                "plugin_name": None,
                "duration_ms": 300.0,
                "result": _make_function_result("Generated answer"),
                "arguments": {"input": "Based on results..."},
            },
        ]

        steps = adapter._build_steps_from_captures(captures)

        assert len(steps) == 2

        # First step: tool_call (WebSearch contains "search")
        assert steps[0].step_index == 0
        assert steps[0].action == "tool_call"
        assert steps[0].tool_name == "SearchPlugin.WebSearch"
        assert steps[0].duration_ms == 150.0
        assert steps[0].metadata["sk_plugin"] == "SearchPlugin"

        # Second step: llm_response (chat_completion contains "chat")
        assert steps[1].step_index == 1
        assert steps[1].action == "llm_response"
        assert steps[1].duration_ms == 300.0


# ===================================================================
# Test: Edge cases
# ===================================================================


class TestEdgeCases:
    """Edge-case and integration-level tests."""

    def test_run_with_empty_input(self):
        """run() handles empty input_data gracefully."""
        result = _make_function_result(value="ok")
        kernel = _make_mock_kernel(invoke_return=result)
        adapter = SemanticKernelAdapter(kernel=kernel, model="gpt-4o")

        p1, p2 = _patch_sk_imports()
        with p1, p2:
            trace = adapter.run({})

        assert trace.success is True
        assert trace.scenario_id == "default"

    def test_cost_tracked_in_trace(self):
        """Cost from result metadata appears in trace total_cost_usd."""
        result = _make_function_result(
            value="answer",
            metadata={"cost": 0.005},
        )
        kernel = _make_mock_kernel(invoke_return=result)
        adapter = SemanticKernelAdapter(kernel=kernel, model="gpt-4o")

        p1, p2 = _patch_sk_imports()
        with p1, p2:
            trace = adapter.run({"query": "test"})

        assert trace.total_cost_usd == 0.005

    def test_trace_id_is_unique_per_run(self):
        """Each run() call produces a unique trace_id."""
        result = _make_function_result()
        kernel = _make_mock_kernel(invoke_return=result)
        adapter = SemanticKernelAdapter(kernel=kernel, model="gpt-4o")

        p1, p2 = _patch_sk_imports()
        with p1, p2:
            t1 = adapter.run({"query": "a"})
            t2 = adapter.run({"query": "b"})

        assert t1.trace_id != t2.trace_id

    def test_error_trace_has_positive_duration(self):
        """Even failed traces have positive total_duration_ms."""
        kernel = _make_mock_kernel()
        kernel.invoke.side_effect = ValueError("bad input")
        adapter = SemanticKernelAdapter(kernel=kernel, model="gpt-4o")

        p1, p2 = _patch_sk_imports()
        with p1, p2:
            trace = adapter.run({"query": "test"})

        assert trace.success is False
        assert trace.total_duration_ms > 0.0

    def test_kernel_arguments_exclude_scenario_id(self):
        """scenario_id and metadata are filtered from kernel arguments."""
        result = _make_function_result()
        kernel = _make_mock_kernel(invoke_return=result)
        adapter = SemanticKernelAdapter(
            kernel=kernel,
            model="gpt-4o",
            plugin_name="p",
            function_name="f",
        )

        p1, p2 = _patch_sk_imports()
        with p1, p2:
            adapter.run({
                "scenario_id": "s1",
                "metadata": {"x": 1},
                "query": "hello",
            })

        # The invoke call should have been made; inspect the arguments kwarg
        call_kwargs = kernel.invoke.call_args
        # arguments kwarg should be a dict (from our mock KernelArguments)
        args_passed = call_kwargs.kwargs.get(
            "arguments", call_kwargs[1].get("arguments")
        )
        # Should only contain "query", not "scenario_id" or "metadata"
        assert "scenario_id" not in args_passed
        assert "metadata" not in args_passed
        assert args_passed["query"] == "hello"
