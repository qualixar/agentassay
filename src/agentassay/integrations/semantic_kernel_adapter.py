# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Semantic Kernel (Microsoft) framework adapter for AgentAssay.

Wraps a Semantic Kernel ``Kernel`` and captures plugin function invocations
as ``StepTrace`` objects. Uses ``FunctionInvocationFilter`` for non-intrusive
trace capture when available, falling back to direct ``kernel.invoke()`` with
manual step extraction.

All ``semantic_kernel`` imports are **lazy** --- this module can be imported
even when Semantic Kernel is not installed.  The ``ImportError`` is raised
only when ``run()`` or ``to_callable()`` is actually called.

Usage
-----
>>> from agentassay.integrations.semantic_kernel_adapter import SemanticKernelAdapter
>>> adapter = SemanticKernelAdapter(kernel=my_kernel, model="gpt-4o",
...                                 plugin_name="chat", function_name="respond")
>>> trace = adapter.run({"query": "What is the capital of France?"})
>>> runner = TrialRunner(adapter.to_callable(), assay_config, adapter.get_config())
"""

from __future__ import annotations

import logging
import time
import traceback
import uuid
from collections.abc import Callable
from typing import Any

from agentassay.core.models import ExecutionTrace, StepTrace
from agentassay.integrations.base import (
    AgentAdapter,
    FrameworkNotInstalledError,
)

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "Semantic Kernel adapter requires semantic-kernel. "
    "Install with: pip install agentassay[semantic-kernel]"
)


def _check_semantic_kernel_installed() -> None:
    """Verify that semantic-kernel is available, raise clear error if not.

    Raises
    ------
    FrameworkNotInstalledError
        If ``semantic_kernel`` cannot be imported.
    """
    try:
        import semantic_kernel  # noqa: F401
    except ImportError as exc:
        raise FrameworkNotInstalledError(_INSTALL_HINT) from exc


class SemanticKernelAdapter(AgentAdapter):
    """Adapter for Microsoft Semantic Kernel instances.

    Wraps a Semantic Kernel ``Kernel`` object and converts plugin/function
    invocations into AgentAssay's uniform ``ExecutionTrace`` format.

    Semantic Kernel organizes functionality through *plugins* (collections
    of related functions) and *functions* (individual skills).  Each function
    invocation during ``kernel.invoke()`` becomes one ``StepTrace``.

    Parameters
    ----------
    kernel
        A Semantic Kernel ``Kernel`` instance.  The kernel should already
        have plugins and services registered.
    model
        LLM model identifier (e.g. ``"gpt-4o"``, ``"gpt-4o-mini"``).
    plugin_name
        The name of the plugin to invoke.  If ``None``, the adapter
        requires ``function_name`` to be a fully qualified name
        (``"PluginName-FunctionName"``) or uses the default function
        on the kernel.
    function_name
        The name of the function within the plugin to invoke.  Combined
        with ``plugin_name`` to resolve the target function.
    config
        Optional configuration dict passed as extra keyword arguments
        to ``kernel.invoke()``.
    agent_name
        Human-readable name for the agent.  Defaults to
        ``"semantic_kernel-agent"``.
    metadata
        Arbitrary metadata attached to every ``ExecutionTrace`` produced.

    Notes
    -----
    The adapter uses ``"custom"`` as the framework identifier in
    ``AgentConfig`` because ``"semantic_kernel"`` is not yet in the
    canonical framework literal set.  The ``framework`` attribute on
    the class itself is ``"semantic_kernel"`` for display and trace
    identification purposes.
    """

    framework: str = "semantic_kernel"

    def __init__(
        self,
        kernel: Any,
        *,
        model: str = "unknown",
        plugin_name: str | None = None,
        function_name: str | None = None,
        config: dict[str, Any] | None = None,
        agent_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            model=model, agent_name=agent_name, metadata=metadata
        )
        self._kernel = kernel
        self._plugin_name = plugin_name
        self._function_name = function_name
        self._config = config or {}

    # -- Core interface -------------------------------------------------------

    def run(self, input_data: dict[str, Any]) -> ExecutionTrace:
        """Invoke the Semantic Kernel and capture an ExecutionTrace.

        Executes the configured plugin function via ``kernel.invoke()``,
        captures each function invocation as a ``StepTrace``, and wraps
        the result in an ``ExecutionTrace``.

        When the kernel exposes ``FunctionInvocationFilter`` hooks, the
        adapter registers a temporary filter to capture per-function
        timing and metadata without modifying user code.  Otherwise,
        it falls back to wrapping the single ``invoke()`` result.

        Parameters
        ----------
        input_data
            The scenario input dictionary.  Keys are passed as
            ``KernelArguments`` to the kernel function.

        Returns
        -------
        ExecutionTrace
            A complete trace with per-function steps, timing, and output.
            On failure, ``success=False`` with the error message.
        """
        _check_semantic_kernel_installed()

        scenario_id = input_data.get("scenario_id", "default")
        trace_id = str(uuid.uuid4())
        overall_start = time.perf_counter()

        try:
            steps, output, cost = self._invoke_kernel(input_data)
            total_ms = (time.perf_counter() - overall_start) * 1000.0

            return ExecutionTrace(
                trace_id=trace_id,
                scenario_id=scenario_id,
                steps=steps,
                input_data=input_data,
                output_data=output,
                success=True,
                error=None,
                total_duration_ms=total_ms,
                total_cost_usd=cost,
                model=self._model,
                framework=self.framework,
                metadata=self._metadata,
            )

        except FrameworkNotInstalledError:
            raise  # Do not swallow install errors

        except Exception as exc:
            total_ms = (time.perf_counter() - overall_start) * 1000.0
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.error(
                "Semantic Kernel adapter failed: %s\n%s",
                error_msg,
                traceback.format_exc(),
            )
            return ExecutionTrace(
                trace_id=trace_id,
                scenario_id=scenario_id,
                steps=[],
                input_data=input_data,
                output_data=None,
                success=False,
                error=error_msg,
                total_duration_ms=total_ms,
                total_cost_usd=0.0,
                model=self._model,
                framework=self.framework,
                metadata=self._metadata,
            )

    def to_callable(self) -> Callable[[dict[str, Any]], ExecutionTrace]:
        """Return a TrialRunner-compatible callable.

        Returns
        -------
        Callable[[dict[str, Any]], ExecutionTrace]
            A closure that calls ``self.run(input_data)``.
        """
        _check_semantic_kernel_installed()
        return self.run

    # -- Override: get_config with compatible framework -----------------------

    def get_config(self):
        """Build an ``AgentConfig`` describing this adapter's agent.

        Uses ``"custom"`` as the framework literal because the canonical
        ``AgentConfig.framework`` Literal does not yet include
        ``"semantic_kernel"``.  The actual framework identifier is stored
        in ``metadata["actual_framework"]``.

        Returns
        -------
        AgentConfig
            Configuration with ``framework="custom"`` and the actual
            framework name preserved in metadata.
        """
        from agentassay.core.models import AgentConfig

        merged_metadata = {
            **self._metadata,
            "actual_framework": self.framework,
        }
        if self._plugin_name:
            merged_metadata["plugin_name"] = self._plugin_name
        if self._function_name:
            merged_metadata["function_name"] = self._function_name

        return AgentConfig(
            agent_id=str(uuid.uuid4()),
            name=self._agent_name,
            framework="semantic_kernel",
            model=self._model,
            metadata=merged_metadata,
        )

    # -- Internal: kernel invocation ------------------------------------------

    def _invoke_kernel(
        self, input_data: dict[str, Any]
    ) -> tuple[list[StepTrace], Any, float]:
        """Execute the kernel function and return steps, output, and cost.

        Attempts to use ``FunctionInvocationFilter`` for per-function
        trace capture.  Falls back to a single-step trace if filters
        are unavailable.

        Parameters
        ----------
        input_data
            The scenario input dictionary.

        Returns
        -------
        tuple[list[StepTrace], Any, float]
            (steps, final_output, total_cost_usd)
        """
        from semantic_kernel import KernelArguments

        # Build KernelArguments from input_data (exclude scenario_id)
        kernel_args = self._build_kernel_arguments(input_data)

        # Collected steps from filter callbacks
        captured_steps: list[dict[str, Any]] = []

        # Try to register a FunctionInvocationFilter for step capture
        filter_registered = self._try_register_filter(captured_steps)

        try:
            invoke_start = time.perf_counter()
            result = self._execute_invoke(kernel_args)
            invoke_duration_ms = (time.perf_counter() - invoke_start) * 1000.0
        finally:
            # Remove filter if we registered one
            if filter_registered:
                self._try_unregister_filter()

        # Extract output and cost from result
        output = self._extract_output(result)
        cost = self._extract_cost(result)

        # Build StepTrace list
        if captured_steps:
            steps = self._build_steps_from_captures(captured_steps)
        else:
            # Fallback: single-step trace from the invoke result
            steps = self._build_fallback_steps(result, invoke_duration_ms)

        return steps, output, cost

    def _build_kernel_arguments(
        self, input_data: dict[str, Any]
    ) -> Any:
        """Build ``KernelArguments`` from the scenario input_data.

        Filters out internal keys (``scenario_id``, ``metadata``) and
        passes remaining key-value pairs as kernel arguments.

        Parameters
        ----------
        input_data
            Raw scenario input.

        Returns
        -------
        KernelArguments
            Arguments ready for ``kernel.invoke()``.
        """
        from semantic_kernel import KernelArguments

        filtered = {
            k: v
            for k, v in input_data.items()
            if k not in ("scenario_id", "metadata")
        }

        # If there's a single "input" or "query" key, use it as the
        # primary argument for convenience
        return KernelArguments(**filtered)

    def _execute_invoke(self, kernel_args: Any) -> Any:
        """Call ``kernel.invoke()`` with the resolved plugin and function.

        Parameters
        ----------
        kernel_args
            ``KernelArguments`` to pass.

        Returns
        -------
        Any
            The ``FunctionResult`` from kernel invocation.
        """
        invoke_kwargs: dict[str, Any] = {**self._config}

        if self._plugin_name and self._function_name:
            return self._kernel.invoke(
                plugin_name=self._plugin_name,
                function_name=self._function_name,
                arguments=kernel_args,
                **invoke_kwargs,
            )

        if self._function_name:
            # function_name might be fully qualified: "PluginName-FunctionName"
            return self._kernel.invoke(
                function_name=self._function_name,
                arguments=kernel_args,
                **invoke_kwargs,
            )

        # No specific function — invoke the kernel's default (if any)
        return self._kernel.invoke(
            arguments=kernel_args,
            **invoke_kwargs,
        )

    def _try_register_filter(
        self, captured_steps: list[dict[str, Any]]
    ) -> bool:
        """Attempt to register a FunctionInvocationFilter on the kernel.

        The filter captures per-function invocation metadata (name,
        plugin, timing) into ``captured_steps`` without modifying the
        kernel's behavior.

        Parameters
        ----------
        captured_steps
            Mutable list to append captured step dicts to.

        Returns
        -------
        bool
            ``True`` if the filter was successfully registered.
        """
        try:
            # Semantic Kernel v1.x exposes filter hooks on the kernel
            if not hasattr(self._kernel, "add_filter"):
                return False

            def _on_function_invocation(context: Any, next_handler: Any) -> Any:
                """Capture function invocation as a step."""
                step_start = time.perf_counter()
                # Call the next handler in the pipeline
                result = next_handler(context)
                duration_ms = (time.perf_counter() - step_start) * 1000.0

                func_name = getattr(context, "function_name", None) or "unknown"
                plugin = getattr(context, "plugin_name", None)
                func_result = getattr(context, "result", None)

                captured_steps.append({
                    "function_name": func_name,
                    "plugin_name": plugin,
                    "duration_ms": duration_ms,
                    "result": func_result,
                    "arguments": getattr(context, "arguments", None),
                })
                return result

            self._kernel.add_filter(
                "function_invocation", _on_function_invocation
            )
            self._active_filter = _on_function_invocation
            return True

        except Exception:
            # Filter registration failed — degrade gracefully
            logger.debug(
                "Could not register FunctionInvocationFilter; "
                "falling back to single-step trace."
            )
            return False

    def _try_unregister_filter(self) -> None:
        """Remove the previously registered filter, if any."""
        try:
            if hasattr(self, "_active_filter") and hasattr(
                self._kernel, "remove_filter"
            ):
                self._kernel.remove_filter(
                    "function_invocation", self._active_filter
                )
        except Exception:
            pass  # Best-effort cleanup
        finally:
            self._active_filter = None

    # -- Internal: step building ----------------------------------------------

    def _build_steps_from_captures(
        self, captured_steps: list[dict[str, Any]]
    ) -> list[StepTrace]:
        """Convert captured filter events into ``StepTrace`` objects.

        Parameters
        ----------
        captured_steps
            List of dicts captured by the ``FunctionInvocationFilter``.

        Returns
        -------
        list[StepTrace]
            Ordered step traces.
        """
        steps: list[StepTrace] = []
        for idx, capture in enumerate(captured_steps):
            func_name = capture.get("function_name", "unknown")
            plugin_name = capture.get("plugin_name")
            duration_ms = capture.get("duration_ms", 0.0)
            result = capture.get("result")
            arguments = capture.get("arguments")

            action, extra = self._classify_function(
                func_name, plugin_name, result, arguments
            )

            step_metadata: dict[str, Any] = {
                "sk_function": func_name,
            }
            if plugin_name:
                step_metadata["sk_plugin"] = plugin_name

            steps.append(
                StepTrace(
                    step_index=idx,
                    action=action,
                    duration_ms=duration_ms,
                    model=self._model,
                    metadata=step_metadata,
                    **extra,
                )
            )
        return steps

    def _build_fallback_steps(
        self, result: Any, duration_ms: float
    ) -> list[StepTrace]:
        """Build a single-step trace from a ``FunctionResult``.

        Used when filter-based capture is unavailable.

        Parameters
        ----------
        result
            The ``FunctionResult`` from ``kernel.invoke()``.
        duration_ms
            Total invocation duration in milliseconds.

        Returns
        -------
        list[StepTrace]
            A list with a single step.
        """
        output_str = self._extract_output(result)
        func_name = self._function_name or "invoke"
        plugin_name = self._plugin_name

        step_metadata: dict[str, Any] = {
            "sk_function": func_name,
            "mode": "fallback",
        }
        if plugin_name:
            step_metadata["sk_plugin"] = plugin_name

        return [
            StepTrace(
                step_index=0,
                action="llm_response",
                llm_output=str(output_str) if output_str is not None else None,
                duration_ms=duration_ms,
                model=self._model,
                metadata=step_metadata,
            )
        ]

    # -- Internal: classification ---------------------------------------------

    @staticmethod
    def _classify_function(
        func_name: str,
        plugin_name: str | None,
        result: Any,
        arguments: Any,
    ) -> tuple[str, dict[str, Any]]:
        """Classify a Semantic Kernel function invocation into a step action.

        Heuristics:
        - If the function/plugin name suggests a tool or native function
          (contains "tool", "search", "calculate", "http", "file", etc.),
          classify as ``tool_call``.
        - If the function/plugin name suggests an LLM call (contains
          "chat", "complete", "generate", "prompt", "semantic"),
          classify as ``llm_response``.
        - Otherwise classify as ``observation``.

        Parameters
        ----------
        func_name
            The kernel function name.
        plugin_name
            The plugin that owns the function, or ``None``.
        result
            The ``FunctionResult`` (may be ``None``).
        arguments
            The ``KernelArguments`` passed to the function.

        Returns
        -------
        tuple[str, dict[str, Any]]
            ``(action_type, extra_kwargs_for_StepTrace)``
        """
        extra: dict[str, Any] = {}
        combined_name = (
            f"{plugin_name}.{func_name}" if plugin_name else func_name
        ).lower()

        # Extract result value as string
        result_str = None
        if result is not None:
            result_str = str(getattr(result, "value", result))

        # Extract arguments as dict
        args_dict = None
        if arguments is not None:
            if isinstance(arguments, dict):
                args_dict = arguments
            elif hasattr(arguments, "__iter__"):
                try:
                    args_dict = dict(arguments)
                except (TypeError, ValueError):
                    args_dict = {"raw": str(arguments)}
            else:
                args_dict = {"raw": str(arguments)}

        # Tool-like function heuristics
        _tool_indicators = (
            "tool", "search", "calculate", "http", "file", "web",
            "database", "db", "api", "fetch", "read", "write",
            "native", "plugin",
        )
        if any(indicator in combined_name for indicator in _tool_indicators):
            extra["tool_name"] = (
                f"{plugin_name}.{func_name}" if plugin_name else func_name
            )
            extra["tool_input"] = args_dict
            extra["tool_output"] = result_str
            return "tool_call", extra

        # LLM-like function heuristics
        _llm_indicators = (
            "chat", "complete", "generate", "prompt", "semantic",
            "llm", "gpt", "openai", "azure_chat",
        )
        if any(indicator in combined_name for indicator in _llm_indicators):
            # Extract the input prompt if available
            if args_dict:
                input_text = (
                    args_dict.get("input")
                    or args_dict.get("prompt")
                    or args_dict.get("query")
                    or args_dict.get("message")
                )
                if input_text:
                    extra["llm_input"] = str(input_text)
            extra["llm_output"] = result_str
            return "llm_response", extra

        # Default: observation
        extra["llm_output"] = result_str
        return "observation", extra

    # -- Internal: output/cost extraction -------------------------------------

    @staticmethod
    def _extract_output(result: Any) -> Any:
        """Extract the final output from a Semantic Kernel ``FunctionResult``.

        Parameters
        ----------
        result
            The result object from ``kernel.invoke()``.

        Returns
        -------
        Any
            The extracted output value, or a string representation.
        """
        if result is None:
            return None

        # FunctionResult has a .value property
        value = getattr(result, "value", None)
        if value is not None:
            return value

        # Some versions expose .result
        inner_result = getattr(result, "result", None)
        if inner_result is not None:
            return inner_result

        # Fallback to string
        return str(result)

    @staticmethod
    def _extract_cost(result: Any) -> float:
        """Extract token cost from the result metadata if available.

        Semantic Kernel may expose usage metadata through the
        ``FunctionResult.metadata`` dict (provider-dependent).

        Parameters
        ----------
        result
            The result object from ``kernel.invoke()``.

        Returns
        -------
        float
            Estimated cost in USD, or ``0.0`` if unavailable.
        """
        if result is None:
            return 0.0

        # Check for metadata with usage info
        metadata = getattr(result, "metadata", None)
        if isinstance(metadata, dict):
            # Some providers embed cost or usage in metadata
            cost = metadata.get("cost") or metadata.get("total_cost")
            if cost is not None:
                try:
                    return float(cost)
                except (TypeError, ValueError):
                    pass

            # Check for token usage and estimate cost
            usage = metadata.get("usage") or metadata.get("token_usage")
            if usage is not None:
                total_tokens = getattr(usage, "total_tokens", None)
                if total_tokens is None and isinstance(usage, dict):
                    total_tokens = usage.get("total_tokens")
                if total_tokens is not None:
                    # Rough estimate: $0.01 per 1000 tokens (varies by model)
                    try:
                        return float(total_tokens) * 0.00001
                    except (TypeError, ValueError):
                        pass

        return 0.0
