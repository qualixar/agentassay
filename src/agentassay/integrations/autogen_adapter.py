# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""AutoGen framework adapter for AgentAssay.

Wraps a Microsoft AutoGen agent (v0.4+ / AG2) and captures its execution
as an ``ExecutionTrace``. AutoGen uses a multi-agent conversation paradigm
with ``initiate_chat()`` or the newer ``run()`` / ``run_stream()``
patterns depending on the version.

All ``autogen`` / ``autogen_agentchat`` imports are **lazy** — this module
can be imported even when AutoGen is not installed.

Usage
-----
>>> from agentassay.integrations import AutoGenAdapter
>>> adapter = AutoGenAdapter(agent=my_agent, model="gpt-4o")
>>> trace = adapter.run({"message": "Solve this math problem"})
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
    "AutoGen adapter requires autogen-agentchat. "
    "Install with: pip install agentassay[autogen]"
)


def _check_autogen_installed() -> None:
    """Verify that AutoGen is available."""
    try:
        import autogen_agentchat  # noqa: F401
    except ImportError:
        try:
            import autogen  # noqa: F401
        except ImportError as exc:
            raise FrameworkNotInstalledError(_INSTALL_HINT) from exc


class AutoGenAdapter(AgentAdapter):
    """Adapter for Microsoft AutoGen / AG2 agents.

    Supports both AutoGen v0.4+ (``autogen-agentchat``) and the older
    ``pyautogen`` API. The adapter tries the modern API first and falls
    back to the legacy ``initiate_chat()`` pattern.

    Parameters
    ----------
    agent
        An AutoGen agent instance (``AssistantAgent``, ``ConversableAgent``,
        or any agent with a ``run()`` or ``initiate_chat()`` method).
    user_proxy
        Optional ``UserProxyAgent`` for the ``initiate_chat()`` pattern.
        If not provided, a minimal proxy is created internally when needed.
    model
        LLM model identifier. If ``"unknown"``, attempts to read from agent config.
    agent_name
        Human-readable name. Defaults to ``agent.name`` or ``"autogen-agent"``.
    metadata
        Arbitrary metadata attached to every trace.
    """

    framework: str = "autogen"

    def __init__(
        self,
        agent: Any,
        *,
        user_proxy: Any | None = None,
        model: str = "unknown",
        agent_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        resolved_model = model
        if model == "unknown":
            # AutoGen stores model in llm_config
            llm_config = getattr(agent, "llm_config", None)
            if isinstance(llm_config, dict):
                config_list = llm_config.get("config_list", [])
                if config_list and isinstance(config_list, list):
                    resolved_model = config_list[0].get("model", "unknown")
                elif "model" in llm_config:
                    resolved_model = llm_config["model"]

        resolved_name = agent_name
        if resolved_name is None:
            resolved_name = getattr(agent, "name", None) or "autogen-agent"

        super().__init__(
            model=resolved_model, agent_name=resolved_name, metadata=metadata
        )
        self._agent = agent
        self._user_proxy = user_proxy

    # -- Core interface -------------------------------------------------------

    def run(self, input_data: dict[str, Any]) -> ExecutionTrace:
        """Invoke the AutoGen agent and capture an ExecutionTrace.

        Tries these execution strategies in order:
        1. ``agent.run()`` (AutoGen v0.4+ async runner, called synchronously)
        2. ``user_proxy.initiate_chat(agent, message=...)`` (legacy pattern)
        3. ``agent.generate_reply(messages=[...])`` (single-turn fallback)

        Parameters
        ----------
        input_data
            The scenario input. Expects a ``"message"`` or ``"query"`` key.

        Returns
        -------
        ExecutionTrace
            Full trace with conversation steps.
        """
        _check_autogen_installed()

        scenario_id = input_data.get("scenario_id", "default")
        trace_id = str(uuid.uuid4())
        overall_start = time.perf_counter()

        try:
            user_message = self._build_message(input_data)
            steps, output = self._execute_agent(user_message)
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
                total_cost_usd=0.0,
                model=self._model,
                framework=self.framework,
                metadata=self._metadata,
            )

        except FrameworkNotInstalledError:
            raise

        except Exception as exc:
            total_ms = (time.perf_counter() - overall_start) * 1000.0
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.error(
                "AutoGen adapter failed: %s\n%s",
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
        """Return a TrialRunner-compatible callable."""
        _check_autogen_installed()
        return self.run

    # -- Internal: execution strategies ---------------------------------------

    def _execute_agent(
        self, message: str
    ) -> tuple[list[StepTrace], Any]:
        """Try available execution strategies and return steps + output."""
        # Strategy 1: Modern AutoGen v0.4+ run() method
        if hasattr(self._agent, "run"):
            return self._run_modern(message)

        # Strategy 2: Legacy initiate_chat pattern
        if self._user_proxy is not None and hasattr(
            self._user_proxy, "initiate_chat"
        ):
            return self._run_initiate_chat(message)

        # Strategy 3: Single-turn generate_reply fallback
        if hasattr(self._agent, "generate_reply"):
            return self._run_generate_reply(message)

        raise RuntimeError(
            f"AutoGen agent {type(self._agent).__name__} has no recognized "
            "execution method (run, initiate_chat, or generate_reply)."
        )

    def _run_modern(
        self, message: str
    ) -> tuple[list[StepTrace], Any]:
        """Execute using AutoGen v0.4+ ``run()`` or ``run_sync()``."""
        step_start = time.perf_counter()

        # AutoGen v0.4 uses async; try run_sync first
        if hasattr(self._agent, "run_sync"):
            result = self._agent.run_sync(task=message)
        else:
            import asyncio

            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                loop = None

            if loop and loop.is_running():
                import concurrent.futures

                with concurrent.futures.ThreadPoolExecutor() as pool:
                    result = pool.submit(
                        asyncio.run, self._agent.run(task=message)
                    ).result()
            else:
                result = asyncio.run(self._agent.run(task=message))

        duration_ms = (time.perf_counter() - step_start) * 1000.0
        steps, output = self._extract_from_result(result, duration_ms)
        return steps, output

    def _run_initiate_chat(
        self, message: str
    ) -> tuple[list[StepTrace], Any]:
        """Execute using legacy ``initiate_chat()`` pattern."""
        step_start = time.perf_counter()
        chat_result = self._user_proxy.initiate_chat(
            self._agent, message=message
        )
        duration_ms = (time.perf_counter() - step_start) * 1000.0

        return self._extract_from_chat_history(chat_result, duration_ms)

    def _run_generate_reply(
        self, message: str
    ) -> tuple[list[StepTrace], Any]:
        """Execute using single-turn ``generate_reply()``."""
        step_start = time.perf_counter()
        messages = [{"role": "user", "content": message}]
        reply = self._agent.generate_reply(messages=messages)
        duration_ms = (time.perf_counter() - step_start) * 1000.0

        reply_str = str(reply) if reply is not None else ""
        steps = [
            StepTrace(
                step_index=0,
                action="llm_response",
                llm_input=message,
                llm_output=reply_str,
                duration_ms=duration_ms,
                model=self._model,
                metadata={"autogen_method": "generate_reply"},
            )
        ]
        return steps, reply

    # -- Internal: result extraction ------------------------------------------

    def _extract_from_result(
        self, result: Any, total_ms: float
    ) -> tuple[list[StepTrace], Any]:
        """Extract steps from a modern AutoGen RunResult / TaskResult."""
        steps: list[StepTrace] = []

        # AutoGen v0.4+: result.messages is a list of ChatMessage objects
        messages = getattr(result, "messages", None)
        if messages and isinstance(messages, (list, tuple)):
            num_msgs = len(messages)
            per_msg_ms = total_ms / max(num_msgs, 1)

            for idx, msg in enumerate(messages):
                action, kwargs = self._classify_autogen_message(msg)
                steps.append(
                    StepTrace(
                        step_index=idx,
                        action=action,
                        duration_ms=per_msg_ms,
                        model=self._model,
                        metadata={
                            "autogen_method": "run",
                            "autogen_msg_type": type(msg).__name__,
                        },
                        **kwargs,
                    )
                )

            # Final output from last assistant message
            output = getattr(result, "output", None) or (
                str(messages[-1]) if messages else None
            )
            return steps, output

        # Fallback single step
        output = getattr(result, "output", None) or str(result)
        steps.append(
            StepTrace(
                step_index=0,
                action="llm_response",
                llm_output=str(output),
                duration_ms=total_ms,
                model=self._model,
                metadata={"autogen_method": "run", "autogen_fallback": True},
            )
        )
        return steps, output

    def _extract_from_chat_history(
        self, chat_result: Any, total_ms: float
    ) -> tuple[list[StepTrace], Any]:
        """Extract steps from a legacy initiate_chat ChatResult."""
        steps: list[StepTrace] = []

        # Legacy AutoGen: chat_result.chat_history is a list of dicts
        history = getattr(chat_result, "chat_history", None)
        if not history:
            # Try the agent's own chat_messages
            if self._user_proxy is not None:
                history = getattr(
                    self._user_proxy, "chat_messages", {}
                ).get(self._agent, [])

        if history and isinstance(history, (list, tuple)):
            num_msgs = len(history)
            per_msg_ms = total_ms / max(num_msgs, 1)

            for idx, msg in enumerate(history):
                action, kwargs = self._classify_autogen_message(msg)
                steps.append(
                    StepTrace(
                        step_index=idx,
                        action=action,
                        duration_ms=per_msg_ms,
                        model=self._model,
                        metadata={
                            "autogen_method": "initiate_chat",
                            "autogen_msg_index": idx,
                        },
                        **kwargs,
                    )
                )

            output = self._get_last_assistant_content(history)
            return steps, output

        # Fallback
        summary = getattr(chat_result, "summary", str(chat_result))
        steps.append(
            StepTrace(
                step_index=0,
                action="llm_response",
                llm_output=str(summary),
                duration_ms=total_ms,
                model=self._model,
                metadata={"autogen_method": "initiate_chat", "autogen_fallback": True},
            )
        )
        return steps, summary

    @staticmethod
    def _classify_autogen_message(
        msg: Any,
    ) -> tuple[str, dict[str, Any]]:
        """Classify an AutoGen message (dict or object) into a step action."""
        extra: dict[str, Any] = {}

        # Dict-style messages (legacy AutoGen)
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
            tool_calls = msg.get("tool_calls")

            if tool_calls:
                if isinstance(tool_calls, list) and tool_calls:
                    tc = tool_calls[0]
                    func = tc.get("function", tc)
                    extra["tool_name"] = func.get("name", "unknown")
                    extra["tool_input"] = func.get("arguments", {})
                    if isinstance(extra["tool_input"], str):
                        try:
                            import json
                            extra["tool_input"] = json.loads(extra["tool_input"])
                        except (ValueError, TypeError):
                            extra["tool_input"] = {"raw": extra["tool_input"]}
                    return "tool_call", extra

            if role == "tool" or msg.get("tool_call_id"):
                extra["tool_name"] = msg.get("name", "tool_response")
                extra["tool_input"] = {"source": "tool_response"}
                extra["tool_output"] = str(content)
                return "tool_call", extra

            if role in ("assistant", "ai"):
                extra["llm_output"] = str(content)
                return "llm_response", extra

            extra["llm_input"] = str(content)
            return "observation", extra

        # Object-style messages (AutoGen v0.4+)
        msg_type = type(msg).__name__.lower()
        content = getattr(msg, "content", str(msg))

        if "toolcall" in msg_type:
            extra["tool_name"] = getattr(msg, "name", "unknown")
            extra["tool_input"] = getattr(msg, "arguments", {})
            return "tool_call", extra

        if "assistant" in msg_type or "ai" in msg_type:
            extra["llm_output"] = str(content)
            return "llm_response", extra

        extra["llm_input"] = str(content)
        return "observation", extra

    @staticmethod
    def _get_last_assistant_content(history: list[Any]) -> Any:
        """Find the last assistant message in a chat history."""
        for msg in reversed(history):
            if isinstance(msg, dict):
                if msg.get("role") in ("assistant", "ai"):
                    return msg.get("content")
            elif hasattr(msg, "role"):
                if getattr(msg, "role", "") in ("assistant", "ai"):
                    return getattr(msg, "content", str(msg))
        return str(history[-1]) if history else None

    @staticmethod
    def _build_message(input_data: dict[str, Any]) -> str:
        """Extract or construct the user message from input_data."""
        for key in ("message", "query", "input", "prompt"):
            if key in input_data:
                return str(input_data[key])

        filtered = {
            k: v
            for k, v in input_data.items()
            if k not in ("scenario_id", "metadata")
        }
        if len(filtered) == 1:
            return str(next(iter(filtered.values())))

        import json

        return json.dumps(filtered, default=str)
