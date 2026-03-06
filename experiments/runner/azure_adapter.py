"""Unified Azure AI Foundry client for AgentAssay experiments.

Routes model calls to the correct Azure subscription/endpoint, handles
rate-limit rotation across 3 subscriptions, tracks token usage and cost,
and converts raw API responses into ``ExecutionTrace`` objects.

Architecture:
    - 3 Azure subscriptions, each with its own endpoint and API key.
    - Models are statically mapped to subscriptions for load balancing.
    - On rate-limit (HTTP 429), the client rotates to the next subscription
      that hosts the same model, or waits if all are throttled.
    - Uses the ``openai`` Python SDK with ``AzureOpenAI`` / ``AsyncAzureOpenAI``
      for compatibility with Azure AI Foundry's OpenAI-compatible API surface.

Env vars loaded from .env via python-dotenv:
    AZURE_SUB1_ENDPOINT, AZURE_SUB1_API_KEY, AZURE_SUB1_REGION
    AZURE_SUB2_ENDPOINT, AZURE_SUB2_API_KEY, AZURE_SUB2_REGION
    AZURE_SUB3_ENDPOINT, AZURE_SUB3_API_KEY, AZURE_SUB3_REGION
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Subscription and model mapping
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SubscriptionConfig:
    """Immutable descriptor for one Azure subscription."""

    name: str
    endpoint: str
    api_key: str
    region: str
    api_version: str = "2025-01-01-preview"


# Static model -> subscription mapping (load-balanced by design).
# Each model is homed to a primary subscription; some models appear on
# multiple subs for failover.
MODEL_SUBSCRIPTION_MAP: dict[str, list[str]] = {
    # Sub 1 (GEN_AI_HM, eastus2)
    "gpt-5.2-chat": ["sub1"],
    "Mistral-Large-3": ["sub1"],
    # Sub 3 (GTIC7_DIVAP, swedencentral)
    "claude-sonnet-4-6": ["sub3"],
    "Llama-4-Maverick-17B-128E-Instruct-FP8": ["sub3"],
    "Phi-4": ["sub3"],
}


@dataclass
class _RateLimitState:
    """Tracks per-subscription rate-limit cooldown."""

    blocked_until: float = 0.0  # monotonic timestamp
    consecutive_429s: int = 0

    @property
    def is_blocked(self) -> bool:
        return time.monotonic() < self.blocked_until

    def record_429(self, retry_after: float = 10.0) -> None:
        self.consecutive_429s += 1
        backoff = min(retry_after * (2 ** (self.consecutive_429s - 1)), 120.0)
        self.blocked_until = time.monotonic() + backoff
        logger.warning(
            "Rate-limited — backing off %.1fs (consecutive: %d)",
            backoff,
            self.consecutive_429s,
        )

    def record_success(self) -> None:
        self.consecutive_429s = 0


# ---------------------------------------------------------------------------
# Token / cost accounting
# ---------------------------------------------------------------------------

@dataclass
class UsageRecord:
    """Token usage from a single API call."""

    model: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


# ---------------------------------------------------------------------------
# Main client
# ---------------------------------------------------------------------------

class AzureFoundryClient:
    """Unified async client for Azure AI Foundry across 3 subscriptions.

    Typical usage::

        client = AzureFoundryClient(env_path=".env")
        result = await client.chat_completion(
            model="gpt-5.2-chat",
            messages=[{"role": "user", "content": "Hello"}],
        )

    Parameters
    ----------
    env_path
        Path to the ``.env`` file containing subscription credentials.
        Resolved relative to the project root.
    cost_tracker
        Optional external ``CostTracker`` instance. If provided, every
        API call's usage is forwarded to it automatically.
    """

    def __init__(
        self,
        env_path: str = ".env",
        cost_tracker: Any | None = None,
    ) -> None:
        self._env_path = Path(env_path)
        self._cost_tracker = cost_tracker

        # Load environment
        load_dotenv(self._env_path, override=False)

        # Build subscription configs
        self._subscriptions: dict[str, SubscriptionConfig] = {}
        self._async_clients: dict[str, Any] = {}
        self._rate_limits: dict[str, _RateLimitState] = {}
        self._usage_log: list[UsageRecord] = []

        self._load_subscriptions()
        self._init_clients()

    # -- Subscription loading ------------------------------------------------

    def _load_subscriptions(self) -> None:
        """Read subscription configs from environment variables."""
        for idx in range(1, 4):
            name = f"sub{idx}"
            endpoint = os.getenv(f"AZURE_SUB{idx}_ENDPOINT", "")
            api_key = os.getenv(f"AZURE_SUB{idx}_API_KEY", "")
            region = os.getenv(f"AZURE_SUB{idx}_REGION", "")
            api_version = os.getenv(
                f"AZURE_SUB{idx}_API_VERSION", "2025-01-01-preview"
            )

            if not endpoint or not api_key:
                logger.warning(
                    "Subscription %s missing endpoint or key — skipping.", name
                )
                continue

            self._subscriptions[name] = SubscriptionConfig(
                name=name,
                endpoint=endpoint,
                api_key=api_key,
                region=region,
                api_version=api_version,
            )
            self._rate_limits[name] = _RateLimitState()

        if not self._subscriptions:
            raise RuntimeError(
                "No valid Azure subscriptions found in environment. "
                "Check .env for AZURE_SUB{1,2,3}_{ENDPOINT,API_KEY}."
            )

        logger.info(
            "Loaded %d Azure subscriptions: %s",
            len(self._subscriptions),
            ", ".join(self._subscriptions.keys()),
        )

    def _init_clients(self) -> None:
        """Create one ``AsyncAzureOpenAI`` client per subscription (lazy)."""
        try:
            from openai import AsyncAzureOpenAI
        except ImportError as exc:
            raise ImportError(
                "The 'openai' package is required for the Azure adapter. "
                "Install it: pip install 'openai>=1.50'"
            ) from exc

        for name, sub in self._subscriptions.items():
            self._async_clients[name] = AsyncAzureOpenAI(
                azure_endpoint=sub.endpoint,
                api_key=sub.api_key,
                api_version=sub.api_version,
            )

    # -- Model routing -------------------------------------------------------

    def _resolve_subscriptions(self, model: str) -> list[str]:
        """Return the list of subscription names that host a model."""
        subs = MODEL_SUBSCRIPTION_MAP.get(model)
        if not subs:
            raise ValueError(
                f"Unknown model '{model}'. Known models: "
                f"{sorted(MODEL_SUBSCRIPTION_MAP.keys())}"
            )
        # Filter to actually loaded subs
        available = [s for s in subs if s in self._subscriptions]
        if not available:
            raise RuntimeError(
                f"Model '{model}' is mapped to {subs} but none "
                f"are loaded. Check .env credentials."
            )
        return available

    def _pick_subscription(self, model: str) -> tuple[str, Any]:
        """Pick the best available subscription for a model.

        Prefers subscriptions that are not rate-limited. If all are
        blocked, picks the one whose cooldown expires soonest.

        Returns
        -------
        tuple[str, AsyncAzureOpenAI]
            Subscription name and the async client.
        """
        sub_names = self._resolve_subscriptions(model)

        # Prefer unblocked
        for name in sub_names:
            if not self._rate_limits[name].is_blocked:
                return name, self._async_clients[name]

        # All blocked — pick the one that unblocks soonest
        earliest = min(
            sub_names, key=lambda n: self._rate_limits[n].blocked_until
        )
        wait_time = max(
            0.0, self._rate_limits[earliest].blocked_until - time.monotonic()
        )
        logger.info(
            "All subscriptions for '%s' rate-limited. Waiting %.1fs for %s.",
            model,
            wait_time,
            earliest,
        )
        return earliest, self._async_clients[earliest]

    # -- Cost estimation -----------------------------------------------------

    # Approximate pricing per 1M tokens: (input, output)
    PRICING: dict[str, tuple[float, float]] = {
        "gpt-5.2-chat": (2.50, 10.00),
        "claude-sonnet-4-6": (3.00, 15.00),
        "Mistral-Large-3": (2.00, 6.00),
        "Llama-4-Maverick-17B-128E-Instruct-FP8": (0.00, 0.00),
        "Phi-4": (0.00, 0.00),
    }

    def _estimate_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Estimate cost in USD for a single API call."""
        input_rate, output_rate = self.PRICING.get(model, (0.0, 0.0))
        cost = (prompt_tokens / 1_000_000) * input_rate + (
            completion_tokens / 1_000_000
        ) * output_rate
        return round(cost, 6)

    # -- Chat completion (core) ----------------------------------------------

    async def chat_completion(
        self,
        model: str,
        messages: list[dict[str, Any]],
        tools: list[dict[str, Any]] | None = None,
        temperature: float = 0.7,
        max_completion_tokens: int = 4096,
        max_retries: int = 3,
    ) -> dict[str, Any]:
        """Send a chat completion request, routing to the correct subscription.

        Handles rate-limit rotation and retries transparently.

        Parameters
        ----------
        model
            Model deployment name (must be in ``MODEL_SUBSCRIPTION_MAP``).
        messages
            OpenAI-format message list.
        tools
            Optional tool definitions for function calling.
        temperature
            Sampling temperature.
        max_completion_tokens
            Maximum completion tokens.
        max_retries
            Maximum retry attempts on transient errors.

        Returns
        -------
        dict
            Keys: ``content``, ``tool_calls``, ``usage``, ``model``,
            ``finish_reason``, ``estimated_cost_usd``.

        Raises
        ------
        RuntimeError
            If all retries are exhausted.
        """
        last_error: Exception | None = None

        for attempt in range(1, max_retries + 1):
            sub_name, client = self._pick_subscription(model)

            # If the chosen sub is blocked, sleep until it opens
            rl = self._rate_limits[sub_name]
            if rl.is_blocked:
                wait = max(0.0, rl.blocked_until - time.monotonic())
                logger.info(
                    "Waiting %.1fs for rate-limit cooldown on %s",
                    wait,
                    sub_name,
                )
                await asyncio.sleep(wait)

            try:
                # Some models (reasoning: GPT-5.2, DeepSeek-R1) only support
                # temperature=1 and don't accept max_completion_tokens.
                # Build kwargs defensively per model capabilities.
                _NO_MAX_TOKENS_MODELS = {
                    "gpt-5.2-chat", "DeepSeek-R1-0528", "Kimi-K2-Thinking",
                    "o1", "o1-mini", "o1-preview", "o3", "o3-mini",
                    "Mistral-Large-3", "Phi-4",
                }
                is_reasoning = any(r in model for r in _NO_MAX_TOKENS_MODELS)

                kwargs: dict[str, Any] = {
                    "model": model,
                    "messages": messages,
                }
                if not is_reasoning:
                    kwargs["temperature"] = temperature
                    kwargs["max_completion_tokens"] = max_completion_tokens
                if tools:
                    kwargs["tools"] = tools
                    kwargs["tool_choice"] = "auto"

                response = await client.chat.completions.create(**kwargs)

                # Parse response
                choice = response.choices[0]
                usage = response.usage

                prompt_tokens = (getattr(usage, 'prompt_tokens', 0) or 0) if usage else 0
                completion_tokens = (getattr(usage, 'completion_tokens', 0) or 0) if usage else 0
                total_tokens = (getattr(usage, 'total_tokens', 0) or 0) if usage else 0

                if not usage or total_tokens == 0:
                    logger.warning(
                        "Zero or missing token usage for model=%s. usage=%s",
                        model, usage,
                    )

                estimated_cost = self._estimate_cost(
                    model, prompt_tokens, completion_tokens
                )

                # Record usage
                record = UsageRecord(
                    model=model,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    total_tokens=total_tokens,
                    estimated_cost_usd=estimated_cost,
                )
                self._usage_log.append(record)

                # Forward to cost tracker
                if self._cost_tracker is not None:
                    self._cost_tracker.record(
                        model=model,
                        prompt_tokens=prompt_tokens,
                        completion_tokens=completion_tokens,
                        cost_usd=estimated_cost,
                    )

                self._rate_limits[sub_name].record_success()

                # Extract tool calls if present
                tool_calls_raw: list[dict[str, Any]] = []
                if choice.message.tool_calls:
                    for tc in choice.message.tool_calls:
                        tool_calls_raw.append({
                            "id": tc.id,
                            "type": tc.type,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        })

                return {
                    "content": choice.message.content or "",
                    "tool_calls": tool_calls_raw,
                    "usage": {
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                    },
                    "model": model,
                    "finish_reason": choice.finish_reason,
                    "estimated_cost_usd": estimated_cost,
                    "subscription": sub_name,
                }

            except Exception as exc:
                last_error = exc
                error_str = str(exc).lower()

                # Detect rate limiting (429)
                is_rate_limit = (
                    "429" in error_str
                    or "rate" in error_str
                    or "throttl" in error_str
                    or "too many requests" in error_str
                )

                if is_rate_limit:
                    # Parse Retry-After if available
                    retry_after = 10.0
                    if hasattr(exc, "response") and hasattr(exc.response, "headers"):
                        ra = exc.response.headers.get("Retry-After")
                        if ra:
                            try:
                                retry_after = float(ra)
                            except (ValueError, TypeError):
                                pass

                    self._rate_limits[sub_name].record_429(retry_after)
                    logger.warning(
                        "Rate limit on %s (attempt %d/%d): %s",
                        sub_name,
                        attempt,
                        max_retries,
                        exc,
                    )
                    continue  # Retry with potentially different sub

                # Transient server error — retry with backoff
                is_server_error = any(
                    code in error_str for code in ("500", "502", "503", "504")
                )
                if is_server_error and attempt < max_retries:
                    wait = 2.0 ** attempt
                    logger.warning(
                        "Server error on %s (attempt %d/%d), retrying in %.1fs: %s",
                        sub_name,
                        attempt,
                        max_retries,
                        wait,
                        exc,
                    )
                    await asyncio.sleep(wait)
                    continue

                # Non-retryable error
                logger.error(
                    "Non-retryable error on %s (attempt %d/%d): %s",
                    sub_name,
                    attempt,
                    max_retries,
                    exc,
                )
                raise

        raise RuntimeError(
            f"All {max_retries} retries exhausted for model '{model}'. "
            f"Last error: {last_error}"
        )

    # -- Multi-turn agent loop -----------------------------------------------

    async def run_agent(
        self,
        model: str,
        system_prompt: str,
        user_input: str,
        tools: list[dict[str, Any]],
        max_steps: int = 10,
        temperature: float = 0.7,
        tool_executor: Any | None = None,
    ) -> dict[str, Any]:
        """Execute a multi-turn agent loop with tool calling.

        The loop continues until the model stops calling tools or
        ``max_steps`` is reached. Each tool call is resolved by
        ``tool_executor`` (a callable that takes tool name + args and
        returns a string result).

        Parameters
        ----------
        model
            Model deployment name.
        system_prompt
            The system-level instruction for the agent.
        user_input
            The user's task/query.
        tools
            OpenAI-format tool definitions.
        max_steps
            Maximum agent steps (each tool call round is one step).
        temperature
            Sampling temperature.
        tool_executor
            Callable: ``(tool_name: str, arguments: dict) -> str``.
            If None, tool calls receive a stub response.

        Returns
        -------
        dict
            Keys: ``steps``, ``final_output``, ``success``, ``error``,
            ``total_duration_ms``, ``total_cost_usd``, ``total_tokens``,
            ``model``, ``step_count``.
        """
        messages: list[dict[str, Any]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_input},
        ]

        steps: list[dict[str, Any]] = []
        total_cost = 0.0
        total_tokens = 0
        overall_start = time.monotonic()

        try:
            for step_idx in range(max_steps):
                step_start = time.monotonic()

                response = await self.chat_completion(
                    model=model,
                    messages=messages,
                    tools=tools if tools else None,
                    temperature=temperature,
                )

                step_duration_ms = (time.monotonic() - step_start) * 1000.0
                total_cost += response.get("estimated_cost_usd", 0.0)
                total_tokens += response.get("usage", {}).get("total_tokens", 0)

                tool_calls = response.get("tool_calls", [])

                if not tool_calls:
                    # Model finished — no more tool calls
                    steps.append({
                        "step_index": step_idx,
                        "action": "llm_response",
                        "llm_output": response.get("content", ""),
                        "model": model,
                        "duration_ms": step_duration_ms,
                        "usage": response.get("usage", {}),
                    })

                    total_duration_ms = (
                        time.monotonic() - overall_start
                    ) * 1000.0

                    return {
                        "steps": steps,
                        "final_output": response.get("content", ""),
                        "success": True,
                        "error": None,
                        "total_duration_ms": total_duration_ms,
                        "total_cost_usd": total_cost,
                        "total_tokens": total_tokens,
                        "model": model,
                        "step_count": len(steps),
                    }

                # Process tool calls
                # Add assistant message with tool calls
                assistant_msg: dict[str, Any] = {
                    "role": "assistant",
                    "content": response.get("content") or None,
                    "tool_calls": tool_calls,
                }
                messages.append(assistant_msg)

                for tc in tool_calls:
                    func_name = tc["function"]["name"]
                    try:
                        func_args = json.loads(tc["function"]["arguments"])
                    except (json.JSONDecodeError, TypeError):
                        func_args = {}

                    # Execute the tool
                    tool_start = time.monotonic()
                    if tool_executor is not None:
                        try:
                            tool_result = str(tool_executor(func_name, func_args))
                        except Exception as tex:
                            tool_result = f"Error executing {func_name}: {tex}"
                    else:
                        tool_result = json.dumps({
                            "status": "ok",
                            "result": f"Stub result for {func_name}",
                        })
                    tool_duration_ms = (time.monotonic() - tool_start) * 1000.0

                    # Record the step
                    steps.append({
                        "step_index": step_idx,
                        "action": "tool_call",
                        "tool_name": func_name,
                        "tool_input": func_args,
                        "tool_output": tool_result,
                        "model": model,
                        "duration_ms": step_duration_ms + tool_duration_ms,
                    })

                    # Add tool result to messages
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tc["id"],
                        "content": tool_result,
                    })

            # Exhausted max_steps
            total_duration_ms = (time.monotonic() - overall_start) * 1000.0
            return {
                "steps": steps,
                "final_output": steps[-1].get("llm_output", "") if steps else "",
                "success": True,
                "error": None,
                "total_duration_ms": total_duration_ms,
                "total_cost_usd": total_cost,
                "total_tokens": total_tokens,
                "model": model,
                "step_count": len(steps),
            }

        except Exception as exc:
            total_duration_ms = (time.monotonic() - overall_start) * 1000.0
            logger.error("Agent loop failed: %s", exc, exc_info=True)
            return {
                "steps": steps,
                "final_output": None,
                "success": False,
                "error": f"{type(exc).__name__}: {exc}",
                "total_duration_ms": total_duration_ms,
                "total_cost_usd": total_cost,
                "total_tokens": total_tokens,
                "model": model,
                "step_count": len(steps),
            }

    # -- Conversion to ExecutionTrace ----------------------------------------

    def to_execution_trace(
        self,
        agent_result: dict[str, Any],
        scenario_id: str,
        framework: str = "azure_foundry",
    ) -> "ExecutionTrace":
        """Convert a ``run_agent`` result dict into an ``ExecutionTrace``.

        Lazy-imports the model class to avoid circular dependencies when
        this module is used standalone.

        Parameters
        ----------
        agent_result
            The dict returned by ``run_agent``.
        scenario_id
            The scenario ID for this trace.
        framework
            Framework label. Defaults to ``"azure_foundry"``.

        Returns
        -------
        ExecutionTrace
        """
        from agentassay.core.models import ExecutionTrace, StepTrace

        step_traces: list[StepTrace] = []
        for raw_step in agent_result.get("steps", []):
            step_traces.append(
                StepTrace(
                    step_index=raw_step.get("step_index", 0),
                    action=raw_step.get("action", "unknown"),
                    tool_name=raw_step.get("tool_name"),
                    tool_input=raw_step.get("tool_input"),
                    tool_output=raw_step.get("tool_output"),
                    llm_input=None,
                    llm_output=raw_step.get("llm_output"),
                    model=raw_step.get("model"),
                    duration_ms=raw_step.get("duration_ms", 0.0),
                    metadata=raw_step.get("usage", {}),
                )
            )

        return ExecutionTrace(
            trace_id=str(uuid.uuid4()),
            scenario_id=scenario_id,
            steps=step_traces,
            input_data={},
            output_data=agent_result.get("final_output"),
            success=agent_result.get("success", False),
            error=agent_result.get("error"),
            total_duration_ms=agent_result.get("total_duration_ms", 0.0),
            total_cost_usd=agent_result.get("total_cost_usd", 0.0),
            model=agent_result.get("model", "unknown"),
            framework=framework,
        )

    # -- Utility -------------------------------------------------------------

    @property
    def usage_log(self) -> list[UsageRecord]:
        """All recorded usage entries."""
        return list(self._usage_log)

    @property
    def total_cost_usd(self) -> float:
        """Running total estimated cost across all calls."""
        return sum(r.estimated_cost_usd for r in self._usage_log)

    @property
    def total_tokens(self) -> int:
        """Running total token count across all calls."""
        return sum(r.total_tokens for r in self._usage_log)

    @property
    def subscriptions(self) -> dict[str, SubscriptionConfig]:
        """Loaded subscription configs (read-only copy)."""
        return dict(self._subscriptions)

    def available_models(self) -> list[str]:
        """Return models that have at least one loaded subscription."""
        result: list[str] = []
        for model, subs in MODEL_SUBSCRIPTION_MAP.items():
            if any(s in self._subscriptions for s in subs):
                result.append(model)
        return sorted(result)

    async def close(self) -> None:
        """Close all underlying HTTP clients."""
        for name, client in self._async_clients.items():
            try:
                await client.close()
            except Exception:
                logger.debug("Error closing client %s", name, exc_info=True)
        self._async_clients.clear()

    def __repr__(self) -> str:
        return (
            f"AzureFoundryClient("
            f"subscriptions={list(self._subscriptions.keys())}, "
            f"models={self.available_models()}, "
            f"total_cost=${self.total_cost_usd:.4f})"
        )
