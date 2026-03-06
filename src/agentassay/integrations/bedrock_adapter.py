"""AWS Bedrock Agents adapter for AgentAssay.

Wraps an AWS Bedrock Agent via the ``bedrock-agent-runtime`` service and
captures each orchestration step, action group invocation, and knowledge
base lookup as a ``StepTrace``. Uses ``invoke_agent()`` with
``enableTrace=True`` to obtain fine-grained orchestration traces including
rationale, action, and observation sub-traces.

All ``boto3`` imports are **lazy** -- this module can be imported even when
boto3 is not installed.  The ``ImportError`` is raised only when ``run()``
or ``to_callable()`` is actually called.

Usage
-----
>>> from agentassay.integrations.bedrock_adapter import BedrockAgentsAdapter
>>> adapter = BedrockAgentsAdapter(
...     agent_id="ABCDEFGHIJ",
...     agent_alias_id="TSTALIASID",
...     model="anthropic.claude-3-sonnet",
... )
>>> trace = adapter.run({"query": "What are our Q4 results?"})
>>> runner = TrialRunner(adapter.to_callable(), assay_config, adapter.get_config())
"""

from __future__ import annotations

import json
import logging
import time
import traceback
import uuid
from collections.abc import Callable
from typing import Any

from agentassay.core.models import AgentConfig, ExecutionTrace, StepTrace
from agentassay.integrations.base import (
    AgentAdapter,
    FrameworkNotInstalledError,
)

logger = logging.getLogger(__name__)

_INSTALL_HINT = (
    "Bedrock Agents adapter requires boto3. "
    "Install with: pip install agentassay[bedrock]"
)


def _check_boto3_installed() -> None:
    """Verify that boto3 is available, raise clear error if not.

    Raises
    ------
    FrameworkNotInstalledError
        If ``boto3`` is not installed.
    """
    try:
        import boto3  # noqa: F401
    except ImportError as exc:
        raise FrameworkNotInstalledError(_INSTALL_HINT) from exc


class BedrockAgentsAdapter(AgentAdapter):
    """Adapter for AWS Bedrock Agents via the bedrock-agent-runtime API.

    Invokes a Bedrock Agent using ``invoke_agent()`` with orchestration
    tracing enabled.  Each trace event (action group invocation, knowledge
    base lookup, rationale/observation) becomes a ``StepTrace``.

    Parameters
    ----------
    agent_id
        The Bedrock Agent identifier (10-character alphanumeric string).
    agent_alias_id
        The agent alias identifier (``TSTALIASID`` for the draft version
        or a deployed alias ID).
    session_id
        Optional session identifier for multi-turn conversations.  If
        ``None``, a new UUID is generated per ``run()`` call.
    region
        AWS region name (e.g. ``"us-east-1"``).  If ``None``, boto3
        uses its default region resolution chain.
    model
        LLM model identifier used by the agent (e.g.
        ``"anthropic.claude-3-sonnet"``).  Informational only.
    agent_name
        Human-readable name for this agent.  Defaults to
        ``"bedrock-agent"``.
    metadata
        Arbitrary metadata attached to every ``ExecutionTrace`` produced.
    """

    framework: str = "bedrock"

    def __init__(
        self,
        agent_id: str,
        agent_alias_id: str,
        *,
        session_id: str | None = None,
        region: str | None = None,
        model: str = "unknown",
        agent_name: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(model=model, agent_name=agent_name, metadata=metadata)
        self._agent_id = agent_id
        self._agent_alias_id = agent_alias_id
        self._session_id = session_id
        self._region = region
        self._client: Any = None  # Lazy-created boto3 client

    # -- Core interface -------------------------------------------------------

    def run(self, input_data: dict[str, Any]) -> ExecutionTrace:
        """Invoke the Bedrock Agent and capture an ExecutionTrace.

        Calls ``invoke_agent()`` with ``enableTrace=True``, then parses
        the EventStream response to extract orchestration steps, action
        group invocations, and knowledge base lookups.

        Parameters
        ----------
        input_data
            The scenario input dictionary.  Must contain a ``"query"`` or
            ``"input"`` key with the user message, or the entire dict is
            serialized as the prompt.

        Returns
        -------
        ExecutionTrace
            A complete trace with per-step timing and output.  On failure,
            ``success=False`` with the error message.
        """
        _check_boto3_installed()

        scenario_id = input_data.get("scenario_id", "default")
        trace_id = str(uuid.uuid4())
        session_id = self._session_id or str(uuid.uuid4())
        overall_start = time.perf_counter()

        try:
            client = self._get_client()
            user_input = self._build_user_input(input_data)

            response = client.invoke_agent(
                agentId=self._agent_id,
                agentAliasId=self._agent_alias_id,
                sessionId=session_id,
                inputText=user_input,
                enableTrace=True,
            )

            steps, output_text = self._parse_event_stream(response)
            total_ms = (time.perf_counter() - overall_start) * 1000.0

            return ExecutionTrace(
                trace_id=trace_id,
                scenario_id=scenario_id,
                steps=steps,
                input_data=input_data,
                output_data=output_text,
                success=True,
                error=None,
                total_duration_ms=total_ms,
                total_cost_usd=0.0,
                model=self._model,
                framework=self.framework,
                metadata={
                    **self._metadata,
                    "bedrock_agent_id": self._agent_id,
                    "bedrock_alias_id": self._agent_alias_id,
                    "bedrock_session_id": session_id,
                },
            )

        except FrameworkNotInstalledError:
            raise

        except Exception as exc:
            total_ms = (time.perf_counter() - overall_start) * 1000.0
            error_msg = f"{type(exc).__name__}: {exc}"
            logger.error(
                "Bedrock Agents adapter failed: %s\n%s",
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
        _check_boto3_installed()
        return self.run

    def get_config(self) -> AgentConfig:
        """Build an ``AgentConfig`` describing this adapter's agent.

        Uses ``"custom"`` as the ``AgentConfig.framework`` value because
        ``"bedrock"`` is not yet in the ``AgentConfig`` framework literal.
        The actual framework identifier is stored in ``metadata["framework"]``.

        Returns
        -------
        AgentConfig
            Configuration with framework, model, and metadata.
        """
        import uuid as _uuid

        return AgentConfig(
            agent_id=str(_uuid.uuid4()),
            name=self._agent_name,
            framework="bedrock",
            model=self._model,
            metadata={
                **self._metadata,
                "bedrock_agent_id": self._agent_id,
                "bedrock_alias_id": self._agent_alias_id,
            },
        )

    # -- Internal: boto3 client -----------------------------------------------

    def _get_client(self) -> Any:
        """Lazily create the ``bedrock-agent-runtime`` boto3 client.

        Returns
        -------
        Any
            A boto3 ``bedrock-agent-runtime`` service client.
        """
        if self._client is None:
            import boto3

            kwargs: dict[str, Any] = {}
            if self._region is not None:
                kwargs["region_name"] = self._region
            self._client = boto3.client("bedrock-agent-runtime", **kwargs)
        return self._client

    # -- Internal: user input -------------------------------------------------

    @staticmethod
    def _build_user_input(input_data: dict[str, Any]) -> str:
        """Extract or construct the user prompt from input_data.

        Checks keys in priority order: ``query``, ``input``, ``prompt``,
        ``message``.  Falls back to serializing the entire dict.

        Parameters
        ----------
        input_data
            The raw scenario input dictionary.

        Returns
        -------
        str
            The user prompt string to send to the agent.
        """
        for key in ("query", "input", "prompt", "message"):
            if key in input_data:
                return str(input_data[key])

        filtered = {
            k: v
            for k, v in input_data.items()
            if k not in ("scenario_id", "metadata")
        }
        if len(filtered) == 1:
            return str(next(iter(filtered.values())))

        return json.dumps(filtered, default=str)

    # -- Internal: EventStream parsing ----------------------------------------

    def _parse_event_stream(
        self, response: dict[str, Any]
    ) -> tuple[list[StepTrace], str]:
        """Parse the Bedrock ``invoke_agent`` EventStream response.

        The ``completion`` field is an EventStream yielding events of type:

        - ``chunk``: Final text output bytes.
        - ``trace``: Orchestration trace with ``orchestrationTrace``
          sub-fields (``rationale``, ``invocationInput``,
          ``invocationOutput``, ``observation``).
        - ``returnControl``: Return-of-control event for action groups.
        - ``files``: File output events.

        Parameters
        ----------
        response
            The raw ``invoke_agent`` response dict from boto3.

        Returns
        -------
        tuple[list[StepTrace], str]
            (ordered list of steps, final output text).
        """
        steps: list[StepTrace] = []
        output_chunks: list[str] = []
        step_index = 0

        completion = response.get("completion", [])
        for event in completion:
            event_start = time.perf_counter()

            # -- Text chunk: final agent output bytes -----------------------
            if "chunk" in event:
                chunk_data = event["chunk"]
                text = chunk_data.get("bytes", b"").decode("utf-8", errors="replace")
                if text:
                    output_chunks.append(text)

            # -- Trace event: orchestration details -------------------------
            elif "trace" in event:
                trace_data = event["trace"].get("trace", {})
                orch_trace = trace_data.get("orchestrationTrace", {})

                new_steps = self._parse_orchestration_trace(
                    orch_trace, step_index, event_start
                )
                steps.extend(new_steps)
                step_index += len(new_steps)

            # -- Return of control: action group invocation -----------------
            elif "returnControl" in event:
                roc = event["returnControl"]
                duration_ms = (time.perf_counter() - event_start) * 1000.0

                invocation_inputs = roc.get("invocationInputs", [])
                for inv_input in invocation_inputs:
                    api_input = inv_input.get(
                        "apiInvocationInput",
                        inv_input.get("functionInvocationInput", {}),
                    )
                    action_group = api_input.get("actionGroupName", "unknown")
                    api_path = api_input.get("apiPath", api_input.get("function", ""))
                    parameters = api_input.get("parameters", api_input.get("actionGroupFunction", {}))

                    steps.append(
                        StepTrace(
                            step_index=step_index,
                            action="tool_call",
                            tool_name=f"{action_group}:{api_path}" if api_path else action_group,
                            tool_input=parameters if isinstance(parameters, dict) else {"params": parameters},
                            tool_output=None,
                            duration_ms=duration_ms,
                            model=self._model,
                            metadata={"bedrock_event": "return_control"},
                        )
                    )
                    step_index += 1

        output_text = "".join(output_chunks)
        return steps, output_text

    def _parse_orchestration_trace(
        self,
        orch_trace: dict[str, Any],
        start_index: int,
        event_start: float,
    ) -> list[StepTrace]:
        """Parse an orchestration trace block into StepTrace objects.

        An orchestration trace may contain:

        - ``rationale``: The agent's reasoning text.
        - ``invocationInput``: Action group or knowledge base invocation.
        - ``invocationOutput``: Result from an action group invocation.
        - ``observation``: Final observation after tool/KB execution.
        - ``modelInvocationInput``: The prompt sent to the foundation model.

        Parameters
        ----------
        orch_trace
            The ``orchestrationTrace`` dict from the Bedrock trace event.
        start_index
            The step index to start numbering from.
        event_start
            The ``time.perf_counter()`` timestamp for the event.

        Returns
        -------
        list[StepTrace]
            Zero or more steps extracted from this trace block.
        """
        steps: list[StepTrace] = []
        current_index = start_index
        duration_ms = (time.perf_counter() - event_start) * 1000.0

        # -- Rationale (agent reasoning) ------------------------------------
        rationale = orch_trace.get("rationale")
        if rationale:
            text = rationale.get("text", "")
            steps.append(
                StepTrace(
                    step_index=current_index,
                    action="llm_response",
                    llm_output=text,
                    duration_ms=duration_ms,
                    model=self._model,
                    metadata={"bedrock_event": "rationale"},
                )
            )
            current_index += 1

        # -- Invocation input (action group or KB call) ---------------------
        inv_input = orch_trace.get("invocationInput")
        if inv_input:
            step = self._parse_invocation_input(inv_input, current_index, duration_ms)
            if step is not None:
                steps.append(step)
                current_index += 1

        # -- Invocation output (action group result) ------------------------
        inv_output = orch_trace.get("invocationOutput")
        if inv_output:
            step = self._parse_invocation_output(inv_output, current_index, duration_ms)
            if step is not None:
                steps.append(step)
                current_index += 1

        # -- Observation (final observation text) ---------------------------
        observation = orch_trace.get("observation")
        if observation:
            obs_type = observation.get("type", "unknown")
            obs_text = observation.get("finalResponse", {}).get(
                "text",
                observation.get("repromptResponse", {}).get(
                    "text",
                    observation.get("knowledgeBaseLookupOutput", {}).get(
                        "text", str(observation)
                    ),
                ),
            )
            steps.append(
                StepTrace(
                    step_index=current_index,
                    action="observation",
                    llm_output=str(obs_text),
                    duration_ms=duration_ms,
                    model=self._model,
                    metadata={
                        "bedrock_event": "observation",
                        "observation_type": obs_type,
                    },
                )
            )
            current_index += 1

        # -- Model invocation input (prompt to FM) --------------------------
        model_input = orch_trace.get("modelInvocationInput")
        if model_input:
            prompt_text = model_input.get("text", "")
            steps.append(
                StepTrace(
                    step_index=current_index,
                    action="observation",
                    llm_input=prompt_text if isinstance(prompt_text, str) else str(prompt_text),
                    duration_ms=duration_ms,
                    model=self._model,
                    metadata={"bedrock_event": "model_invocation_input"},
                )
            )
            current_index += 1

        return steps

    def _parse_invocation_input(
        self,
        inv_input: dict[str, Any],
        step_index: int,
        duration_ms: float,
    ) -> StepTrace | None:
        """Parse an invocation input into a StepTrace.

        Handles both action group invocations and knowledge base lookups.

        Parameters
        ----------
        inv_input
            The ``invocationInput`` dict from the orchestration trace.
        step_index
            The step index for this trace entry.
        duration_ms
            Duration in milliseconds.

        Returns
        -------
        StepTrace | None
            A step trace, or ``None`` if the input is empty/unrecognized.
        """
        inv_type = inv_input.get("invocationType", "")

        # Action group invocation
        ag_input = inv_input.get("actionGroupInvocationInput")
        if ag_input:
            action_group = ag_input.get("actionGroupName", "unknown")
            api_path = ag_input.get("apiPath", ag_input.get("function", ""))
            parameters = ag_input.get("requestBody", ag_input.get("parameters", {}))
            return StepTrace(
                step_index=step_index,
                action="tool_call",
                tool_name=f"{action_group}:{api_path}" if api_path else action_group,
                tool_input=parameters if isinstance(parameters, dict) else {"raw": parameters},
                duration_ms=duration_ms,
                model=self._model,
                metadata={
                    "bedrock_event": "invocation_input",
                    "invocation_type": inv_type or "ACTION_GROUP",
                },
            )

        # Knowledge base lookup
        kb_input = inv_input.get("knowledgeBaseLookupInput")
        if kb_input:
            kb_id = kb_input.get("knowledgeBaseId", "unknown")
            query_text = kb_input.get("text", "")
            return StepTrace(
                step_index=step_index,
                action="retrieval",
                tool_name=f"knowledge_base:{kb_id}",
                tool_input={"query": query_text, "knowledge_base_id": kb_id},
                duration_ms=duration_ms,
                model=self._model,
                metadata={
                    "bedrock_event": "invocation_input",
                    "invocation_type": inv_type or "KNOWLEDGE_BASE",
                },
            )

        return None

    def _parse_invocation_output(
        self,
        inv_output: dict[str, Any],
        step_index: int,
        duration_ms: float,
    ) -> StepTrace | None:
        """Parse an invocation output into a StepTrace.

        Parameters
        ----------
        inv_output
            The ``invocationOutput`` dict from the orchestration trace.
        step_index
            The step index for this trace entry.
        duration_ms
            Duration in milliseconds.

        Returns
        -------
        StepTrace | None
            A step trace, or ``None`` if the output is empty/unrecognized.
        """
        # Action group output
        ag_output = inv_output.get("actionGroupInvocationOutput")
        if ag_output:
            text = ag_output.get("text", str(ag_output))
            return StepTrace(
                step_index=step_index,
                action="tool_call",
                tool_name="action_group_response",
                tool_input={"source": "action_group"},
                tool_output=text,
                duration_ms=duration_ms,
                model=self._model,
                metadata={"bedrock_event": "invocation_output"},
            )

        # Knowledge base output
        kb_output = inv_output.get("knowledgeBaseLookupOutput")
        if kb_output:
            references = kb_output.get("retrievedReferences", [])
            return StepTrace(
                step_index=step_index,
                action="retrieval",
                tool_name="knowledge_base_response",
                tool_input={"source": "knowledge_base"},
                tool_output={
                    "num_references": len(references),
                    "references": [
                        {
                            "content": ref.get("content", {}).get("text", ""),
                            "location": ref.get("location", {}),
                        }
                        for ref in references[:5]
                    ],
                },
                duration_ms=duration_ms,
                model=self._model,
                metadata={"bedrock_event": "invocation_output"},
            )

        return None
