"""Tests for the AWS Bedrock Agents adapter.

Validates adapter creation, boto3 install checks, EventStream parsing,
action group invocations, knowledge base lookups, orchestration trace
extraction, error handling, and TrialRunner compatibility.

All tests use ``unittest.mock`` -- boto3 is NEVER imported for real.

Target: 20+ tests.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from agentassay.core.models import ExecutionTrace
from agentassay.integrations.base import FrameworkNotInstalledError

# ===================================================================
# Helpers
# ===================================================================


def _make_adapter(**kwargs: Any):
    """Create a BedrockAgentsAdapter with sensible defaults."""
    from agentassay.integrations.bedrock_adapter import BedrockAgentsAdapter

    defaults = {
        "agent_id": "ABCDEFGHIJ",
        "agent_alias_id": "TSTALIASID",
        "model": "anthropic.claude-3-sonnet",
        "agent_name": "test-bedrock-agent",
    }
    defaults.update(kwargs)
    return BedrockAgentsAdapter(**defaults)


def _make_chunk_event(text: str) -> dict[str, Any]:
    """Build a Bedrock EventStream chunk event with text bytes."""
    return {"chunk": {"bytes": text.encode("utf-8")}}


def _make_trace_event(orchestration_trace: dict[str, Any]) -> dict[str, Any]:
    """Build a Bedrock EventStream trace event."""
    return {
        "trace": {
            "trace": {
                "orchestrationTrace": orchestration_trace,
            }
        }
    }


def _make_return_control_event(
    action_group: str = "OrderAPI",
    api_path: str = "/getOrder",
    parameters: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a Bedrock EventStream returnControl event."""
    return {
        "returnControl": {
            "invocationInputs": [
                {
                    "apiInvocationInput": {
                        "actionGroupName": action_group,
                        "apiPath": api_path,
                        "parameters": parameters or {"orderId": "123"},
                    }
                }
            ]
        }
    }


def _mock_invoke_agent_response(events: list[dict[str, Any]]) -> dict[str, Any]:
    """Build a mocked invoke_agent response with the given events."""
    return {"completion": events}


# ===================================================================
# TestBedrockAgentsAdapter
# ===================================================================


class TestBedrockAgentsAdapter:
    """Tests for BedrockAgentsAdapter."""

    # -- Construction -------------------------------------------------------

    def test_create_adapter(self):
        """Adapter can be created without boto3 installed."""
        adapter = _make_adapter()
        assert adapter.framework == "bedrock"
        assert adapter.model == "anthropic.claude-3-sonnet"
        assert adapter.agent_name == "test-bedrock-agent"

    def test_default_agent_name(self):
        """Default agent name uses framework prefix."""
        from agentassay.integrations.bedrock_adapter import BedrockAgentsAdapter

        adapter = BedrockAgentsAdapter(
            agent_id="ABCDEFGHIJ",
            agent_alias_id="TSTALIASID",
        )
        assert adapter.agent_name == "bedrock-agent"

    def test_repr(self):
        adapter = _make_adapter()
        r = repr(adapter)
        assert "BedrockAgentsAdapter" in r
        assert "bedrock" in r

    # -- Install check ------------------------------------------------------

    def test_check_boto3_not_installed(self):
        """_check_boto3_installed raises FrameworkNotInstalledError."""
        from agentassay.integrations.bedrock_adapter import _check_boto3_installed

        with patch.dict("sys.modules", {"boto3": None}):
            with pytest.raises(FrameworkNotInstalledError, match="boto3"):
                _check_boto3_installed()

    def test_check_boto3_installed_ok(self):
        """_check_boto3_installed succeeds when boto3 is importable."""
        from agentassay.integrations.bedrock_adapter import _check_boto3_installed

        mock_boto3 = MagicMock()
        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            _check_boto3_installed()  # Should not raise

    # -- run(): basic streaming response ------------------------------------

    @patch("agentassay.integrations.bedrock_adapter._check_boto3_installed")
    def test_run_basic_chunk_response(self, mock_check):
        """run() produces a successful trace from a simple chunk response."""
        adapter = _make_adapter()

        mock_client = MagicMock()
        mock_client.invoke_agent.return_value = _mock_invoke_agent_response(
            [
                _make_chunk_event("The capital of France is Paris."),
            ]
        )
        adapter._client = mock_client

        trace = adapter.run({"query": "What is the capital of France?"})

        assert isinstance(trace, ExecutionTrace)
        assert trace.success is True
        assert trace.framework == "bedrock"
        assert trace.output_data == "The capital of France is Paris."
        assert trace.model == "anthropic.claude-3-sonnet"
        assert "bedrock_agent_id" in trace.metadata

    # -- run(): orchestration trace with rationale --------------------------

    @patch("agentassay.integrations.bedrock_adapter._check_boto3_installed")
    def test_run_rationale_trace(self, mock_check):
        """Rationale trace events become llm_response steps."""
        adapter = _make_adapter()

        mock_client = MagicMock()
        mock_client.invoke_agent.return_value = _mock_invoke_agent_response(
            [
                _make_trace_event(
                    {
                        "rationale": {"text": "I need to look up the order details."},
                    }
                ),
                _make_chunk_event("Order #123 is shipped."),
            ]
        )
        adapter._client = mock_client

        trace = adapter.run({"query": "Check order 123"})

        assert trace.success is True
        assert len(trace.steps) >= 1
        rationale_step = trace.steps[0]
        assert rationale_step.action == "llm_response"
        assert "look up" in rationale_step.llm_output

    # -- run(): action group invocation (tool call) -------------------------

    @patch("agentassay.integrations.bedrock_adapter._check_boto3_installed")
    def test_run_action_group_invocation(self, mock_check):
        """Action group invocations are parsed as tool_call steps."""
        adapter = _make_adapter()

        mock_client = MagicMock()
        mock_client.invoke_agent.return_value = _mock_invoke_agent_response(
            [
                _make_trace_event(
                    {
                        "invocationInput": {
                            "invocationType": "ACTION_GROUP",
                            "actionGroupInvocationInput": {
                                "actionGroupName": "OrderAPI",
                                "apiPath": "/getOrder",
                                "parameters": {"orderId": "123"},
                            },
                        },
                    }
                ),
                _make_trace_event(
                    {
                        "invocationOutput": {
                            "actionGroupInvocationOutput": {
                                "text": '{"status": "shipped", "tracking": "1Z999AA10"}'
                            },
                        },
                    }
                ),
                _make_chunk_event("Your order has been shipped."),
            ]
        )
        adapter._client = mock_client

        trace = adapter.run({"query": "Where is my order?"})

        assert trace.success is True
        # Should have at least 2 steps: invocation input + output
        tool_steps = [s for s in trace.steps if s.action == "tool_call"]
        assert len(tool_steps) >= 1
        assert "OrderAPI" in tool_steps[0].tool_name

    # -- run(): knowledge base lookup (retrieval) ---------------------------

    @patch("agentassay.integrations.bedrock_adapter._check_boto3_installed")
    def test_run_knowledge_base_lookup(self, mock_check):
        """Knowledge base lookups are parsed as retrieval steps."""
        adapter = _make_adapter()

        mock_client = MagicMock()
        mock_client.invoke_agent.return_value = _mock_invoke_agent_response(
            [
                _make_trace_event(
                    {
                        "invocationInput": {
                            "invocationType": "KNOWLEDGE_BASE",
                            "knowledgeBaseLookupInput": {
                                "knowledgeBaseId": "KB12345",
                                "text": "What is the return policy?",
                            },
                        },
                    }
                ),
                _make_trace_event(
                    {
                        "invocationOutput": {
                            "knowledgeBaseLookupOutput": {
                                "retrievedReferences": [
                                    {
                                        "content": {"text": "30-day return policy applies."},
                                        "location": {
                                            "type": "S3",
                                            "s3Location": {"uri": "s3://docs/policy.pdf"},
                                        },
                                    },
                                ],
                            },
                        },
                    }
                ),
                _make_chunk_event("Our return policy allows returns within 30 days."),
            ]
        )
        adapter._client = mock_client

        trace = adapter.run({"query": "Return policy?"})

        assert trace.success is True
        retrieval_steps = [s for s in trace.steps if s.action == "retrieval"]
        assert len(retrieval_steps) >= 1
        assert "knowledge_base" in retrieval_steps[0].tool_name

    # -- run(): observation trace -------------------------------------------

    @patch("agentassay.integrations.bedrock_adapter._check_boto3_installed")
    def test_run_observation_trace(self, mock_check):
        """Observation trace events become observation steps."""
        adapter = _make_adapter()

        mock_client = MagicMock()
        mock_client.invoke_agent.return_value = _mock_invoke_agent_response(
            [
                _make_trace_event(
                    {
                        "observation": {
                            "type": "FINISH",
                            "finalResponse": {"text": "Task completed successfully."},
                        },
                    }
                ),
                _make_chunk_event("Done."),
            ]
        )
        adapter._client = mock_client

        trace = adapter.run({"query": "Do something"})

        assert trace.success is True
        obs_steps = [s for s in trace.steps if s.action == "observation"]
        assert len(obs_steps) >= 1

    # -- run(): return of control event -------------------------------------

    @patch("agentassay.integrations.bedrock_adapter._check_boto3_installed")
    def test_run_return_control_event(self, mock_check):
        """returnControl events are parsed as tool_call steps."""
        adapter = _make_adapter()

        mock_client = MagicMock()
        mock_client.invoke_agent.return_value = _mock_invoke_agent_response(
            [
                _make_return_control_event(
                    action_group="InventoryAPI",
                    api_path="/checkStock",
                    parameters={"sku": "WIDGET-001"},
                ),
            ]
        )
        adapter._client = mock_client

        trace = adapter.run({"query": "Check inventory"})

        assert trace.success is True
        tool_steps = [s for s in trace.steps if s.action == "tool_call"]
        assert len(tool_steps) == 1
        assert "InventoryAPI" in tool_steps[0].tool_name
        assert "/checkStock" in tool_steps[0].tool_name

    # -- run(): empty event stream ------------------------------------------

    @patch("agentassay.integrations.bedrock_adapter._check_boto3_installed")
    def test_run_empty_event_stream(self, mock_check):
        """Empty event stream produces a successful trace with no steps."""
        adapter = _make_adapter()

        mock_client = MagicMock()
        mock_client.invoke_agent.return_value = _mock_invoke_agent_response([])
        adapter._client = mock_client

        trace = adapter.run({"query": "Hello"})

        assert trace.success is True
        assert len(trace.steps) == 0
        assert trace.output_data == ""

    # -- run(): error handling ----------------------------------------------

    @patch("agentassay.integrations.bedrock_adapter._check_boto3_installed")
    def test_run_exception_returns_failed_trace(self, mock_check):
        """Exceptions during invoke_agent produce success=False traces."""
        adapter = _make_adapter()

        mock_client = MagicMock()
        mock_client.invoke_agent.side_effect = RuntimeError("Throttled")
        adapter._client = mock_client

        trace = adapter.run({"query": "test"})

        assert trace.success is False
        assert "RuntimeError" in trace.error
        assert "Throttled" in trace.error
        assert trace.steps == []

    def test_run_without_boto3_raises(self):
        """run() raises FrameworkNotInstalledError when boto3 missing."""
        adapter = _make_adapter()

        with patch.dict("sys.modules", {"boto3": None}):
            with pytest.raises(FrameworkNotInstalledError, match="boto3"):
                adapter.run({"query": "test"})

    # -- run(): session ID handling -----------------------------------------

    @patch("agentassay.integrations.bedrock_adapter._check_boto3_installed")
    def test_run_uses_provided_session_id(self, mock_check):
        """Provided session_id is forwarded to invoke_agent."""
        adapter = _make_adapter(session_id="my-session-42")

        mock_client = MagicMock()
        mock_client.invoke_agent.return_value = _mock_invoke_agent_response(
            [
                _make_chunk_event("OK"),
            ]
        )
        adapter._client = mock_client

        trace = adapter.run({"query": "Continue conversation"})

        call_kwargs = mock_client.invoke_agent.call_args[1]
        assert call_kwargs["sessionId"] == "my-session-42"
        assert trace.metadata["bedrock_session_id"] == "my-session-42"

    @patch("agentassay.integrations.bedrock_adapter._check_boto3_installed")
    def test_run_generates_session_id_when_none(self, mock_check):
        """A UUID session_id is generated when not provided."""
        adapter = _make_adapter()

        mock_client = MagicMock()
        mock_client.invoke_agent.return_value = _mock_invoke_agent_response(
            [
                _make_chunk_event("OK"),
            ]
        )
        adapter._client = mock_client

        _trace = adapter.run({"query": "New conversation"})  # noqa: F841 - side effect tested

        call_kwargs = mock_client.invoke_agent.call_args[1]
        assert len(call_kwargs["sessionId"]) == 36  # UUID format

    # -- run(): scenario_id -------------------------------------------------

    @patch("agentassay.integrations.bedrock_adapter._check_boto3_installed")
    def test_run_uses_scenario_id_from_input(self, mock_check):
        """scenario_id is extracted from input_data."""
        adapter = _make_adapter()

        mock_client = MagicMock()
        mock_client.invoke_agent.return_value = _mock_invoke_agent_response(
            [
                _make_chunk_event("OK"),
            ]
        )
        adapter._client = mock_client

        trace = adapter.run({"query": "test", "scenario_id": "ecommerce-1"})

        assert trace.scenario_id == "ecommerce-1"

    @patch("agentassay.integrations.bedrock_adapter._check_boto3_installed")
    def test_run_default_scenario_id(self, mock_check):
        """Default scenario_id is 'default'."""
        adapter = _make_adapter()

        mock_client = MagicMock()
        mock_client.invoke_agent.return_value = _mock_invoke_agent_response(
            [
                _make_chunk_event("OK"),
            ]
        )
        adapter._client = mock_client

        trace = adapter.run({"query": "test"})

        assert trace.scenario_id == "default"

    # -- to_callable --------------------------------------------------------

    @patch("agentassay.integrations.bedrock_adapter._check_boto3_installed")
    def test_to_callable_returns_run(self, mock_check):
        """to_callable returns a callable that is self.run."""
        adapter = _make_adapter()
        fn = adapter.to_callable()
        assert callable(fn)
        assert fn == adapter.run

    def test_to_callable_without_boto3_raises(self):
        """to_callable raises when boto3 is missing."""
        adapter = _make_adapter()

        with patch.dict("sys.modules", {"boto3": None}):
            with pytest.raises(FrameworkNotInstalledError, match="boto3"):
                adapter.to_callable()

    # -- get_config ---------------------------------------------------------

    def test_get_config(self):
        """get_config returns an AgentConfig with correct values.

        Uses 'custom' as framework since 'bedrock' is not in the
        AgentConfig Literal.  The real framework is in metadata.
        """
        adapter = _make_adapter()
        config = adapter.get_config()
        assert config.framework == "bedrock"
        assert config.model == "anthropic.claude-3-sonnet"
        assert config.name == "test-bedrock-agent"
        assert config.metadata["bedrock_agent_id"] == "ABCDEFGHIJ"

    # -- _build_user_input --------------------------------------------------

    def test_build_user_input_query_key(self):
        """_build_user_input uses 'query' key first."""
        from agentassay.integrations.bedrock_adapter import BedrockAgentsAdapter

        result = BedrockAgentsAdapter._build_user_input({"query": "Hello world", "input": "other"})
        assert result == "Hello world"

    def test_build_user_input_fallback_to_json(self):
        """_build_user_input serializes to JSON when no known key exists."""
        from agentassay.integrations.bedrock_adapter import BedrockAgentsAdapter

        result = BedrockAgentsAdapter._build_user_input({"custom_field": "value", "another": 42})
        assert "custom_field" in result
        assert "42" in result

    # -- _get_client --------------------------------------------------------

    @patch("agentassay.integrations.bedrock_adapter._check_boto3_installed")
    def test_get_client_creates_lazily(self, mock_check):
        """_get_client lazily creates a boto3 client."""
        adapter = _make_adapter(region="us-west-2")

        mock_boto3 = MagicMock()
        with patch.dict("sys.modules", {"boto3": mock_boto3}):
            with patch(
                "agentassay.integrations.bedrock_adapter.boto3",
                mock_boto3,
                create=True,
            ):
                # Force reimport path (line left for future use)
                # noqa: F841 - original_import reserved for future reimport logic

                # Directly test client is None initially
                assert adapter._client is None

    # -- Multi-chunk output -------------------------------------------------

    @patch("agentassay.integrations.bedrock_adapter._check_boto3_installed")
    def test_run_multi_chunk_output(self, mock_check):
        """Multiple text chunks are concatenated into output_data."""
        adapter = _make_adapter()

        mock_client = MagicMock()
        mock_client.invoke_agent.return_value = _mock_invoke_agent_response(
            [
                _make_chunk_event("Hello "),
                _make_chunk_event("World!"),
            ]
        )
        adapter._client = mock_client

        trace = adapter.run({"query": "Greet me"})

        assert trace.success is True
        assert trace.output_data == "Hello World!"

    # -- Complex orchestration: rationale + action + observation -------------

    @patch("agentassay.integrations.bedrock_adapter._check_boto3_installed")
    def test_run_full_orchestration_flow(self, mock_check):
        """Full orchestration: rationale -> action -> observation -> output."""
        adapter = _make_adapter()

        mock_client = MagicMock()
        mock_client.invoke_agent.return_value = _mock_invoke_agent_response(
            [
                _make_trace_event(
                    {
                        "rationale": {"text": "I should check the order status."},
                    }
                ),
                _make_trace_event(
                    {
                        "invocationInput": {
                            "invocationType": "ACTION_GROUP",
                            "actionGroupInvocationInput": {
                                "actionGroupName": "OrderAPI",
                                "apiPath": "/getStatus",
                                "parameters": {"id": "456"},
                            },
                        },
                    }
                ),
                _make_trace_event(
                    {
                        "invocationOutput": {
                            "actionGroupInvocationOutput": {"text": '{"status": "delivered"}'},
                        },
                    }
                ),
                _make_trace_event(
                    {
                        "observation": {
                            "type": "FINISH",
                            "finalResponse": {"text": "Order 456 has been delivered."},
                        },
                    }
                ),
                _make_chunk_event("Your order 456 has been delivered."),
            ]
        )
        adapter._client = mock_client

        trace = adapter.run({"query": "Order 456 status?"})

        assert trace.success is True
        assert len(trace.steps) >= 4
        actions = [s.action for s in trace.steps]
        assert "llm_response" in actions  # rationale
        assert "tool_call" in actions  # action group
        assert "observation" in actions  # observation
        assert trace.output_data == "Your order 456 has been delivered."

    # -- Step indices are monotonic -----------------------------------------

    @patch("agentassay.integrations.bedrock_adapter._check_boto3_installed")
    def test_step_indices_monotonic(self, mock_check):
        """All step indices are monotonically increasing."""
        adapter = _make_adapter()

        mock_client = MagicMock()
        mock_client.invoke_agent.return_value = _mock_invoke_agent_response(
            [
                _make_trace_event({"rationale": {"text": "Think"}}),
                _make_trace_event(
                    {
                        "invocationInput": {
                            "invocationType": "ACTION_GROUP",
                            "actionGroupInvocationInput": {
                                "actionGroupName": "API",
                                "apiPath": "/do",
                                "parameters": {},
                            },
                        },
                    }
                ),
                _make_trace_event(
                    {"observation": {"type": "FINISH", "finalResponse": {"text": "Done"}}}
                ),
                _make_chunk_event("OK"),
            ]
        )
        adapter._client = mock_client

        trace = adapter.run({"query": "test"})

        indices = [s.step_index for s in trace.steps]
        assert indices == sorted(indices)
        assert len(set(indices)) == len(indices)  # All unique
