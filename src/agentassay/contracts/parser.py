# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Safe condition parser for AgentAssay contract evaluation.

Evaluates condition strings from ContractSpec YAML files using ONLY
regex matching and explicit comparisons. No dynamic code interpretation
of any kind -- only whitelisted functions and comparison operators
are supported.

Security design:
    - Condition strings are matched against a fixed set of regex patterns.
    - Only two built-in functions are recognized: ``uses_tool()`` and
      ``output_contains()``.
    - Comparison operators are dispatched through an explicit mapping.
    - No eval, no ast.literal_eval, no dynamic code paths.
"""

from __future__ import annotations

import logging
import operator
import re
from collections.abc import Callable
from typing import Any

from agentassay.core.models import ExecutionTrace

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Comparison operators -- the only operators we support
# ---------------------------------------------------------------------------

_COMPARISON_OPS: dict[str, Callable[[Any, Any], bool]] = {
    "<=": operator.le,
    ">=": operator.ge,
    "!=": operator.ne,
    "==": operator.eq,
    "<": operator.lt,
    ">": operator.gt,
}

# Pattern: `identifier op value` or `not func(args)`
_COMPARISON_RE = re.compile(
    r"^(\w+)\s*(<=|>=|!=|==|<|>)\s*(.+)$"
)

# Pattern: function call like `uses_tool('search')` or `output_contains('hello')`
_FUNC_CALL_RE = re.compile(
    r"""^(\w+)\(\s*['"]([^'"]*)['"]\s*\)$"""
)

# Pattern: negated function call like `not uses_tool('delete_database')`
_NOT_FUNC_RE = re.compile(
    r"""^not\s+(\w+)\(\s*['"]([^'"]*)['"]\s*\)$"""
)

# Pattern: bare identifier like `success`
_BARE_IDENT_RE = re.compile(
    r"^(\w+)$"
)

# Pattern: negated bare identifier like `not success`
_NOT_BARE_RE = re.compile(
    r"^not\s+(\w+)$"
)


# ---------------------------------------------------------------------------
# Value parsing
# ---------------------------------------------------------------------------


def _resolve_value(raw: str) -> int | float | str | bool:
    """Parse a literal value from a condition string.

    Handles integers, floats, booleans, and quoted strings.
    Uses explicit type parsing only -- no dynamic interpretation.

    Parameters
    ----------
    raw
        The raw string from the right-hand side of a comparison.

    Returns
    -------
    int | float | str | bool
        The parsed value.
    """
    stripped = raw.strip()

    # Boolean literals
    if stripped.lower() == "true":
        return True
    if stripped.lower() == "false":
        return False

    # Quoted string
    if (stripped.startswith("'") and stripped.endswith("'")) or (
        stripped.startswith('"') and stripped.endswith('"')
    ):
        return stripped[1:-1]

    # Integer
    try:
        return int(stripped)
    except ValueError:
        pass

    # Float
    try:
        return float(stripped)
    except ValueError:
        pass

    # Fall back to string
    return stripped


# ---------------------------------------------------------------------------
# Trace context extraction
# ---------------------------------------------------------------------------


def build_trace_context(trace: ExecutionTrace) -> dict[str, Any]:
    """Extract all evaluable properties from a trace into a flat dict.

    This is the "symbol table" for condition evaluation. Every property
    that conditions can reference is extracted here.

    Parameters
    ----------
    trace
        The execution trace to extract context from.

    Returns
    -------
    dict[str, Any]
        Flat mapping of property names to their values.
    """
    tools = trace.tools_used  # set[str] via @property

    output_str = ""
    if trace.output_data is not None:
        output_str = str(trace.output_data).lower()

    return {
        "step_count": trace.step_count,
        "total_cost_usd": trace.total_cost_usd,
        "total_duration_ms": trace.total_duration_ms,
        "success": trace.success,
        "tools_used": tools,
        "output_str": output_str,
        "output_data": trace.output_data,
        "input_data": trace.input_data,
    }


# ---------------------------------------------------------------------------
# Condition evaluation
# ---------------------------------------------------------------------------


def evaluate_condition(
    condition: str,
    context: dict[str, Any],
) -> tuple[bool, str]:
    """Safely evaluate a single condition string against a trace context.

    Returns (satisfied, actual_description).

    Security: This function uses ONLY regex matching and explicit
    comparisons. No dynamic code interpretation of any kind.

    Parameters
    ----------
    condition
        The condition string from the contract (e.g., "step_count <= 10").
    context
        The trace context dict from ``build_trace_context``.

    Returns
    -------
    tuple[bool, str]
        A pair of (is_satisfied, description_of_actual_value).
    """
    cond = condition.strip()

    # --- Pattern 1: `not func('arg')` ---
    m = _NOT_FUNC_RE.match(cond)
    if m:
        func_name, arg = m.group(1), m.group(2)
        satisfied, actual = _call_builtin(func_name, arg, context)
        return not satisfied, f"not {func_name}('{arg}') -> not {actual}"

    # --- Pattern 2: `func('arg')` ---
    m = _FUNC_CALL_RE.match(cond)
    if m:
        func_name, arg = m.group(1), m.group(2)
        satisfied, actual = _call_builtin(func_name, arg, context)
        return satisfied, actual

    # --- Pattern 3: `identifier op value` ---
    m = _COMPARISON_RE.match(cond)
    if m:
        ident, op_str, rhs_raw = m.group(1), m.group(2), m.group(3)
        return _evaluate_comparison(ident, op_str, rhs_raw, context)

    # --- Pattern 4: `not identifier` ---
    m = _NOT_BARE_RE.match(cond)
    if m:
        ident = m.group(1)
        if ident in context:
            val = context[ident]
            return not bool(val), f"not {ident} = not {val}"
        return False, f"unknown identifier '{ident}'"

    # --- Pattern 5: bare `identifier` (e.g., `success`) ---
    m = _BARE_IDENT_RE.match(cond)
    if m:
        ident = m.group(1)
        if ident in context:
            val = context[ident]
            return bool(val), f"{ident} = {val}"
        return False, f"unknown identifier '{ident}'"

    # --- Unknown pattern ---
    logger.warning("Cannot parse condition: '%s' -- treating as failed", cond)
    return False, f"unparseable condition: '{cond}'"


def _call_builtin(
    func_name: str,
    arg: str,
    context: dict[str, Any],
) -> tuple[bool, str]:
    """Dispatch a whitelisted built-in function call.

    Parameters
    ----------
    func_name
        Function name (e.g., "uses_tool", "output_contains").
    arg
        The string argument passed to the function.
    context
        The trace context dict.

    Returns
    -------
    tuple[bool, str]
        (result, description).
    """
    if func_name == "uses_tool":
        tools: set[str] = context.get("tools_used", set())
        found = arg in tools
        return found, f"uses_tool('{arg}') = {found} (tools: {sorted(tools)})"

    if func_name == "output_contains":
        output_str: str = context.get("output_str", "")
        found = arg.lower() in output_str
        preview = output_str[:100] + "..." if len(output_str) > 100 else output_str
        return found, f"output_contains('{arg}') = {found} (output: '{preview}')"

    logger.warning("Unknown built-in function: '%s' -- treating as failed", func_name)
    return False, f"unknown function '{func_name}'"


def _evaluate_comparison(
    ident: str,
    op_str: str,
    rhs_raw: str,
    context: dict[str, Any],
) -> tuple[bool, str]:
    """Evaluate a comparison expression like ``step_count <= 10``.

    Parameters
    ----------
    ident
        Left-hand side identifier (e.g., "step_count").
    op_str
        Comparison operator string (e.g., "<=").
    rhs_raw
        Right-hand side value string (e.g., "10").
    context
        The trace context dict.

    Returns
    -------
    tuple[bool, str]
        (result, description).
    """
    if ident not in context:
        return False, f"unknown identifier '{ident}'"

    lhs = context[ident]
    rhs = _resolve_value(rhs_raw)
    op_fn = _COMPARISON_OPS.get(op_str)

    if op_fn is None:
        return False, f"unknown operator '{op_str}'"

    try:
        result = op_fn(lhs, rhs)
        return bool(result), f"{ident} = {lhs} (threshold: {op_str} {rhs})"
    except TypeError:
        return False, (
            f"type mismatch: cannot compare {type(lhs).__name__} "
            f"{op_str} {type(rhs).__name__}"
        )
