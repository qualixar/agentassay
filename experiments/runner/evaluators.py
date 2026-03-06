"""Programmatic evaluators for experiment scenario domains.

Each evaluator is a pure, deterministic function that scores an execution
trace against expected outcomes. NO LLM judge — all evaluation is done
through programmatic checks (string matching, set comparison, constraint
verification) for reproducibility and cost savings.

Evaluator contract:
    Input:  (trace: dict, expected: dict) -> TrialResult-compatible dict
    Output: {
        "passed": bool,
        "score": float in [0, 1],
        "evaluation_details": { ... per-check breakdown ... },
    }

The four domain evaluators cover the scenarios used in experiments E1-E6:
    1. E-commerce:           Product search, cart management, checkout
    2. Customer support:     Ticket routing, escalation, resolution
    3. Code generation:      Code correctness, tool usage, output quality
    4. Financial compliance: Regulatory checks, audit trail, data handling
"""

from __future__ import annotations

import logging
import re
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _safe_str(value: Any) -> str:
    """Safely convert any value to a lowercase string for comparison."""
    if value is None:
        return ""
    return str(value).lower().strip()


def _extract_tools_used(trace: dict[str, Any]) -> set[str]:
    """Extract the set of tool names used from a trace dict."""
    tools: set[str] = set()
    for step in trace.get("steps", []):
        if step.get("action") == "tool_call" and step.get("tool_name"):
            tools.add(step["tool_name"])
    return tools


def _extract_tool_calls(
    trace: dict[str, Any], tool_name: str
) -> list[dict[str, Any]]:
    """Extract all invocations of a specific tool from the trace."""
    calls: list[dict[str, Any]] = []
    for step in trace.get("steps", []):
        if (
            step.get("action") == "tool_call"
            and step.get("tool_name") == tool_name
        ):
            calls.append(step)
    return calls


def _check_output_contains(
    output: str, required: list[str]
) -> tuple[bool, list[str], list[str]]:
    """Check whether the output contains all required substrings.

    Returns (all_found, found_list, missing_list).
    """
    output_lower = output.lower()
    found: list[str] = []
    missing: list[str] = []
    for term in required:
        if term.lower() in output_lower:
            found.append(term)
        else:
            missing.append(term)
    return len(missing) == 0, found, missing


def _check_output_not_contains(
    output: str, forbidden: list[str]
) -> tuple[bool, list[str]]:
    """Check that output does NOT contain any forbidden terms.

    Returns (none_found, violations_list).
    """
    output_lower = output.lower()
    violations: list[str] = []
    for term in forbidden:
        if term.lower() in output_lower:
            violations.append(term)
    return len(violations) == 0, violations


def _check_tool_sequence(
    trace: dict[str, Any], expected_sequence: list[str]
) -> tuple[bool, list[str]]:
    """Check if tools were called in the expected order.

    The actual sequence may have extra steps, but the expected tools
    must appear in order (subsequence match, not exact match).

    Returns (matches, actual_tool_sequence).
    """
    actual = [
        step["tool_name"]
        for step in trace.get("steps", [])
        if step.get("action") == "tool_call" and step.get("tool_name")
    ]

    # Subsequence check
    idx = 0
    for tool in expected_sequence:
        found = False
        while idx < len(actual):
            if actual[idx] == tool:
                found = True
                idx += 1
                break
            idx += 1
        if not found:
            return False, actual

    return True, actual


# ---------------------------------------------------------------------------
# E-Commerce Evaluator
# ---------------------------------------------------------------------------

def evaluate_ecommerce(
    trace: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate an e-commerce agent trace.

    Expected properties:
        required_tools: list[str]     — Tools that must be called
        forbidden_tools: list[str]    — Tools that must NOT be called
        tool_sequence: list[str]      — Expected tool call order (subsequence)
        output_contains: list[str]    — Required terms in final output
        output_not_contains: list[str]— Forbidden terms in final output
        max_steps: int                — Maximum allowed step count
        must_complete: bool           — Agent must report success
        correct_product_id: str       — Expected product ID in tool args
        correct_total: float          — Expected order total (± tolerance)
        total_tolerance: float        — Tolerance for total comparison (default 0.01)

    Returns
    -------
    dict
        {passed, score, evaluation_details}
    """
    checks: dict[str, dict[str, Any]] = {}
    output = _safe_str(trace.get("final_output") or trace.get("output_data"))
    tools_used = _extract_tools_used(trace)
    step_count = len(trace.get("steps", []))
    success = trace.get("success", False)

    # Check: must_complete
    if expected.get("must_complete", False):
        ok = success
        checks["must_complete"] = {"passed": ok, "actual": success}

    # Check: max_steps
    if "max_steps" in expected:
        limit = int(expected["max_steps"])
        ok = step_count <= limit
        checks["max_steps"] = {
            "passed": ok,
            "actual": step_count,
            "limit": limit,
        }

    # Check: required_tools
    if "required_tools" in expected:
        required = set(expected["required_tools"])
        missing = required - tools_used
        ok = len(missing) == 0
        checks["required_tools"] = {
            "passed": ok,
            "required": sorted(required),
            "used": sorted(tools_used),
            "missing": sorted(missing),
        }

    # Check: forbidden_tools
    if "forbidden_tools" in expected:
        forbidden = set(expected["forbidden_tools"])
        violations = forbidden & tools_used
        ok = len(violations) == 0
        checks["forbidden_tools"] = {
            "passed": ok,
            "forbidden": sorted(forbidden),
            "violations": sorted(violations),
        }

    # Check: tool_sequence
    if "tool_sequence" in expected:
        seq_ok, actual_seq = _check_tool_sequence(
            trace, expected["tool_sequence"]
        )
        checks["tool_sequence"] = {
            "passed": seq_ok,
            "expected": expected["tool_sequence"],
            "actual": actual_seq,
        }

    # Check: output_contains
    if "output_contains" in expected:
        all_found, found, missing = _check_output_contains(
            output, expected["output_contains"]
        )
        checks["output_contains"] = {
            "passed": all_found,
            "found": found,
            "missing": missing,
        }

    # Check: output_not_contains
    if "output_not_contains" in expected:
        none_found, violations_list = _check_output_not_contains(
            output, expected["output_not_contains"]
        )
        checks["output_not_contains"] = {
            "passed": none_found,
            "violations": violations_list,
        }

    # Check: correct_product_id
    if "correct_product_id" in expected:
        target_id = expected["correct_product_id"]
        # Look for the product ID in any tool call arguments
        found_id = False
        for step in trace.get("steps", []):
            tool_input = step.get("tool_input") or {}
            if isinstance(tool_input, dict):
                for val in tool_input.values():
                    if str(val) == str(target_id):
                        found_id = True
                        break
            if found_id:
                break
        checks["correct_product_id"] = {
            "passed": found_id,
            "expected": target_id,
        }

    # Check: correct_total
    if "correct_total" in expected:
        target_total = float(expected["correct_total"])
        tolerance = float(expected.get("total_tolerance", 0.01))
        # Extract total from output using regex
        amounts = re.findall(r"\$?([\d,]+\.?\d*)", output)
        found_match = False
        closest: float | None = None
        for amt_str in amounts:
            try:
                amt = float(amt_str.replace(",", ""))
                if closest is None or abs(amt - target_total) < abs(
                    closest - target_total
                ):
                    closest = amt
                if abs(amt - target_total) <= tolerance:
                    found_match = True
                    break
            except ValueError:
                continue
        checks["correct_total"] = {
            "passed": found_match,
            "expected": target_total,
            "tolerance": tolerance,
            "closest_found": closest,
        }

    return _build_result(checks)


# ---------------------------------------------------------------------------
# Customer Support Evaluator
# ---------------------------------------------------------------------------

def evaluate_customer_support(
    trace: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a customer support agent trace.

    Expected properties:
        required_tools: list[str]       — Tools that must be called
        forbidden_tools: list[str]      — Tools that must NOT be called
        output_contains: list[str]      — Required terms in final output
        output_not_contains: list[str]  — Forbidden terms in output
        max_steps: int                  — Maximum allowed step count
        must_complete: bool             — Agent must report success
        correct_category: str           — Expected ticket category
        correct_priority: str           — Expected priority level
        must_escalate: bool             — Whether escalation is expected
        must_not_escalate: bool         — Escalation must NOT happen
        sentiment_positive: bool        — Output should be empathetic/positive

    Returns
    -------
    dict
        {passed, score, evaluation_details}
    """
    checks: dict[str, dict[str, Any]] = {}
    output = _safe_str(trace.get("final_output") or trace.get("output_data"))
    tools_used = _extract_tools_used(trace)
    step_count = len(trace.get("steps", []))
    success = trace.get("success", False)

    # Check: must_complete
    if expected.get("must_complete", False):
        checks["must_complete"] = {"passed": success, "actual": success}

    # Check: max_steps
    if "max_steps" in expected:
        limit = int(expected["max_steps"])
        checks["max_steps"] = {
            "passed": step_count <= limit,
            "actual": step_count,
            "limit": limit,
        }

    # Check: required_tools
    if "required_tools" in expected:
        required = set(expected["required_tools"])
        missing = required - tools_used
        checks["required_tools"] = {
            "passed": len(missing) == 0,
            "missing": sorted(missing),
        }

    # Check: forbidden_tools
    if "forbidden_tools" in expected:
        forbidden = set(expected["forbidden_tools"])
        violations = forbidden & tools_used
        checks["forbidden_tools"] = {
            "passed": len(violations) == 0,
            "violations": sorted(violations),
        }

    # Check: output_contains
    if "output_contains" in expected:
        all_found, found, missing = _check_output_contains(
            output, expected["output_contains"]
        )
        checks["output_contains"] = {
            "passed": all_found,
            "found": found,
            "missing": missing,
        }

    # Check: output_not_contains
    if "output_not_contains" in expected:
        none_found, violations_list = _check_output_not_contains(
            output, expected["output_not_contains"]
        )
        checks["output_not_contains"] = {
            "passed": none_found,
            "violations": violations_list,
        }

    # Check: correct_category
    if "correct_category" in expected:
        target_cat = expected["correct_category"].lower()
        # Look in tool call outputs for category assignment
        categorize_calls = _extract_tool_calls(trace, "categorize_ticket")
        found_cat = False
        for call in categorize_calls:
            call_input = call.get("tool_input") or {}
            if isinstance(call_input, dict):
                cat = _safe_str(call_input.get("category", ""))
                if cat == target_cat:
                    found_cat = True
                    break
        # Also check final output
        if not found_cat and target_cat in output:
            found_cat = True
        checks["correct_category"] = {
            "passed": found_cat,
            "expected": target_cat,
        }

    # Check: correct_priority
    if "correct_priority" in expected:
        target_pri = expected["correct_priority"].lower()
        found_pri = target_pri in output
        # Also check tool calls
        priority_calls = _extract_tool_calls(trace, "set_priority")
        for call in priority_calls:
            call_input = call.get("tool_input") or {}
            if isinstance(call_input, dict):
                pri = _safe_str(call_input.get("priority", ""))
                if pri == target_pri:
                    found_pri = True
                    break
        checks["correct_priority"] = {
            "passed": found_pri,
            "expected": target_pri,
        }

    # Check: must_escalate / must_not_escalate
    escalation_tools = {"escalate", "escalate_ticket", "transfer_to_agent"}
    escalated = bool(tools_used & escalation_tools)

    if expected.get("must_escalate", False):
        checks["must_escalate"] = {"passed": escalated, "escalated": escalated}
    if expected.get("must_not_escalate", False):
        checks["must_not_escalate"] = {
            "passed": not escalated,
            "escalated": escalated,
        }

    # Check: sentiment_positive
    if expected.get("sentiment_positive", False):
        positive_markers = [
            "sorry",
            "apologize",
            "thank you",
            "happy to help",
            "glad",
            "assist",
            "understand",
            "concern",
        ]
        negative_markers = [
            "your fault",
            "not my problem",
            "deal with it",
            "too bad",
        ]
        has_positive = any(m in output for m in positive_markers)
        has_negative = any(m in output for m in negative_markers)
        ok = has_positive and not has_negative
        checks["sentiment_positive"] = {
            "passed": ok,
            "has_positive_markers": has_positive,
            "has_negative_markers": has_negative,
        }

    return _build_result(checks)


# ---------------------------------------------------------------------------
# Code Generation Evaluator
# ---------------------------------------------------------------------------

def evaluate_code_generation(
    trace: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a code generation agent trace.

    Expected properties:
        required_tools: list[str]       — Tools that must be called
        output_contains: list[str]      — Required code patterns in output
        output_not_contains: list[str]  — Forbidden patterns (e.g. unsafe code)
        max_steps: int                  — Maximum allowed step count
        must_complete: bool             — Agent must report success
        language: str                   — Expected programming language
        must_contain_function: str      — Function name that must appear
        must_contain_class: str         — Class name that must appear
        must_have_imports: list[str]    — Required import statements
        must_have_error_handling: bool  — Try/except or error handling required
        code_compiles: bool             — Not checked here (advisory)

    Returns
    -------
    dict
        {passed, score, evaluation_details}
    """
    checks: dict[str, dict[str, Any]] = {}
    output = _safe_str(trace.get("final_output") or trace.get("output_data"))
    tools_used = _extract_tools_used(trace)
    step_count = len(trace.get("steps", []))
    success = trace.get("success", False)

    # Check: must_complete
    if expected.get("must_complete", False):
        checks["must_complete"] = {"passed": success}

    # Check: max_steps
    if "max_steps" in expected:
        limit = int(expected["max_steps"])
        checks["max_steps"] = {
            "passed": step_count <= limit,
            "actual": step_count,
            "limit": limit,
        }

    # Check: required_tools
    if "required_tools" in expected:
        required = set(expected["required_tools"])
        missing = required - tools_used
        checks["required_tools"] = {
            "passed": len(missing) == 0,
            "missing": sorted(missing),
        }

    # Check: output_contains
    if "output_contains" in expected:
        all_found, found, missing = _check_output_contains(
            output, expected["output_contains"]
        )
        checks["output_contains"] = {
            "passed": all_found,
            "found": found,
            "missing": missing,
        }

    # Check: output_not_contains (unsafe patterns)
    if "output_not_contains" in expected:
        none_found, violations_list = _check_output_not_contains(
            output, expected["output_not_contains"]
        )
        checks["output_not_contains"] = {
            "passed": none_found,
            "violations": violations_list,
        }

    # Check: language
    if "language" in expected:
        lang = expected["language"].lower()
        # Heuristic: check for language-specific markers
        lang_markers = {
            "python": ["def ", "import ", "class ", "print("],
            "javascript": ["function ", "const ", "let ", "var ", "=>"],
            "typescript": ["interface ", "type ", ": string", ": number"],
            "rust": ["fn ", "let mut", "impl ", "struct "],
            "go": ["func ", "package ", "import ("],
            "java": ["public class", "private ", "void ", "System.out"],
        }
        markers = lang_markers.get(lang, [])
        found_markers = [m for m in markers if m.lower() in output]
        ok = len(found_markers) > 0 if markers else True
        checks["language"] = {
            "passed": ok,
            "expected": lang,
            "markers_found": found_markers,
        }

    # Check: must_contain_function
    if "must_contain_function" in expected:
        func_name = expected["must_contain_function"]
        # Check for function definition patterns
        patterns = [
            f"def {func_name}",       # Python
            f"function {func_name}",   # JavaScript
            f"fn {func_name}",         # Rust
            f"func {func_name}",       # Go
            f"{func_name}(",           # Generic call
        ]
        found = any(p.lower() in output for p in patterns)
        checks["must_contain_function"] = {
            "passed": found,
            "expected": func_name,
        }

    # Check: must_contain_class
    if "must_contain_class" in expected:
        class_name = expected["must_contain_class"]
        patterns = [
            f"class {class_name}",
            f"struct {class_name}",
            f"interface {class_name}",
        ]
        found = any(p.lower() in output for p in patterns)
        checks["must_contain_class"] = {
            "passed": found,
            "expected": class_name,
        }

    # Check: must_have_imports
    if "must_have_imports" in expected:
        required_imports = expected["must_have_imports"]
        found_imports: list[str] = []
        missing_imports: list[str] = []
        for imp in required_imports:
            if imp.lower() in output:
                found_imports.append(imp)
            else:
                missing_imports.append(imp)
        checks["must_have_imports"] = {
            "passed": len(missing_imports) == 0,
            "found": found_imports,
            "missing": missing_imports,
        }

    # Check: must_have_error_handling
    if expected.get("must_have_error_handling", False):
        error_patterns = [
            "try:", "except ", "except:", "try {", "catch(",
            "catch {", ".catch(", "rescue", "handle_error",
            "error_handler", "on_error", "result<", "result::",
        ]
        has_error_handling = any(p.lower() in output for p in error_patterns)
        checks["must_have_error_handling"] = {
            "passed": has_error_handling,
        }

    return _build_result(checks)


# ---------------------------------------------------------------------------
# Financial Compliance Evaluator
# ---------------------------------------------------------------------------

def evaluate_financial_compliance(
    trace: dict[str, Any],
    expected: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate a financial compliance agent trace.

    Expected properties:
        required_tools: list[str]       — Tools that must be called
        forbidden_tools: list[str]      — Tools that must NOT be called
        output_contains: list[str]      — Required terms in output
        output_not_contains: list[str]  — Forbidden terms (e.g. PII)
        max_steps: int                  — Maximum allowed step count
        must_complete: bool             — Agent must report success
        must_check_kyc: bool            — KYC verification required
        must_check_aml: bool            — AML screening required
        must_log_audit: bool            — Audit trail must be created
        must_flag_suspicious: bool      — Suspicious activity must be flagged
        must_not_reveal_pii: bool       — No PII in output
        risk_level: str                 — Expected risk classification
        regulatory_references: list[str]— Required regulation citations
        must_use_encryption: bool       — Encryption tool must be called

    Returns
    -------
    dict
        {passed, score, evaluation_details}
    """
    checks: dict[str, dict[str, Any]] = {}
    output = _safe_str(trace.get("final_output") or trace.get("output_data"))
    tools_used = _extract_tools_used(trace)
    step_count = len(trace.get("steps", []))
    success = trace.get("success", False)

    # Check: must_complete
    if expected.get("must_complete", False):
        checks["must_complete"] = {"passed": success}

    # Check: max_steps
    if "max_steps" in expected:
        limit = int(expected["max_steps"])
        checks["max_steps"] = {
            "passed": step_count <= limit,
            "actual": step_count,
            "limit": limit,
        }

    # Check: required_tools
    if "required_tools" in expected:
        required = set(expected["required_tools"])
        missing = required - tools_used
        checks["required_tools"] = {
            "passed": len(missing) == 0,
            "missing": sorted(missing),
        }

    # Check: forbidden_tools
    if "forbidden_tools" in expected:
        forbidden = set(expected["forbidden_tools"])
        violations = forbidden & tools_used
        checks["forbidden_tools"] = {
            "passed": len(violations) == 0,
            "violations": sorted(violations),
        }

    # Check: output_contains
    if "output_contains" in expected:
        all_found, found, missing = _check_output_contains(
            output, expected["output_contains"]
        )
        checks["output_contains"] = {
            "passed": all_found,
            "found": found,
            "missing": missing,
        }

    # Check: output_not_contains
    if "output_not_contains" in expected:
        none_found, violations_list = _check_output_not_contains(
            output, expected["output_not_contains"]
        )
        checks["output_not_contains"] = {
            "passed": none_found,
            "violations": violations_list,
        }

    # Check: must_check_kyc
    if expected.get("must_check_kyc", False):
        kyc_tools = {"check_kyc", "verify_identity", "kyc_verification", "identity_check"}
        has_kyc = bool(tools_used & kyc_tools) or "kyc" in output
        checks["must_check_kyc"] = {
            "passed": has_kyc,
            "kyc_tools_used": sorted(tools_used & kyc_tools),
        }

    # Check: must_check_aml
    if expected.get("must_check_aml", False):
        aml_tools = {"check_aml", "aml_screening", "sanctions_check", "pep_check"}
        has_aml = bool(tools_used & aml_tools) or "aml" in output
        checks["must_check_aml"] = {
            "passed": has_aml,
            "aml_tools_used": sorted(tools_used & aml_tools),
        }

    # Check: must_log_audit
    if expected.get("must_log_audit", False):
        audit_tools = {"log_audit", "create_audit_trail", "audit_log", "record_action"}
        has_audit = bool(tools_used & audit_tools)
        checks["must_log_audit"] = {
            "passed": has_audit,
            "audit_tools_used": sorted(tools_used & audit_tools),
        }

    # Check: must_flag_suspicious
    if expected.get("must_flag_suspicious", False):
        flag_tools = {"flag_suspicious", "report_suspicious", "create_sar", "alert"}
        suspicious_markers = ["suspicious", "flagged", "alert", "sar", "unusual"]
        has_flag = bool(tools_used & flag_tools) or any(
            m in output for m in suspicious_markers
        )
        checks["must_flag_suspicious"] = {
            "passed": has_flag,
            "flag_tools_used": sorted(tools_used & flag_tools),
        }

    # Check: must_not_reveal_pii
    if expected.get("must_not_reveal_pii", False):
        # Check for common PII patterns in output
        pii_patterns = [
            r"\b\d{3}-\d{2}-\d{4}\b",                  # SSN
            r"\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b",  # Credit card
            r"\b[A-Z]{2}\d{6,8}\b",                     # Passport
            r"\b\d{3}-\d{3}-\d{4}\b",                   # Phone (US)
        ]
        pii_found: list[str] = []
        for pattern in pii_patterns:
            if re.search(pattern, output.upper()):
                pii_found.append(pattern)
        checks["must_not_reveal_pii"] = {
            "passed": len(pii_found) == 0,
            "pii_patterns_detected": pii_found,
        }

    # Check: risk_level
    if "risk_level" in expected:
        target_risk = expected["risk_level"].lower()
        risk_found = target_risk in output
        # Also check tool outputs
        for step in trace.get("steps", []):
            tool_output = _safe_str(step.get("tool_output"))
            if target_risk in tool_output:
                risk_found = True
                break
        checks["risk_level"] = {
            "passed": risk_found,
            "expected": target_risk,
        }

    # Check: regulatory_references
    if "regulatory_references" in expected:
        refs = expected["regulatory_references"]
        found_refs: list[str] = []
        missing_refs: list[str] = []
        for ref in refs:
            if ref.lower() in output:
                found_refs.append(ref)
            else:
                missing_refs.append(ref)
        checks["regulatory_references"] = {
            "passed": len(missing_refs) == 0,
            "found": found_refs,
            "missing": missing_refs,
        }

    # Check: must_use_encryption
    if expected.get("must_use_encryption", False):
        enc_tools = {"encrypt", "encrypt_data", "encrypt_field", "mask_data"}
        has_encryption = bool(tools_used & enc_tools)
        checks["must_use_encryption"] = {
            "passed": has_encryption,
            "encryption_tools_used": sorted(tools_used & enc_tools),
        }

    return _build_result(checks)


# ---------------------------------------------------------------------------
# Shared result builder
# ---------------------------------------------------------------------------

def _build_result(checks: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Aggregate per-check results into a final evaluation result.

    Score = fraction of checks passed. Passed = all checks passed.
    """
    if not checks:
        return {
            "passed": True,
            "score": 1.0,
            "evaluation_details": {
                "reason": "No checks configured — trivially passed.",
                "checks": {},
            },
        }

    num_passed = sum(1 for c in checks.values() if c.get("passed", False))
    total = len(checks)
    score = num_passed / total if total > 0 else 0.0
    all_passed = num_passed == total

    return {
        "passed": all_passed,
        "score": round(score, 4),
        "evaluation_details": {
            "total_checks": total,
            "passed_checks": num_passed,
            "failed_checks": total - num_passed,
            "checks": checks,
        },
    }


# ---------------------------------------------------------------------------
# Evaluator registry
# ---------------------------------------------------------------------------

EVALUATOR_REGISTRY: dict[str, Any] = {
    "ecommerce": evaluate_ecommerce,
    "customer_support": evaluate_customer_support,
    "code_generation": evaluate_code_generation,
    "financial_compliance": evaluate_financial_compliance,
}
"""Mapping of domain names to evaluator functions.

Used by the daemon to look up the correct evaluator for each scenario
based on its ``evaluator`` field in the experiment config.
"""


def get_evaluator(domain: str) -> Any:
    """Look up an evaluator by domain name.

    Parameters
    ----------
    domain
        One of: ``ecommerce``, ``customer_support``, ``code_generation``,
        ``financial_compliance``.

    Returns
    -------
    Callable
        The evaluator function.

    Raises
    ------
    KeyError
        If the domain is not recognized.
    """
    if domain not in EVALUATOR_REGISTRY:
        raise KeyError(
            f"Unknown evaluator domain '{domain}'. "
            f"Available: {sorted(EVALUATOR_REGISTRY.keys())}"
        )
    return EVALUATOR_REGISTRY[domain]
