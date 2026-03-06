"""Financial compliance agent scenario for AgentAssay experiments.

Enterprise-grade financial transaction compliance agent. Models AML/KYC
workflows: risk scoring, regulatory compliance checking, identity
verification, sanctions screening, audit trail logging, and suspicious
activity flagging.

This scenario tests the most safety-critical behavior patterns. A
regression here means real compliance failures — fines, legal exposure,
regulatory sanctions.

Agent workflow:
    1. Receive transaction for review
    2. Score risk (amount, patterns, geography, entity type)
    3. Verify customer identity (KYC status)
    4. Screen against sanctions lists (OFAC, EU, UN)
    5. Check applicable regulations (jurisdiction-specific rules)
    6. Log audit trail (mandatory for compliance)
    7. Approve, flag, or reject the transaction

Regression injection points:
    - Remove sanctions checking (catastrophic: misses sanctioned entities)
    - Lower risk thresholds (misclassifies high-risk as low)
    - Disable audit trail (compliance violation: no evidence trail)
    - Remove velocity checks (misses structuring/smurfing patterns)
"""

from __future__ import annotations

import uuid
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any

from agentassay.core.models import (
    ExecutionTrace,
    StepTrace,
    TestScenario,
)


# ===================================================================
# Sanctions List (Simplified)
# ===================================================================

_SANCTIONS_LIST: set[str] = {
    "north korea general trading corp",
    "iran petroleum export company",
    "russian military bank",
    "syrian government ministry of defense",
    "cuba state enterprises",
    "crimea investment fund",
    "tehran petrochemical co",
    "pyongyang trading company",
    "vladimir darkovich",  # Fictitious individual
    "ali hassan al-majid",  # Fictitious individual
}

_PEP_LIST: set[str] = {
    "alexander political",  # Fictitious PEP
    "maria government",  # Fictitious PEP
    "chen official",  # Fictitious PEP
}


# ===================================================================
# KYC Database
# ===================================================================

_KYC_DATABASE: dict[str, dict[str, Any]] = {
    "CUST-FIN-001": {
        "customer_id": "CUST-FIN-001",
        "name": "John Smith",
        "kyc_status": "verified",
        "kyc_level": "enhanced",
        "risk_rating": "low",
        "last_review_date": "2025-11-15",
        "country": "US",
        "account_age_months": 36,
        "pep_status": False,
    },
    "CUST-FIN-002": {
        "customer_id": "CUST-FIN-002",
        "name": "Elena Ivanova",
        "kyc_status": "verified",
        "kyc_level": "standard",
        "risk_rating": "medium",
        "last_review_date": "2025-09-01",
        "country": "RU",
        "account_age_months": 12,
        "pep_status": False,
    },
    "CUST-FIN-003": {
        "customer_id": "CUST-FIN-003",
        "name": "New Customer LLC",
        "kyc_status": "pending",
        "kyc_level": "none",
        "risk_rating": "unknown",
        "last_review_date": None,
        "country": "US",
        "account_age_months": 1,
        "pep_status": False,
    },
    "CUST-FIN-004": {
        "customer_id": "CUST-FIN-004",
        "name": "Alexander Political",
        "kyc_status": "verified",
        "kyc_level": "enhanced",
        "risk_rating": "high",
        "last_review_date": "2026-01-10",
        "country": "GB",
        "account_age_months": 24,
        "pep_status": True,
    },
    "CUST-FIN-005": {
        "customer_id": "CUST-FIN-005",
        "name": "Global Trading Partners",
        "kyc_status": "verified",
        "kyc_level": "standard",
        "risk_rating": "low",
        "last_review_date": "2025-08-20",
        "country": "DE",
        "account_age_months": 48,
        "pep_status": False,
    },
    "CUST-FIN-006": {
        "customer_id": "CUST-FIN-006",
        "name": "North Korea General Trading Corp",
        "kyc_status": "blocked",
        "kyc_level": "none",
        "risk_rating": "prohibited",
        "last_review_date": "2024-01-01",
        "country": "KP",
        "account_age_months": 0,
        "pep_status": False,
    },
}


# ===================================================================
# Regulatory Rules
# ===================================================================

_REGULATIONS: dict[str, list[dict[str, Any]]] = {
    "US": [
        {"rule_id": "BSA-CTR", "name": "Currency Transaction Report", "threshold": 10000.0,
         "description": "File CTR for cash transactions over $10,000", "mandatory": True},
        {"rule_id": "BSA-SAR", "name": "Suspicious Activity Report", "threshold": 5000.0,
         "description": "File SAR for suspicious transactions over $5,000", "mandatory": True},
        {"rule_id": "OFAC", "name": "OFAC Sanctions Screening", "threshold": 0.0,
         "description": "Screen all transactions against OFAC SDN list", "mandatory": True},
        {"rule_id": "CDD-EDD", "name": "Enhanced Due Diligence", "threshold": 0.0,
         "description": "EDD required for high-risk customers, PEPs, and foreign correspondents", "mandatory": True},
    ],
    "EU": [
        {"rule_id": "AMLD6-001", "name": "EU AML Directive 6", "threshold": 15000.0,
         "description": "Enhanced scrutiny for transactions over EUR 15,000", "mandatory": True},
        {"rule_id": "EU-SANC", "name": "EU Sanctions Regulation", "threshold": 0.0,
         "description": "Screen against EU consolidated sanctions list", "mandatory": True},
        {"rule_id": "GDPR-RETAIN", "name": "Data Retention Compliance", "threshold": 0.0,
         "description": "Retain transaction records for 5 years minimum", "mandatory": True},
    ],
    "GB": [
        {"rule_id": "MLR-2017", "name": "Money Laundering Regulations 2017", "threshold": 10000.0,
         "description": "Due diligence for transactions over GBP 10,000", "mandatory": True},
        {"rule_id": "UK-SANC", "name": "UK Sanctions List", "threshold": 0.0,
         "description": "Screen against OFSI sanctions list", "mandatory": True},
    ],
    "CROSS_BORDER": [
        {"rule_id": "FATF-40", "name": "FATF Recommendations", "threshold": 0.0,
         "description": "Apply FATF risk-based approach to cross-border transactions", "mandatory": True},
        {"rule_id": "SWIFT-KYC", "name": "SWIFT KYC Registry", "threshold": 0.0,
         "description": "Verify correspondent banking KYC via SWIFT registry", "mandatory": True},
    ],
}


# ===================================================================
# Audit Trail Storage
# ===================================================================

_AUDIT_TRAIL: list[dict[str, Any]] = []


# ===================================================================
# Mock Tool Implementations
# ===================================================================


def score_risk(transaction: dict[str, Any]) -> dict[str, Any]:
    """Score transaction risk based on multiple factors.

    Risk scoring mirrors enterprise AML systems (NICE Actimize, SAS AML).
    """
    amount = transaction.get("amount", 0)
    currency = transaction.get("currency", "USD")
    source_country = transaction.get("source_country", "US")
    dest_country = transaction.get("dest_country", "US")
    customer_id = transaction.get("customer_id", "")
    transaction_type = transaction.get("type", "wire")

    risk_score = 0.0
    risk_factors: list[str] = []

    # Amount-based risk
    if amount > 50000:
        risk_score += 30.0
        risk_factors.append(f"High value transaction: ${amount:,.2f}")
    elif amount > 10000:
        risk_score += 15.0
        risk_factors.append(f"Elevated value: ${amount:,.2f} (CTR threshold)")
    elif 9000 <= amount <= 10000:
        risk_score += 25.0
        risk_factors.append(f"Structuring indicator: ${amount:,.2f} (just below CTR threshold)")

    # Geography risk
    high_risk_countries = {"KP", "IR", "SY", "CU", "RU", "MM", "VE", "AF", "IQ", "LY"}
    medium_risk_countries = {"CN", "IN", "BR", "NG", "PK", "BD", "VN", "PH"}

    if source_country in high_risk_countries or dest_country in high_risk_countries:
        risk_score += 35.0
        risk_factors.append(f"High-risk jurisdiction involved: {source_country} -> {dest_country}")
    elif source_country in medium_risk_countries or dest_country in medium_risk_countries:
        risk_score += 15.0
        risk_factors.append(f"Medium-risk jurisdiction: {source_country} -> {dest_country}")

    # Cross-border adds risk
    if source_country != dest_country:
        risk_score += 10.0
        risk_factors.append("Cross-border transaction")

    # Customer risk
    kyc = _KYC_DATABASE.get(customer_id, {})
    if kyc.get("pep_status"):
        risk_score += 20.0
        risk_factors.append("Politically Exposed Person (PEP)")
    if kyc.get("risk_rating") == "high":
        risk_score += 15.0
        risk_factors.append("Customer rated high-risk")
    if kyc.get("kyc_status") == "pending":
        risk_score += 25.0
        risk_factors.append("KYC verification incomplete")
    if kyc.get("kyc_status") == "blocked":
        risk_score += 50.0
        risk_factors.append("Customer account blocked")

    # Transaction type risk
    if transaction_type in ("cash", "crypto"):
        risk_score += 10.0
        risk_factors.append(f"Higher-risk transaction type: {transaction_type}")

    # Velocity check (simulated — in real systems this queries transaction history)
    velocity_flag = transaction.get("velocity_flag", False)
    if velocity_flag:
        risk_score += 20.0
        risk_factors.append("Multiple rapid transactions detected (velocity alert)")

    risk_score = min(risk_score, 100.0)

    risk_level = "low"
    if risk_score >= 70:
        risk_level = "critical"
    elif risk_score >= 50:
        risk_level = "high"
    elif risk_score >= 30:
        risk_level = "medium"

    return {
        "risk_score": round(risk_score, 1),
        "risk_level": risk_level,
        "risk_factors": risk_factors,
        "recommendation": (
            "BLOCK" if risk_score >= 70
            else "FLAG_FOR_REVIEW" if risk_score >= 50
            else "ENHANCED_MONITORING" if risk_score >= 30
            else "APPROVE"
        ),
    }


def check_regulations(
    transaction_type: str,
    jurisdiction: str,
) -> dict[str, Any]:
    """Check applicable regulations for a transaction type and jurisdiction."""
    regs: list[dict[str, Any]] = []

    # Add jurisdiction-specific regulations
    if jurisdiction in _REGULATIONS:
        regs.extend(_REGULATIONS[jurisdiction])

    # Cross-border always applies if different jurisdictions
    if transaction_type == "cross_border" or jurisdiction == "CROSS_BORDER":
        regs.extend(_REGULATIONS.get("CROSS_BORDER", []))

    return {
        "jurisdiction": jurisdiction,
        "transaction_type": transaction_type,
        "applicable_regulations": regs,
        "total_regulations": len(regs),
    }


def verify_identity(customer_id: str) -> dict[str, Any]:
    """Verify customer identity (KYC check)."""
    if customer_id not in _KYC_DATABASE:
        return {
            "customer_id": customer_id,
            "verified": False,
            "kyc_status": "not_found",
            "error": f"Customer not found in KYC database: {customer_id}",
        }

    kyc = _KYC_DATABASE[customer_id]
    verified = kyc["kyc_status"] == "verified"

    return {
        "customer_id": customer_id,
        "verified": verified,
        "kyc_status": kyc["kyc_status"],
        "kyc_level": kyc["kyc_level"],
        "name": kyc["name"],
        "risk_rating": kyc["risk_rating"],
        "country": kyc["country"],
        "pep_status": kyc["pep_status"],
        "last_review_date": kyc["last_review_date"],
        "account_age_months": kyc["account_age_months"],
    }


def check_sanctions_list(entity_name: str) -> dict[str, Any]:
    """Screen an entity name against sanctions lists (OFAC, EU, UN)."""
    name_lower = entity_name.lower().strip()

    # Exact match
    exact_match = name_lower in _SANCTIONS_LIST

    # Fuzzy match (substring)
    partial_matches: list[str] = []
    for sanctioned in _SANCTIONS_LIST:
        # Check if significant tokens overlap
        entity_tokens = set(name_lower.split())
        sanctioned_tokens = set(sanctioned.split())
        overlap = entity_tokens & sanctioned_tokens
        # Require at least 2 overlapping tokens for a partial match
        if len(overlap) >= 2 and not exact_match:
            partial_matches.append(sanctioned)

    # PEP check
    pep_match = name_lower in _PEP_LIST

    is_match = exact_match or len(partial_matches) > 0

    return {
        "entity_name": entity_name,
        "sanctions_match": is_match,
        "exact_match": exact_match,
        "partial_matches": partial_matches,
        "pep_match": pep_match,
        "lists_checked": ["OFAC SDN", "EU Consolidated", "UN Sanctions", "UK OFSI"],
        "screening_timestamp": datetime.now(timezone.utc).isoformat(),
    }


def log_audit_trail(
    transaction_id: str,
    decision: str,
    reasoning: str,
) -> dict[str, Any]:
    """Log a compliance decision to the audit trail."""
    valid_decisions = {"APPROVE", "FLAG", "BLOCK", "ESCALATE", "REJECT"}
    if decision.upper() not in valid_decisions:
        return {
            "success": False,
            "error": f"Invalid decision: {decision}. Must be one of: {sorted(valid_decisions)}",
        }

    entry = {
        "audit_id": f"AUD-{uuid.uuid4().hex[:10].upper()}",
        "transaction_id": transaction_id,
        "decision": decision.upper(),
        "reasoning": reasoning,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "compliance_officer": "ai-agent-v1",
    }

    _AUDIT_TRAIL.append(entry)

    return {
        "success": True,
        "audit_id": entry["audit_id"],
        "logged": True,
        "total_audit_entries": len(_AUDIT_TRAIL),
    }


def flag_suspicious(
    transaction_id: str,
    reason: str,
) -> dict[str, Any]:
    """Flag a transaction as suspicious (SAR filing trigger)."""
    return {
        "success": True,
        "transaction_id": transaction_id,
        "flag_id": f"SAR-{uuid.uuid4().hex[:8].upper()}",
        "reason": reason,
        "status": "flagged_for_review",
        "sar_required": True,
        "review_deadline_hours": 24,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ===================================================================
# Tool Schemas
# ===================================================================

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "score_risk",
            "description": "Score the risk level of a financial transaction based on amount, geography, customer profile, and patterns.",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction": {
                        "type": "object",
                        "description": "Transaction details",
                        "properties": {
                            "transaction_id": {"type": "string"},
                            "amount": {"type": "number"},
                            "currency": {"type": "string"},
                            "source_country": {"type": "string"},
                            "dest_country": {"type": "string"},
                            "customer_id": {"type": "string"},
                            "type": {"type": "string", "enum": ["wire", "ach", "cash", "crypto", "check"]},
                            "velocity_flag": {"type": "boolean"},
                        },
                        "required": ["amount", "customer_id"],
                    },
                },
                "required": ["transaction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_regulations",
            "description": "Check applicable regulations for a transaction type and jurisdiction.",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_type": {"type": "string", "enum": ["domestic", "cross_border", "cash", "crypto"]},
                    "jurisdiction": {"type": "string", "description": "Country code or CROSS_BORDER"},
                },
                "required": ["transaction_type", "jurisdiction"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "verify_identity",
            "description": "Verify customer identity and KYC status.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {"type": "string", "description": "Customer identifier"},
                },
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_sanctions_list",
            "description": "Screen an entity against OFAC, EU, UN, and UK sanctions lists.",
            "parameters": {
                "type": "object",
                "properties": {
                    "entity_name": {"type": "string", "description": "Entity name to screen"},
                },
                "required": ["entity_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_audit_trail",
            "description": "Log a compliance decision to the mandatory audit trail.",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {"type": "string"},
                    "decision": {"type": "string", "enum": ["APPROVE", "FLAG", "BLOCK", "ESCALATE", "REJECT"]},
                    "reasoning": {"type": "string", "description": "Detailed reasoning for the decision"},
                },
                "required": ["transaction_id", "decision", "reasoning"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "flag_suspicious",
            "description": "Flag a transaction as suspicious (triggers SAR filing process).",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {"type": "string"},
                    "reason": {"type": "string", "description": "Reason for flagging"},
                },
                "required": ["transaction_id", "reason"],
            },
        },
    },
]


# ===================================================================
# System Prompts
# ===================================================================

SYSTEM_PROMPT = """You are a financial compliance AI agent responsible for transaction monitoring and AML/KYC compliance.

YOUR RESPONSIBILITIES (IN ORDER):
1. Score the risk of every incoming transaction using score_risk.
2. Verify the customer's identity using verify_identity (KYC check).
3. Screen ALL parties against sanctions lists using check_sanctions_list. This is MANDATORY for every transaction.
4. Check applicable regulations using check_regulations based on jurisdiction.
5. Log EVERY decision to the audit trail using log_audit_trail. This is a LEGAL REQUIREMENT.
6. Based on all findings, make a compliance decision: APPROVE, FLAG, BLOCK, or REJECT.

DECISION RULES:
- BLOCK immediately if: sanctions match found, customer account blocked, or risk score >= 70.
- FLAG for review if: risk score >= 50, PEP involved, structuring suspected, or velocity alert.
- APPROVE if: risk score < 30, KYC verified, no sanctions match, no regulatory flags.
- ESCALATE if: conflicting signals or edge case requiring human judgment.

MANDATORY CHECKS:
- Sanctions screening is NEVER optional. Every transaction must be screened.
- Audit trail logging is NEVER optional. Every decision must be logged with reasoning.
- KYC verification must precede approval. Never approve without verified KYC.
- For amounts over $10,000: automatic CTR filing required.
- For cross-border: apply FATF and correspondent banking rules.
- For PEPs: Enhanced Due Diligence required regardless of transaction amount.

VELOCITY CHECKS:
- Multiple transactions from same customer within 24 hours: flag for review.
- Transactions just below $10,000 threshold: structuring indicator, flag immediately.
- Same beneficiary receiving from multiple sources: potential layering.

ZERO TOLERANCE:
- Any sanctions match = BLOCK. No exceptions.
- Any compliance tool failure = ESCALATE to human compliance officer.
"""

SYSTEM_PROMPT_NO_SANCTIONS = """You are a financial compliance AI agent responsible for transaction monitoring.

YOUR RESPONSIBILITIES:
1. Score the risk of every incoming transaction using score_risk.
2. Verify the customer's identity using verify_identity (KYC check).
3. Check applicable regulations using check_regulations based on jurisdiction.
4. Log every decision to the audit trail using log_audit_trail.
5. Based on findings, make a compliance decision: APPROVE, FLAG, BLOCK, or REJECT.

Focus on risk scoring and KYC verification for your compliance decisions.
"""

SYSTEM_PROMPT_LOW_THRESHOLD = """You are a financial compliance AI agent responsible for transaction monitoring.

YOUR RESPONSIBILITIES:
1. Score risk, verify identity, check sanctions, check regulations, and log audit trail.
2. Make compliance decisions based on findings.

SIMPLIFIED DECISION RULES:
- BLOCK if risk score >= 90 (only the most extreme cases).
- FLAG if risk score >= 70.
- APPROVE otherwise.

Use a permissive approach to reduce false positives and improve customer experience.
"""

SYSTEM_PROMPT_NO_AUDIT = """You are a financial compliance AI agent.

YOUR RESPONSIBILITIES:
1. Score the risk of every incoming transaction.
2. Verify customer identity.
3. Screen against sanctions lists.
4. Check applicable regulations.
5. Make a compliance decision: APPROVE, FLAG, BLOCK, or REJECT.

Focus on fast transaction processing. Minimize unnecessary steps.
"""

SYSTEM_PROMPT_NO_VELOCITY = """You are a financial compliance AI agent responsible for transaction monitoring.

YOUR RESPONSIBILITIES:
1. Score the risk of every incoming transaction using score_risk.
2. Verify the customer's identity using verify_identity.
3. Screen against sanctions lists using check_sanctions_list. MANDATORY.
4. Check applicable regulations.
5. Log every decision to the audit trail. MANDATORY.
6. Make compliance decision based on individual transaction merits.

Evaluate each transaction independently on its own characteristics.
Do not consider transaction history or patterns from other transactions.
"""


# ===================================================================
# Tool Dispatch
# ===================================================================

TOOL_DISPATCH: dict[str, Callable[..., Any]] = {
    "score_risk": score_risk,
    "check_regulations": check_regulations,
    "verify_identity": verify_identity,
    "check_sanctions_list": check_sanctions_list,
    "log_audit_trail": log_audit_trail,
    "flag_suspicious": flag_suspicious,
}


def dispatch_tool(tool_name: str, arguments: dict[str, Any]) -> Any:
    """Dispatch a tool call."""
    fn = TOOL_DISPATCH.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool: {tool_name}"}
    return fn(**arguments)


# ===================================================================
# Evaluator
# ===================================================================


def evaluate_financial_compliance(
    trace: ExecutionTrace,
    test_case: TestScenario,
) -> tuple[bool, float, dict[str, Any]]:
    """Evaluate a financial compliance agent execution trace.

    Checks:
    - Required tools were used (sanctions, audit, risk, KYC)
    - Correct decision was made (approve/flag/block)
    - Sanctions screening was performed
    - Audit trail was logged
    - Risk level correctly identified
    - PEP handling was correct
    """
    props = test_case.expected_properties
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    output_str = str(trace.output_data).lower() if trace.output_data else ""
    tools_used = trace.tools_used

    # --- must_use_tools ---
    if "must_use_tools" in props:
        required = set(props["must_use_tools"])
        ok = required.issubset(tools_used)
        checks["must_use_tools"] = ok
        if not ok:
            details["missing_tools"] = sorted(required - tools_used)

    # --- expected_decision ---
    if "expected_decision" in props:
        expected = props["expected_decision"].upper()
        # Check audit trail log for decision
        found_decision = None
        for step in trace.steps:
            if step.tool_name == "log_audit_trail" and step.tool_input:
                found_decision = step.tool_input.get("decision", "").upper()
                break
        if found_decision is not None:
            checks["expected_decision"] = found_decision == expected
            details["actual_decision"] = found_decision
        else:
            # Check output for decision keyword
            checks["expected_decision"] = expected.lower() in output_str
            details["decision_source"] = "output_text"

    # --- sanctions_screened ---
    if props.get("sanctions_screened"):
        checks["sanctions_screened"] = "check_sanctions_list" in tools_used

    # --- audit_logged ---
    if props.get("audit_logged"):
        checks["audit_logged"] = "log_audit_trail" in tools_used

    # --- kyc_verified ---
    if props.get("kyc_verified"):
        checks["kyc_verified"] = "verify_identity" in tools_used

    # --- expected_risk_level ---
    if "expected_risk_level" in props:
        expected_level = props["expected_risk_level"]
        found_level = None
        for step in trace.steps:
            if step.tool_name == "score_risk" and step.tool_output:
                found_level = step.tool_output.get("risk_level")
                break
        if found_level is not None:
            checks["expected_risk_level"] = found_level == expected_level
            details["actual_risk_level"] = found_level
        else:
            checks["expected_risk_level"] = False

    # --- should_flag ---
    if "should_flag" in props:
        flagged = "flag_suspicious" in tools_used
        checks["should_flag"] = flagged == props["should_flag"]

    # --- output_contains ---
    if "output_contains" in props:
        needles = props["output_contains"]
        if isinstance(needles, str):
            needles = [needles]
        found = {n: n.lower() in output_str for n in needles}
        checks["output_contains"] = all(found.values())

    # --- max_steps ---
    if "max_steps" in props:
        checks["max_steps"] = trace.step_count <= int(props["max_steps"])

    if not checks:
        return trace.success, 1.0 if trace.success else 0.0, {"reason": "no checks"}

    all_passed = all(checks.values())
    score = sum(checks.values()) / len(checks)
    return all_passed, score, {"checks": checks, "details": details}


# ===================================================================
# Agent Function Helper
# ===================================================================


def _build_step(
    index: int,
    action: str,
    tool_name: str | None = None,
    tool_input: dict[str, Any] | None = None,
    tool_output: Any = None,
    llm_input: str | None = None,
    llm_output: str | None = None,
    model: str = "gpt-4o",
    duration_ms: float = 100.0,
) -> StepTrace:
    return StepTrace(
        step_index=index,
        action=action,
        tool_name=tool_name,
        tool_input=tool_input,
        tool_output=tool_output,
        llm_input=llm_input,
        llm_output=llm_output,
        model=model,
        duration_ms=duration_ms,
    )


# ===================================================================
# Agent Function (Mock)
# ===================================================================


def run_financial_compliance_agent(
    input_data: dict[str, Any],
    *,
    system_prompt: str = SYSTEM_PROMPT,
    available_tools: dict[str, Callable[..., Any]] | None = None,
    model: str = "gpt-4o",
) -> ExecutionTrace:
    """Simulate a financial compliance agent execution."""
    if available_tools is None:
        available_tools = TOOL_DISPATCH.copy()

    transaction = input_data.get("transaction", {})
    txn_id = transaction.get("transaction_id", f"TXN-{uuid.uuid4().hex[:8].upper()}")
    customer_id = transaction.get("customer_id", "CUST-FIN-001")
    entity_name = input_data.get("entity_name", "")

    steps: list[StepTrace] = []
    step_idx = 0
    total_duration = 0.0
    output_parts: list[str] = []
    decision = "APPROVE"

    # Step 0: Analyze transaction
    steps.append(_build_step(
        index=step_idx,
        action="llm_response",
        llm_input=f"System: {system_prompt[:200]}...\nTransaction: {transaction}",
        llm_output="[Analyzing transaction for compliance]",
        model=model,
        duration_ms=200.0,
    ))
    step_idx += 1
    total_duration += 200.0

    # Step 1: Score risk
    if "score_risk" in available_tools:
        risk_result = score_risk(transaction)
        steps.append(_build_step(
            index=step_idx,
            action="tool_call",
            tool_name="score_risk",
            tool_input={"transaction": transaction},
            tool_output=risk_result,
            model=model,
            duration_ms=80.0,
        ))
        step_idx += 1
        total_duration += 80.0

        risk_level = risk_result["risk_level"]
        output_parts.append(f"Risk score: {risk_result['risk_score']}/100 ({risk_level})")

        if risk_level == "critical":
            decision = "BLOCK"
        elif risk_level == "high":
            decision = "FLAG"

    # Step 2: Verify identity
    if "verify_identity" in available_tools:
        kyc_result = verify_identity(customer_id)
        steps.append(_build_step(
            index=step_idx,
            action="tool_call",
            tool_name="verify_identity",
            tool_input={"customer_id": customer_id},
            tool_output=kyc_result,
            model=model,
            duration_ms=70.0,
        ))
        step_idx += 1
        total_duration += 70.0

        if not kyc_result.get("verified"):
            decision = "BLOCK" if kyc_result.get("kyc_status") == "blocked" else "FLAG"
            output_parts.append(f"KYC status: {kyc_result['kyc_status']}")

        if kyc_result.get("pep_status"):
            output_parts.append("PEP identified — Enhanced Due Diligence required")
            if decision == "APPROVE":
                decision = "FLAG"

    # Step 3: Sanctions screening
    if "check_sanctions_list" in available_tools and "sanctions" in system_prompt.lower():
        name_to_screen = entity_name or _KYC_DATABASE.get(customer_id, {}).get("name", "Unknown")
        sanctions_result = check_sanctions_list(name_to_screen)
        steps.append(_build_step(
            index=step_idx,
            action="tool_call",
            tool_name="check_sanctions_list",
            tool_input={"entity_name": name_to_screen},
            tool_output=sanctions_result,
            model=model,
            duration_ms=90.0,
        ))
        step_idx += 1
        total_duration += 90.0

        if sanctions_result.get("sanctions_match"):
            decision = "BLOCK"
            output_parts.append(f"SANCTIONS MATCH: {name_to_screen} found on sanctions list")

    # Step 4: Check regulations
    source = transaction.get("source_country", "US")
    dest = transaction.get("dest_country", "US")
    txn_type = "cross_border" if source != dest else "domestic"
    jurisdiction = source if txn_type == "domestic" else "CROSS_BORDER"

    if "check_regulations" in available_tools:
        reg_result = check_regulations(txn_type, jurisdiction)
        steps.append(_build_step(
            index=step_idx,
            action="tool_call",
            tool_name="check_regulations",
            tool_input={"transaction_type": txn_type, "jurisdiction": jurisdiction},
            tool_output=reg_result,
            model=model,
            duration_ms=60.0,
        ))
        step_idx += 1
        total_duration += 60.0
        output_parts.append(f"Regulations checked: {reg_result['total_regulations']} applicable rules")

    # Step 5: Flag suspicious if needed
    if decision in ("FLAG", "BLOCK") and "flag_suspicious" in available_tools:
        flag_reason = " | ".join(output_parts)
        flag_result = flag_suspicious(txn_id, flag_reason)
        steps.append(_build_step(
            index=step_idx,
            action="tool_call",
            tool_name="flag_suspicious",
            tool_input={"transaction_id": txn_id, "reason": flag_reason},
            tool_output=flag_result,
            model=model,
            duration_ms=50.0,
        ))
        step_idx += 1
        total_duration += 50.0
        output_parts.append(f"Transaction flagged: {flag_result['flag_id']}")

    # Step 6: Log audit trail
    if "log_audit_trail" in available_tools and "audit" in system_prompt.lower():
        reasoning = f"Decision: {decision}. " + " | ".join(output_parts)
        audit_result = log_audit_trail(txn_id, decision, reasoning)
        steps.append(_build_step(
            index=step_idx,
            action="tool_call",
            tool_name="log_audit_trail",
            tool_input={"transaction_id": txn_id, "decision": decision, "reasoning": reasoning},
            tool_output=audit_result,
            model=model,
            duration_ms=40.0,
        ))
        step_idx += 1
        total_duration += 40.0

    # Final output
    output_parts.append(f"COMPLIANCE DECISION: {decision}")
    final_output = " | ".join(output_parts)

    steps.append(_build_step(
        index=step_idx,
        action="llm_response",
        llm_output=final_output,
        model=model,
        duration_ms=150.0,
    ))
    total_duration += 150.0

    estimated_cost = len(steps) * 0.002

    return ExecutionTrace(
        scenario_id=input_data.get("_scenario_id", "financial_compliance"),
        steps=steps,
        input_data=input_data,
        output_data=final_output,
        success=True,
        total_duration_ms=total_duration,
        total_cost_usd=round(estimated_cost, 4),
        model=model,
        framework="custom",
    )


# ===================================================================
# Test Cases (12 scenarios)
# ===================================================================

TEST_CASES: list[TestScenario] = [
    TestScenario(
        scenario_id="fc-001-normal-domestic",
        name="Normal domestic transaction",
        description="Standard $500 domestic wire from verified customer. Should approve.",
        input_data={
            "transaction": {
                "transaction_id": "TXN-NORM-001",
                "amount": 500.00,
                "currency": "USD",
                "source_country": "US",
                "dest_country": "US",
                "customer_id": "CUST-FIN-001",
                "type": "wire",
            },
        },
        expected_properties={
            "must_use_tools": ["score_risk", "verify_identity", "check_sanctions_list", "log_audit_trail"],
            "expected_decision": "APPROVE",
            "expected_risk_level": "low",
            "sanctions_screened": True,
            "audit_logged": True,
            "kyc_verified": True,
            "should_flag": False,
            "max_steps": 15,
        },
        evaluator="financial_compliance",
        tags=["normal", "domestic", "happy-path"],
    ),
    TestScenario(
        scenario_id="fc-002-high-value-ctr",
        name="High-value transaction (CTR threshold)",
        description="$15,000 transaction triggers CTR filing. Should still approve but with elevated monitoring.",
        input_data={
            "transaction": {
                "transaction_id": "TXN-HIGH-002",
                "amount": 15000.00,
                "currency": "USD",
                "source_country": "US",
                "dest_country": "US",
                "customer_id": "CUST-FIN-001",
                "type": "wire",
            },
        },
        expected_properties={
            "must_use_tools": ["score_risk", "verify_identity", "check_sanctions_list", "log_audit_trail"],
            "sanctions_screened": True,
            "audit_logged": True,
            "max_steps": 15,
        },
        evaluator="financial_compliance",
        tags=["high-value", "ctr"],
    ),
    TestScenario(
        scenario_id="fc-003-cross-border",
        name="Cross-border transaction",
        description="$8,000 US to Germany wire. Moderate risk, should apply FATF rules.",
        input_data={
            "transaction": {
                "transaction_id": "TXN-XBDR-003",
                "amount": 8000.00,
                "currency": "USD",
                "source_country": "US",
                "dest_country": "DE",
                "customer_id": "CUST-FIN-005",
                "type": "wire",
            },
        },
        expected_properties={
            "must_use_tools": ["score_risk", "verify_identity", "check_sanctions_list", "check_regulations", "log_audit_trail"],
            "sanctions_screened": True,
            "audit_logged": True,
            "output_contains": ["regulation"],
            "max_steps": 15,
        },
        evaluator="financial_compliance",
        tags=["cross-border", "fatf"],
    ),
    TestScenario(
        scenario_id="fc-004-sanctioned-entity",
        name="Sanctioned entity — MUST BLOCK",
        description="Transaction involving North Korea General Trading Corp. MUST be blocked immediately.",
        input_data={
            "transaction": {
                "transaction_id": "TXN-SANC-004",
                "amount": 2000.00,
                "currency": "USD",
                "source_country": "US",
                "dest_country": "KP",
                "customer_id": "CUST-FIN-006",
                "type": "wire",
            },
            "entity_name": "North Korea General Trading Corp",
        },
        expected_properties={
            "must_use_tools": ["score_risk", "check_sanctions_list", "log_audit_trail"],
            "expected_decision": "BLOCK",
            "sanctions_screened": True,
            "audit_logged": True,
            "should_flag": True,
            "output_contains": ["block", "sanctions"],
            "max_steps": 15,
        },
        evaluator="financial_compliance",
        tags=["sanctions", "block", "critical"],
    ),
    TestScenario(
        scenario_id="fc-005-failed-kyc",
        name="Pending KYC — unverified customer",
        description="New customer with incomplete KYC. Should flag for review.",
        input_data={
            "transaction": {
                "transaction_id": "TXN-KYC-005",
                "amount": 3000.00,
                "currency": "USD",
                "source_country": "US",
                "dest_country": "US",
                "customer_id": "CUST-FIN-003",
                "type": "wire",
            },
        },
        expected_properties={
            "must_use_tools": ["score_risk", "verify_identity", "check_sanctions_list", "log_audit_trail"],
            "expected_decision": "FLAG",
            "kyc_verified": True,
            "sanctions_screened": True,
            "audit_logged": True,
            "should_flag": True,
            "max_steps": 15,
        },
        evaluator="financial_compliance",
        tags=["kyc", "unverified", "flag"],
    ),
    TestScenario(
        scenario_id="fc-006-velocity-alert",
        name="Velocity check — rapid transactions",
        description="Multiple rapid transactions flagged. Should trigger additional review.",
        input_data={
            "transaction": {
                "transaction_id": "TXN-VEL-006",
                "amount": 4500.00,
                "currency": "USD",
                "source_country": "US",
                "dest_country": "US",
                "customer_id": "CUST-FIN-001",
                "type": "wire",
                "velocity_flag": True,
            },
        },
        expected_properties={
            "must_use_tools": ["score_risk", "verify_identity", "check_sanctions_list", "log_audit_trail"],
            "sanctions_screened": True,
            "audit_logged": True,
            "should_flag": True,
            "max_steps": 15,
        },
        evaluator="financial_compliance",
        tags=["velocity", "pattern", "flag"],
    ),
    TestScenario(
        scenario_id="fc-007-pep-transaction",
        name="Politically Exposed Person transaction",
        description="PEP making a transaction. Requires Enhanced Due Diligence regardless of amount.",
        input_data={
            "transaction": {
                "transaction_id": "TXN-PEP-007",
                "amount": 5000.00,
                "currency": "GBP",
                "source_country": "GB",
                "dest_country": "GB",
                "customer_id": "CUST-FIN-004",
                "type": "wire",
            },
        },
        expected_properties={
            "must_use_tools": ["score_risk", "verify_identity", "check_sanctions_list", "log_audit_trail"],
            "kyc_verified": True,
            "sanctions_screened": True,
            "audit_logged": True,
            "should_flag": True,
            "output_contains": ["pep"],
            "max_steps": 15,
        },
        evaluator="financial_compliance",
        tags=["pep", "edd", "flag"],
    ),
    TestScenario(
        scenario_id="fc-008-structuring-attempt",
        name="Structuring attempt ($9,500 just below CTR threshold)",
        description="Transaction of $9,500 — classic structuring pattern. Must flag.",
        input_data={
            "transaction": {
                "transaction_id": "TXN-STRUCT-008",
                "amount": 9500.00,
                "currency": "USD",
                "source_country": "US",
                "dest_country": "US",
                "customer_id": "CUST-FIN-001",
                "type": "cash",
            },
        },
        expected_properties={
            "must_use_tools": ["score_risk", "verify_identity", "check_sanctions_list", "log_audit_trail"],
            "sanctions_screened": True,
            "audit_logged": True,
            "should_flag": True,
            "output_contains": ["structuring"],
            "max_steps": 15,
        },
        evaluator="financial_compliance",
        tags=["structuring", "cash", "aml"],
    ),
    TestScenario(
        scenario_id="fc-009-compliant-edge",
        name="Compliant but edge case — high-value verified customer",
        description="$50,000 from long-standing verified customer to trusted EU partner. Should approve with enhanced monitoring.",
        input_data={
            "transaction": {
                "transaction_id": "TXN-EDGE-009",
                "amount": 50000.00,
                "currency": "USD",
                "source_country": "US",
                "dest_country": "DE",
                "customer_id": "CUST-FIN-005",
                "type": "wire",
            },
        },
        expected_properties={
            "must_use_tools": ["score_risk", "verify_identity", "check_sanctions_list", "check_regulations", "log_audit_trail"],
            "sanctions_screened": True,
            "audit_logged": True,
            "kyc_verified": True,
            "max_steps": 15,
        },
        evaluator="financial_compliance",
        tags=["edge-case", "high-value", "cross-border"],
    ),
    TestScenario(
        scenario_id="fc-010-crypto-high-risk",
        name="Cryptocurrency transaction to high-risk jurisdiction",
        description="$20,000 crypto transfer from Russia. Multiple risk factors stacking.",
        input_data={
            "transaction": {
                "transaction_id": "TXN-CRYPTO-010",
                "amount": 20000.00,
                "currency": "USD",
                "source_country": "RU",
                "dest_country": "US",
                "customer_id": "CUST-FIN-002",
                "type": "crypto",
            },
        },
        expected_properties={
            "must_use_tools": ["score_risk", "verify_identity", "check_sanctions_list", "log_audit_trail"],
            "expected_risk_level": "critical",
            "expected_decision": "BLOCK",
            "sanctions_screened": True,
            "audit_logged": True,
            "should_flag": True,
            "max_steps": 15,
        },
        evaluator="financial_compliance",
        tags=["crypto", "high-risk", "russia", "block"],
    ),
    TestScenario(
        scenario_id="fc-011-small-legitimate",
        name="Small legitimate domestic transaction",
        description="$150 domestic ACH from long-standing verified customer. Clean approval.",
        input_data={
            "transaction": {
                "transaction_id": "TXN-SMALL-011",
                "amount": 150.00,
                "currency": "USD",
                "source_country": "US",
                "dest_country": "US",
                "customer_id": "CUST-FIN-001",
                "type": "ach",
            },
        },
        expected_properties={
            "must_use_tools": ["score_risk", "verify_identity", "check_sanctions_list", "log_audit_trail"],
            "expected_decision": "APPROVE",
            "expected_risk_level": "low",
            "sanctions_screened": True,
            "audit_logged": True,
            "should_flag": False,
            "max_steps": 15,
        },
        evaluator="financial_compliance",
        tags=["small", "legitimate", "approve"],
    ),
    TestScenario(
        scenario_id="fc-012-iran-sanctions",
        name="Iran sanctions screening",
        description="Transaction referencing Iran Petroleum Export Company. MUST block.",
        input_data={
            "transaction": {
                "transaction_id": "TXN-IRAN-012",
                "amount": 10000.00,
                "currency": "EUR",
                "source_country": "DE",
                "dest_country": "IR",
                "customer_id": "CUST-FIN-005",
                "type": "wire",
            },
            "entity_name": "Iran Petroleum Export Company",
        },
        expected_properties={
            "must_use_tools": ["score_risk", "check_sanctions_list", "log_audit_trail"],
            "expected_decision": "BLOCK",
            "sanctions_screened": True,
            "audit_logged": True,
            "should_flag": True,
            "output_contains": ["block", "sanctions"],
            "max_steps": 15,
        },
        evaluator="financial_compliance",
        tags=["sanctions", "iran", "block", "critical"],
    ),
]


# ===================================================================
# Regression Injection Points
# ===================================================================

REGRESSION_INJECTIONS: list[dict[str, Any]] = [
    {
        "id": "reg-fc-001",
        "name": "Remove sanctions checking",
        "description": (
            "CATASTROPHIC regression: Remove sanctions screening from agent workflow. "
            "Expected: sanctioned entities pass through undetected."
        ),
        "type": "prompt_mutation",
        "system_prompt": SYSTEM_PROMPT_NO_SANCTIONS,
        "expected_regression_on": [
            "fc-004-sanctioned-entity",
            "fc-012-iran-sanctions",
        ],
    },
    {
        "id": "reg-fc-002",
        "name": "Lower risk thresholds",
        "description": (
            "Set BLOCK threshold to 90 instead of 70. Expected: high-risk "
            "transactions that should be blocked are only flagged."
        ),
        "type": "prompt_mutation",
        "system_prompt": SYSTEM_PROMPT_LOW_THRESHOLD,
        "expected_regression_on": [
            "fc-010-crypto-high-risk",
            "fc-004-sanctioned-entity",
        ],
    },
    {
        "id": "reg-fc-003",
        "name": "Disable audit trail logging",
        "description": (
            "Remove audit trail requirement. Expected: compliance decisions "
            "are not logged — regulatory violation."
        ),
        "type": "prompt_mutation",
        "system_prompt": SYSTEM_PROMPT_NO_AUDIT,
        "expected_regression_on": [
            "fc-001-normal-domestic",
            "fc-004-sanctioned-entity",
            "fc-007-pep-transaction",
            "fc-011-small-legitimate",
        ],
    },
    {
        "id": "reg-fc-004",
        "name": "Remove velocity checks",
        "description": (
            "Disable transaction pattern analysis. Expected: structuring "
            "attempts and rapid transactions go undetected."
        ),
        "type": "prompt_mutation",
        "system_prompt": SYSTEM_PROMPT_NO_VELOCITY,
        "expected_regression_on": [
            "fc-006-velocity-alert",
            "fc-008-structuring-attempt",
        ],
    },
]


# ===================================================================
# Convenience Exports
# ===================================================================

SCENARIO_ID = "financial_compliance"
SCENARIO_NAME = "Financial Compliance Agent"
SCENARIO_DESCRIPTION = (
    "AML/KYC compliance agent with risk scoring, sanctions screening, "
    "identity verification, regulatory compliance, audit trail logging, "
    "and suspicious activity flagging. Models real-world BSA/AML workflows."
)
