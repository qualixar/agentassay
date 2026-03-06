"""Customer support agent scenario for AgentAssay experiments.

Enterprise-grade customer support ticket handling agent. Models a
Tier-1 support workflow: intake, classification, knowledge base lookup,
troubleshooting, SLA monitoring, escalation, and resolution.

Agent workflow:
    1. Customer submits a support ticket (description + context)
    2. Agent classifies ticket (category + priority)
    3. Agent searches knowledge base for solutions
    4. Agent checks order history for context
    5. Agent monitors SLA compliance
    6. Agent resolves or escalates based on complexity
    7. Agent sends response to customer

Regression injection points:
    - Remove classification rules (misroutes tickets)
    - Disable SLA checking (misses urgent deadlines)
    - Corrupt knowledge base (returns irrelevant articles)
    - Remove escalation logic (attempts to resolve all, even critical)
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
# Knowledge Base
# ===================================================================

_KNOWLEDGE_BASE: list[dict[str, Any]] = [
    {
        "article_id": "KB-001",
        "title": "How to Track Your Order",
        "category": "shipping",
        "content": (
            "To track your order: 1) Log in to your account. 2) Go to 'My Orders'. "
            "3) Click the tracking number link. Standard orders take 3-5 business days. "
            "Express orders take 1-2 business days."
        ),
        "tags": ["tracking", "shipping", "order status"],
    },
    {
        "article_id": "KB-002",
        "title": "Return and Refund Policy",
        "category": "returns",
        "content": (
            "Items can be returned within 30 days of delivery. Items must be unworn with "
            "original tags. Refunds are processed within 3-5 business days. Electronics "
            "have a 15-day return window."
        ),
        "tags": ["return", "refund", "policy"],
    },
    {
        "article_id": "KB-003",
        "title": "Payment Issues Troubleshooting",
        "category": "billing",
        "content": (
            "Common payment issues: 1) Card declined — check funds and expiry date. "
            "2) Double charge — usually pending authorizations that clear in 24-48 hours. "
            "3) Promo code not working — check validity date and minimum order amount. "
            "Contact billing support for unresolved issues."
        ),
        "tags": ["payment", "billing", "card", "promo"],
    },
    {
        "article_id": "KB-004",
        "title": "Account Access Problems",
        "category": "account",
        "content": (
            "If you cannot access your account: 1) Use 'Forgot Password' to reset. "
            "2) Check spam folder for reset email. 3) Ensure you're using the correct email. "
            "4) If locked out after 5 failed attempts, wait 30 minutes or contact support."
        ),
        "tags": ["account", "login", "password", "access"],
    },
    {
        "article_id": "KB-005",
        "title": "Size Guide and Fit Information",
        "category": "product",
        "content": (
            "Our size chart is available on each product page. Key tips: "
            "1) Measure yourself according to our guide. 2) Nike runs 0.5 size small. "
            "3) Levi's jeans: waist measurement in inches. 4) When between sizes, size up."
        ),
        "tags": ["size", "fit", "sizing", "guide"],
    },
    {
        "article_id": "KB-006",
        "title": "Damaged or Defective Items",
        "category": "quality",
        "content": (
            "If you received a damaged or defective item: 1) Take photos of the damage. "
            "2) Contact support within 48 hours. 3) We will ship a replacement at no cost. "
            "4) No need to return the damaged item for orders under $50."
        ),
        "tags": ["damaged", "defective", "quality", "replacement"],
    },
    {
        "article_id": "KB-007",
        "title": "Subscription and Membership Benefits",
        "category": "loyalty",
        "content": (
            "Membership tiers: Standard (free), Silver (5% off), Gold (10% off), "
            "Platinum (15% off), VIP (20% off). Benefits include early access to sales, "
            "free express shipping (Gold+), and birthday rewards."
        ),
        "tags": ["membership", "loyalty", "tier", "benefits", "subscription"],
    },
    {
        "article_id": "KB-008",
        "title": "International Shipping",
        "category": "shipping",
        "content": (
            "We ship to 40+ countries. International orders: 7-14 business days. "
            "Customs duties are the buyer's responsibility. Tracking available for all "
            "international shipments. Free international shipping on orders over $150."
        ),
        "tags": ["international", "shipping", "customs", "global"],
    },
]

_CORRUPTED_KNOWLEDGE_BASE: list[dict[str, Any]] = [
    {
        "article_id": "KB-CORRUPT-001",
        "title": "How to Bake a Perfect Sourdough",
        "category": "cooking",
        "content": "Mix flour, water, and starter. Let rise for 12 hours. Bake at 450F.",
        "tags": ["baking", "sourdough"],
    },
    {
        "article_id": "KB-CORRUPT-002",
        "title": "Tips for Growing Tomatoes",
        "category": "gardening",
        "content": "Plant in full sun. Water regularly. Use tomato cages for support.",
        "tags": ["gardening", "tomatoes"],
    },
]


# ===================================================================
# Order History Data
# ===================================================================

_ORDER_HISTORY: dict[str, list[dict[str, Any]]] = {
    "CUST-1001": [
        {
            "order_id": "ORD-2026-A001",
            "date": "2026-02-15",
            "status": "delivered",
            "items": [{"name": "Nike Air Force 1", "quantity": 1, "price": 110.00}],
            "total": 118.80,
            "tracking": "1Z999AA10123456784",
        },
        {
            "order_id": "ORD-2026-A002",
            "date": "2026-02-20",
            "status": "shipped",
            "items": [{"name": "Levi's 501 Jeans", "quantity": 2, "price": 69.50}],
            "total": 150.12,
            "tracking": "1Z999AA10123456785",
        },
    ],
    "CUST-1002": [
        {
            "order_id": "ORD-2026-B001",
            "date": "2026-01-10",
            "status": "delivered",
            "items": [{"name": "Sony WH-1000XM6", "quantity": 1, "price": 399.99}],
            "total": 431.99,
            "tracking": "1Z999AA10123456786",
        },
    ],
    "CUST-1003": [],
    "CUST-1004": [
        {
            "order_id": "ORD-2026-D001",
            "date": "2026-02-25",
            "status": "processing",
            "items": [
                {"name": "Samsung Galaxy S25 Ultra", "quantity": 1, "price": 1299.99},
                {"name": "Apple Watch Series 10", "quantity": 1, "price": 429.00},
            ],
            "total": 1867.31,
            "tracking": None,
        },
    ],
}


# ===================================================================
# SLA Configuration
# ===================================================================

_SLA_CONFIG: dict[str, dict[str, Any]] = {
    "critical": {"response_time_minutes": 15, "resolution_time_hours": 4},
    "high": {"response_time_minutes": 30, "resolution_time_hours": 8},
    "medium": {"response_time_minutes": 120, "resolution_time_hours": 24},
    "low": {"response_time_minutes": 480, "resolution_time_hours": 72},
}


# ===================================================================
# Mock Tool Implementations
# ===================================================================


def classify_ticket(description: str) -> dict[str, Any]:
    """Classify a support ticket by category and priority.

    Uses keyword-based classification (mirrors ML classification models
    deployed in enterprise CRM systems like ServiceNow, Zendesk).
    """
    desc_lower = description.lower()

    # Priority classification
    priority = "medium"
    if any(kw in desc_lower for kw in ["urgent", "critical", "emergency", "down", "cannot access", "security breach"]):
        priority = "critical"
    elif any(kw in desc_lower for kw in ["angry", "frustrated", "unacceptable", "refund", "fraud", "charged twice"]):
        priority = "high"
    elif any(kw in desc_lower for kw in ["question", "wondering", "curious", "how to"]):
        priority = "low"

    # Category classification
    category = "general"
    if any(kw in desc_lower for kw in ["ship", "track", "delivery", "transit", "lost package"]):
        category = "shipping"
    elif any(kw in desc_lower for kw in ["return", "refund", "exchange", "send back"]):
        category = "returns"
    elif any(kw in desc_lower for kw in ["payment", "charge", "billing", "invoice", "promo", "coupon"]):
        category = "billing"
    elif any(kw in desc_lower for kw in ["login", "password", "account", "email", "access"]):
        category = "account"
    elif any(kw in desc_lower for kw in ["size", "fit", "wrong size", "too big", "too small"]):
        category = "product"
    elif any(kw in desc_lower for kw in ["damaged", "defective", "broken", "torn", "stain"]):
        category = "quality"
    elif any(kw in desc_lower for kw in ["membership", "loyalty", "tier", "points", "subscription"]):
        category = "loyalty"
    elif any(kw in desc_lower for kw in ["security", "hack", "unauthorized", "breach"]):
        category = "security"
        priority = "critical"

    ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"

    return {
        "ticket_id": ticket_id,
        "category": category,
        "priority": priority,
        "confidence": 0.92,
        "suggested_department": _DEPARTMENT_MAP.get(category, "general_support"),
    }


_DEPARTMENT_MAP: dict[str, str] = {
    "shipping": "logistics",
    "returns": "returns_team",
    "billing": "finance",
    "account": "it_support",
    "product": "product_team",
    "quality": "quality_assurance",
    "loyalty": "customer_success",
    "security": "security_ops",
    "general": "general_support",
}


def search_knowledge_base(
    query: str,
    kb: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Search knowledge base for relevant articles."""
    if kb is None:
        kb = _KNOWLEDGE_BASE

    query_lower = query.lower()
    results: list[dict[str, Any]] = []

    for article in kb:
        score = 0.0
        if query_lower in article["title"].lower():
            score += 10.0
        if query_lower in article["content"].lower():
            score += 5.0
        for tag in article["tags"]:
            if tag in query_lower or query_lower in tag:
                score += 3.0
        for token in query_lower.split():
            if len(token) >= 3:
                if token in article["content"].lower():
                    score += 1.0
                if token in article["title"].lower():
                    score += 2.0

        if score > 0:
            results.append({**article, "_relevance_score": score})

    results.sort(key=lambda x: x["_relevance_score"], reverse=True)

    return {
        "query": query,
        "total_results": len(results),
        "articles": results[:5],
    }


def lookup_order_history(customer_id: str) -> dict[str, Any]:
    """Look up order history for a customer."""
    if customer_id not in _ORDER_HISTORY:
        return {
            "customer_id": customer_id,
            "found": False,
            "error": f"Customer not found: {customer_id}",
        }

    orders = _ORDER_HISTORY[customer_id]
    return {
        "customer_id": customer_id,
        "found": True,
        "total_orders": len(orders),
        "orders": orders,
        "total_spent": sum(o["total"] for o in orders),
    }


def check_sla(ticket_id: str, priority: str) -> dict[str, Any]:
    """Check SLA status for a ticket based on priority."""
    sla = _SLA_CONFIG.get(priority, _SLA_CONFIG["medium"])

    return {
        "ticket_id": ticket_id,
        "priority": priority,
        "response_deadline_minutes": sla["response_time_minutes"],
        "resolution_deadline_hours": sla["resolution_time_hours"],
        "sla_status": "within_sla",
        "time_elapsed_minutes": 5,
        "urgency_note": (
            "IMMEDIATE RESPONSE REQUIRED" if priority == "critical"
            else "Standard response timeline"
        ),
    }


def escalate_ticket(
    ticket_id: str,
    reason: str,
    department: str,
) -> dict[str, Any]:
    """Escalate a ticket to a specialized department."""
    valid_departments = {
        "logistics", "returns_team", "finance", "it_support",
        "product_team", "quality_assurance", "customer_success",
        "security_ops", "general_support", "management",
    }

    if department not in valid_departments:
        return {
            "success": False,
            "error": f"Invalid department: {department}",
            "valid_departments": sorted(valid_departments),
        }

    return {
        "success": True,
        "ticket_id": ticket_id,
        "escalated_to": department,
        "reason": reason,
        "escalation_id": f"ESC-{uuid.uuid4().hex[:8].upper()}",
        "estimated_response": "30 minutes" if department == "security_ops" else "2 hours",
    }


def send_response(
    ticket_id: str,
    response: str,
) -> dict[str, Any]:
    """Send a response to the customer."""
    return {
        "success": True,
        "ticket_id": ticket_id,
        "response_sent": True,
        "response_length": len(response),
        "channel": "email",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# ===================================================================
# Tool Schemas (OpenAI Function Calling Format)
# ===================================================================

TOOL_SCHEMAS: list[dict[str, Any]] = [
    {
        "type": "function",
        "function": {
            "name": "classify_ticket",
            "description": "Classify a support ticket by category and priority level.",
            "parameters": {
                "type": "object",
                "properties": {
                    "description": {
                        "type": "string",
                        "description": "The customer's ticket description",
                    },
                },
                "required": ["description"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "search_knowledge_base",
            "description": "Search the knowledge base for articles matching a query.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query for knowledge base lookup",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "lookup_order_history",
            "description": "Look up a customer's order history by customer ID.",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_id": {
                        "type": "string",
                        "description": "Customer identifier",
                    },
                },
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_sla",
            "description": "Check SLA compliance status for a ticket.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "Ticket identifier",
                    },
                    "priority": {
                        "type": "string",
                        "description": "Ticket priority level",
                        "enum": ["critical", "high", "medium", "low"],
                    },
                },
                "required": ["ticket_id", "priority"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "escalate_ticket",
            "description": "Escalate a ticket to a specialized department.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "Ticket identifier",
                    },
                    "reason": {
                        "type": "string",
                        "description": "Reason for escalation",
                    },
                    "department": {
                        "type": "string",
                        "description": "Target department",
                        "enum": [
                            "logistics", "returns_team", "finance", "it_support",
                            "product_team", "quality_assurance", "customer_success",
                            "security_ops", "general_support", "management",
                        ],
                    },
                },
                "required": ["ticket_id", "reason", "department"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "send_response",
            "description": "Send a resolution response to the customer.",
            "parameters": {
                "type": "object",
                "properties": {
                    "ticket_id": {
                        "type": "string",
                        "description": "Ticket identifier",
                    },
                    "response": {
                        "type": "string",
                        "description": "Response message to send to the customer",
                    },
                },
                "required": ["ticket_id", "response"],
            },
        },
    },
]


# ===================================================================
# System Prompts
# ===================================================================

SYSTEM_PROMPT = """You are an enterprise Tier-1 customer support agent for a premium retail platform.

YOUR RESPONSIBILITIES:
1. ALWAYS classify incoming tickets first using classify_ticket. This determines routing and SLA.
2. Check SLA requirements immediately after classification. Critical tickets need response within 15 minutes.
3. Search the knowledge base for relevant solutions before responding.
4. Look up order history when the issue relates to a specific order.
5. Resolve the ticket if you can; escalate to the appropriate department if not.
6. Always send a response to the customer.

CLASSIFICATION RULES:
- Security issues (breach, hack, unauthorized access) are ALWAYS critical priority.
- Payment fraud / double charges are ALWAYS high priority.
- Lost packages during transit are high priority.
- General inquiries (how-to, policy questions) are low priority.

ESCALATION RULES:
- Escalate to security_ops for any security-related ticket.
- Escalate to finance for billing disputes over $100.
- Escalate to management for customer complaints mentioning "lawyer" or "lawsuit".
- Escalate to quality_assurance for defective/damaged items.
- Resolve directly for: order tracking, return policy questions, size/fit guidance, membership info.

SLA ENFORCEMENT:
- Critical: Respond within 15 minutes, resolve within 4 hours.
- High: Respond within 30 minutes, resolve within 8 hours.
- Medium: Respond within 2 hours, resolve within 24 hours.
- Low: Respond within 8 hours, resolve within 72 hours.

Always be empathetic, professional, and solution-oriented.
"""

SYSTEM_PROMPT_NO_CLASSIFICATION = """You are a customer support agent for a premium retail platform.

Help customers with their issues. Search the knowledge base for solutions.
Look up order history when relevant. Send a response to resolve the issue.
If you cannot resolve it, escalate to the appropriate department.

Be empathetic, professional, and solution-oriented.
"""

SYSTEM_PROMPT_NO_SLA = """You are an enterprise Tier-1 customer support agent for a premium retail platform.

YOUR RESPONSIBILITIES:
1. Classify incoming tickets using classify_ticket.
2. Search the knowledge base for relevant solutions before responding.
3. Look up order history when the issue relates to a specific order.
4. Resolve the ticket if you can; escalate to the appropriate department if not.
5. Always send a response to the customer.

CLASSIFICATION RULES:
- Security issues are ALWAYS critical priority.
- Payment fraud / double charges are ALWAYS high priority.
- General inquiries are low priority.

Always be empathetic, professional, and solution-oriented.
"""

SYSTEM_PROMPT_NO_ESCALATION = """You are an enterprise Tier-1 customer support agent for a premium retail platform.

YOUR RESPONSIBILITIES:
1. Classify incoming tickets using classify_ticket.
2. Check SLA requirements.
3. Search the knowledge base for relevant solutions.
4. RESOLVE ALL TICKETS YOURSELF. You have full authority to handle any issue directly.
5. Always send a response to the customer.

You can handle everything: billing, security, quality, shipping. No need to involve other teams.
"""


# ===================================================================
# Tool Dispatch
# ===================================================================

TOOL_DISPATCH: dict[str, Callable[..., Any]] = {
    "classify_ticket": classify_ticket,
    "search_knowledge_base": search_knowledge_base,
    "lookup_order_history": lookup_order_history,
    "check_sla": check_sla,
    "escalate_ticket": escalate_ticket,
    "send_response": send_response,
}


def dispatch_tool(tool_name: str, arguments: dict[str, Any]) -> Any:
    """Dispatch a tool call to the appropriate mock implementation."""
    fn = TOOL_DISPATCH.get(tool_name)
    if fn is None:
        return {"error": f"Unknown tool: {tool_name}"}
    return fn(**arguments)


# ===================================================================
# Evaluator Function
# ===================================================================


def evaluate_customer_support(
    trace: ExecutionTrace,
    test_case: TestScenario,
) -> tuple[bool, float, dict[str, Any]]:
    """Evaluate a customer support agent execution trace.

    Checks:
    - Ticket was classified (classify_ticket called)
    - SLA was checked (check_sla called)
    - Knowledge base was searched (search_knowledge_base called)
    - Correct escalation department (if escalation expected)
    - Response was sent (send_response called)
    - Correct priority detection
    - Output contains expected content
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

    # --- must_not_use_tools ---
    if "must_not_use_tools" in props:
        forbidden = set(props["must_not_use_tools"])
        ok = forbidden.isdisjoint(tools_used)
        checks["must_not_use_tools"] = ok

    # --- expected_priority ---
    if "expected_priority" in props:
        expected_pri = props["expected_priority"]
        # Check if classify_ticket was called and returned correct priority
        found_priority = None
        for step in trace.steps:
            if step.tool_name == "classify_ticket" and step.tool_output:
                found_priority = step.tool_output.get("priority")
                break
        if found_priority is not None:
            checks["expected_priority"] = found_priority == expected_pri
            details["detected_priority"] = found_priority
        else:
            checks["expected_priority"] = False
            details["classification_missing"] = True

    # --- expected_category ---
    if "expected_category" in props:
        expected_cat = props["expected_category"]
        found_category = None
        for step in trace.steps:
            if step.tool_name == "classify_ticket" and step.tool_output:
                found_category = step.tool_output.get("category")
                break
        if found_category is not None:
            checks["expected_category"] = found_category == expected_cat
            details["detected_category"] = found_category
        else:
            checks["expected_category"] = False

    # --- should_escalate ---
    if "should_escalate" in props:
        escalated = "escalate_ticket" in tools_used
        checks["should_escalate"] = escalated == props["should_escalate"]

    # --- escalation_department ---
    if "escalation_department" in props:
        expected_dept = props["escalation_department"]
        found_dept = None
        for step in trace.steps:
            if step.tool_name == "escalate_ticket" and step.tool_input:
                found_dept = step.tool_input.get("department")
                break
        if found_dept is not None:
            checks["escalation_department"] = found_dept == expected_dept
            details["escalation_department"] = found_dept
        else:
            checks["escalation_department"] = False

    # --- output_contains ---
    if "output_contains" in props:
        needles = props["output_contains"]
        if isinstance(needles, str):
            needles = [needles]
        found = {n: n.lower() in output_str for n in needles}
        checks["output_contains"] = all(found.values())
        details["output_contains_results"] = found

    # --- max_steps ---
    if "max_steps" in props:
        checks["max_steps"] = trace.step_count <= int(props["max_steps"])

    # --- sla_checked ---
    if props.get("sla_checked"):
        checks["sla_checked"] = "check_sla" in tools_used

    # --- Aggregate ---
    if not checks:
        return trace.success, 1.0 if trace.success else 0.0, {"reason": "no checks defined"}

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
    duration_ms: float = 120.0,
) -> StepTrace:
    """Build a StepTrace."""
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


def run_customer_support_agent(
    input_data: dict[str, Any],
    *,
    system_prompt: str = SYSTEM_PROMPT,
    available_tools: dict[str, Callable[..., Any]] | None = None,
    model: str = "gpt-4o",
    use_corrupted_kb: bool = False,
) -> ExecutionTrace:
    """Simulate a customer support agent execution.

    Produces realistic execution traces following the Tier-1 support workflow.
    """
    if available_tools is None:
        available_tools = TOOL_DISPATCH.copy()

    ticket_description = input_data.get("ticket_description", "")
    customer_id = input_data.get("customer_id", "CUST-1001")
    steps: list[StepTrace] = []
    step_idx = 0
    total_duration = 0.0
    output_parts: list[str] = []

    # Step 0: Process ticket
    steps.append(_build_step(
        index=step_idx,
        action="llm_response",
        llm_input=f"System: {system_prompt[:200]}...\nTicket: {ticket_description}",
        llm_output="[Analyzing customer support ticket]",
        model=model,
        duration_ms=180.0,
    ))
    step_idx += 1
    total_duration += 180.0

    ticket_id = None
    priority = "medium"
    category = "general"

    # Step 1: Classify ticket
    if "classify_ticket" in available_tools and "classify" in system_prompt.lower():
        classification = classify_ticket(ticket_description)
        ticket_id = classification["ticket_id"]
        priority = classification["priority"]
        category = classification["category"]

        steps.append(_build_step(
            index=step_idx,
            action="tool_call",
            tool_name="classify_ticket",
            tool_input={"description": ticket_description},
            tool_output=classification,
            model=model,
            duration_ms=80.0,
        ))
        step_idx += 1
        total_duration += 80.0
    else:
        ticket_id = f"TKT-{uuid.uuid4().hex[:8].upper()}"

    # Step 2: Check SLA
    if "check_sla" in available_tools and "sla" in system_prompt.lower():
        sla_result = check_sla(ticket_id, priority)
        steps.append(_build_step(
            index=step_idx,
            action="tool_call",
            tool_name="check_sla",
            tool_input={"ticket_id": ticket_id, "priority": priority},
            tool_output=sla_result,
            model=model,
            duration_ms=50.0,
        ))
        step_idx += 1
        total_duration += 50.0

        if priority == "critical":
            output_parts.append(f"[URGENT] Ticket {ticket_id} classified as {priority}. Immediate response required.")

    # Step 3: Look up order history (if relevant)
    order_related = any(kw in ticket_description.lower() for kw in
                        ["order", "ship", "track", "deliver", "package", "received", "bought"])
    if order_related and "lookup_order_history" in available_tools:
        order_result = lookup_order_history(customer_id)
        steps.append(_build_step(
            index=step_idx,
            action="tool_call",
            tool_name="lookup_order_history",
            tool_input={"customer_id": customer_id},
            tool_output=order_result,
            model=model,
            duration_ms=70.0,
        ))
        step_idx += 1
        total_duration += 70.0

    # Step 4: Search knowledge base
    if "search_knowledge_base" in available_tools:
        kb = _CORRUPTED_KNOWLEDGE_BASE if use_corrupted_kb else None
        kb_result = search_knowledge_base(ticket_description, kb)
        steps.append(_build_step(
            index=step_idx,
            action="tool_call",
            tool_name="search_knowledge_base",
            tool_input={"query": ticket_description},
            tool_output=kb_result,
            model=model,
            duration_ms=60.0,
        ))
        step_idx += 1
        total_duration += 60.0

        if kb_result["articles"]:
            output_parts.append(f"Found {kb_result['total_results']} relevant KB articles.")

    # Step 5: Determine action — resolve or escalate
    should_escalate = False
    escalation_dept = "general_support"

    if category == "security":
        should_escalate = True
        escalation_dept = "security_ops"
    elif category == "quality":
        should_escalate = True
        escalation_dept = "quality_assurance"
    elif category == "billing" and any(kw in ticket_description.lower() for kw in ["fraud", "unauthorized", "dispute"]):
        should_escalate = True
        escalation_dept = "finance"
    elif any(kw in ticket_description.lower() for kw in ["lawyer", "lawsuit", "legal"]):
        should_escalate = True
        escalation_dept = "management"

    # Skip escalation if prompt says to resolve everything
    if "resolve all tickets yourself" in system_prompt.lower():
        should_escalate = False

    if should_escalate and "escalate_ticket" in available_tools:
        reason = f"Ticket requires specialized handling by {escalation_dept}. Category: {category}, Priority: {priority}."
        esc_result = escalate_ticket(ticket_id, reason, escalation_dept)
        steps.append(_build_step(
            index=step_idx,
            action="tool_call",
            tool_name="escalate_ticket",
            tool_input={"ticket_id": ticket_id, "reason": reason, "department": escalation_dept},
            tool_output=esc_result,
            model=model,
            duration_ms=80.0,
        ))
        step_idx += 1
        total_duration += 80.0
        output_parts.append(
            f"Ticket {ticket_id} has been escalated to {escalation_dept}. "
            f"A specialist will respond within {esc_result.get('estimated_response', '2 hours')}."
        )
    else:
        # Resolve directly
        resolution = _generate_resolution(category, ticket_description)
        output_parts.append(resolution)

    # Step 6: Send response
    response_text = " ".join(output_parts) if output_parts else "Thank you for contacting us. We are looking into your issue."
    if "send_response" in available_tools:
        send_result = send_response(ticket_id, response_text)
        steps.append(_build_step(
            index=step_idx,
            action="tool_call",
            tool_name="send_response",
            tool_input={"ticket_id": ticket_id, "response": response_text},
            tool_output=send_result,
            model=model,
            duration_ms=50.0,
        ))
        step_idx += 1
        total_duration += 50.0

    # Final output
    final_output = response_text

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
        scenario_id=input_data.get("_scenario_id", "customer_support"),
        steps=steps,
        input_data=input_data,
        output_data=final_output,
        success=True,
        total_duration_ms=total_duration,
        total_cost_usd=round(estimated_cost, 4),
        model=model,
        framework="custom",
    )


def _generate_resolution(category: str, description: str) -> str:
    """Generate a resolution message based on ticket category."""
    resolutions: dict[str, str] = {
        "shipping": (
            "I've checked your order status. Your package is in transit and "
            "expected to arrive within 3-5 business days. You can track it "
            "using the tracking number in your order confirmation email."
        ),
        "returns": (
            "You can return items within 30 days of delivery. Please ensure "
            "items are unworn with original tags. I'll initiate the return process "
            "and email you a prepaid shipping label."
        ),
        "billing": (
            "I've reviewed your billing concern. If you see a pending charge, "
            "it should clear within 24-48 hours. For verified double charges, "
            "I've flagged your account for an immediate refund."
        ),
        "account": (
            "I can help with your account access. I've sent a password reset "
            "link to your registered email. If you don't receive it within "
            "5 minutes, please check your spam folder."
        ),
        "product": (
            "Based on our size guide, I recommend checking the measurements "
            "on the product page. If the item doesn't fit, we offer free "
            "exchanges within 30 days."
        ),
        "quality": (
            "I'm sorry about the quality issue. Please share photos of the "
            "damage and we'll ship a replacement at no cost within 48 hours."
        ),
        "loyalty": (
            "Your current membership tier provides exclusive benefits including "
            "discounts, early sale access, and free express shipping. "
            "I can review your tier status and available rewards."
        ),
    }
    return resolutions.get(category, (
        "Thank you for reaching out. I've reviewed your concern and will work "
        "on a resolution. Please allow 24 hours for a complete response."
    ))


# ===================================================================
# Test Cases (12 scenarios)
# ===================================================================

TEST_CASES: list[TestScenario] = [
    TestScenario(
        scenario_id="cs-001-shipping-inquiry",
        name="Standard shipping inquiry",
        description="Customer asks about package tracking. Low priority, resolve directly.",
        input_data={
            "ticket_description": "Where is my order? I placed it last week and haven't received it yet.",
            "customer_id": "CUST-1001",
        },
        expected_properties={
            "must_use_tools": ["classify_ticket", "search_knowledge_base", "lookup_order_history", "send_response"],
            "expected_priority": "medium",
            "expected_category": "shipping",
            "should_escalate": False,
            "output_contains": ["track"],
            "max_steps": 15,
        },
        evaluator="customer_support",
        tags=["shipping", "tracking", "happy-path"],
    ),
    TestScenario(
        scenario_id="cs-002-billing-dispute",
        name="Billing dispute — double charge",
        description="Customer reports being charged twice. High priority.",
        input_data={
            "ticket_description": "I was charged twice for my last order! This is unacceptable. I need a refund immediately.",
            "customer_id": "CUST-1002",
        },
        expected_properties={
            "must_use_tools": ["classify_ticket", "check_sla", "send_response"],
            "expected_priority": "high",
            "expected_category": "billing",
            "sla_checked": True,
            "output_contains": ["refund", "charge"],
            "max_steps": 15,
        },
        evaluator="customer_support",
        tags=["billing", "dispute", "high-priority"],
    ),
    TestScenario(
        scenario_id="cs-003-security-breach",
        name="Account security breach — critical",
        description="Customer reports unauthorized access. Must be classified critical and escalated to security_ops.",
        input_data={
            "ticket_description": "URGENT: Someone hacked my account and placed unauthorized orders. I need immediate help. This is a security breach.",
            "customer_id": "CUST-1001",
        },
        expected_properties={
            "must_use_tools": ["classify_ticket", "check_sla", "escalate_ticket", "send_response"],
            "expected_priority": "critical",
            "expected_category": "security",
            "should_escalate": True,
            "escalation_department": "security_ops",
            "sla_checked": True,
            "output_contains": ["escalate", "security"],
            "max_steps": 15,
        },
        evaluator="customer_support",
        tags=["security", "critical", "escalation"],
    ),
    TestScenario(
        scenario_id="cs-004-return-request",
        name="Simple return request",
        description="Customer wants to return an item. Resolve directly with policy info.",
        input_data={
            "ticket_description": "I'd like to return the jeans I bought last week. They don't fit right.",
            "customer_id": "CUST-1001",
        },
        expected_properties={
            "must_use_tools": ["classify_ticket", "search_knowledge_base", "send_response"],
            "expected_category": "returns",
            "should_escalate": False,
            "output_contains": ["return", "30 days"],
            "max_steps": 15,
        },
        evaluator="customer_support",
        tags=["returns", "resolve-direct"],
    ),
    TestScenario(
        scenario_id="cs-005-defective-product",
        name="Defective product report",
        description="Customer received damaged headphones. Escalate to quality_assurance.",
        input_data={
            "ticket_description": "My Sony headphones arrived with a broken hinge. The right earcup is damaged and won't fold.",
            "customer_id": "CUST-1002",
        },
        expected_properties={
            "must_use_tools": ["classify_ticket", "escalate_ticket", "send_response"],
            "expected_category": "quality",
            "should_escalate": True,
            "escalation_department": "quality_assurance",
            "output_contains": ["damage", "replacement"],
            "max_steps": 15,
        },
        evaluator="customer_support",
        tags=["quality", "defective", "escalation"],
    ),
    TestScenario(
        scenario_id="cs-006-account-locked",
        name="Account locked after failed logins",
        description="Customer cannot access account. Resolve directly.",
        input_data={
            "ticket_description": "I cannot login to my account. I've tried my password multiple times and now it says my account is locked.",
            "customer_id": "CUST-1003",
        },
        expected_properties={
            "must_use_tools": ["classify_ticket", "search_knowledge_base", "send_response"],
            "expected_category": "account",
            "should_escalate": False,
            "output_contains": ["password", "reset"],
            "max_steps": 12,
        },
        evaluator="customer_support",
        tags=["account", "login", "resolve-direct"],
    ),
    TestScenario(
        scenario_id="cs-007-legal-threat",
        name="Customer threatening legal action",
        description="Angry customer mentioning lawyer. Must escalate to management.",
        input_data={
            "ticket_description": "This is the third time my order was wrong. I've had enough. I'm contacting my lawyer about this. Your service is terrible.",
            "customer_id": "CUST-1004",
        },
        expected_properties={
            "must_use_tools": ["classify_ticket", "escalate_ticket", "send_response"],
            "should_escalate": True,
            "escalation_department": "management",
            "max_steps": 15,
        },
        evaluator="customer_support",
        tags=["escalation", "legal", "management"],
    ),
    TestScenario(
        scenario_id="cs-008-sla-critical",
        name="SLA enforcement on critical ticket",
        description="Verify SLA checking is performed for critical tickets.",
        input_data={
            "ticket_description": "EMERGENCY: Our corporate account has been compromised. Unauthorized bulk orders placed. Need immediate lockdown.",
            "customer_id": "CUST-1004",
        },
        expected_properties={
            "must_use_tools": ["classify_ticket", "check_sla", "escalate_ticket", "send_response"],
            "expected_priority": "critical",
            "sla_checked": True,
            "should_escalate": True,
            "output_contains": ["urgent", "immediate"],
            "max_steps": 15,
        },
        evaluator="customer_support",
        tags=["sla", "critical", "enforcement"],
    ),
    TestScenario(
        scenario_id="cs-009-membership-inquiry",
        name="Loyalty membership benefits inquiry",
        description="Customer asks about their membership tier. Low priority, resolve directly.",
        input_data={
            "ticket_description": "How do I check my membership tier? I want to know what benefits I get and how to earn more points.",
            "customer_id": "CUST-1001",
        },
        expected_properties={
            "must_use_tools": ["classify_ticket", "search_knowledge_base", "send_response"],
            "expected_priority": "low",
            "expected_category": "loyalty",
            "should_escalate": False,
            "output_contains": ["membership", "tier"],
            "max_steps": 12,
        },
        evaluator="customer_support",
        tags=["loyalty", "low-priority", "resolve-direct"],
    ),
    TestScenario(
        scenario_id="cs-010-fraud-report",
        name="Payment fraud escalation",
        description="Customer reports fraudulent charges. Must classify high and escalate to finance.",
        input_data={
            "ticket_description": "There are charges on my account I never made. Someone used my saved payment method to place orders. This is fraud.",
            "customer_id": "CUST-1002",
        },
        expected_properties={
            "must_use_tools": ["classify_ticket", "check_sla", "escalate_ticket", "send_response"],
            "expected_priority": "high",
            "expected_category": "billing",
            "should_escalate": True,
            "escalation_department": "finance",
            "sla_checked": True,
            "max_steps": 15,
        },
        evaluator="customer_support",
        tags=["fraud", "billing", "escalation"],
    ),
    TestScenario(
        scenario_id="cs-011-size-question",
        name="Product size guidance request",
        description="Customer needs sizing help. Resolve directly with KB article.",
        input_data={
            "ticket_description": "What size should I get in Nike shoes? I usually wear 9.5 in Adidas. Do Nike shoes run small?",
            "customer_id": "CUST-1001",
        },
        expected_properties={
            "must_use_tools": ["classify_ticket", "search_knowledge_base", "send_response"],
            "expected_category": "product",
            "should_escalate": False,
            "output_contains": ["size"],
            "max_steps": 12,
        },
        evaluator="customer_support",
        tags=["product", "sizing", "resolve-direct"],
    ),
    TestScenario(
        scenario_id="cs-012-international-shipping",
        name="International shipping inquiry",
        description="Customer asks about international shipping. Resolve with KB info.",
        input_data={
            "ticket_description": "Do you ship internationally? I'm in Germany and want to order. What are the shipping times and costs?",
            "customer_id": "CUST-1003",
        },
        expected_properties={
            "must_use_tools": ["classify_ticket", "search_knowledge_base", "send_response"],
            "expected_category": "shipping",
            "expected_priority": "low",
            "should_escalate": False,
            "output_contains": ["international", "shipping"],
            "max_steps": 12,
        },
        evaluator="customer_support",
        tags=["shipping", "international", "low-priority"],
    ),
]


# ===================================================================
# Regression Injection Points
# ===================================================================

REGRESSION_INJECTIONS: list[dict[str, Any]] = [
    {
        "id": "reg-cs-001",
        "name": "Remove classification rules",
        "description": (
            "Remove classification logic from system prompt. Expected regression: "
            "tickets are not classified, SLA/routing breaks."
        ),
        "type": "prompt_mutation",
        "system_prompt": SYSTEM_PROMPT_NO_CLASSIFICATION,
        "expected_regression_on": [
            "cs-003-security-breach",
            "cs-008-sla-critical",
            "cs-010-fraud-report",
        ],
    },
    {
        "id": "reg-cs-002",
        "name": "Disable SLA checking",
        "description": (
            "Remove SLA enforcement from system prompt. Expected regression: "
            "critical tickets not prioritized, SLA breaches."
        ),
        "type": "prompt_mutation",
        "system_prompt": SYSTEM_PROMPT_NO_SLA,
        "expected_regression_on": [
            "cs-002-billing-dispute",
            "cs-003-security-breach",
            "cs-008-sla-critical",
            "cs-010-fraud-report",
        ],
    },
    {
        "id": "reg-cs-003",
        "name": "Corrupt knowledge base responses",
        "description": (
            "Replace KB with irrelevant articles. Expected regression: "
            "agent provides incorrect or nonsensical solutions."
        ),
        "type": "data_corruption",
        "use_corrupted_kb": True,
        "expected_regression_on": [
            "cs-001-shipping-inquiry",
            "cs-004-return-request",
            "cs-006-account-locked",
            "cs-009-membership-inquiry",
        ],
    },
    {
        "id": "reg-cs-004",
        "name": "Remove escalation capability",
        "description": (
            "Instruct agent to resolve all tickets itself. Expected regression: "
            "critical/security tickets not escalated to specialists."
        ),
        "type": "prompt_mutation",
        "system_prompt": SYSTEM_PROMPT_NO_ESCALATION,
        "expected_regression_on": [
            "cs-003-security-breach",
            "cs-005-defective-product",
            "cs-007-legal-threat",
            "cs-008-sla-critical",
            "cs-010-fraud-report",
        ],
    },
]


# ===================================================================
# Convenience Exports
# ===================================================================

SCENARIO_ID = "customer_support"
SCENARIO_NAME = "Enterprise Customer Support Agent"
SCENARIO_DESCRIPTION = (
    "Tier-1 customer support agent with ticket classification, SLA monitoring, "
    "knowledge base lookup, order history, escalation, and resolution workflows."
)
