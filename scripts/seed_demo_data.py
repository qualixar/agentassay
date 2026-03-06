"""Seed the AgentAssay dashboard with realistic demo data.

Run this once to populate the local SQLite database with fake test runs,
verdicts, coverage, fingerprints, costs, and gate decisions. Then launch
the dashboard to see all 4 views populated.

Usage:
    python scripts/seed_demo_data.py
    agentassay dashboard
"""

from __future__ import annotations

import json
import random
import uuid
from datetime import datetime, timedelta, timezone

from agentassay.persistence import ResultStore


def _uuid() -> str:
    return str(uuid.uuid4())


def _ts(days_ago: int = 0, hours_ago: int = 0) -> str:
    dt = datetime.now(timezone.utc) - timedelta(days=days_ago, hours=hours_ago)
    return dt.isoformat()


def seed(db_path: str | None = None) -> None:
    """Populate the database with realistic demo data."""
    store = ResultStore(db_path=db_path) if db_path else ResultStore()

    # Create a project
    project_id = _uuid()
    project_id = store.save_project("demo-ecommerce-agent")

    agents = [
        ("search-agent", "2.1.0", "langgraph"),
        ("search-agent", "2.0.0", "langgraph"),
        ("booking-agent", "1.3.0", "crewai"),
    ]
    models = ["gpt-4o", "claude-sonnet-4-6", "gpt-4o-mini"]
    scenarios = [
        "basic-search", "multi-tool-chain", "error-recovery",
        "empty-query-edge", "concurrent-tools", "timeout-handling",
    ]

    print("Seeding demo data...")

    for day in range(14, -1, -1):
        # 1-3 runs per day
        for run_idx in range(random.randint(1, 3)):
            agent_name, agent_version, framework = random.choice(agents)
            model = random.choice(models)
            run_id = _uuid()
            started = _ts(days_ago=day, hours_ago=random.randint(0, 12))
            num_trials = random.randint(20, 60)
            total_cost = round(random.uniform(0.40, 2.50), 2)

            # Simulate improving pass rate over time
            base_rate = min(0.95, 0.70 + (14 - day) * 0.015)
            # Add some noise
            run_pass_rate = max(0.50, min(0.99, base_rate + random.gauss(0, 0.05)))

            run_id = store.save_run(
                project_id=project_id,
                agent_name=agent_name,
                agent_version=agent_version,
                model=model,
                framework=framework,
                config_json=json.dumps({"num_trials": num_trials, "alpha": 0.05}),
                started_at=started,
                status="completed",
                total_cost=total_cost,
            )

            # Trials
            pass_count = 0
            for t in range(num_trials):
                sc = random.choice(scenarios)
                success = random.random() < run_pass_rate
                if success:
                    pass_count += 1
                cost = round(random.uniform(0.005, 0.04), 4)

                store.save_trial(
                    run_id=run_id,
                    scenario_id=sc,
                    trial_num=t,
                    success=success,
                    latency_ms=round(random.uniform(500, 4000), 1),
                    cost=cost,
                    token_count=random.randint(200, 2000),
                    step_count=random.randint(2, 8),
                    trace_json=json.dumps({"steps": []}),
                )

            actual_rate = pass_count / num_trials if num_trials > 0 else 0

            # Verdicts per scenario
            for sc in scenarios:
                sc_rate = max(0.40, min(1.0, actual_rate + random.gauss(0, 0.08)))
                sc_trials = random.randint(5, 15)
                store.save_verdict(
                    run_id=run_id,
                    scenario_id=sc,
                    status="pass" if sc_rate >= 0.80 else ("fail" if sc_rate < 0.65 else "inconclusive"),
                    pass_rate=round(sc_rate, 3),
                    ci_lower=round(max(0, sc_rate - random.uniform(0.08, 0.15)), 3),
                    ci_upper=round(min(1, sc_rate + random.uniform(0.08, 0.15)), 3),
                    p_value=round(random.uniform(0.001, 0.5), 4) if sc_rate < 0.80 else None,
                    effect_size=round(random.uniform(0.05, 0.70), 3) if sc_rate < 0.80 else None,
                    n_trials=sc_trials,
                    method="fisher_exact",
                )

            # Coverage (5 dimensions)
            for dim in ["tool", "path", "state", "boundary", "model"]:
                score = round(random.uniform(0.55, 0.95), 3)
                store.save_coverage(
                    run_id=run_id,
                    dimension=dim,
                    score=score,
                    details_json=json.dumps({"items_covered": random.randint(5, 20)}),
                )

            # Fingerprints per scenario
            for sc in scenarios:
                vector = [round(random.uniform(0.5, 1.0), 3) for _ in range(6)]
                store.save_fingerprint(
                    run_id=run_id,
                    scenario_id=sc,
                    vector_json=json.dumps(vector),
                )

            # Costs per model
            store.save_cost(
                run_id=run_id,
                model=model,
                input_tokens=random.randint(5000, 50000),
                output_tokens=random.randint(1000, 15000),
                total_cost=total_cost,
                trial_count=num_trials,
            )

            # Gate decision
            decision = "deploy" if actual_rate >= 0.82 else ("block" if actual_rate < 0.70 else "abstain")
            store.save_gate_decision(
                run_id=run_id,
                pipeline="production",
                decision=decision,
                reason=f"Pass rate {actual_rate:.1%}",
                rules_json=json.dumps({"min_pass_rate": 0.85, "coverage_floor": 0.70}),
                commit_sha=uuid.uuid4().hex[:7],
            )

    print("Done! Seeded 14 days of demo data.")
    print(f"Database: {store._db_path}")
    print()
    print("Launch the dashboard:")
    print("  agentassay dashboard")


if __name__ == "__main__":
    seed()
