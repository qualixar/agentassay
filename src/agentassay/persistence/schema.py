"""Schema DDL and helpers for the AgentAssay SQLite persistence layer.

Contains all table definitions, index definitions, and low-level utility
functions shared by the storage and reader modules.  This is the DDL layer
-- no business logic lives here.

Tables
------
projects, runs, trials, verdicts, coverage, fingerprints, gate_decisions, costs

Thread safety
-------------
The helpers in this module are pure functions with no shared state.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _uuid() -> str:
    """Generate a new UUID4 string."""
    return str(uuid.uuid4())


def _now_iso() -> str:
    """UTC-aware ISO-8601 timestamp string for SQLite."""
    return datetime.now(timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Known tables -- used for injection-safe ``count()``
# ---------------------------------------------------------------------------

ALLOWED_TABLES: frozenset[str] = frozenset({
    "projects", "runs", "trials", "verdicts",
    "coverage", "fingerprints", "gate_decisions", "costs",
})


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS projects (
    id         TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS runs (
    id            TEXT PRIMARY KEY,
    project_id    TEXT REFERENCES projects(id),
    agent_name    TEXT NOT NULL,
    agent_version TEXT,
    model         TEXT NOT NULL,
    framework     TEXT NOT NULL,
    config_json   TEXT NOT NULL,
    started_at    TIMESTAMP NOT NULL,
    completed_at  TIMESTAMP,
    status        TEXT NOT NULL,
    total_trials  INTEGER,
    total_cost    REAL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS trials (
    id          TEXT PRIMARY KEY,
    run_id      TEXT REFERENCES runs(id),
    scenario_id TEXT NOT NULL,
    trial_num   INTEGER NOT NULL,
    success     BOOLEAN NOT NULL,
    latency_ms  REAL NOT NULL,
    cost        REAL DEFAULT 0,
    token_count INTEGER DEFAULT 0,
    step_count  INTEGER NOT NULL,
    error_msg   TEXT,
    trace_json  TEXT NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS verdicts (
    id          TEXT PRIMARY KEY,
    run_id      TEXT REFERENCES runs(id),
    scenario_id TEXT NOT NULL,
    status      TEXT NOT NULL,
    pass_rate   REAL NOT NULL,
    ci_lower    REAL NOT NULL,
    ci_upper    REAL NOT NULL,
    p_value     REAL,
    effect_size REAL,
    n_trials    INTEGER NOT NULL,
    method      TEXT NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS coverage (
    id           TEXT PRIMARY KEY,
    run_id       TEXT REFERENCES runs(id),
    dimension    TEXT NOT NULL,
    score        REAL NOT NULL,
    details_json TEXT,
    created_at   TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS fingerprints (
    id          TEXT PRIMARY KEY,
    run_id      TEXT REFERENCES runs(id),
    scenario_id TEXT NOT NULL,
    vector_json TEXT NOT NULL,
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS gate_decisions (
    id         TEXT PRIMARY KEY,
    run_id     TEXT REFERENCES runs(id),
    pipeline   TEXT NOT NULL,
    decision   TEXT NOT NULL,
    reason     TEXT,
    rules_json TEXT NOT NULL,
    commit_sha TEXT,
    pr_number  INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS costs (
    id            TEXT PRIMARY KEY,
    run_id        TEXT REFERENCES runs(id),
    model         TEXT NOT NULL,
    input_tokens  INTEGER DEFAULT 0,
    output_tokens INTEGER DEFAULT 0,
    total_cost    REAL DEFAULT 0,
    trial_count   INTEGER DEFAULT 0,
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
"""

INDEX_SQL = """
CREATE INDEX IF NOT EXISTS idx_runs_project
    ON runs(project_id, started_at DESC);
CREATE INDEX IF NOT EXISTS idx_trials_run
    ON trials(run_id, scenario_id);
CREATE INDEX IF NOT EXISTS idx_verdicts_run
    ON verdicts(run_id);
CREATE INDEX IF NOT EXISTS idx_coverage_run
    ON coverage(run_id);
CREATE INDEX IF NOT EXISTS idx_fingerprints_run
    ON fingerprints(run_id, scenario_id);
CREATE INDEX IF NOT EXISTS idx_gate_decisions_pipeline
    ON gate_decisions(pipeline, created_at DESC);
CREATE INDEX IF NOT EXISTS idx_costs_run
    ON costs(run_id, model);
"""
