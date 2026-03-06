# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""SQLite-based persistence layer for AgentAssay.

Provides durable storage for test runs, trials, verdicts, coverage metrics,
behavioral fingerprints, deployment gate decisions, and cost accounting.
All data is stored in a single SQLite database with a normalized schema
optimized for both write throughput during test execution and read-heavy
dashboard queries.

Three entry points:

1. **ResultStore** (``storage``)
   Core CRUD engine backed by SQLite.  Thread-safe, zero external
   dependencies beyond the Python stdlib.  Designed for concurrent writes
   from parallel trial runners.

2. **QueryAPI** (``queries``)
   Read-only analytical layer on top of ResultStore.  Provides trend
   aggregations, cross-run comparisons, and summary statistics for the
   AgentAssay dashboard.

3. **EventBus** (``events``)
   Synchronous, in-process publish/subscribe bus.  Emits lifecycle events
   (trial_complete, verdict_ready, run_complete, ...) so that dashboard
   components can react without polling the database.

Typical usage::

    from agentassay.persistence import ResultStore, QueryAPI, EventBus

    store = ResultStore("./results.db")
    bus = EventBus()
    api = QueryAPI(store)

    bus.subscribe("trial_complete", lambda d: print(d))
    store.save_trial(...)
    bus.emit("trial_complete", {"run_id": "...", "trial_num": 1})
"""

from agentassay.persistence.events import EventBus
from agentassay.persistence.queries import QueryAPI
from agentassay.persistence.storage import ResultStore

__all__ = [
    "ResultStore",
    "QueryAPI",
    "EventBus",
]
