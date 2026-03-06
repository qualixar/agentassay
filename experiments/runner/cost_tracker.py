"""Real-time cost tracking and budget enforcement for experiment campaigns.

Tracks token usage (input/output) per model per experiment, estimates cost
using configurable pricing tables, writes running totals to a persistent
JSON log, and enforces budget guards to prevent runaway spend.

Thread-safe: uses a threading lock around all mutable state. The JSON log
is written atomically (temp file + rename) on every flush.

Design principles:
    - Costs are ESTIMATES. Azure billing may differ slightly due to
      rounding, minimum charges, and token counting differences.
    - The budget guard is advisory by default (warns) but can be
      configured to hard-stop the daemon.
    - All amounts are in USD.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pricing table
# ---------------------------------------------------------------------------

# Approximate pricing per 1,000,000 tokens: (input_rate, output_rate)
# These are Azure AI Foundry rates as of Feb 2026.
# Zero-cost entries are for models with included serverless compute.
DEFAULT_PRICING: dict[str, tuple[float, float]] = {
    "gpt-5.2-chat": (2.50, 10.00),
    "claude-sonnet-4-6": (3.00, 15.00),
    "claude-haiku-4-5": (0.80, 4.00),
    "DeepSeek-R1-0528": (0.55, 2.19),
    "Llama-3.3-70B-Instruct": (0.00, 0.00),
    "Mistral-Large-3": (2.00, 6.00),
    "Llama-4-Maverick-17B-128E-Instruct-FP8": (0.00, 0.00),
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ModelUsage:
    """Accumulated usage for a single model."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    total_calls: int = 0
    estimated_cost_usd: float = 0.0
    last_updated: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
            "total_calls": self.total_calls,
            "estimated_cost_usd": round(self.estimated_cost_usd, 6),
            "last_updated": self.last_updated,
        }


@dataclass
class ExperimentUsage:
    """Accumulated usage for a single experiment."""

    models: dict[str, ModelUsage] = field(default_factory=dict)
    total_cost_usd: float = 0.0
    total_calls: int = 0
    started_at: str = ""
    last_updated: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "models": {m: u.to_dict() for m, u in self.models.items()},
            "total_cost_usd": round(self.total_cost_usd, 6),
            "total_calls": self.total_calls,
            "started_at": self.started_at,
            "last_updated": self.last_updated,
        }


# ---------------------------------------------------------------------------
# Budget guard
# ---------------------------------------------------------------------------

class BudgetExceededError(Exception):
    """Raised when the total experiment cost exceeds the hard budget limit."""


# ---------------------------------------------------------------------------
# Main tracker
# ---------------------------------------------------------------------------

class CostTracker:
    """Real-time cost tracker with persistent JSON log and budget enforcement.

    Parameters
    ----------
    log_path
        Path to the JSON cost log file. Created on first flush.
    budget_usd
        Total budget cap in USD. When exceeded, either warns or raises.
    warn_at_pct
        Percentage of budget at which to start logging warnings.
        Default 0.80 (80%).
    hard_stop
        If True, raises ``BudgetExceededError`` when budget is exceeded.
        If False (default), only logs a warning.
    pricing
        Custom pricing table. Falls back to ``DEFAULT_PRICING`` for
        unknown models.

    Example
    -------
    >>> tracker = CostTracker(budget_usd=175.0, hard_stop=True)
    >>> tracker.record(
    ...     model="gpt-5.2-chat",
    ...     prompt_tokens=1200,
    ...     completion_tokens=340,
    ...     experiment_id="E1",
    ... )
    >>> print(tracker.total_cost_usd)
    0.006400
    """

    def __init__(
        self,
        log_path: str = "experiments/results/cost_log.json",
        budget_usd: float = 175.0,
        warn_at_pct: float = 0.80,
        hard_stop: bool = False,
        pricing: dict[str, tuple[float, float]] | None = None,
    ) -> None:
        self._log_path = Path(log_path)
        self._budget_usd = max(0.0, budget_usd)
        self._warn_at_pct = max(0.0, min(1.0, warn_at_pct))
        self._hard_stop = hard_stop
        self._pricing = dict(DEFAULT_PRICING)
        if pricing:
            self._pricing.update(pricing)

        self._lock = threading.Lock()
        self._experiments: dict[str, ExperimentUsage] = {}
        self._grand_total_cost: float = 0.0
        self._grand_total_calls: int = 0
        self._grand_total_tokens: int = 0
        self._started_at: str = datetime.now(timezone.utc).isoformat()

        # Load existing log if present (resume support)
        self._load_existing()

    # -- Recording -----------------------------------------------------------

    def record(
        self,
        model: str,
        prompt_tokens: int,
        completion_tokens: int,
        cost_usd: float | None = None,
        experiment_id: str = "default",
    ) -> float:
        """Record token usage from a single API call.

        Parameters
        ----------
        model
            Model name (must be in the pricing table for accurate costs).
        prompt_tokens
            Number of input tokens consumed.
        completion_tokens
            Number of output tokens generated.
        cost_usd
            Pre-computed cost. If None, estimated from the pricing table.
        experiment_id
            Experiment this call belongs to. Default "default".

        Returns
        -------
        float
            Estimated cost for this single call.

        Raises
        ------
        BudgetExceededError
            If ``hard_stop=True`` and total cost exceeds budget.
        """
        if cost_usd is None:
            cost_usd = self._estimate_cost(model, prompt_tokens, completion_tokens)

        now_iso = datetime.now(timezone.utc).isoformat()

        with self._lock:
            # Ensure experiment entry
            if experiment_id not in self._experiments:
                self._experiments[experiment_id] = ExperimentUsage(
                    started_at=now_iso
                )
            exp = self._experiments[experiment_id]

            # Ensure model entry
            if model not in exp.models:
                exp.models[model] = ModelUsage()
            mu = exp.models[model]

            # Update model-level
            mu.prompt_tokens += prompt_tokens
            mu.completion_tokens += completion_tokens
            mu.total_tokens += prompt_tokens + completion_tokens
            mu.total_calls += 1
            mu.estimated_cost_usd += cost_usd
            mu.last_updated = now_iso

            # Update experiment-level
            exp.total_cost_usd += cost_usd
            exp.total_calls += 1
            exp.last_updated = now_iso

            # Update grand totals
            self._grand_total_cost += cost_usd
            self._grand_total_calls += 1
            self._grand_total_tokens += prompt_tokens + completion_tokens

            # Budget checks
            self._check_budget()

        return cost_usd

    # -- Budget enforcement --------------------------------------------------

    def _check_budget(self) -> None:
        """Check budget thresholds. Called under lock.

        Logs warnings at the warn threshold and either logs or raises
        at the hard budget limit.
        """
        if self._budget_usd <= 0:
            return

        pct = self._grand_total_cost / self._budget_usd

        if pct >= 1.0:
            msg = (
                f"BUDGET EXCEEDED: ${self._grand_total_cost:.4f} / "
                f"${self._budget_usd:.2f} ({pct:.1%})"
            )
            if self._hard_stop:
                logger.critical(msg)
                raise BudgetExceededError(msg)
            else:
                logger.warning(msg)

        elif pct >= self._warn_at_pct:
            logger.warning(
                "Budget warning: ${:.4f} / ${:.2f} ({:.1%})",
                self._grand_total_cost,
                self._budget_usd,
                pct,
            )

    @property
    def budget_remaining_usd(self) -> float:
        """Remaining budget in USD (may be negative if exceeded)."""
        with self._lock:
            return self._budget_usd - self._grand_total_cost

    @property
    def budget_utilization_pct(self) -> float:
        """Budget utilization as a fraction in [0, inf)."""
        with self._lock:
            if self._budget_usd <= 0:
                return 0.0
            return self._grand_total_cost / self._budget_usd

    def is_budget_exceeded(self) -> bool:
        """Check if the budget has been exceeded."""
        with self._lock:
            return self._grand_total_cost >= self._budget_usd > 0

    # -- Cost estimation -----------------------------------------------------

    def _estimate_cost(
        self, model: str, prompt_tokens: int, completion_tokens: int
    ) -> float:
        """Estimate cost from the pricing table."""
        input_rate, output_rate = self._pricing.get(model, (0.0, 0.0))
        cost = (
            (prompt_tokens / 1_000_000) * input_rate
            + (completion_tokens / 1_000_000) * output_rate
        )
        return round(cost, 6)

    # -- Accessors -----------------------------------------------------------

    @property
    def total_cost_usd(self) -> float:
        """Grand total estimated cost across all experiments."""
        with self._lock:
            return self._grand_total_cost

    @property
    def total_calls(self) -> int:
        """Grand total API calls."""
        with self._lock:
            return self._grand_total_calls

    @property
    def total_tokens(self) -> int:
        """Grand total tokens consumed."""
        with self._lock:
            return self._grand_total_tokens

    def cost_by_model(self) -> dict[str, float]:
        """Aggregate cost per model across all experiments."""
        with self._lock:
            totals: dict[str, float] = defaultdict(float)
            for exp in self._experiments.values():
                for model, mu in exp.models.items():
                    totals[model] += mu.estimated_cost_usd
            return dict(totals)

    def cost_by_experiment(self) -> dict[str, float]:
        """Total cost per experiment."""
        with self._lock:
            return {
                eid: exp.total_cost_usd
                for eid, exp in self._experiments.items()
            }

    def summary(self) -> dict[str, Any]:
        """Full summary suitable for JSON export or status display."""
        with self._lock:
            return {
                "grand_total_cost_usd": round(self._grand_total_cost, 6),
                "grand_total_calls": self._grand_total_calls,
                "grand_total_tokens": self._grand_total_tokens,
                "budget_usd": self._budget_usd,
                "budget_remaining_usd": round(
                    self._budget_usd - self._grand_total_cost, 6
                ),
                "budget_utilization_pct": round(
                    (self._grand_total_cost / self._budget_usd * 100)
                    if self._budget_usd > 0
                    else 0.0,
                    2,
                ),
                "experiments": {
                    eid: exp.to_dict()
                    for eid, exp in self._experiments.items()
                },
                "started_at": self._started_at,
                "last_updated": datetime.now(timezone.utc).isoformat(),
            }

    # -- Persistence ---------------------------------------------------------

    def flush(self) -> Path:
        """Write the current cost log to disk atomically.

        Returns the path to the written file.
        """
        self._log_path.parent.mkdir(parents=True, exist_ok=True)

        data = self.summary()

        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._log_path.parent),
            prefix=".cost_tmp_",
            suffix=".json",
        )
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, self._log_path)
        except Exception:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        logger.debug(
            "Cost log flushed: $%.4f total (%d calls)",
            self._grand_total_cost,
            self._grand_total_calls,
        )
        return self._log_path

    def _load_existing(self) -> None:
        """Load an existing cost log for resume support."""
        if not self._log_path.exists():
            return

        try:
            with open(self._log_path) as f:
                data = json.load(f)

            self._grand_total_cost = data.get("grand_total_cost_usd", 0.0)
            self._grand_total_calls = data.get("grand_total_calls", 0)
            self._grand_total_tokens = data.get("grand_total_tokens", 0)
            self._started_at = data.get("started_at", self._started_at)

            for eid, exp_data in data.get("experiments", {}).items():
                exp = ExperimentUsage(
                    total_cost_usd=exp_data.get("total_cost_usd", 0.0),
                    total_calls=exp_data.get("total_calls", 0),
                    started_at=exp_data.get("started_at", ""),
                    last_updated=exp_data.get("last_updated", ""),
                )
                for model_name, mu_data in exp_data.get("models", {}).items():
                    exp.models[model_name] = ModelUsage(
                        prompt_tokens=mu_data.get("prompt_tokens", 0),
                        completion_tokens=mu_data.get("completion_tokens", 0),
                        total_tokens=mu_data.get("total_tokens", 0),
                        total_calls=mu_data.get("total_calls", 0),
                        estimated_cost_usd=mu_data.get("estimated_cost_usd", 0.0),
                        last_updated=mu_data.get("last_updated", ""),
                    )
                self._experiments[eid] = exp

            logger.info(
                "Cost log loaded: $%.4f from %d prior calls",
                self._grand_total_cost,
                self._grand_total_calls,
            )

        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("Could not load existing cost log: %s", exc)

    # -- Dunder --------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"CostTracker("
            f"total=${self._grand_total_cost:.4f}, "
            f"calls={self._grand_total_calls}, "
            f"budget=${self._budget_usd:.2f}, "
            f"remaining=${self.budget_remaining_usd:.4f})"
        )
