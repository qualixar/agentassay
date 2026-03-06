"""Daemon fixes for Session 6 - bulletproof self-healing.

This module contains the 5 critical fixes:
1. Checkpoint-on-error: Save state before continuing after exceptions
2. Blob-push-on-checkpoint: Immediate upload after each checkpoint
3. Heartbeat logging: Progress visibility every 5 minutes
4. Token validation: Reject success=true with tokens=0
5. Graceful shutdown: Save state on SIGTERM

These functions will be monkey-patched into the daemon.
"""

import asyncio
import logging
import os
import signal
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger("agentassay.daemon")

# Global state for heartbeat task
_heartbeat_task: asyncio.Task | None = None
_heartbeat_stop_event: asyncio.Event | None = None
_heartbeat_stats: dict[str, Any] = {
    "last_beat": 0.0,
    "trials_completed": 0,
    "current_experiment": "",
    "current_model": "",
}


# ---------------------------------------------------------------------------
# FIX 1 + FIX 2: Checkpoint with immediate blob push
# ---------------------------------------------------------------------------

def save_checkpoint_with_blob_push(
    checkpoint_manager: Any,
    push_blob_fn: callable,
    results_dir: Path,
    log_dir: Path,
    status_file: Path,
    experiment_id: str,
    model: str,
    scenario: str,
    completed: int,
    results: list[dict[str, Any]],
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Save checkpoint AND immediately push to blob (no 30-min wait).

    This fixes the zombie daemon issue where checkpoints were written
    but never uploaded to blob storage.
    """
    # Save checkpoint (atomic write with file locking)
    checkpoint_path = checkpoint_manager.save(
        experiment_id=experiment_id,
        model=model,
        scenario=scenario,
        completed=completed,
        results=results,
        metadata=metadata,
    )

    # IMMEDIATE blob push (force=True bypasses 30-min interval)
    try:
        push_blob_fn(results_dir, log_dir, status_file, force=True)
        logger.info(
            "Checkpoint + blob push complete: %s (%d trials)",
            checkpoint_path.name,
            completed,
        )
    except Exception as exc:
        logger.warning(
            "Checkpoint saved locally but blob push failed: %s", exc
        )

    return checkpoint_path


# ---------------------------------------------------------------------------
# FIX 3: Heartbeat logging (every 5 minutes)
# ---------------------------------------------------------------------------

async def heartbeat_task(interval_sec: float = 300.0) -> None:
    """Background task that logs progress every 5 minutes.

    Provides visibility into daemon health and prevents silent failures.
    """
    global _heartbeat_stats, _heartbeat_stop_event

    _heartbeat_stop_event = asyncio.Event()

    logger.info("Heartbeat task started (interval: %.0f sec)", interval_sec)

    while not _heartbeat_stop_event.is_set():
        try:
            await asyncio.wait_for(
                _heartbeat_stop_event.wait(),
                timeout=interval_sec,
            )
        except asyncio.TimeoutError:
            # Timeout = heartbeat due
            now = time.monotonic()
            elapsed = now - _heartbeat_stats.get("last_beat", 0)

            logger.info(
                "❤️  HEARTBEAT [%.0fs elapsed] | Experiment: %s | Model: %s | Trials: %d",
                elapsed,
                _heartbeat_stats.get("current_experiment", "NONE"),
                _heartbeat_stats.get("current_model", "NONE"),
                _heartbeat_stats.get("trials_completed", 0),
            )

            _heartbeat_stats["last_beat"] = now


def update_heartbeat_stats(
    experiment: str = "",
    model: str = "",
    trials: int = 0,
) -> None:
    """Update heartbeat statistics (called from trial runners)."""
    global _heartbeat_stats

    if experiment:
        _heartbeat_stats["current_experiment"] = experiment
    if model:
        _heartbeat_stats["current_model"] = model
    if trials > 0:
        _heartbeat_stats["trials_completed"] = trials


def start_heartbeat(interval_sec: float = 300.0) -> asyncio.Task:
    """Start the heartbeat background task."""
    global _heartbeat_task

    if _heartbeat_task is None or _heartbeat_task.done():
        _heartbeat_task = asyncio.create_task(heartbeat_task(interval_sec))

    return _heartbeat_task


def stop_heartbeat() -> None:
    """Stop the heartbeat background task."""
    global _heartbeat_stop_event, _heartbeat_task

    if _heartbeat_stop_event:
        _heartbeat_stop_event.set()

    if _heartbeat_task and not _heartbeat_task.done():
        _heartbeat_task.cancel()
        logger.info("Heartbeat task stopped")


# ---------------------------------------------------------------------------
# FIX 4: Token validation
# ---------------------------------------------------------------------------

def validate_trial_result(result: dict[str, Any]) -> bool:
    """Validate a trial result for data integrity.

    Checks:
    - If success=True, tokens must be > 0 (no fake data)
    - If passed=True and success=False, warn but allow

    Returns True if valid, False if corrupted (should be discarded).
    """
    success = result.get("success", False)
    tokens = result.get("tokens", 0)
    passed = result.get("passed", False)
    model = result.get("model", "UNKNOWN")
    trial_idx = result.get("trial_index", -1)

    # CRITICAL: success=True but tokens=0 is FAKE DATA
    if success and tokens == 0:
        logger.error(
            "INVALID RESULT [%s trial %d]: success=True but tokens=0 (FAKE DATA) — DISCARDING",
            model,
            trial_idx,
        )
        return False

    # Warning: passed=True but success=False (logic error but not corrupted data)
    if passed and not success:
        logger.warning(
            "Suspicious result [%s trial %d]: passed=True but success=False",
            model,
            trial_idx,
        )

    return True


# ---------------------------------------------------------------------------
# FIX 5: Graceful shutdown handler
# ---------------------------------------------------------------------------

_shutdown_handlers: list[callable] = []


def register_shutdown_handler(handler: callable) -> None:
    """Register a function to call on SIGTERM/SIGINT."""
    global _shutdown_handlers
    _shutdown_handlers.append(handler)


def signal_handler(signum: int, frame: Any) -> None:
    """Handle SIGTERM/SIGINT gracefully.

    Saves all checkpoints and pushes to blob before exiting.
    """
    sig_name = signal.Signals(signum).name
    logger.warning(
        "Received %s — shutting down gracefully (saving state...)",
        sig_name,
    )

    # Call all registered handlers (checkpoint saves, blob pushes, etc.)
    for handler in _shutdown_handlers:
        try:
            handler()
        except Exception as exc:
            logger.error("Shutdown handler failed: %s", exc)

    stop_heartbeat()

    logger.info("Graceful shutdown complete. Exiting.")
    exit(0)


def setup_signal_handlers() -> None:
    """Install SIGTERM and SIGINT handlers."""
    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)
    logger.info("Signal handlers installed (SIGTERM, SIGINT)")
