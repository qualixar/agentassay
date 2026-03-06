"""Checkpoint and resume manager for long-running experiment campaigns.

Saves incremental state after every N trials so that experiments can be
interrupted (crash, budget pause, session end) and resumed without losing
progress.

Design:
    - One JSON file per (experiment, model, scenario) triple.
    - File locking via ``fcntl.flock`` for thread safety on POSIX systems.
    - Atomic writes: write to a temp file, then rename, to prevent
      corruption on crash.
    - The checkpoint directory is auto-created at first save.

File layout::

    experiments/results/checkpoints/
        E1__gpt-5.2-chat__ecommerce-basic.json
        E1__claude-sonnet-4-6__ecommerce-basic.json
        ...
"""

from __future__ import annotations

import fcntl
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def _checkpoint_filename(
    experiment_id: str, model: str, scenario: str
) -> str:
    """Build a deterministic, filesystem-safe checkpoint filename.

    Replaces characters that are problematic in filenames.
    """
    safe_exp = experiment_id.replace("/", "_").replace(" ", "_")
    safe_model = model.replace("/", "_").replace(" ", "_")
    safe_scenario = scenario.replace("/", "_").replace(" ", "_")
    return f"{safe_exp}__{safe_model}__{safe_scenario}.json"


class CheckpointManager:
    """Thread-safe checkpoint/resume manager for experiment state.

    Parameters
    ----------
    checkpoint_dir
        Directory where checkpoint files are stored. Created on first save.
    save_interval
        Save a checkpoint every N completed trials. Default 25.

    Example
    -------
    >>> mgr = CheckpointManager("experiments/results/checkpoints")
    >>> # On resume:
    >>> state = mgr.load("E1", "gpt-5.2-chat", "ecommerce-basic")
    >>> if state:
    ...     completed = state["completed"]
    ...     results = state["results"]
    ...     print(f"Resuming from trial {completed}")
    >>> # During execution:
    >>> mgr.save("E1", "gpt-5.2-chat", "ecommerce-basic", completed=50, results=[...])
    """

    def __init__(
        self,
        checkpoint_dir: str = "experiments/results/checkpoints",
        save_interval: int = 25,
    ) -> None:
        self._checkpoint_dir = Path(checkpoint_dir)
        self._save_interval = max(1, save_interval)
        self._trial_counters: dict[str, int] = {}

    @property
    def checkpoint_dir(self) -> Path:
        """The directory where checkpoints are stored."""
        return self._checkpoint_dir

    @property
    def save_interval(self) -> int:
        """How often (in trials) checkpoints are written."""
        return self._save_interval

    # -- Save ----------------------------------------------------------------

    def save(
        self,
        experiment_id: str,
        model: str,
        scenario: str,
        completed: int,
        results: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> Path:
        """Persist experiment state to a checkpoint file.

        Uses atomic write (temp file + rename) and file locking to
        guarantee data integrity even under concurrent access.

        Parameters
        ----------
        experiment_id
            Experiment identifier (e.g. "E1").
        model
            Model name.
        scenario
            Scenario identifier.
        completed
            Number of completed trials.
        results
            List of serializable result dicts.
        metadata
            Optional additional metadata to store.

        Returns
        -------
        Path
            Path to the written checkpoint file.
        """
        self._checkpoint_dir.mkdir(parents=True, exist_ok=True)

        filename = _checkpoint_filename(experiment_id, model, scenario)
        filepath = self._checkpoint_dir / filename

        state = {
            "experiment_id": experiment_id,
            "model": model,
            "scenario": scenario,
            "completed": completed,
            "total_results": len(results),
            "results": results,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": 1,
        }

        # Atomic write: write to temp file in same directory, then rename
        fd, tmp_path = tempfile.mkstemp(
            dir=str(self._checkpoint_dir),
            prefix=".ckpt_tmp_",
            suffix=".json",
        )
        try:
            with os.fdopen(fd, "w") as f:
                # Acquire exclusive lock
                fcntl.flock(f.fileno(), fcntl.LOCK_EX)
                try:
                    json.dump(state, f, indent=2, default=str)
                    f.flush()
                    os.fsync(f.fileno())
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            # Atomic rename (POSIX guarantees atomicity for same filesystem)
            os.replace(tmp_path, filepath)
        except Exception:
            # Clean up temp file on failure
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise

        logger.debug(
            "Checkpoint saved: %s (completed=%d, results=%d)",
            filepath.name,
            completed,
            len(results),
        )
        return filepath

    def maybe_save(
        self,
        experiment_id: str,
        model: str,
        scenario: str,
        completed: int,
        results: list[dict[str, Any]],
        metadata: dict[str, Any] | None = None,
    ) -> Path | None:
        """Save a checkpoint if the save interval has been reached.

        Call this after every trial. It only writes to disk when
        ``completed`` is a multiple of ``save_interval`` or when
        ``completed`` exceeds the last-saved count by ``save_interval``.

        Returns
        -------
        Path | None
            Path to checkpoint if saved, None if skipped.
        """
        key = _checkpoint_filename(experiment_id, model, scenario)
        last_saved = self._trial_counters.get(key, 0)

        if completed - last_saved >= self._save_interval:
            self._trial_counters[key] = completed
            return self.save(
                experiment_id, model, scenario, completed, results, metadata
            )
        return None

    # -- Load ----------------------------------------------------------------

    def load(
        self,
        experiment_id: str,
        model: str,
        scenario: str,
    ) -> dict[str, Any] | None:
        """Load checkpoint state if it exists.

        Parameters
        ----------
        experiment_id
            Experiment identifier.
        model
            Model name.
        scenario
            Scenario identifier.

        Returns
        -------
        dict | None
            The checkpoint state dict, or None if no checkpoint exists.
            Keys include: ``experiment_id``, ``model``, ``scenario``,
            ``completed``, ``results``, ``metadata``, ``timestamp``.
        """
        filename = _checkpoint_filename(experiment_id, model, scenario)
        filepath = self._checkpoint_dir / filename

        if not filepath.exists():
            logger.debug("No checkpoint found: %s", filepath.name)
            return None

        try:
            with open(filepath) as f:
                fcntl.flock(f.fileno(), fcntl.LOCK_SH)
                try:
                    state = json.load(f)
                finally:
                    fcntl.flock(f.fileno(), fcntl.LOCK_UN)

            logger.info(
                "Checkpoint loaded: %s (completed=%d)",
                filepath.name,
                state.get("completed", 0),
            )
            return state

        except (json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "Corrupted checkpoint %s — ignoring: %s", filepath.name, exc
            )
            return None

    # -- Query ---------------------------------------------------------------

    def is_complete(
        self,
        experiment_id: str,
        model: str,
        scenario: str,
        total: int,
    ) -> bool:
        """Check whether all trials for a triple are already complete.

        Parameters
        ----------
        experiment_id
            Experiment identifier.
        model
            Model name.
        scenario
            Scenario identifier.
        total
            Total number of trials expected.

        Returns
        -------
        bool
            True if checkpoint exists and completed >= total.
        """
        state = self.load(experiment_id, model, scenario)
        if state is None:
            return False
        return state.get("completed", 0) >= total

    def completed_count(
        self,
        experiment_id: str,
        model: str,
        scenario: str,
    ) -> int:
        """Return the number of completed trials, or 0 if no checkpoint."""
        state = self.load(experiment_id, model, scenario)
        if state is None:
            return 0
        return state.get("completed", 0)

    def list_checkpoints(self) -> list[dict[str, Any]]:
        """List all checkpoint files with summary info.

        Returns
        -------
        list[dict]
            Each dict has keys: ``filename``, ``experiment_id``, ``model``,
            ``scenario``, ``completed``, ``timestamp``.
        """
        if not self._checkpoint_dir.exists():
            return []

        summaries: list[dict[str, Any]] = []
        for path in sorted(self._checkpoint_dir.glob("*.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
                summaries.append({
                    "filename": path.name,
                    "experiment_id": data.get("experiment_id", ""),
                    "model": data.get("model", ""),
                    "scenario": data.get("scenario", ""),
                    "completed": data.get("completed", 0),
                    "timestamp": data.get("timestamp", ""),
                })
            except (json.JSONDecodeError, OSError):
                logger.debug("Skipping unreadable checkpoint: %s", path.name)
                continue

        return summaries

    def clear(
        self,
        experiment_id: str | None = None,
        model: str | None = None,
        scenario: str | None = None,
    ) -> int:
        """Delete checkpoint files matching the given filters.

        All parameters are optional; omitted parameters match everything.

        Returns
        -------
        int
            Number of checkpoint files deleted.
        """
        if not self._checkpoint_dir.exists():
            return 0

        deleted = 0
        for path in list(self._checkpoint_dir.glob("*.json")):
            try:
                with open(path) as f:
                    data = json.load(f)
            except (json.JSONDecodeError, OSError):
                continue

            match = True
            if experiment_id is not None:
                match = match and data.get("experiment_id") == experiment_id
            if model is not None:
                match = match and data.get("model") == model
            if scenario is not None:
                match = match and data.get("scenario") == scenario

            if match:
                path.unlink(missing_ok=True)
                deleted += 1
                logger.debug("Deleted checkpoint: %s", path.name)

        return deleted

    def __repr__(self) -> str:
        return (
            f"CheckpointManager("
            f"dir='{self._checkpoint_dir}', "
            f"interval={self._save_interval})"
        )
