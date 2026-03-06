"""Self-healing experiment daemon for AgentAssay.

Reads experiment configs from YAML files, runs experiments E1-E7 in
sequence, executes all models in parallel within each experiment using
asyncio, and writes structured results, logs, checkpoints, and cost data.

The daemon is designed to be resilient:
    - Catches all exceptions at every level (trial, model, experiment).
    - Logs errors and continues to the next trial/model/experiment.
    - Saves checkpoints after every N trials for crash recovery.
    - Tracks cost in real time and stops if budget is exceeded.
    - Writes a live ``status.json`` for external monitoring.

E7 (Token-Efficient Testing) is a specialized experiment that compares
five testing approaches (fixed-N, SPRT-only, SPRT+fingerprint,
SPRT+fingerprint+budget, full system) across all models and scenarios.
It measures cost savings while maintaining equivalent statistical power.

Usage::

    # Run all experiments
    python -m experiments.runner.daemon

    # Run with custom config dir and resume
    python -m experiments.runner.daemon --config experiments/configs/ --resume

    # Run a single experiment
    python -m experiments.runner.daemon --experiment E1
    python -m experiments.runner.daemon --experiment e7_efficiency
"""

from __future__ import annotations

import argparse
import asyncio
import glob as glob_mod
import json
import logging
import os
import subprocess
import sys
import time
import traceback
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from experiments.runner.azure_adapter import AzureFoundryClient
from experiments.runner.checkpoint import CheckpointManager
from experiments.runner.cost_tracker import BudgetExceededError, CostTracker
from experiments.runner.evaluators import EVALUATOR_REGISTRY, get_evaluator

# Session 6 fixes: bulletproof self-healing
from experiments.runner.daemon_fixes import (
    save_checkpoint_with_blob_push,
    validate_trial_result,
    start_heartbeat,
    stop_heartbeat,
    update_heartbeat_stats,
    setup_signal_handlers,
    register_shutdown_handler,
)

# Lazy imports for E7 efficiency module (only loaded when E7 runs)
# from agentassay.efficiency import BehavioralFingerprint, AdaptiveBudgetOptimizer, TraceStore
# from agentassay.statistics import SPRTRunner, fisher_exact_regression

logger = logging.getLogger("agentassay.daemon")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CONFIG_DIR = PROJECT_ROOT / "experiments" / "configs"
DEFAULT_RESULTS_DIR = PROJECT_ROOT / "experiments" / "results"
DEFAULT_LOG_DIR = PROJECT_ROOT / "experiments" / "logs"
STATUS_FILE = PROJECT_ROOT / "status.json"

ALL_MODELS = [
    "gpt-5.2-chat",
    "claude-sonnet-4-6",
    "Mistral-Large-3",
    "Llama-4-Maverick-17B-128E-Instruct-FP8",
    "Phi-4",
]


# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _setup_logging(log_dir: Path, verbose: bool = False) -> None:
    """Configure structured logging to both file and console."""
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"daemon_{timestamp}.log"

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG if verbose else logging.INFO)

    # File handler — full detail
    fh = logging.FileHandler(log_file, encoding="utf-8")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    ))
    root_logger.addHandler(fh)

    # Console handler — summary
    ch = logging.StreamHandler(sys.stderr)
    ch.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch.setFormatter(logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S",
    ))
    root_logger.addHandler(ch)

    logger.info("Logging initialized: %s", log_file)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------

def load_experiment_configs(config_dir: Path) -> list[dict[str, Any]]:
    """Load and validate experiment configs from YAML files.

    Files are sorted by name to ensure deterministic ordering
    (E1.yaml < E2.yaml < ... < E7.yaml).

    Returns a list of experiment config dicts, each with at minimum:
        - experiment_id: str
        - name: str
        - scenarios: list[dict]
        - models: list[str]
        - trials_per_scenario: int
        - evaluator: str

    E7 uses a nested format with ``experiment.id`` — this loader
    normalises both formats into a flat dict with ``experiment_id``.
    """
    config_dir.mkdir(parents=True, exist_ok=True)
    configs: list[dict[str, Any]] = []

    yaml_files = sorted(config_dir.glob("*.yaml")) + sorted(
        config_dir.glob("*.yml")
    )

    if not yaml_files:
        logger.warning("No experiment configs found in %s", config_dir)
        return []

    for path in yaml_files:
        try:
            with open(path) as f:
                raw = yaml.safe_load(f)

            if not isinstance(raw, dict):
                logger.warning("Skipping %s — not a dict", path.name)
                continue

            # Validate required fields — support both flat and nested formats
            exp_id = raw.get("experiment_id")
            if not exp_id:
                # Try nested format: experiment.id (used by E7)
                exp_section = raw.get("experiment", {})
                if isinstance(exp_section, dict):
                    exp_id = exp_section.get("id")

            if not exp_id:
                logger.warning("Skipping %s — missing experiment_id", path.name)
                continue

            # Normalise: set top-level experiment_id for downstream code
            raw["experiment_id"] = exp_id

            # For nested configs (E7), preserve the raw YAML for the handler
            if "experiment" in raw and isinstance(raw["experiment"], dict):
                raw["_raw_yaml"] = dict(raw)

            # Normalise models — handle list-of-dicts format (E7)
            models_raw = raw.get("models", ALL_MODELS)
            if isinstance(models_raw, list) and models_raw and isinstance(models_raw[0], dict):
                raw["models"] = [m.get("name", m) for m in models_raw]
            else:
                raw.setdefault("models", ALL_MODELS)

            # Apply defaults for E1-E6 style configs
            raw.setdefault("trials_per_scenario", 50)
            raw.setdefault("evaluator", "ecommerce")
            raw.setdefault("scenarios", [])
            raw.setdefault("temperature", 0.7)
            raw.setdefault("max_steps", 10)
            raw["_source_file"] = str(path)

            configs.append(raw)
            logger.info(
                "Loaded experiment config: %s (%d scenarios, %d models)",
                exp_id,
                len(raw.get("scenarios", [])),
                len(raw.get("models", [])),
            )

        except Exception as exc:
            logger.error(
                "Failed to load config %s: %s", path.name, exc
            )
            continue

    return configs


# ---------------------------------------------------------------------------
# Status writer
# ---------------------------------------------------------------------------

def write_status(
    status_file: Path,
    current_experiment: str,
    progress: dict[str, Any],
    cost_summary: dict[str, Any],
    phase: str = "running",
    error: str | None = None,
) -> None:
    """Write a live status file for external monitoring.

    The status file is written atomically and includes:
        - Current experiment being run
        - Per-model progress (completed/total trials)
        - ETA estimates
        - Total cost so far
        - Phase (starting, running, completed, error)
    """
    status = {
        "phase": phase,
        "current_experiment": current_experiment,
        "progress": progress,
        "cost": cost_summary,
        "error": error,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "pid": os.getpid(),
    }

    try:
        import tempfile
        fd, tmp = tempfile.mkstemp(
            dir=str(status_file.parent),
            prefix=".status_",
            suffix=".json",
        )
        with os.fdopen(fd, "w") as f:
            json.dump(status, f, indent=2, default=str)
        os.replace(tmp, status_file)
    except Exception as exc:
        logger.debug("Failed to write status: %s", exc)


# ---------------------------------------------------------------------------
# Azure Blob Storage push (self-healing — never crashes the daemon)
# ---------------------------------------------------------------------------

# Track last blob push time to enforce 30-minute intervals
_last_blob_push_time: float = 0.0
_BLOB_PUSH_INTERVAL_SEC: float = 30.0 * 60.0  # 30 minutes


def _try_blob_push_sdk(
    results_dir: Path,
    log_dir: Path,
    status_file: Path,
) -> bool:
    """Push results to Azure Blob Storage using azure-storage-blob SDK.

    Returns True if push succeeded, False otherwise.
    This function NEVER raises — all exceptions are caught and logged.
    """
    conn_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING", "")
    container = os.environ.get("AZURE_STORAGE_CONTAINER", "experiment-results")

    if not conn_string:
        logger.debug("AZURE_STORAGE_CONNECTION_STRING not set — skipping blob push")
        return False

    try:
        from azure.storage.blob import BlobServiceClient

        service = BlobServiceClient.from_connection_string(conn_string)
        container_client = service.get_container_client(container)

        uploaded = 0

        # Push results/*.json
        if results_dir.exists():
            for fpath in results_dir.rglob("*.json"):
                blob_name = f"results/{fpath.relative_to(results_dir)}"
                try:
                    with open(fpath, "rb") as data:
                        container_client.upload_blob(
                            name=blob_name,
                            data=data,
                            overwrite=True,
                        )
                    uploaded += 1
                except Exception as exc:
                    logger.debug("Blob upload failed for %s: %s", blob_name, exc)

        # Push logs/*.log
        if log_dir.exists():
            for fpath in log_dir.rglob("*.log"):
                blob_name = f"logs/{fpath.relative_to(log_dir)}"
                try:
                    with open(fpath, "rb") as data:
                        container_client.upload_blob(
                            name=blob_name,
                            data=data,
                            overwrite=True,
                        )
                    uploaded += 1
                except Exception as exc:
                    logger.debug("Blob upload failed for %s: %s", blob_name, exc)

        # Push status.json
        if status_file.exists():
            try:
                with open(status_file, "rb") as data:
                    container_client.upload_blob(
                        name="status.json",
                        data=data,
                        overwrite=True,
                    )
                uploaded += 1
            except Exception as exc:
                logger.debug("Blob upload failed for status.json: %s", exc)

        # Push cost_log.json (lives in results dir)
        cost_log = results_dir / "cost_log.json"
        if cost_log.exists():
            try:
                with open(cost_log, "rb") as data:
                    container_client.upload_blob(
                        name="cost_log.json",
                        data=data,
                        overwrite=True,
                    )
                uploaded += 1
            except Exception as exc:
                logger.debug("Blob upload failed for cost_log.json: %s", exc)

        logger.info("Blob push complete: %d files uploaded to %s/%s", uploaded, container, "")
        return True

    except ImportError:
        logger.warning(
            "azure-storage-blob not installed — falling back to az CLI for blob push"
        )
        return False
    except Exception as exc:
        logger.warning("Blob push (SDK) failed: %s", exc)
        return False


def _try_blob_push_cli(
    results_dir: Path,
    log_dir: Path,
    status_file: Path,
) -> bool:
    """Fallback: push results using az storage blob upload-batch CLI.

    Returns True if push succeeded, False otherwise.
    """
    account = os.environ.get("AZURE_STORAGE_ACCOUNT", "")
    sas_token = os.environ.get("AZURE_STORAGE_SAS_TOKEN", "")
    container = os.environ.get("AZURE_STORAGE_CONTAINER", "experiment-results")

    if not account or not sas_token:
        logger.debug("AZURE_STORAGE_ACCOUNT or SAS_TOKEN not set — skipping CLI blob push")
        return False

    try:
        uploaded = 0

        # Upload results directory
        if results_dir.exists() and any(results_dir.rglob("*.json")):
            result = subprocess.run(
                [
                    "az", "storage", "blob", "upload-batch",
                    "--account-name", account,
                    "--destination", container,
                    "--source", str(results_dir),
                    "--destination-path", "results",
                    "--sas-token", sas_token,
                    "--overwrite",
                    "--no-progress",
                    "-o", "none",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                uploaded += 1
            else:
                logger.debug("az blob upload-batch (results) failed: %s", result.stderr)

        # Upload logs directory
        if log_dir.exists() and any(log_dir.rglob("*.log")):
            result = subprocess.run(
                [
                    "az", "storage", "blob", "upload-batch",
                    "--account-name", account,
                    "--destination", container,
                    "--source", str(log_dir),
                    "--destination-path", "logs",
                    "--sas-token", sas_token,
                    "--overwrite",
                    "--no-progress",
                    "-o", "none",
                ],
                capture_output=True,
                text=True,
                timeout=300,
            )
            if result.returncode == 0:
                uploaded += 1
            else:
                logger.debug("az blob upload-batch (logs) failed: %s", result.stderr)

        # Upload status.json (single file)
        if status_file.exists():
            result = subprocess.run(
                [
                    "az", "storage", "blob", "upload",
                    "--account-name", account,
                    "--container-name", container,
                    "--file", str(status_file),
                    "--name", "status.json",
                    "--sas-token", sas_token,
                    "--overwrite",
                    "--no-progress",
                    "-o", "none",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                uploaded += 1

        # Upload cost_log.json
        cost_log = results_dir / "cost_log.json"
        if cost_log.exists():
            result = subprocess.run(
                [
                    "az", "storage", "blob", "upload",
                    "--account-name", account,
                    "--container-name", container,
                    "--file", str(cost_log),
                    "--name", "cost_log.json",
                    "--sas-token", sas_token,
                    "--overwrite",
                    "--no-progress",
                    "-o", "none",
                ],
                capture_output=True,
                text=True,
                timeout=60,
            )
            if result.returncode == 0:
                uploaded += 1

        logger.info("Blob push (CLI) complete: %d batches uploaded", uploaded)
        return uploaded > 0

    except Exception as exc:
        logger.warning("Blob push (CLI) failed: %s", exc)
        return False


def push_results_to_blob(
    results_dir: Path,
    log_dir: Path,
    status_file: Path,
    force: bool = False,
) -> None:
    """Push experiment results to Azure Blob Storage.

    Called after each experiment batch or every 30 minutes.
    Self-healing: if push fails, logs a warning and continues.
    The daemon NEVER crashes due to a blob push failure.

    Parameters
    ----------
    results_dir
        Directory containing experiment result JSON files.
    log_dir
        Directory containing daemon log files.
    status_file
        Path to the live status.json file.
    force
        If True, push regardless of the 30-minute interval.
    """
    global _last_blob_push_time

    now = time.monotonic()
    elapsed = now - _last_blob_push_time

    if not force and elapsed < _BLOB_PUSH_INTERVAL_SEC:
        logger.debug(
            "Skipping blob push: %.0fs since last push (interval: %.0fs)",
            elapsed,
            _BLOB_PUSH_INTERVAL_SEC,
        )
        return

    logger.info("Pushing results to Azure Blob Storage...")

    # Try SDK first (faster, more reliable), fall back to CLI
    success = _try_blob_push_sdk(results_dir, log_dir, status_file)
    if not success:
        success = _try_blob_push_cli(results_dir, log_dir, status_file)

    if success:
        _last_blob_push_time = now
        logger.info("Blob push succeeded at %s", datetime.now(timezone.utc).isoformat())
    else:
        logger.warning(
            "Blob push failed (both SDK and CLI). "
            "Experiments continue — will retry in %.0f minutes.",
            _BLOB_PUSH_INTERVAL_SEC / 60.0,
        )


# ---------------------------------------------------------------------------
# Single trial execution
# ---------------------------------------------------------------------------

async def run_single_trial(
    client: AzureFoundryClient,
    model: str,
    scenario: dict[str, Any],
    experiment_config: dict[str, Any],
    evaluator_fn: Any,
    trial_index: int,
) -> dict[str, Any]:
    """Execute one agent trial and evaluate the result.

    Returns a serializable result dict suitable for checkpointing.
    """
    trial_id = str(uuid.uuid4())
    scenario_id = scenario.get("scenario_id", "unknown")
    start_time = time.monotonic()

    try:
        # Build tool definitions from scenario
        tools = scenario.get("tools", [])
        system_prompt = scenario.get("system_prompt", "You are a helpful agent.")
        user_input = scenario.get("user_input", "")
        max_steps = experiment_config.get("max_steps", 10)
        temperature = experiment_config.get("temperature", 0.7)

        # Build a tool executor from scenario tool definitions
        tool_responses = scenario.get("tool_responses", {})

        def tool_executor(name: str, args: dict) -> str:
            """Look up pre-defined tool responses for deterministic evaluation."""
            if name in tool_responses:
                response = tool_responses[name]
                if isinstance(response, dict):
                    return json.dumps(response)
                return str(response)
            return json.dumps({"status": "ok", "result": f"Result for {name}"})

        # Run the agent
        agent_result = await client.run_agent(
            model=model,
            system_prompt=system_prompt,
            user_input=user_input,
            tools=tools,
            max_steps=max_steps,
            temperature=temperature,
            tool_executor=tool_executor,
        )

        duration_ms = (time.monotonic() - start_time) * 1000.0

        # Evaluate
        expected = scenario.get("expected", {})
        eval_result = evaluator_fn(agent_result, expected)

        return {
            "trial_id": trial_id,
            "trial_index": trial_index,
            "scenario_id": scenario_id,
            "model": model,
            "passed": eval_result.get("passed", False),
            "score": eval_result.get("score", 0.0),
            "evaluation_details": eval_result.get("evaluation_details", {}),
            "success": agent_result.get("success", False),
            "error": agent_result.get("error"),
            "duration_ms": round(duration_ms, 2),
            "cost_usd": agent_result.get("total_cost_usd", 0.0),
            "tokens": agent_result.get("total_tokens", 0),
            "step_count": agent_result.get("step_count", 0),
            "_steps": agent_result.get("steps", []),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    except Exception as exc:
        duration_ms = (time.monotonic() - start_time) * 1000.0
        logger.error(
            "Trial %s failed (model=%s, scenario=%s): %s\n%s",
            trial_id,
            model,
            scenario_id,
            exc,
            traceback.format_exc(),
        )
        return {
            "trial_id": trial_id,
            "trial_index": trial_index,
            "scenario_id": scenario_id,
            "model": model,
            "passed": False,
            "score": 0.0,
            "evaluation_details": {"error": str(exc)},
            "success": False,
            "error": f"{type(exc).__name__}: {exc}",
            "duration_ms": round(duration_ms, 2),
            "cost_usd": 0.0,
            "tokens": 0,
            "step_count": 0,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }


# ---------------------------------------------------------------------------
# Model-level execution (all trials for one model on one scenario)
# ---------------------------------------------------------------------------

async def run_model_trials(
    client: AzureFoundryClient,
    model: str,
    scenario: dict[str, Any],
    experiment_config: dict[str, Any],
    evaluator_fn: Any,
    checkpoint: CheckpointManager,
    cost: CostTracker,
    experiment_id: str,
    total_trials: int,
) -> list[dict[str, Any]]:
    """Run all trials for one model on one scenario, with checkpoint/resume.

    Returns the complete list of trial results.
    """
    scenario_id = scenario.get("scenario_id", "unknown")

    # Check for existing checkpoint
    existing = checkpoint.load(experiment_id, model, scenario_id)
    start_from = 0
    results: list[dict[str, Any]] = []

    if existing:
        start_from = existing.get("completed", 0)
        results = existing.get("results", [])
        if start_from >= total_trials:
            logger.info(
                "Skipping %s/%s — already complete (%d/%d)",
                model,
                scenario_id,
                start_from,
                total_trials,
            )
            return results
        logger.info(
            "Resuming %s/%s from trial %d/%d",
            model,
            scenario_id,
            start_from,
            total_trials,
        )

    for trial_idx in range(start_from, total_trials):
        # Budget check
        if cost.is_budget_exceeded():
            logger.warning(
                "Budget exceeded — stopping %s/%s at trial %d",
                model,
                scenario_id,
                trial_idx,
            )
            break

        try:
            result = await run_single_trial(
                client=client,
                model=model,
                scenario=scenario,
                experiment_config=experiment_config,
                evaluator_fn=evaluator_fn,
                trial_index=trial_idx,
            )

            # FIX 4: Token validation - reject fake data
            if validate_trial_result(result):
                results.append(result)
            else:
                # Invalid result - append a failed trial instead
                logger.error(
                    "Trial %d for %s/%s produced invalid data — marking as failed",
                    trial_idx,
                    model,
                    scenario_id,
                )
                results.append({
                    "trial_id": str(uuid.uuid4()),
                    "trial_index": trial_idx,
                    "scenario_id": scenario_id,
                    "model": model,
                    "passed": False,
                    "score": 0.0,
                    "error": "INVALID_DATA: success=True but tokens=0",
                    "success": False,
                    "duration_ms": 0.0,
                    "cost_usd": 0.0,
                    "tokens": 0,
                    "step_count": 0,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                })

        except Exception as exc:
            logger.error(
                "Unhandled error in trial %d for %s/%s: %s",
                trial_idx,
                model,
                scenario_id,
                exc,
            )
            results.append({
                "trial_id": str(uuid.uuid4()),
                "trial_index": trial_idx,
                "scenario_id": scenario_id,
                "model": model,
                "passed": False,
                "score": 0.0,
                "error": str(exc),
                "success": False,
                "duration_ms": 0.0,
                "cost_usd": 0.0,
                "tokens": 0,
                "step_count": 0,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })

            # FIX 1: Checkpoint-on-error - save state before continuing
            logger.warning(
                "Saving emergency checkpoint after error (trial %d, %s/%s)",
                trial_idx,
                model,
                scenario_id,
            )
            save_checkpoint_with_blob_push(
                checkpoint_manager=checkpoint,
                push_blob_fn=push_results_to_blob,
                results_dir=DEFAULT_RESULTS_DIR,
                log_dir=DEFAULT_LOG_DIR,
                status_file=STATUS_FILE,
                experiment_id=experiment_id,
                model=model,
                scenario=scenario_id,
                completed=len(results),
                results=results,
                metadata={"last_error": str(exc), "error_at_trial": trial_idx},
            )

        # FIX 2: Periodic checkpoint with immediate blob push
        if len(results) % checkpoint.save_interval == 0:
            save_checkpoint_with_blob_push(
                checkpoint_manager=checkpoint,
                push_blob_fn=push_results_to_blob,
                results_dir=DEFAULT_RESULTS_DIR,
                log_dir=DEFAULT_LOG_DIR,
                status_file=STATUS_FILE,
                experiment_id=experiment_id,
                model=model,
                scenario=scenario_id,
                completed=len(results),
                results=results,
            )

        # FIX 3: Update heartbeat stats
        update_heartbeat_stats(
            experiment=experiment_id,
            model=model,
            trials=len(results),
        )

    # FIX 2: Final checkpoint save with immediate blob push
    save_checkpoint_with_blob_push(
        checkpoint_manager=checkpoint,
        push_blob_fn=push_results_to_blob,
        results_dir=DEFAULT_RESULTS_DIR,
        log_dir=DEFAULT_LOG_DIR,
        status_file=STATUS_FILE,
        experiment_id=experiment_id,
        model=model,
        scenario=scenario_id,
        completed=len(results),
        results=results,
    )

    passed = sum(1 for r in results if r.get("passed"))
    logger.info(
        "Completed %s/%s: %d/%d trials, %d passed (%.1f%%)",
        model,
        scenario_id,
        len(results),
        total_trials,
        passed,
        (passed / len(results) * 100) if results else 0,
    )

    return results


# ---------------------------------------------------------------------------
# E7: Token-Efficient Testing — approach runners
# ---------------------------------------------------------------------------

def _load_e7_dependencies() -> dict[str, Any]:
    """Lazy-load efficiency module classes for E7.

    Returns a dict of imported symbols. This avoids importing heavy
    dependencies for experiments E1-E6 that do not need them.
    """
    from agentassay.efficiency import (
        BehavioralFingerprint,
        AdaptiveBudgetOptimizer,
        TraceStore,
    )
    from agentassay.statistics import SPRTRunner, fisher_exact_regression

    return {
        "BehavioralFingerprint": BehavioralFingerprint,
        "AdaptiveBudgetOptimizer": AdaptiveBudgetOptimizer,
        "TraceStore": TraceStore,
        "SPRTRunner": SPRTRunner,
        "fisher_exact_regression": fisher_exact_regression,
    }


async def _run_e7_baseline_trials(
    client: AzureFoundryClient,
    model: str,
    scenario: dict[str, Any],
    experiment_config: dict[str, Any],
    evaluator_fn: Any,
    n_trials: int,
) -> list[dict[str, Any]]:
    """Run N baseline trials for E7 (shared across all approaches).

    These trials establish the baseline pass rate and are reused by every
    approach so that cost comparisons are fair.
    """
    results: list[dict[str, Any]] = []

    for idx in range(n_trials):
        result = await run_single_trial(
            client=client,
            model=model,
            scenario=scenario,
            experiment_config=experiment_config,
            evaluator_fn=evaluator_fn,
            trial_index=idx,
        )
        results.append(result)

    return results


def _inject_e7_regression(
    scenario: dict[str, Any],
    regression_type: str = "prompt_degradation",
) -> dict[str, Any]:
    """Create a degraded copy of a scenario for regression injection.

    The returned scenario is identical to the input except the system
    prompt has critical instructions removed, simulating a realistic
    regression.
    """
    degraded = dict(scenario)  # shallow copy
    original_prompt = degraded.get("system_prompt", "")

    if regression_type == "prompt_degradation":
        # Remove the last 30% of the system prompt to simulate
        # instruction loss (a common real-world regression vector)
        cutoff = int(len(original_prompt) * 0.7)
        degraded["system_prompt"] = original_prompt[:cutoff]
    elif regression_type == "temperature_spike":
        # Raise temperature to inject stochastic degradation
        degraded["temperature"] = min(
            experiment_config_temperature(degraded) + 0.5, 2.0
        )
    else:
        degraded["system_prompt"] = original_prompt[:int(len(original_prompt) * 0.7)]

    return degraded


def experiment_config_temperature(config: dict[str, Any]) -> float:
    """Extract temperature from config, defaulting to 0.7."""
    return float(config.get("temperature", 0.7))


async def _run_e7_approach_fixed_n(
    client: AzureFoundryClient,
    model: str,
    degraded_scenario: dict[str, Any],
    experiment_config: dict[str, Any],
    evaluator_fn: Any,
    baseline_results: list[dict[str, Any]],
    approach_config: dict[str, Any],
    deps: dict[str, Any],
) -> dict[str, Any]:
    """Run the fixed-N approach: 100 candidate trials + Fisher exact test.

    This is the baseline approach with no optimization. Every candidate
    trial is run regardless of accumulating evidence.
    """
    n_trials = approach_config.get("trials", 100)
    candidate_results: list[dict[str, Any]] = []
    total_tokens = 0
    total_cost = 0.0

    for idx in range(n_trials):
        result = await run_single_trial(
            client=client,
            model=model,
            scenario=degraded_scenario,
            experiment_config=experiment_config,
            evaluator_fn=evaluator_fn,
            trial_index=idx,
        )
        candidate_results.append(result)
        total_tokens += result.get("tokens", 0)
        total_cost += result.get("cost_usd", 0.0)

    # Fisher exact test
    fisher = deps["fisher_exact_regression"]
    baseline_passes = sum(1 for r in baseline_results if r.get("passed"))
    candidate_passes = sum(1 for r in candidate_results if r.get("passed"))

    test_result = fisher(
        baseline_passes=baseline_passes,
        baseline_n=len(baseline_results),
        current_passes=candidate_passes,
        current_n=len(candidate_results),
    )

    return {
        "approach": "fixed_n",
        "trials_used": n_trials,
        "tokens_used": total_tokens,
        "cost_usd": total_cost,
        "verdict": "FAIL" if test_result.significant else "PASS",
        "p_value": test_result.p_value,
        "power": 1.0 if test_result.significant else 0.0,
        "baseline_pass_rate": baseline_passes / len(baseline_results) if baseline_results else 0.0,
        "candidate_pass_rate": candidate_passes / n_trials if n_trials > 0 else 0.0,
    }


async def _run_e7_approach_sprt_only(
    client: AzureFoundryClient,
    model: str,
    degraded_scenario: dict[str, Any],
    experiment_config: dict[str, Any],
    evaluator_fn: Any,
    baseline_results: list[dict[str, Any]],
    approach_config: dict[str, Any],
    params: dict[str, Any],
    deps: dict[str, Any],
) -> dict[str, Any]:
    """Run the SPRT-only approach: sequential stopping without fingerprinting.

    The SPRT monitors each trial sequentially and stops as soon as there
    is sufficient evidence to accept H0 (no regression) or H1 (regression).
    """
    max_trials = approach_config.get("max_trials", 200)
    alpha = params.get("alpha", 0.05)
    beta = params.get("beta", 0.10)
    delta = params.get("delta", 0.10)

    baseline_passes = sum(1 for r in baseline_results if r.get("passed"))
    # Clamp p0 to open interval (0,1) — SPRT requires strict bounds
    # Ensure minimum gap of 0.005 between p0 and p1 for numerical stability
    p0_raw = baseline_passes / len(baseline_results) if baseline_results else 0.9
    p0 = max(0.01, min(0.999, p0_raw))
    p1 = max(0.001, p0 - delta)
    if p1 >= p0 - 0.005:
        p1 = max(0.001, p0 / 2.0)  # Fall back to half of p0

    SPRTRunner = deps["SPRTRunner"]
    sprt = SPRTRunner(p0=p0, p1=p1, alpha=alpha, beta=beta)

    total_tokens = 0
    total_cost = 0.0
    trials_used = 0
    sprt_result = None

    for idx in range(max_trials):
        result = await run_single_trial(
            client=client,
            model=model,
            scenario=degraded_scenario,
            experiment_config=experiment_config,
            evaluator_fn=evaluator_fn,
            trial_index=idx,
        )
        total_tokens += result.get("tokens", 0)
        total_cost += result.get("cost_usd", 0.0)
        trials_used += 1

        sprt_result = sprt.update(passed=result.get("passed", False))
        if sprt_result.decision != "continue":
            break

    decision = sprt_result.decision if sprt_result else "continue"
    verdict = "FAIL" if decision == "accept_h1" else ("PASS" if decision == "accept_h0" else "INCONCLUSIVE")

    return {
        "approach": "sprt_only",
        "trials_used": trials_used,
        "tokens_used": total_tokens,
        "cost_usd": total_cost,
        "verdict": verdict,
        "sprt_decision": decision,
        "power": 1.0 if verdict == "FAIL" else 0.0,
        "baseline_pass_rate": p0,
    }


async def _run_e7_approach_sprt_fingerprint(
    client: AzureFoundryClient,
    model: str,
    degraded_scenario: dict[str, Any],
    experiment_config: dict[str, Any],
    evaluator_fn: Any,
    baseline_results: list[dict[str, Any]],
    approach_config: dict[str, Any],
    params: dict[str, Any],
    deps: dict[str, Any],
) -> dict[str, Any]:
    """Run SPRT + behavioral fingerprinting approach.

    Uses Hotelling's T-squared on behavioral fingerprints to potentially
    detect regressions in fewer trials. The fingerprints capture
    multi-dimensional behavioral signals (tool usage patterns, step
    counts, output characteristics) that may diverge before pass/fail
    rates do.
    """
    max_trials = approach_config.get("max_trials", 200)
    alpha = params.get("alpha", 0.05)
    beta = params.get("beta", 0.10)
    delta = params.get("delta", 0.10)

    baseline_passes = sum(1 for r in baseline_results if r.get("passed"))
    # Clamp p0 to open interval (0,1) — SPRT requires strict bounds
    # Ensure minimum gap of 0.005 between p0 and p1 for numerical stability
    p0_raw = baseline_passes / len(baseline_results) if baseline_results else 0.9
    p0 = max(0.01, min(0.999, p0_raw))
    p1 = max(0.001, p0 - delta)
    if p1 >= p0 - 0.005:
        p1 = max(0.001, p0 / 2.0)  # Fall back to half of p0

    SPRTRunner = deps["SPRTRunner"]
    BehavioralFingerprint = deps["BehavioralFingerprint"]

    sprt = SPRTRunner(p0=p0, p1=p1, alpha=alpha, beta=beta)

    # Compute baseline fingerprints
    baseline_fingerprints = [
        BehavioralFingerprint.from_trial_result(r) for r in baseline_results
    ]

    total_tokens = 0
    total_cost = 0.0
    trials_used = 0
    candidate_fingerprints: list[Any] = []
    sprt_result = None
    fingerprint_diverged = False

    for idx in range(max_trials):
        result = await run_single_trial(
            client=client,
            model=model,
            scenario=degraded_scenario,
            experiment_config=experiment_config,
            evaluator_fn=evaluator_fn,
            trial_index=idx,
        )
        total_tokens += result.get("tokens", 0)
        total_cost += result.get("cost_usd", 0.0)
        trials_used += 1

        # Update SPRT
        sprt_result = sprt.update(passed=result.get("passed", False))

        # Update fingerprints — check for divergence after minimum samples
        candidate_fingerprints.append(
            BehavioralFingerprint.from_trial_result(result)
        )

        if len(candidate_fingerprints) >= 10:
            try:
                diverged = BehavioralFingerprint.hotelling_t2_test(
                    baseline_fingerprints,
                    candidate_fingerprints,
                    alpha=alpha,
                )
                if diverged:
                    fingerprint_diverged = True
            except Exception:
                # If fingerprint test fails (e.g., singular covariance),
                # fall back to SPRT-only decision
                pass

        # Stop if SPRT reaches a decision OR fingerprints diverge
        if sprt_result.decision != "continue":
            break
        if fingerprint_diverged and trials_used >= 20:
            # Fingerprint divergence is strong evidence of regression,
            # but require at least 20 trials to avoid false positives
            break

    decision = sprt_result.decision if sprt_result else "continue"
    if fingerprint_diverged and decision == "continue":
        decision = "accept_h1"  # fingerprints detected the regression

    verdict = "FAIL" if decision == "accept_h1" else ("PASS" if decision == "accept_h0" else "INCONCLUSIVE")

    return {
        "approach": "sprt_fingerprint",
        "trials_used": trials_used,
        "tokens_used": total_tokens,
        "cost_usd": total_cost,
        "verdict": verdict,
        "sprt_decision": sprt_result.decision if sprt_result else "continue",
        "fingerprint_diverged": fingerprint_diverged,
        "power": 1.0 if verdict == "FAIL" else 0.0,
        "baseline_pass_rate": p0,
    }


async def _run_e7_approach_sprt_fp_budget(
    client: AzureFoundryClient,
    model: str,
    degraded_scenario: dict[str, Any],
    experiment_config: dict[str, Any],
    evaluator_fn: Any,
    baseline_results: list[dict[str, Any]],
    approach_config: dict[str, Any],
    params: dict[str, Any],
    deps: dict[str, Any],
) -> dict[str, Any]:
    """Run SPRT + fingerprinting + adaptive budget calibration.

    First runs a small calibration batch (default 10 traces), then uses
    the AdaptiveBudgetOptimizer to compute an optimal N for the
    remaining trials. This avoids over-testing high-variance agents
    and under-testing low-variance ones.
    """
    calibration_size = approach_config.get("calibration_size", 10)
    alpha = params.get("alpha", 0.05)
    beta = params.get("beta", 0.10)
    delta = params.get("delta", 0.10)

    baseline_passes = sum(1 for r in baseline_results if r.get("passed"))
    # Clamp p0 to open interval (0,1) — SPRT requires strict bounds
    # Ensure minimum gap of 0.005 between p0 and p1 for numerical stability
    p0_raw = baseline_passes / len(baseline_results) if baseline_results else 0.9
    p0 = max(0.01, min(0.999, p0_raw))
    p1 = max(0.001, p0 - delta)
    if p1 >= p0 - 0.005:
        p1 = max(0.001, p0 / 2.0)  # Fall back to half of p0

    SPRTRunner = deps["SPRTRunner"]
    BehavioralFingerprint = deps["BehavioralFingerprint"]
    AdaptiveBudgetOptimizer = deps["AdaptiveBudgetOptimizer"]

    total_tokens = 0
    total_cost = 0.0
    trials_used = 0

    # Phase 1: calibration — run a small batch to estimate variance
    calibration_results: list[dict[str, Any]] = []
    for idx in range(calibration_size):
        result = await run_single_trial(
            client=client,
            model=model,
            scenario=degraded_scenario,
            experiment_config=experiment_config,
            evaluator_fn=evaluator_fn,
            trial_index=idx,
        )
        calibration_results.append(result)
        total_tokens += result.get("tokens", 0)
        total_cost += result.get("cost_usd", 0.0)
        trials_used += 1

    # Phase 2: compute optimal budget from calibration fingerprints
    optimizer = AdaptiveBudgetOptimizer(
        alpha=alpha, beta=beta, delta=delta
    )
    calibration_fingerprints = [
        BehavioralFingerprint.from_trial_result(r) for r in calibration_results
    ]
    per_trial_cost = sum(
        r.get("cost_usd", 0.0) for r in calibration_results
    ) / max(len(calibration_results), 1)
    budget_estimate = optimizer.calibrate_from_fingerprints(
        fingerprints=calibration_fingerprints,
        per_trial_cost_usd=per_trial_cost,
    )
    optimal_n = budget_estimate.recommended_n
    # Cap at 200 and account for calibration trials already run
    remaining_n = min(optimal_n, 200) - calibration_size
    remaining_n = max(remaining_n, 0)

    # Phase 3: SPRT with fingerprinting on remaining trials
    sprt = SPRTRunner(p0=p0, p1=p1, alpha=alpha, beta=beta)

    # Feed calibration results into SPRT first
    sprt_result = None
    for r in calibration_results:
        sprt_result = sprt.update(passed=r.get("passed", False))
        if sprt_result.decision != "continue":
            break

    baseline_fingerprints = [
        BehavioralFingerprint.from_trial_result(r) for r in baseline_results
    ]
    candidate_fingerprints = [
        BehavioralFingerprint.from_trial_result(r) for r in calibration_results
    ]
    fingerprint_diverged = False

    if sprt_result is None or sprt_result.decision == "continue":
        for idx in range(remaining_n):
            result = await run_single_trial(
                client=client,
                model=model,
                scenario=degraded_scenario,
                experiment_config=experiment_config,
                evaluator_fn=evaluator_fn,
                trial_index=calibration_size + idx,
            )
            total_tokens += result.get("tokens", 0)
            total_cost += result.get("cost_usd", 0.0)
            trials_used += 1

            sprt_result = sprt.update(passed=result.get("passed", False))
            candidate_fingerprints.append(
                BehavioralFingerprint.from_trial_result(result)
            )

            # Fingerprint check
            if len(candidate_fingerprints) >= 10:
                try:
                    diverged = BehavioralFingerprint.hotelling_t2_test(
                        baseline_fingerprints,
                        candidate_fingerprints,
                        alpha=alpha,
                    )
                    if diverged:
                        fingerprint_diverged = True
                except Exception:
                    pass

            if sprt_result.decision != "continue":
                break
            if fingerprint_diverged and trials_used >= 20:
                break

    decision = sprt_result.decision if sprt_result else "continue"
    if fingerprint_diverged and decision == "continue":
        decision = "accept_h1"

    verdict = "FAIL" if decision == "accept_h1" else ("PASS" if decision == "accept_h0" else "INCONCLUSIVE")

    return {
        "approach": "sprt_fp_budget",
        "trials_used": trials_used,
        "tokens_used": total_tokens,
        "cost_usd": total_cost,
        "verdict": verdict,
        "optimal_n": optimal_n,
        "calibration_size": calibration_size,
        "fingerprint_diverged": fingerprint_diverged,
        "power": 1.0 if verdict == "FAIL" else 0.0,
        "baseline_pass_rate": p0,
    }


async def _run_e7_approach_full_system(
    client: AzureFoundryClient,
    model: str,
    degraded_scenario: dict[str, Any],
    experiment_config: dict[str, Any],
    evaluator_fn: Any,
    baseline_results: list[dict[str, Any]],
    approach_config: dict[str, Any],
    params: dict[str, Any],
    deps: dict[str, Any],
) -> dict[str, Any]:
    """Run the full system: trace-first + fingerprinting + budget + SPRT.

    The trace store enables offline analysis of previously collected
    traces before running any new live trials. This is the most
    cost-efficient approach: it may detect regressions purely from
    trace analysis (zero additional API calls) or determine that only
    a small number of confirmatory trials are needed.
    """
    calibration_size = approach_config.get("calibration_size", 10)
    alpha = params.get("alpha", 0.05)
    beta = params.get("beta", 0.10)
    delta = params.get("delta", 0.10)

    baseline_passes = sum(1 for r in baseline_results if r.get("passed"))
    # Clamp p0 to open interval (0,1) — SPRT requires strict bounds
    # Ensure minimum gap of 0.005 between p0 and p1 for numerical stability
    p0_raw = baseline_passes / len(baseline_results) if baseline_results else 0.9
    p0 = max(0.01, min(0.999, p0_raw))
    p1 = max(0.001, p0 - delta)
    if p1 >= p0 - 0.005:
        p1 = max(0.001, p0 / 2.0)  # Fall back to half of p0

    SPRTRunner = deps["SPRTRunner"]
    BehavioralFingerprint = deps["BehavioralFingerprint"]
    AdaptiveBudgetOptimizer = deps["AdaptiveBudgetOptimizer"]
    TraceStore = deps["TraceStore"]

    total_tokens = 0
    total_cost = 0.0
    trials_used = 0

    # Phase 1: trace-first analysis — check if baseline traces already
    # contain enough signal to detect changes without new API calls
    trace_store = TraceStore()
    for r in baseline_results:
        # Convert trial result to ExecutionTrace for trace store
        from agentassay.core.models import ExecutionTrace, StepTrace
        steps_raw = r.get("_steps", [])
        step_traces = []
        for s in steps_raw:
            st = StepTrace(
                step_index=s.get("step_index", 0),
                action=s.get("action", "llm_response"),
                tool_name=s.get("tool_name"),
                tool_input=s.get("tool_input"),
                tool_output=s.get("tool_output"),
                llm_output=s.get("llm_output"),
                model=s.get("model"),
                duration_ms=s.get("duration_ms", 0.0),
                metadata=s.get("usage", {}) if s.get("usage") else {},
            )
            step_traces.append(st)
        trace = ExecutionTrace(
            trace_id=r.get("trial_id", ""),
            scenario_id=r.get("scenario_id", "unknown"),
            steps=step_traces,
            output_data=r.get("evaluation_details", {}),
            success=r.get("success", False),
            error=r.get("error"),
            total_duration_ms=r.get("duration_ms", 0.0),
            total_cost_usd=r.get("cost_usd", 0.0),
            model=r.get("model", "unknown"),
            framework="azure_foundry",
            metadata={"tokens": r.get("tokens", 0)},
        )
        trace_store.record(trace)

    # Attempt offline regression detection via drift_detection
    try:
        drift_results = trace_store.drift_detection(alpha=alpha)
        if drift_results and any(d.get("drift_detected") for d in drift_results):
            offline_verdict = "regression"
        elif drift_results:
            offline_verdict = "no_regression"
        else:
            offline_verdict = None
    except Exception:
        offline_verdict = None

    if offline_verdict is not None and offline_verdict != "inconclusive":
        # Trace store was able to make a determination without any
        # new API calls — maximum cost savings
        verdict = "FAIL" if offline_verdict == "regression" else "PASS"
        return {
            "approach": "full_system",
            "trials_used": 0,
            "tokens_used": 0,
            "cost_usd": 0.0,
            "verdict": verdict,
            "trace_first_resolved": True,
            "optimal_n": 0,
            "calibration_size": 0,
            "fingerprint_diverged": False,
            "power": 1.0 if verdict == "FAIL" else 0.0,
            "baseline_pass_rate": p0,
        }

    # Phase 2: calibration batch (small set of live trials)
    calibration_results: list[dict[str, Any]] = []
    for idx in range(calibration_size):
        result = await run_single_trial(
            client=client,
            model=model,
            scenario=degraded_scenario,
            experiment_config=experiment_config,
            evaluator_fn=evaluator_fn,
            trial_index=idx,
        )
        calibration_results.append(result)
        total_tokens += result.get("tokens", 0)
        total_cost += result.get("cost_usd", 0.0)
        trials_used += 1

    # Phase 3: adaptive budget from calibration fingerprints
    optimizer = AdaptiveBudgetOptimizer(
        alpha=alpha, beta=beta, delta=delta
    )
    calibration_fingerprints = [
        BehavioralFingerprint.from_trial_result(r) for r in calibration_results
    ]
    per_trial_cost = sum(
        r.get("cost_usd", 0.0) for r in calibration_results
    ) / max(len(calibration_results), 1)
    budget_estimate = optimizer.calibrate_from_fingerprints(
        fingerprints=calibration_fingerprints,
        per_trial_cost_usd=per_trial_cost,
    )
    optimal_n = budget_estimate.recommended_n
    remaining_n = min(optimal_n, 200) - calibration_size
    remaining_n = max(remaining_n, 0)

    # Phase 4: SPRT + fingerprinting on remaining budget
    sprt = SPRTRunner(p0=p0, p1=p1, alpha=alpha, beta=beta)

    sprt_result = None
    for r in calibration_results:
        sprt_result = sprt.update(passed=r.get("passed", False))
        if sprt_result.decision != "continue":
            break

    baseline_fingerprints = [
        BehavioralFingerprint.from_trial_result(r) for r in baseline_results
    ]
    candidate_fingerprints = [
        BehavioralFingerprint.from_trial_result(r) for r in calibration_results
    ]
    fingerprint_diverged = False

    if sprt_result is None or sprt_result.decision == "continue":
        for idx in range(remaining_n):
            result = await run_single_trial(
                client=client,
                model=model,
                scenario=degraded_scenario,
                experiment_config=experiment_config,
                evaluator_fn=evaluator_fn,
                trial_index=calibration_size + idx,
            )
            total_tokens += result.get("tokens", 0)
            total_cost += result.get("cost_usd", 0.0)
            trials_used += 1

            sprt_result = sprt.update(passed=result.get("passed", False))
            candidate_fingerprints.append(
                BehavioralFingerprint.from_trial_result(result)
            )

            if len(candidate_fingerprints) >= 10:
                try:
                    diverged = BehavioralFingerprint.hotelling_t2_test(
                        baseline_fingerprints,
                        candidate_fingerprints,
                        alpha=alpha,
                    )
                    if diverged:
                        fingerprint_diverged = True
                except Exception:
                    pass

            if sprt_result.decision != "continue":
                break
            if fingerprint_diverged and trials_used >= 15:
                break

    decision = sprt_result.decision if sprt_result else "continue"
    if fingerprint_diverged and decision == "continue":
        decision = "accept_h1"

    verdict = "FAIL" if decision == "accept_h1" else ("PASS" if decision == "accept_h0" else "INCONCLUSIVE")

    return {
        "approach": "full_system",
        "trials_used": trials_used,
        "tokens_used": total_tokens,
        "cost_usd": total_cost,
        "verdict": verdict,
        "trace_first_resolved": False,
        "optimal_n": optimal_n,
        "calibration_size": calibration_size,
        "fingerprint_diverged": fingerprint_diverged,
        "power": 1.0 if verdict == "FAIL" else 0.0,
        "baseline_pass_rate": p0,
    }


# Dispatch table for E7 approach runners
_E7_APPROACH_RUNNERS: dict[str, Any] = {
    "fixed_n": _run_e7_approach_fixed_n,
    "sprt_only": _run_e7_approach_sprt_only,
    "sprt_fingerprint": _run_e7_approach_sprt_fingerprint,
    "sprt_fp_budget": _run_e7_approach_sprt_fp_budget,
    "full_system": _run_e7_approach_full_system,
}


async def run_e7_experiment(
    config: dict[str, Any],
    client: AzureFoundryClient,
    checkpoint: CheckpointManager,
    cost: CostTracker,
    results_dir: Path,
) -> dict[str, Any]:
    """Run E7: Token-Efficient Testing Evaluation.

    This experiment is structurally different from E1-E6: instead of
    running uniform trial batches, it compares 5 testing approaches on
    each (model x scenario) pair, repeating each comparison multiple
    times for statistical reliability.

    Protocol per (model, scenario, repetition):
        1. Run 100 baseline trials (shared across approaches, run once)
        2. Inject regression into the scenario
        3. For each of the 5 approaches:
           a. Run the approach on the degraded scenario
           b. Record: trials_used, tokens_consumed, cost, verdict
        4. Save results

    Parameters
    ----------
    config
        E7 experiment configuration from YAML.
    client
        Azure AI Foundry client.
    checkpoint
        Checkpoint manager for resume support.
    cost
        Cost tracker with budget enforcement.
    results_dir
        Directory for writing E7 results.

    Returns
    -------
    dict
        Experiment-level summary.
    """
    experiment_id = config.get("experiment_id", "e7_efficiency")
    raw = config.get("_raw_yaml", config)

    # Parse config — handle both flat and nested structures
    exp_section = raw.get("experiment", {})
    if exp_section and isinstance(exp_section, dict):
        experiment_id = exp_section.get("id", experiment_id)

    models_raw = raw.get("models", config.get("models", ALL_MODELS))
    if isinstance(models_raw, list) and models_raw and isinstance(models_raw[0], dict):
        models = [m["name"] for m in models_raw]
    else:
        models = list(models_raw) if models_raw else list(ALL_MODELS)

    scenarios_raw = raw.get("scenarios", config.get("scenarios", []))
    params = raw.get("parameters", config.get("parameters", {}))
    approaches = params.get("approaches", [])
    repetitions = params.get("repetitions", 50)
    alpha = params.get("alpha", 0.05)
    beta = params.get("beta", 0.10)
    delta = params.get("delta", 0.10)

    logger.info(
        "=" * 60 + "\n"
        "Starting E7: Token-Efficient Testing\n"
        "  Models: %d | Scenarios: %d | Approaches: %d | Repetitions: %d\n"
        "=" * 60,
        len(models),
        len(scenarios_raw),
        len(approaches),
        repetitions,
    )

    # Load efficiency module dependencies
    try:
        deps = _load_e7_dependencies()
    except ImportError as exc:
        logger.error(
            "E7 requires the efficiency module. Install with: "
            "pip install -e '.[dev]'. Error: %s",
            exc,
        )
        return {"experiment_id": experiment_id, "error": f"Import failed: {exc}"}

    # Filter to available models
    available = set(client.available_models())
    active_models = [m for m in models if m in available]
    skipped_models = [m for m in models if m not in available]
    if skipped_models:
        logger.warning("Skipping unavailable models: %s", skipped_models)

    if not active_models:
        logger.error("No available models for E7")
        return {"experiment_id": experiment_id, "error": "No available models"}

    experiment_start = time.monotonic()
    all_results: dict[str, dict[str, list[dict[str, Any]]]] = {}

    # Resolve evaluator for each scenario
    scenario_evaluators: dict[str, Any] = {}
    for sc in scenarios_raw:
        sc_name = sc if isinstance(sc, str) else sc.get("scenario_id", "unknown")
        try:
            scenario_evaluators[sc_name] = get_evaluator(sc_name)
        except KeyError:
            # Fall back to ecommerce evaluator
            scenario_evaluators[sc_name] = get_evaluator("ecommerce")

    exp_results_dir = results_dir / experiment_id
    exp_results_dir.mkdir(parents=True, exist_ok=True)

    for sc_raw in scenarios_raw:
        sc_name = sc_raw if isinstance(sc_raw, str) else sc_raw.get("scenario_id", "unknown")
        evaluator_fn = scenario_evaluators.get(sc_name)
        if evaluator_fn is None:
            logger.warning("Skipping scenario %s — no evaluator", sc_name)
            continue

        # Build a minimal scenario dict for the runner
        scenario: dict[str, Any] = {
            "scenario_id": sc_name,
            "name": sc_name,
            "system_prompt": f"You are a helpful {sc_name} agent.",
            "user_input": f"Complete the {sc_name} task.",
            "tools": [],
            "tool_responses": {},
            "expected": {},
        }

        model_results: dict[str, list[dict[str, Any]]] = {}

        async def _run_e7_model(model: str) -> tuple[str, list[dict[str, Any]]]:
            """Run all E7 repetitions for one model. Returns (model_name, results)."""
            logger.info(
                "--- E7: %s / %s --- (%d repetitions x %d approaches)",
                model, sc_name, repetitions, len(approaches),
            )

            repetition_results: list[dict[str, Any]] = []

            for rep_idx in range(repetitions):
                # Budget check
                if cost.is_budget_exceeded():
                    logger.warning("Budget exceeded at rep %d", rep_idx)
                    break

                # Check for existing checkpoint
                ckpt_scenario = f"{sc_name}__rep{rep_idx:03d}"
                existing = checkpoint.load(experiment_id, model, ckpt_scenario)
                if existing and existing.get("completed", 0) >= len(approaches):
                    repetition_results.extend(existing.get("results", []))
                    continue

                rep_start = time.monotonic()

                try:
                    # Step 1: Run shared baseline (100 trials)
                    baseline_results = await _run_e7_baseline_trials(
                        client=client,
                        model=model,
                        scenario=scenario,
                        experiment_config=config,
                        evaluator_fn=evaluator_fn,
                        n_trials=100,
                    )
                    baseline_cost = sum(
                        r.get("cost_usd", 0.0) for r in baseline_results
                    )
                    baseline_tokens = sum(
                        r.get("tokens", 0) for r in baseline_results
                    )

                    # Step 2: Create degraded scenario
                    degraded_scenario = _inject_e7_regression(scenario)

                    # Step 3: Run each approach
                    rep_approach_results: list[dict[str, Any]] = []

                    for approach_cfg in approaches:
                        approach_name = approach_cfg.get("name", "unknown")

                        if cost.is_budget_exceeded():
                            logger.warning(
                                "Budget exceeded during approach %s",
                                approach_name,
                            )
                            break

                        try:
                            runner_fn = _E7_APPROACH_RUNNERS.get(approach_name)
                            if runner_fn is None:
                                logger.warning(
                                    "Unknown approach: %s — skipping",
                                    approach_name,
                                )
                                continue

                            # Build args — some approaches need extra params
                            kwargs: dict[str, Any] = {
                                "client": client,
                                "model": model,
                                "degraded_scenario": degraded_scenario,
                                "experiment_config": config,
                                "evaluator_fn": evaluator_fn,
                                "baseline_results": baseline_results,
                                "approach_config": approach_cfg,
                                "deps": deps,
                            }

                            # SPRT-based approaches need params
                            if approach_name != "fixed_n":
                                kwargs["params"] = params

                            approach_result = await runner_fn(**kwargs)

                            # Add metadata
                            approach_result["repetition"] = rep_idx
                            approach_result["model"] = model
                            approach_result["scenario"] = sc_name
                            approach_result["baseline_cost_usd"] = baseline_cost
                            approach_result["baseline_tokens"] = baseline_tokens
                            approach_result["timestamp"] = datetime.now(
                                timezone.utc
                            ).isoformat()

                            rep_approach_results.append(approach_result)

                        except Exception as exc:
                            logger.error(
                                "E7 approach %s failed (model=%s, scenario=%s, rep=%d): %s",
                                approach_name, model, sc_name, rep_idx, exc,
                            )
                            rep_approach_results.append({
                                "approach": approach_name,
                                "repetition": rep_idx,
                                "model": model,
                                "scenario": sc_name,
                                "error": str(exc),
                                "trials_used": 0,
                                "tokens_used": 0,
                                "cost_usd": 0.0,
                                "verdict": "ERROR",
                                "power": 0.0,
                            })

                    repetition_results.extend(rep_approach_results)

                    # FIX 2: Checkpoint after each repetition with immediate blob push
                    save_checkpoint_with_blob_push(
                        checkpoint_manager=checkpoint,
                        push_blob_fn=push_results_to_blob,
                        results_dir=DEFAULT_RESULTS_DIR,
                        log_dir=DEFAULT_LOG_DIR,
                        status_file=STATUS_FILE,
                        experiment_id=experiment_id,
                        model=model,
                        scenario=ckpt_scenario,
                        completed=len(approaches) * (rep_idx + 1),
                        results=repetition_results,
                    )

                    # FIX 3: Update heartbeat
                    update_heartbeat_stats(
                        experiment=experiment_id,
                        model=model,
                        trials=len(repetition_results),
                    )

                    rep_duration = (time.monotonic() - rep_start) * 1000.0
                    logger.debug(
                        "E7 rep %d/%d for %s/%s: %.1fms, %d approach results",
                        rep_idx + 1,
                        repetitions,
                        model,
                        sc_name,
                        rep_duration,
                        len(rep_approach_results),
                    )

                except Exception as exc:
                    logger.error(
                        "E7 repetition %d failed (model=%s, scenario=%s): %s\n%s",
                        rep_idx, model, sc_name, exc, traceback.format_exc(),
                    )

            # Write per-model results
            safe_model = model.replace("/", "_").replace(" ", "_")
            result_file = exp_results_dir / f"{sc_name}__{safe_model}.json"
            with open(result_file, "w") as f:
                json.dump(
                    {
                        "experiment_id": experiment_id,
                        "scenario": sc_name,
                        "model": model,
                        "repetitions": repetitions,
                        "approaches": [a.get("name") for a in approaches],
                        "results": repetition_results,
                    },
                    f,
                    indent=2,
                    default=str,
                )

            return model, repetition_results

        # Run all models in PARALLEL (models are on different subscriptions)
        gathered = await asyncio.gather(
            *(_run_e7_model(m) for m in active_models),
            return_exceptions=True,
        )

        for result in gathered:
            if isinstance(result, Exception):
                logger.error("E7 model task failed: %s", result)
            else:
                model_name, results = result
                model_results[model_name] = results

        all_results[sc_name] = model_results

        # Update status
        write_status(
            STATUS_FILE,
            current_experiment=experiment_id,
            progress={
                m: {
                    "completed": len(r),
                    "total": repetitions * len(approaches),
                }
                for m, r in model_results.items()
            },
            cost_summary=cost.summary(),
        )

    experiment_duration_ms = (time.monotonic() - experiment_start) * 1000.0

    # Write experiment summary
    summary = {
        "experiment_id": experiment_id,
        "duration_ms": round(experiment_duration_ms, 2),
        "models": active_models,
        "skipped_models": skipped_models,
        "scenarios": [
            s if isinstance(s, str) else s.get("scenario_id", "?")
            for s in scenarios_raw
        ],
        "approaches": [a.get("name") for a in approaches],
        "repetitions": repetitions,
        "cost_usd": cost.cost_by_experiment().get(experiment_id, 0.0),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    summary_file = exp_results_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    cost.flush()

    logger.info(
        "E7 complete in %.1fs — cost: $%.4f",
        experiment_duration_ms / 1000,
        cost.cost_by_experiment().get(experiment_id, 0.0),
    )

    return summary


# ---------------------------------------------------------------------------
# Experiment-level execution (all models in parallel)
# ---------------------------------------------------------------------------

async def run_experiment(
    config: dict[str, Any],
    client: AzureFoundryClient,
    checkpoint: CheckpointManager,
    cost: CostTracker,
    results_dir: Path,
) -> dict[str, Any]:
    """Run one complete experiment across all models in parallel.

    Parameters
    ----------
    config
        Experiment configuration dict (from YAML).
    client
        Azure AI Foundry client.
    checkpoint
        Checkpoint manager for resume support.
    cost
        Cost tracker with budget enforcement.
    results_dir
        Directory for writing experiment results.

    Returns
    -------
    dict
        Experiment-level summary with per-model results.
    """
    experiment_id = config["experiment_id"]

    # E7 has a fundamentally different structure — dispatch to specialized handler
    if experiment_id == "e7_efficiency":
        return await run_e7_experiment(
            config=config,
            client=client,
            checkpoint=checkpoint,
            cost=cost,
            results_dir=results_dir,
        )

    models = config.get("models", ALL_MODELS)
    scenarios = config.get("scenarios", [])
    trials_per_scenario = config.get("trials_per_scenario", 50)
    evaluator_name = config.get("evaluator", "ecommerce")

    logger.info(
        "=" * 60 + "\n"
        "Starting experiment %s: %d scenarios x %d models x %d trials\n"
        "Evaluator: %s\n"
        "=" * 60,
        experiment_id,
        len(scenarios),
        len(models),
        trials_per_scenario,
        evaluator_name,
    )

    # Resolve evaluator
    try:
        evaluator_fn = get_evaluator(evaluator_name)
    except KeyError:
        logger.error(
            "Unknown evaluator '%s' for experiment %s — skipping",
            evaluator_name,
            experiment_id,
        )
        return {"experiment_id": experiment_id, "error": f"Unknown evaluator: {evaluator_name}"}

    # Filter to available models
    available = set(client.available_models())
    active_models = [m for m in models if m in available]
    skipped_models = [m for m in models if m not in available]
    if skipped_models:
        logger.warning(
            "Skipping unavailable models: %s", skipped_models
        )

    if not active_models:
        logger.error("No available models for experiment %s", experiment_id)
        return {"experiment_id": experiment_id, "error": "No available models"}

    if not scenarios:
        logger.error("No scenarios defined for experiment %s", experiment_id)
        return {"experiment_id": experiment_id, "error": "No scenarios"}

    experiment_start = time.monotonic()
    all_results: dict[str, dict[str, list[dict[str, Any]]]] = {}

    for scenario_raw in scenarios:
        # Handle both string and dict scenario formats
        if isinstance(scenario_raw, str):
            scenario: dict[str, Any] = {
                "scenario_id": scenario_raw,
                "name": scenario_raw,
                "system_prompt": f"You are a helpful {scenario_raw} agent.",
                "user_input": f"Complete the {scenario_raw} task.",
                "tools": [],
                "tool_responses": {},
                "expected": {},
            }
        else:
            scenario = scenario_raw

        scenario_id = scenario.get("scenario_id", "unknown")
        logger.info(
            "--- Scenario: %s (%s) ---",
            scenario_id,
            scenario.get("name", "unnamed"),
        )

        # Run all models in parallel for this scenario
        tasks = []
        for model in active_models:
            task = run_model_trials(
                client=client,
                model=model,
                scenario=scenario,
                experiment_config=config,
                evaluator_fn=evaluator_fn,
                checkpoint=checkpoint,
                cost=cost,
                experiment_id=experiment_id,
                total_trials=trials_per_scenario,
            )
            tasks.append((model, task))

        # Execute all models concurrently
        model_results: dict[str, list[dict[str, Any]]] = {}
        gathered = await asyncio.gather(
            *(t for _, t in tasks),
            return_exceptions=True,
        )

        for (model, _), result in zip(tasks, gathered):
            if isinstance(result, Exception):
                logger.error(
                    "Model %s failed on scenario %s: %s\n%s",
                    model,
                    scenario_id,
                    result,
                    traceback.format_exc(),
                )
                model_results[model] = []
            else:
                model_results[model] = result

        all_results[scenario_id] = model_results

        # Update status file
        progress = {}
        for m, res in model_results.items():
            total = trials_per_scenario
            completed = len(res)
            passed = sum(1 for r in res if r.get("passed"))
            progress[m] = {
                "completed": completed,
                "total": total,
                "passed": passed,
                "pass_rate": round(passed / completed, 4) if completed > 0 else 0.0,
            }

        write_status(
            STATUS_FILE,
            current_experiment=experiment_id,
            progress=progress,
            cost_summary=cost.summary(),
        )

    experiment_duration_ms = (time.monotonic() - experiment_start) * 1000.0

    # Write experiment results to disk
    exp_results_dir = results_dir / experiment_id
    exp_results_dir.mkdir(parents=True, exist_ok=True)

    summary = {
        "experiment_id": experiment_id,
        "duration_ms": round(experiment_duration_ms, 2),
        "models": active_models,
        "skipped_models": skipped_models,
        "scenarios": [
            s if isinstance(s, str) else s.get("scenario_id", "?")
            for s in scenarios
        ],
        "trials_per_scenario": trials_per_scenario,
        "evaluator": evaluator_name,
        "cost_usd": cost.cost_by_experiment().get(experiment_id, 0.0),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }

    # Write per-model result files
    for scenario_id, model_results in all_results.items():
        for model, results in model_results.items():
            safe_model = model.replace("/", "_").replace(" ", "_")
            result_file = (
                exp_results_dir / f"{scenario_id}__{safe_model}.json"
            )
            with open(result_file, "w") as f:
                json.dump(
                    {
                        "experiment_id": experiment_id,
                        "scenario_id": scenario_id,
                        "model": model,
                        "total_trials": len(results),
                        "passed": sum(1 for r in results if r.get("passed")),
                        "results": results,
                    },
                    f,
                    indent=2,
                    default=str,
                )

    # Write experiment summary
    summary_file = exp_results_dir / "summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    # Flush cost log
    cost.flush()

    logger.info(
        "Experiment %s complete in %.1fs — cost: $%.4f",
        experiment_id,
        experiment_duration_ms / 1000,
        cost.cost_by_experiment().get(experiment_id, 0.0),
    )

    return summary


# ---------------------------------------------------------------------------
# Main daemon loop
# ---------------------------------------------------------------------------

async def main(
    config_dir: Path = DEFAULT_CONFIG_DIR,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    log_dir: Path = DEFAULT_LOG_DIR,
    resume: bool = True,
    experiment_filter: str | None = None,
    budget_usd: float = 175.0,
    verbose: bool = False,
    env_path: str = ".env",
) -> None:
    """Main daemon loop — iterate through the experiment queue.

    Parameters
    ----------
    config_dir
        Directory containing experiment YAML configs.
    results_dir
        Directory for writing experiment results.
    log_dir
        Directory for daemon log files.
    resume
        If True, use checkpoints to resume interrupted experiments.
    experiment_filter
        If set, only run experiments matching this ID (e.g. "E1").
    budget_usd
        Total budget cap in USD.
    verbose
        Enable debug-level logging.
    env_path
        Path to .env file for Azure credentials.
    """
    _setup_logging(log_dir, verbose=verbose)

    # FIX 5: Setup signal handlers for graceful shutdown
    setup_signal_handlers()

    logger.info("=" * 60)
    logger.info("AgentAssay Experiment Daemon starting (Session 6 - bulletproof)")
    logger.info("Config dir:  %s", config_dir)
    logger.info("Results dir: %s", results_dir)
    logger.info("Budget:      $%.2f", budget_usd)
    logger.info("Resume:      %s", resume)
    logger.info("Fixes:       checkpoint-on-error, blob-push-on-checkpoint, heartbeat, token-validation, graceful-shutdown")
    logger.info("=" * 60)

    # Initialize infrastructure
    cost = CostTracker(
        log_path=str(results_dir / "cost_log.json"),
        budget_usd=budget_usd,
        warn_at_pct=0.80,
        hard_stop=False,
    )

    checkpoint = CheckpointManager(
        checkpoint_dir=str(results_dir / "checkpoints"),
        save_interval=25,
    )

    # FIX 3: Start heartbeat background task (logs progress every 5 min)
    start_heartbeat(interval_sec=300.0)

    # FIX 5: Register shutdown handler to save state on SIGTERM
    def final_checkpoint_save() -> None:
        """Save all state before shutdown."""
        logger.info("Running final checkpoint save...")
        push_results_to_blob(results_dir, log_dir, STATUS_FILE, force=True)
        cost.flush()

    register_shutdown_handler(final_checkpoint_save)

    # Resolve env_path relative to project root
    env_full_path = PROJECT_ROOT / env_path
    if not env_full_path.exists():
        logger.error("Environment file not found: %s", env_full_path)
        logger.error(
            "Create a .env file with AZURE_SUB{1,2,3}_{ENDPOINT,KEY,REGION}"
        )
        write_status(
            STATUS_FILE,
            current_experiment="none",
            progress={},
            cost_summary=cost.summary(),
            phase="error",
            error=f"Missing .env file: {env_full_path}",
        )
        return

    try:
        client = AzureFoundryClient(
            env_path=str(env_full_path),
            cost_tracker=cost,
        )
    except Exception as exc:
        logger.error("Failed to initialize Azure client: %s", exc)
        write_status(
            STATUS_FILE,
            current_experiment="none",
            progress={},
            cost_summary=cost.summary(),
            phase="error",
            error=str(exc),
        )
        return

    logger.info("Available models: %s", client.available_models())

    # Load experiment configs
    configs = load_experiment_configs(config_dir)
    if not configs:
        logger.error("No experiment configs to run. Exiting.")
        return

    # Apply filter if specified
    if experiment_filter:
        configs = [
            c for c in configs if c["experiment_id"] == experiment_filter
        ]
        if not configs:
            logger.error(
                "No experiment matching filter '%s'", experiment_filter
            )
            return

    logger.info(
        "Experiment queue: %s",
        [c["experiment_id"] for c in configs],
    )

    write_status(
        STATUS_FILE,
        current_experiment="starting",
        progress={},
        cost_summary=cost.summary(),
        phase="starting",
    )

    # Run experiments in sequence
    summaries: list[dict[str, Any]] = []

    for config in configs:
        experiment_id = config["experiment_id"]

        # Budget check before each experiment
        if cost.is_budget_exceeded():
            logger.warning(
                "Budget exceeded before experiment %s — stopping.",
                experiment_id,
            )
            break

        try:
            summary = await run_experiment(
                config=config,
                client=client,
                checkpoint=checkpoint,
                cost=cost,
                results_dir=results_dir,
            )
            summaries.append(summary)

            # Push results to Blob Storage after each experiment completes
            push_results_to_blob(results_dir, log_dir, STATUS_FILE)

        except BudgetExceededError as exc:
            logger.warning("Budget exceeded during %s: %s", experiment_id, exc)
            summaries.append({
                "experiment_id": experiment_id,
                "error": f"Budget exceeded: {exc}",
            })
            # Force-push final state before stopping
            push_results_to_blob(results_dir, log_dir, STATUS_FILE, force=True)
            break

        except Exception as exc:
            logger.error(
                "Experiment %s failed: %s\n%s",
                experiment_id,
                exc,
                traceback.format_exc(),
            )
            summaries.append({
                "experiment_id": experiment_id,
                "error": f"{type(exc).__name__}: {exc}",
            })
            # Push current state even on failure
            push_results_to_blob(results_dir, log_dir, STATUS_FILE)
            # Continue to next experiment — self-healing

    # Final cost flush
    cost.flush()

    # Write final campaign summary
    campaign_summary = {
        "experiments": summaries,
        "total_cost_usd": cost.total_cost_usd,
        "total_calls": cost.total_calls,
        "total_tokens": cost.total_tokens,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }

    campaign_file = results_dir / "campaign_summary.json"
    with open(campaign_file, "w") as f:
        json.dump(campaign_summary, f, indent=2, default=str)

    write_status(
        STATUS_FILE,
        current_experiment="done",
        progress={},
        cost_summary=cost.summary(),
        phase="completed",
    )

    # Final force push — ensure all results are in Blob Storage
    push_results_to_blob(results_dir, log_dir, STATUS_FILE, force=True)

    # Clean up
    await client.close()

    # FIX 3: Stop heartbeat task
    stop_heartbeat()

    logger.info("=" * 60)
    logger.info("Experiment campaign complete!")
    logger.info("Total cost:   $%.4f", cost.total_cost_usd)
    logger.info("Total calls:  %d", cost.total_calls)
    logger.info("Total tokens: %d", cost.total_tokens)
    logger.info("Results:      %s", results_dir)
    logger.info("=" * 60)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def cli() -> None:
    """Parse CLI arguments and run the daemon."""
    parser = argparse.ArgumentParser(
        prog="agentassay-daemon",
        description="AgentAssay experiment daemon — runs E1-E7 across all models.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=str(DEFAULT_CONFIG_DIR),
        help="Path to experiment config directory (default: experiments/configs/)",
    )
    parser.add_argument(
        "--results",
        type=str,
        default=str(DEFAULT_RESULTS_DIR),
        help="Path to results directory (default: experiments/results/)",
    )
    parser.add_argument(
        "--logs",
        type=str,
        default=str(DEFAULT_LOG_DIR),
        help="Path to log directory (default: experiments/logs/)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        default=True,
        help="Resume from checkpoints (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Start fresh, ignoring checkpoints",
    )
    parser.add_argument(
        "--experiment",
        type=str,
        default=None,
        help="Run only this experiment (e.g. E1)",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=175.0,
        help="Total budget in USD (default: 175.0)",
    )
    parser.add_argument(
        "--env",
        type=str,
        default=".env",
        help="Path to .env file (default: .env)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )

    args = parser.parse_args()

    resume = args.resume and not args.no_resume

    asyncio.run(
        main(
            config_dir=Path(args.config),
            results_dir=Path(args.results),
            log_dir=Path(args.logs),
            resume=resume,
            experiment_filter=args.experiment,
            budget_usd=args.budget,
            verbose=args.verbose,
            env_path=args.env,
        )
    )


if __name__ == "__main__":
    cli()
