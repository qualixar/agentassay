# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Contract YAML loader and validator for AgentAssay.

Loads ContractSpec-compatible YAML files and normalizes them into a
validated internal representation. AgentAssert is an OPTIONAL dependency:
this loader parses the YAML independently using PyYAML so the contracts
module works standalone.

Contract YAML format (simplified ContractSpec-compatible)::

    contract:
      name: "my_agent_contract"
      version: "1.0"               # optional
      description: "..."           # optional
      constraints:
        - name: "max_steps"
          type: "invariant"
          severity: "hard"
          condition: "step_count <= 10"
        - name: "cost_limit"
          type: "guardrail"
          severity: "hard"
          condition: "total_cost_usd <= 1.0"
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_VALID_CONSTRAINT_TYPES = frozenset(
    {"precondition", "postcondition", "invariant", "guardrail"}
)

_VALID_SEVERITIES = frozenset({"hard", "soft"})


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------

class ContractLoadError(Exception):
    """Raised when a contract YAML file cannot be loaded or validated."""


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------

class ContractLoader:
    """Load and validate contract YAML files or dicts.

    This is a stateless utility: each method takes raw input and returns
    a normalized, validated dict. No instance state is required, but the
    class provides a clear API boundary and is easy to mock in tests.

    The loader validates:
        - Required top-level ``contract`` key
        - Required ``contract.name`` (non-empty string)
        - Each constraint has ``name``, ``type``, ``severity``, ``condition``
        - ``type`` is one of: precondition, postcondition, invariant, guardrail
        - ``severity`` is one of: hard, soft
        - ``condition`` is a non-empty string
    """

    @staticmethod
    def load_yaml(path: str | Path) -> dict[str, Any]:
        """Load a contract from a YAML file.

        Parameters
        ----------
        path
            Filesystem path to a ``.yaml`` or ``.yml`` contract file.

        Returns
        -------
        dict[str, Any]
            Validated and normalized contract dict.

        Raises
        ------
        ContractLoadError
            If the file cannot be read, is not valid YAML, or fails
            structural validation.
        FileNotFoundError
            If the file does not exist.
        """
        resolved = Path(path).expanduser().resolve()

        if not resolved.is_file():
            raise FileNotFoundError(
                f"Contract file not found: {resolved}"
            )

        try:
            import yaml
        except ImportError as exc:
            raise ContractLoadError(
                "PyYAML is required to load contract YAML files. "
                "Install it with: pip install pyyaml"
            ) from exc

        try:
            with open(resolved, encoding="utf-8") as fh:
                raw = yaml.safe_load(fh)
        except yaml.YAMLError as exc:
            raise ContractLoadError(
                f"Failed to parse YAML from {resolved}: {exc}"
            ) from exc

        if not isinstance(raw, dict):
            raise ContractLoadError(
                f"Expected a YAML mapping at top level, got {type(raw).__name__}"
            )

        return ContractLoader.load_dict(raw)

    @staticmethod
    def load_dict(data: dict[str, Any]) -> dict[str, Any]:
        """Validate and normalize a contract dict.

        Parameters
        ----------
        data
            Raw dict, typically from ``yaml.safe_load()`` or constructed
            programmatically. Must contain a ``contract`` key.

        Returns
        -------
        dict[str, Any]
            Normalized contract dict with all fields validated.

        Raises
        ------
        ContractLoadError
            If required fields are missing or have invalid values.
        """
        if not isinstance(data, dict):
            raise ContractLoadError(
                f"Expected a dict, got {type(data).__name__}"
            )

        # --- Top-level contract key ---
        if "contract" not in data:
            raise ContractLoadError(
                "Missing required top-level key 'contract'. "
                "Expected: contract: { name: ..., constraints: [...] }"
            )

        contract = data["contract"]
        if not isinstance(contract, dict):
            raise ContractLoadError(
                f"'contract' must be a mapping, got {type(contract).__name__}"
            )

        # --- contract.name ---
        name = contract.get("name")
        if not name or not isinstance(name, str) or not name.strip():
            raise ContractLoadError(
                "contract.name is required and must be a non-empty string"
            )
        contract["name"] = name.strip()

        # --- contract.version (optional, default "1.0") ---
        if "version" not in contract:
            contract["version"] = "1.0"

        # --- contract.description (optional, default "") ---
        if "description" not in contract:
            contract["description"] = ""

        # --- contract.constraints ---
        constraints = contract.get("constraints")
        if constraints is None:
            constraints = []
        if not isinstance(constraints, list):
            raise ContractLoadError(
                f"contract.constraints must be a list, "
                f"got {type(constraints).__name__}"
            )

        validated: list[dict[str, Any]] = []
        seen_names: set[str] = set()

        for idx, raw_constraint in enumerate(constraints):
            validated.append(
                ContractLoader._validate_constraint(
                    raw_constraint, idx, seen_names
                )
            )

        contract["constraints"] = validated
        return data

    # -- Internal helpers ----------------------------------------------------

    @staticmethod
    def _validate_constraint(
        raw: Any,
        idx: int,
        seen_names: set[str],
    ) -> dict[str, Any]:
        """Validate a single constraint entry.

        Parameters
        ----------
        raw
            The raw constraint value from YAML (should be a dict).
        idx
            Position in the constraints list, for error messages.
        seen_names
            Set of constraint names already seen, to detect duplicates.

        Returns
        -------
        dict[str, Any]
            Validated and normalized constraint dict.

        Raises
        ------
        ContractLoadError
            If the constraint is invalid.
        """
        prefix = f"constraints[{idx}]"

        if not isinstance(raw, dict):
            raise ContractLoadError(
                f"{prefix}: expected a mapping, got {type(raw).__name__}"
            )

        # --- name ---
        c_name = raw.get("name")
        if not c_name or not isinstance(c_name, str) or not c_name.strip():
            raise ContractLoadError(
                f"{prefix}: 'name' is required and must be a non-empty string"
            )
        c_name = c_name.strip()

        if c_name in seen_names:
            raise ContractLoadError(
                f"{prefix}: duplicate constraint name '{c_name}'"
            )
        seen_names.add(c_name)

        # --- type ---
        c_type = raw.get("type")
        if not c_type or not isinstance(c_type, str):
            raise ContractLoadError(
                f"{prefix} ('{c_name}'): 'type' is required and must be a string"
            )
        c_type = c_type.strip().lower()
        if c_type not in _VALID_CONSTRAINT_TYPES:
            raise ContractLoadError(
                f"{prefix} ('{c_name}'): invalid type '{c_type}'. "
                f"Must be one of: {', '.join(sorted(_VALID_CONSTRAINT_TYPES))}"
            )

        # --- severity ---
        c_severity = raw.get("severity", "hard")
        if not isinstance(c_severity, str):
            raise ContractLoadError(
                f"{prefix} ('{c_name}'): 'severity' must be a string"
            )
        c_severity = c_severity.strip().lower()
        if c_severity not in _VALID_SEVERITIES:
            raise ContractLoadError(
                f"{prefix} ('{c_name}'): invalid severity '{c_severity}'. "
                f"Must be one of: {', '.join(sorted(_VALID_SEVERITIES))}"
            )

        # --- condition ---
        c_condition = raw.get("condition")
        if not c_condition or not isinstance(c_condition, str) or not c_condition.strip():
            raise ContractLoadError(
                f"{prefix} ('{c_name}'): 'condition' is required and "
                "must be a non-empty string"
            )
        c_condition = c_condition.strip()

        # --- description (optional) ---
        c_desc = raw.get("description", "")

        return {
            "name": c_name,
            "type": c_type,
            "severity": c_severity,
            "condition": c_condition,
            "description": str(c_desc),
        }
