#!/bin/bash
###############################################################################
# AgentAssay — Pull Experiment Results from Azure Blob Storage
#
# Downloads experiment results from Azure Blob Storage to local Mac.
# NO SSH required — works even when Mac was off for days.
# Incremental: only downloads blobs newer than last pull.
#
# Architecture:
#   VM pushes results to Blob Storage every 30 minutes.
#   This script pulls from Blob Storage using a read-only SAS token.
#
# Usage:
#   ./experiments/infra/pull-results.sh [OPTIONS]
#
# Options:
#   --full               Download everything (not incremental)
#   --results-only       Only pull results/ blobs
#   --logs-only          Only pull logs/ blobs
#   --output-dir DIR     Local output directory (default: experiments/)
#   --help               Show this help
#
# Prerequisites:
#   - VM created via setup-vm.sh (for .vm-state.json with SAS token)
#   - az CLI installed (uses az storage blob commands)
###############################################################################
set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STATE_FILE="${SCRIPT_DIR}/.vm-state.json"
LAST_PULL_FILE="${SCRIPT_DIR}/.last-pull-timestamp"
OUTPUT_DIR="${PROJECT_ROOT}/experiments"
FULL_DOWNLOAD=false
RESULTS_ONLY=false
LOGS_ONLY=false

# ─── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ─── Help ────────────────────────────────────────────────────────────────────
show_help() {
    sed -n '/^# Usage:/,/^# Prerequisites:/p' "$0" | head -n -1 | sed 's/^# //'
    exit 0
}

# ─── Parse Args ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --full)          FULL_DOWNLOAD=true; shift ;;
        --results-only)  RESULTS_ONLY=true; shift ;;
        --logs-only)     LOGS_ONLY=true; shift ;;
        --output-dir)    OUTPUT_DIR="$2"; shift 2 ;;
        --help|-h)       show_help ;;
        *)               err "Unknown option: $1"; show_help ;;
    esac
done

# ─── Load VM State (Blob Storage credentials) ───────────────────────────────
if [[ ! -f "${STATE_FILE}" ]]; then
    err "VM state file not found: ${STATE_FILE}"
    err "Run ./experiments/infra/setup-vm.sh first"
    exit 1
fi

STORAGE_ACCOUNT=$(jq -r '.storage_account' "${STATE_FILE}")
STORAGE_CONTAINER=$(jq -r '.storage_container' "${STATE_FILE}")
MAC_SAS_TOKEN=$(jq -r '.mac_sas_token' "${STATE_FILE}")

if [[ -z "${STORAGE_ACCOUNT}" || "${STORAGE_ACCOUNT}" == "null" ]]; then
    err "Storage account not found in state file. Re-run setup-vm.sh."
    exit 1
fi

info "Pulling from Blob Storage: ${STORAGE_ACCOUNT}/${STORAGE_CONTAINER}"

# ─── Determine blob prefix filters ──────────────────────────────────────────
BLOB_PREFIXES=()
if [[ "${RESULTS_ONLY}" == "true" ]]; then
    BLOB_PREFIXES=("results/")
elif [[ "${LOGS_ONLY}" == "true" ]]; then
    BLOB_PREFIXES=("logs/")
else
    BLOB_PREFIXES=("results/" "logs/" "status.json" "cost_log.json")
fi

# ─── Create output directories ──────────────────────────────────────────────
mkdir -p "${OUTPUT_DIR}/results"
mkdir -p "${OUTPUT_DIR}/logs"

# ─── Download blobs ──────────────────────────────────────────────────────────
TOTAL_DOWNLOADED=0

for PREFIX in "${BLOB_PREFIXES[@]}"; do
    info "Downloading blobs with prefix: ${PREFIX}"

    # For status.json and cost_log.json (single files, not directories)
    if [[ "${PREFIX}" == "status.json" || "${PREFIX}" == "cost_log.json" ]]; then
        az storage blob download \
            --account-name "${STORAGE_ACCOUNT}" \
            --container-name "${STORAGE_CONTAINER}" \
            --name "${PREFIX}" \
            --file "${OUTPUT_DIR}/${PREFIX}" \
            --sas-token "${MAC_SAS_TOKEN}" \
            --no-progress \
            -o none 2>/dev/null || {
                warn "Blob '${PREFIX}' not found (may not exist yet)"
                continue
            }
        TOTAL_DOWNLOADED=$((TOTAL_DOWNLOADED + 1))
        continue
    fi

    # For directories (results/, logs/) — use download-batch
    DEST_DIR="${OUTPUT_DIR}/${PREFIX%/}"
    mkdir -p "${DEST_DIR}"

    # Build the download command
    DOWNLOAD_ARGS=(
        --account-name "${STORAGE_ACCOUNT}"
        --source "${STORAGE_CONTAINER}"
        --destination "${DEST_DIR}"
        --pattern "${PREFIX}*"
        --sas-token "${MAC_SAS_TOKEN}"
        --no-progress
        -o none
    )

    # Incremental: only download blobs modified after last pull
    if [[ "${FULL_DOWNLOAD}" == "false" && -f "${LAST_PULL_FILE}" ]]; then
        LAST_PULL_TS=$(cat "${LAST_PULL_FILE}")
        info "  Incremental: blobs newer than ${LAST_PULL_TS}"
        # az storage blob download-batch doesn't have --if-modified-since,
        # so we list blobs first and filter
        BLOB_LIST=$(az storage blob list \
            --account-name "${STORAGE_ACCOUNT}" \
            --container-name "${STORAGE_CONTAINER}" \
            --prefix "${PREFIX}" \
            --sas-token "${MAC_SAS_TOKEN}" \
            --query "[?properties.lastModified>'${LAST_PULL_TS}'].name" \
            -o tsv 2>/dev/null || echo "")

        if [[ -z "${BLOB_LIST}" ]]; then
            info "  No new blobs since last pull"
            continue
        fi

        # Download each new blob individually
        BLOB_COUNT=0
        while IFS= read -r BLOB_NAME; do
            [[ -z "${BLOB_NAME}" ]] && continue
            # Construct local path: strip the prefix and place in dest dir
            LOCAL_PATH="${OUTPUT_DIR}/${BLOB_NAME}"
            LOCAL_DIR=$(dirname "${LOCAL_PATH}")
            mkdir -p "${LOCAL_DIR}"

            az storage blob download \
                --account-name "${STORAGE_ACCOUNT}" \
                --container-name "${STORAGE_CONTAINER}" \
                --name "${BLOB_NAME}" \
                --file "${LOCAL_PATH}" \
                --sas-token "${MAC_SAS_TOKEN}" \
                --no-progress \
                -o none 2>/dev/null || warn "Failed to download: ${BLOB_NAME}"

            BLOB_COUNT=$((BLOB_COUNT + 1))
        done <<< "${BLOB_LIST}"

        TOTAL_DOWNLOADED=$((TOTAL_DOWNLOADED + BLOB_COUNT))
        ok "  Downloaded ${BLOB_COUNT} new blobs from ${PREFIX}"
    else
        # Full download
        az storage blob download-batch \
            "${DOWNLOAD_ARGS[@]}" 2>/dev/null || {
                warn "No blobs found with prefix '${PREFIX}'"
                continue
            }

        # Count downloaded
        BLOB_COUNT=$(az storage blob list \
            --account-name "${STORAGE_ACCOUNT}" \
            --container-name "${STORAGE_CONTAINER}" \
            --prefix "${PREFIX}" \
            --sas-token "${MAC_SAS_TOKEN}" \
            --query "length(@)" \
            -o tsv 2>/dev/null || echo "0")

        TOTAL_DOWNLOADED=$((TOTAL_DOWNLOADED + BLOB_COUNT))
        ok "  Downloaded ${BLOB_COUNT} blobs from ${PREFIX}"
    fi
done

# ─── Show Summary ────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}━━━ Pull Summary ━━━${NC}"
echo "  Total blobs downloaded: ${TOTAL_DOWNLOADED}"

# Show status.json summary if present
STATUS_FILE_LOCAL="${OUTPUT_DIR}/status.json"
if [[ -f "${STATUS_FILE_LOCAL}" ]]; then
    echo ""
    info "Experiment status:"
    jq -r '
        "  Phase:      \(.phase // "unknown")",
        "  Experiment: \(.current_experiment // "none")",
        "  Cost:       $\(.cost.total_cost_usd // 0 | tostring)",
        "  Timestamp:  \(.timestamp // "unknown")"
    ' "${STATUS_FILE_LOCAL}" 2>/dev/null || warn "Could not parse status.json"
fi

# Show cost_log summary if present
COST_FILE_LOCAL="${OUTPUT_DIR}/cost_log.json"
if [[ -f "${COST_FILE_LOCAL}" ]]; then
    info "Cost summary:"
    jq -r '
        "  Total cost:   $\(.total_cost_usd // 0)",
        "  Total calls:  \(.total_calls // 0)",
        "  Total tokens: \(.total_tokens // 0)"
    ' "${COST_FILE_LOCAL}" 2>/dev/null || true
fi

# Show results directory stats
for DIR_NAME in results logs; do
    LOCAL_PATH="${OUTPUT_DIR}/${DIR_NAME}"
    if [[ -d "${LOCAL_PATH}" ]]; then
        FILE_COUNT=$(find "${LOCAL_PATH}" -type f 2>/dev/null | wc -l | tr -d ' ')
        if [[ "${FILE_COUNT}" -gt 0 ]]; then
            DIR_SIZE=$(du -sh "${LOCAL_PATH}" 2>/dev/null | cut -f1)
            echo "  ${DIR_NAME}/: ${FILE_COUNT} files (${DIR_SIZE})"
        fi
    fi
done

# ─── Update Last Pull Timestamp ─────────────────────────────────────────────
date -u +%Y-%m-%dT%H:%M:%SZ > "${LAST_PULL_FILE}"

# ─── Done ────────────────────────────────────────────────────────────────────
echo ""
ok "Results pulled from Blob Storage (NO SSH used)"
echo ""
echo "  Output: ${OUTPUT_DIR}/"
echo "  Next pull will be incremental (only new blobs)."
echo "  Use --full to re-download everything."
echo ""
