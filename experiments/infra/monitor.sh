#!/bin/bash
###############################################################################
# AgentAssay — Monitor Experiment Status (Blob-Based)
#
# Monitors experiment progress by reading status.json from Azure Blob Storage.
# NO SSH required for default monitoring — fully Blob-based.
# SSH (via az ssh vm AAD tunnel) only used for --shell and --follow modes.
#
# Usage:
#   ./experiments/infra/monitor.sh [OPTIONS]
#
# Options:
#   --follow             Poll Blob Storage every 60s for status updates
#   --cost               Show Azure cost estimate
#   --shell              Open interactive shell on VM (az ssh vm AAD tunnel)
#   --restart            Restart the experiment daemon (requires az ssh vm)
#   --stop               Stop the experiment daemon (requires az ssh vm)
#   --logs N             Show last N log lines (requires az ssh vm)
#   --help               Show this help
#
# Prerequisites:
#   - VM created via setup-vm.sh (for .vm-state.json)
#   - az CLI logged in
###############################################################################
set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="${SCRIPT_DIR}/.vm-state.json"
LOG_LINES=20
FOLLOW=false
SHOW_COST=false
OPEN_SHELL=false
DO_RESTART=false
DO_STOP=false
SHOW_LOGS=false

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
        --logs)      LOG_LINES="$2"; SHOW_LOGS=true; shift 2 ;;
        --follow)    FOLLOW=true; shift ;;
        --cost)      SHOW_COST=true; shift ;;
        --shell)     OPEN_SHELL=true; shift ;;
        --restart)   DO_RESTART=true; shift ;;
        --stop)      DO_STOP=true; shift ;;
        --help|-h)   show_help ;;
        *)           err "Unknown option: $1"; show_help ;;
    esac
done

# ─── Load VM State ──────────────────────────────────────────────────────────
if [[ ! -f "${STATE_FILE}" ]]; then
    err "VM state file not found: ${STATE_FILE}"
    err "Run ./experiments/infra/setup-vm.sh first"
    exit 1
fi

RESOURCE_GROUP=$(jq -r '.resource_group' "${STATE_FILE}")
VM_NAME=$(jq -r '.vm_name' "${STATE_FILE}")
STORAGE_ACCOUNT=$(jq -r '.storage_account' "${STATE_FILE}")
STORAGE_CONTAINER=$(jq -r '.storage_container' "${STATE_FILE}")
MAC_SAS_TOKEN=$(jq -r '.mac_sas_token' "${STATE_FILE}")
CREATED_AT=$(jq -r '.created_at' "${STATE_FILE}")

# ─── Handle: --shell (AAD tunnel to VM) ─────────────────────────────────────
if [[ "${OPEN_SHELL}" == "true" ]]; then
    info "Opening interactive shell on ${VM_NAME} (AAD tunnel, no open ports)..."
    az ssh vm -g "${RESOURCE_GROUP}" -n "${VM_NAME}"
    exit 0
fi

# ─── Handle: --follow (live journalctl via AAD tunnel) ──────────────────────
if [[ "${FOLLOW}" == "true" ]]; then
    info "Following experiment status via Blob Storage (Ctrl+C to exit)..."
    echo ""
    while true; do
        # Download status.json from Blob Storage
        TMP_STATUS="/tmp/agentassay-status-$$.json"
        az storage blob download \
            --account-name "${STORAGE_ACCOUNT}" \
            --container-name "${STORAGE_CONTAINER}" \
            --name "status.json" \
            --file "${TMP_STATUS}" \
            --sas-token "${MAC_SAS_TOKEN}" \
            --no-progress \
            -o none 2>/dev/null || {
                warn "status.json not found in Blob Storage. Retrying in 60s..."
                rm -f "${TMP_STATUS}"
                sleep 60
                continue
            }

        # Clear screen and display
        clear
        echo -e "${BOLD}${CYAN}━━━ AgentAssay Monitor (Blob-based, polling every 60s) ━━━${NC}"
        echo -e "  ${BOLD}Time:${NC} $(date -u +%H:%M:%S) UTC"
        echo ""

        if [[ -f "${TMP_STATUS}" ]]; then
            jq -r '
                "  Phase:       \(.phase // "unknown")",
                "  Experiment:  \(.current_experiment // "none")",
                "  Cost:        $\(.cost.total_cost_usd // 0)",
                "  Error:       \(.error // "none")",
                "  Last update: \(.timestamp // "unknown")",
                "",
                "  Progress:",
                (if .progress then (.progress | to_entries[] | "    \(.key): \(.value)") else "    No progress data" end)
            ' "${TMP_STATUS}" 2>/dev/null || cat "${TMP_STATUS}"
        fi

        rm -f "${TMP_STATUS}"
        echo ""
        echo -e "  ${CYAN}Press Ctrl+C to exit${NC}"
        sleep 60
    done
    exit 0
fi

# ─── Handle: --restart (AAD tunnel) ─────────────────────────────────────────
if [[ "${DO_RESTART}" == "true" ]]; then
    info "Restarting experiment daemon via az ssh vm (AAD tunnel)..."
    az ssh vm \
        --resource-group "${RESOURCE_GROUP}" \
        --name "${VM_NAME}" \
        -- "sudo systemctl restart agentassay-runner.service && sleep 3 && sudo systemctl status agentassay-runner.service --no-pager"
    exit 0
fi

# ─── Handle: --stop (AAD tunnel) ────────────────────────────────────────────
if [[ "${DO_STOP}" == "true" ]]; then
    info "Stopping experiment daemon via az ssh vm (AAD tunnel)..."
    az ssh vm \
        --resource-group "${RESOURCE_GROUP}" \
        --name "${VM_NAME}" \
        -- "sudo systemctl stop agentassay-runner.service && echo 'Daemon stopped.'"
    exit 0
fi

# ─── Handle: --logs N (AAD tunnel) ──────────────────────────────────────────
if [[ "${SHOW_LOGS}" == "true" ]]; then
    info "Fetching last ${LOG_LINES} log lines via az ssh vm (AAD tunnel)..."
    az ssh vm \
        --resource-group "${RESOURCE_GROUP}" \
        --name "${VM_NAME}" \
        -- "sudo journalctl -u agentassay-runner --no-pager -n ${LOG_LINES} --output=short-iso"
    exit 0
fi

# ─── Default: Blob-Based Status Check (NO SSH) ──────────────────────────────
echo ""
echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${BOLD}${CYAN}  AgentAssay Experiment Monitor (Blob-based — no SSH)${NC}"
echo -e "${BOLD}${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "  ${BOLD}VM:${NC}         ${VM_NAME} (NO public IP)"
echo -e "  ${BOLD}Storage:${NC}    ${STORAGE_ACCOUNT}/${STORAGE_CONTAINER}"
echo -e "  ${BOLD}Created:${NC}    ${CREATED_AT}"
echo ""

# Check VM power state
VM_STATE=$(az vm get-instance-view \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${VM_NAME}" \
    --query "instanceView.statuses[?starts_with(code, 'PowerState')].displayStatus" \
    -o tsv 2>/dev/null || echo "UNREACHABLE")

if [[ "${VM_STATE}" == "VM running" ]]; then
    echo -e "  ${BOLD}VM State:${NC}   ${GREEN}${VM_STATE}${NC}"
else
    echo -e "  ${BOLD}VM State:${NC}   ${RED}${VM_STATE}${NC}"
    echo ""
    echo "  Start VM: az vm start -g ${RESOURCE_GROUP} -n ${VM_NAME}"
fi

# ─── Download status.json from Blob Storage ──────────────────────────────────
echo ""
echo -e "${BOLD}━━━ Experiment Progress (from Blob Storage) ━━━${NC}"

TMP_STATUS="/tmp/agentassay-monitor-status-$$.json"
TMP_COST="/tmp/agentassay-monitor-cost-$$.json"

az storage blob download \
    --account-name "${STORAGE_ACCOUNT}" \
    --container-name "${STORAGE_CONTAINER}" \
    --name "status.json" \
    --file "${TMP_STATUS}" \
    --sas-token "${MAC_SAS_TOKEN}" \
    --no-progress \
    -o none 2>/dev/null || TMP_STATUS=""

if [[ -n "${TMP_STATUS}" && -f "${TMP_STATUS}" ]]; then
    jq -r '
        "  Phase:       \(.phase // "unknown")",
        "  Experiment:  \(.current_experiment // "none")",
        "  Error:       \(.error // "none")",
        "  Timestamp:   \(.timestamp // "unknown")",
        "  PID:         \(.pid // "unknown")",
        "",
        "  Cost:"
    ' "${TMP_STATUS}" 2>/dev/null || warn "Could not parse status.json"

    # Extract cost details
    jq -r '
        if .cost then
            "    Total:  $\(.cost.total_cost_usd // 0)",
            "    Calls:  \(.cost.total_calls // 0)",
            "    Tokens: \(.cost.total_tokens // 0)"
        else
            "    No cost data"
        end
    ' "${TMP_STATUS}" 2>/dev/null || true

    # Progress per model
    echo ""
    echo "  Progress per model:"
    jq -r '
        if .progress and (.progress | length > 0) then
            .progress | to_entries[] | "    \(.key): \(.value)"
        else
            "    No per-model progress data yet"
        end
    ' "${TMP_STATUS}" 2>/dev/null || true

    rm -f "${TMP_STATUS}"
else
    warn "status.json not found in Blob Storage."
    warn "Experiments may not have started or not pushed status yet."
    warn "The daemon pushes every 30 minutes."
fi

# ─── Download cost_log.json from Blob Storage ────────────────────────────────
echo ""
echo -e "${BOLD}━━━ Cost Log (from Blob Storage) ━━━${NC}"

az storage blob download \
    --account-name "${STORAGE_ACCOUNT}" \
    --container-name "${STORAGE_CONTAINER}" \
    --name "cost_log.json" \
    --file "${TMP_COST}" \
    --sas-token "${MAC_SAS_TOKEN}" \
    --no-progress \
    -o none 2>/dev/null || TMP_COST=""

if [[ -n "${TMP_COST}" && -f "${TMP_COST}" ]]; then
    jq -r '
        "  Total cost:   $\(.total_cost_usd // 0)",
        "  Total calls:  \(.total_calls // 0)",
        "  Total tokens: \(.total_tokens // 0)",
        "  Budget:       $\(.budget_usd // "unknown")",
        "  Budget used:  \(if .budget_usd and .total_cost_usd then "\((.total_cost_usd / .budget_usd * 100) | floor)%" else "?" end)"
    ' "${TMP_COST}" 2>/dev/null || warn "Could not parse cost_log.json"
    rm -f "${TMP_COST}"
else
    info "cost_log.json not found in Blob Storage yet."
fi

# ─── Blob Storage stats ─────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}━━━ Blob Storage Contents ━━━${NC}"

# Count blobs by prefix
for PREFIX in "results/" "logs/"; do
    BLOB_COUNT=$(az storage blob list \
        --account-name "${STORAGE_ACCOUNT}" \
        --container-name "${STORAGE_CONTAINER}" \
        --prefix "${PREFIX}" \
        --sas-token "${MAC_SAS_TOKEN}" \
        --query "length(@)" \
        -o tsv 2>/dev/null || echo "0")
    echo "  ${PREFIX}: ${BLOB_COUNT} blobs"
done

# ─── Cost Estimate ───────────────────────────────────────────────────────────
if [[ "${SHOW_COST}" == "true" ]]; then
    echo ""
    echo -e "${BOLD}━━━ Azure Infrastructure Cost Estimate ━━━${NC}"
    if command -v python3 &>/dev/null; then
        DAYS_RUNNING=$(python3 -c "
from datetime import datetime, timezone
created = datetime.fromisoformat('${CREATED_AT}'.replace('Z', '+00:00'))
now = datetime.now(timezone.utc)
days = (now - created).total_seconds() / 86400
print(f'{days:.1f}')
" 2>/dev/null || echo "?")
    else
        DAYS_RUNNING="?"
    fi
    echo "  Days running:    ${DAYS_RUNNING}"
    echo "  VM daily rate:   ~\$2.00 (Standard_B2ms)"
    echo "  Storage rate:    ~\$0.01/day"
    if [[ "${DAYS_RUNNING}" != "?" ]]; then
        COST=$(python3 -c "print(f'\${float(${DAYS_RUNNING}) * 2.01:.2f}')" 2>/dev/null || echo "?")
        echo "  Estimated total: ~\$${COST}"
    fi
fi

# ─── Quick Commands ──────────────────────────────────────────────────────────
echo ""
echo -e "${CYAN}Quick commands:${NC}"
echo "  Live follow:     ./experiments/infra/monitor.sh --follow"
echo "  Open shell:      ./experiments/infra/monitor.sh --shell"
echo "  View logs:       ./experiments/infra/monitor.sh --logs 50"
echo "  Restart daemon:  ./experiments/infra/monitor.sh --restart"
echo "  Pull results:    ./experiments/infra/pull-results.sh"
echo "  Show cost:       ./experiments/infra/monitor.sh --cost"
echo ""
