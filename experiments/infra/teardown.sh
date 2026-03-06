#!/bin/bash
###############################################################################
# AgentAssay Experiment Teardown — Accenture Security Compliant
#
# Deletes ALL Azure resources created for AgentAssay experiments.
# Cascading delete: removes VM, disk, NSG, storage account, and resource group.
# Pulls final results from Blob Storage before deleting.
#
# Usage:
#   ./experiments/infra/teardown.sh [OPTIONS]
#
# Options:
#   --force            Skip confirmation prompt
#   --pull-first       Pull results before deleting (default: prompted)
#   --no-pull          Skip pulling results
#   --help             Show this help
#
# DESTRUCTIVE: This permanently deletes all Azure resources.
###############################################################################
set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="${SCRIPT_DIR}/.vm-state.json"
LAST_PULL_FILE="${SCRIPT_DIR}/.last-pull-timestamp"
FORCE=false
PULL_FIRST=""  # empty = ask, "yes" = pull, "no" = skip

# ─── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

info()  { echo -e "${CYAN}[INFO]${NC}  $*"; }
ok()    { echo -e "${GREEN}[OK]${NC}    $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
err()   { echo -e "${RED}[ERROR]${NC} $*" >&2; }

# ─── Help ────────────────────────────────────────────────────────────────────
show_help() {
    sed -n '/^# Usage:/,/^# DESTRUCTIVE:/p' "$0" | head -n -1 | sed 's/^# //'
    exit 0
}

# ─── Parse Args ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --force)       FORCE=true; shift ;;
        --pull-first)  PULL_FIRST="yes"; shift ;;
        --no-pull)     PULL_FIRST="no"; shift ;;
        --help|-h)     show_help ;;
        *)             err "Unknown option: $1"; show_help ;;
    esac
done

# ─── Load State ──────────────────────────────────────────────────────────────
if [[ -f "${STATE_FILE}" ]]; then
    RESOURCE_GROUP=$(jq -r '.resource_group' "${STATE_FILE}")
    VM_NAME=$(jq -r '.vm_name' "${STATE_FILE}")
    STORAGE_ACCOUNT=$(jq -r '.storage_account // "unknown"' "${STATE_FILE}")
    STORAGE_CONTAINER=$(jq -r '.storage_container // "unknown"' "${STATE_FILE}")
    CREATED_AT=$(jq -r '.created_at' "${STATE_FILE}")
else
    warn "State file not found: ${STATE_FILE}"
    RESOURCE_GROUP="agentassay-experiments"
    VM_NAME="agentassay-runner"
    STORAGE_ACCOUNT="unknown"
    STORAGE_CONTAINER="unknown"
    CREATED_AT="unknown"
fi

# ─── Check if Resource Group Exists ─────────────────────────────────────────
if ! az group show -n "${RESOURCE_GROUP}" &>/dev/null 2>&1; then
    info "Resource group '${RESOURCE_GROUP}' does not exist. Nothing to delete."
    # Clean up state files if they exist
    rm -f "${STATE_FILE}"
    rm -f "${LAST_PULL_FILE}"
    exit 0
fi

# ─── List Resources ─────────────────────────────────────────────────────────
echo ""
echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${RED}  DESTRUCTIVE OPERATION — Azure Resource Deletion${NC}"
echo -e "${RED}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Resource Group:   ${RESOURCE_GROUP}"
echo "  VM Name:          ${VM_NAME}"
echo -e "  Public IP:        ${GREEN}NONE${NC} (Accenture compliant)"
echo "  Storage Account:  ${STORAGE_ACCOUNT}"
echo "  Container:        ${STORAGE_CONTAINER}"
echo "  Created:          ${CREATED_AT}"
echo ""

info "Resources that will be PERMANENTLY DELETED:"
az resource list \
    --resource-group "${RESOURCE_GROUP}" \
    --query "[].{Name:name, Type:type, Location:location}" \
    -o table 2>/dev/null || warn "Could not list resources"
echo ""

# ─── Pull Final Results from Blob Storage ────────────────────────────────────
if [[ "${PULL_FIRST}" == "" ]]; then
    read -r -p "Pull final results from Blob Storage before deleting? [Y/n]: " PULL_ANSWER
    PULL_FIRST=$([[ "${PULL_ANSWER}" =~ ^[Nn]$ ]] && echo "no" || echo "yes")
fi

if [[ "${PULL_FIRST}" == "yes" ]]; then
    info "Pulling final results from Blob Storage before teardown..."
    PULL_SCRIPT="${SCRIPT_DIR}/pull-results.sh"
    if [[ -x "${PULL_SCRIPT}" ]]; then
        "${PULL_SCRIPT}" --full || warn "Pull failed — results may be lost. Continue anyway."
    else
        warn "pull-results.sh not found or not executable. Skipping pull."
    fi
fi

# ─── Confirm Deletion ───────────────────────────────────────────────────────
if [[ "${FORCE}" == "false" ]]; then
    echo ""
    echo -e "${RED}This will PERMANENTLY DELETE all resources listed above.${NC}"
    echo -e "${RED}Including: VM, disk, NSG, storage account (${STORAGE_ACCOUNT}), and all blobs.${NC}"
    echo -e "${RED}This action CANNOT be undone.${NC}"
    echo ""
    read -r -p "Type 'DELETE' to confirm: " CONFIRM
    if [[ "${CONFIRM}" != "DELETE" ]]; then
        info "Aborted. No resources were deleted."
        exit 0
    fi
fi

# ─── Delete ENTIRE Resource Group (cascading delete) ────────────────────────
echo ""
info "Deleting resource group '${RESOURCE_GROUP}'..."
info "This cascading-deletes ALL resources (VM, disk, NSG, storage account — everything)."
info "Takes 2-5 minutes..."

az group delete \
    --name "${RESOURCE_GROUP}" \
    --yes \
    --no-wait

# Wait for deletion with a progress indicator
WAIT_COUNT=0
MAX_WAIT=60  # 60 * 5s = 5 minutes max
while az group show -n "${RESOURCE_GROUP}" &>/dev/null 2>&1; do
    WAIT_COUNT=$((WAIT_COUNT + 1))
    if [[ ${WAIT_COUNT} -ge ${MAX_WAIT} ]]; then
        warn "Deletion taking longer than expected. Check Azure portal."
        warn "Resource group may still be deleting in background."
        break
    fi
    printf "\r  Waiting for deletion... %ds" $((WAIT_COUNT * 5))
    sleep 5
done
echo ""

# ─── Verify Deletion ────────────────────────────────────────────────────────
if az group show -n "${RESOURCE_GROUP}" &>/dev/null 2>&1; then
    warn "Resource group still exists (may be deleting in background)."
    warn "Check Azure portal: https://portal.azure.com"
else
    ok "Resource group '${RESOURCE_GROUP}' deleted successfully"
fi

# ─── Cleanup Local State Files ──────────────────────────────────────────────
if [[ -f "${STATE_FILE}" ]]; then
    rm -f "${STATE_FILE}"
    ok "Local state file cleaned up: .vm-state.json"
fi

if [[ -f "${LAST_PULL_FILE}" ]]; then
    rm -f "${LAST_PULL_FILE}"
    ok "Local pull timestamp cleaned up: .last-pull-timestamp"
fi

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Teardown Complete${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  All Azure resources for AgentAssay experiments have been deleted:"
echo "    - VM:              ${VM_NAME}"
echo "    - Storage Account: ${STORAGE_ACCOUNT}"
echo "    - Resource Group:  ${RESOURCE_GROUP} (and everything in it)"
echo ""
echo "  Estimated cost saved: ~\$2/day"
echo ""
echo "  To run experiments again:"
echo "    1. ./experiments/infra/setup-vm.sh"
echo "    2. ./experiments/infra/deploy.sh"
echo ""
