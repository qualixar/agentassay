#!/bin/bash
###############################################################################
# AgentAssay Experiment Deployment — Accenture Security Compliant
#
# Deploys AgentAssay code to the Azure VM and starts the experiment daemon.
# Uses az ssh vm (AAD tunnel) exclusively — no public IP, no open ports.
# Configures Blob Storage connection for results push.
#
# Usage:
#   ./experiments/infra/deploy.sh [OPTIONS]
#
# Options:
#   --env-file PATH      Path to .env file (default: .env in project root)
#   --config-dir PATH    Experiment config dir (default: experiments/configs/)
#   --skip-install       Skip pip install (redeploy code only)
#   --no-start           Deploy without starting the daemon
#   --help               Show this help
#
# Prerequisites:
#   - VM created via ./experiments/infra/setup-vm.sh
#   - .env file with API keys in project root
#   - az CLI logged in
###############################################################################
set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
STATE_FILE="${SCRIPT_DIR}/.vm-state.json"
ENV_FILE="${PROJECT_ROOT}/.env"
CONFIG_DIR="${PROJECT_ROOT}/experiments/configs"
SKIP_INSTALL=false
NO_START=false

REMOTE_USER="agentassay"
REMOTE_DIR="/home/agentassay/agentassay"
ADMIN_USER="azureuser"

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
step()  { echo -e "\n${GREEN}━━━ Step $1: $2 ━━━${NC}"; }

# ─── Help ────────────────────────────────────────────────────────────────────
show_help() {
    sed -n '/^# Usage:/,/^# Prerequisites:/p' "$0" | head -n -1 | sed 's/^# //'
    exit 0
}

# ─── Parse Args ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --env-file)      ENV_FILE="$2"; shift 2 ;;
        --config-dir)    CONFIG_DIR="$2"; shift 2 ;;
        --skip-install)  SKIP_INSTALL=true; shift ;;
        --no-start)      NO_START=true; shift ;;
        --help|-h)       show_help ;;
        *)               err "Unknown option: $1"; show_help ;;
    esac
done

# ─── Load VM State ──────────────────────────────────────────────────────────
step 1 "Loading VM state"

if [[ ! -f "${STATE_FILE}" ]]; then
    err "VM state file not found: ${STATE_FILE}"
    err "Run ./experiments/infra/setup-vm.sh first"
    exit 1
fi

RESOURCE_GROUP=$(jq -r '.resource_group' "${STATE_FILE}")
VM_NAME=$(jq -r '.vm_name' "${STATE_FILE}")
STORAGE_ACCOUNT=$(jq -r '.storage_account' "${STATE_FILE}")
STORAGE_CONTAINER=$(jq -r '.storage_container' "${STATE_FILE}")
STORAGE_CONN_STRING=$(jq -r '.storage_connection_string' "${STATE_FILE}")
VM_SAS_TOKEN=$(jq -r '.vm_sas_token' "${STATE_FILE}")

ok "VM: ${VM_NAME} (NO public IP) in ${RESOURCE_GROUP}"
ok "Storage: ${STORAGE_ACCOUNT}/${STORAGE_CONTAINER}"

# ─── Validate Prerequisites ─────────────────────────────────────────────────
step 2 "Validating prerequisites"

if [[ ! -f "${ENV_FILE}" ]]; then
    err ".env file not found at: ${ENV_FILE}"
    err "Create it with your API keys (AZURE_SUB{1,2,3}_*, etc.)"
    exit 1
fi
ok ".env file found: ${ENV_FILE}"

if [[ ! -f "${PROJECT_ROOT}/pyproject.toml" ]]; then
    err "pyproject.toml not found. Are you in the right directory?"
    exit 1
fi
ok "pyproject.toml found"

if [[ ! -d "${PROJECT_ROOT}/src/agentassay" ]]; then
    err "src/agentassay/ not found. Source code missing."
    exit 1
fi
ok "Source code found"

if [[ ! -d "${CONFIG_DIR}" ]]; then
    warn "Config directory not found: ${CONFIG_DIR}"
    warn "Experiments will fail without configs. Create them first."
fi

# Verify VM is running
VM_STATE=$(az vm get-instance-view \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${VM_NAME}" \
    --query "instanceView.statuses[?starts_with(code, 'PowerState')].displayStatus" \
    -o tsv 2>/dev/null || echo "UNKNOWN")

if [[ "${VM_STATE}" != "VM running" ]]; then
    err "VM is not running. State: ${VM_STATE}"
    err "Start it with: az vm start -g ${RESOURCE_GROUP} -n ${VM_NAME}"
    exit 1
fi
ok "VM is running"

# ─── Create Deployment Tarball ───────────────────────────────────────────────
step 3 "Creating deployment tarball"

TARBALL="/tmp/agentassay-deploy-$(date +%Y%m%d-%H%M%S).tar.gz"

cd "${PROJECT_ROOT}"

# Build file list — include only what's needed on the VM
TAR_FILES=(
    "pyproject.toml"
    "src/"
    "experiments/configs/"
    "experiments/runner/"
    "experiments/scenarios/"
)

# Filter to only existing paths
EXISTING_FILES=()
for f in "${TAR_FILES[@]}"; do
    if [[ -e "${f}" ]]; then
        EXISTING_FILES+=("${f}")
    else
        warn "Skipping missing path: ${f}"
    fi
done

if [[ ${#EXISTING_FILES[@]} -eq 0 ]]; then
    err "No files to deploy"
    exit 1
fi

tar czf "${TARBALL}" "${EXISTING_FILES[@]}"

TARBALL_SIZE=$(du -h "${TARBALL}" | cut -f1)
ok "Tarball created: ${TARBALL} (${TARBALL_SIZE})"

# ─── Upload to VM via az ssh vm (piped tarball) ─────────────────────────────
step 4 "Uploading code to VM (via az ssh vm — AAD tunnel)"

info "Piping tarball through az ssh vm..."

# Stream the tarball directly through the SSH tunnel
cat "${TARBALL}" | az ssh vm \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${VM_NAME}" \
    -- "bash -c 'set -euo pipefail; \
        sudo mkdir -p ${REMOTE_DIR}; \
        sudo chown -R ${REMOTE_USER}:${REMOTE_USER} /home/${REMOTE_USER}/; \
        cat > /tmp/agentassay-deploy.tar.gz; \
        sudo -u ${REMOTE_USER} tar xzf /tmp/agentassay-deploy.tar.gz -C ${REMOTE_DIR}/; \
        rm -f /tmp/agentassay-deploy.tar.gz; \
        sudo -u ${REMOTE_USER} mkdir -p ${REMOTE_DIR}/experiments/results; \
        sudo -u ${REMOTE_USER} mkdir -p ${REMOTE_DIR}/experiments/logs; \
        sudo -u ${REMOTE_USER} mkdir -p ${REMOTE_DIR}/experiments/figures; \
        sudo -u ${REMOTE_USER} mkdir -p ${REMOTE_DIR}/experiments/analysis; \
        echo \"Files deployed:\"; \
        ls -la ${REMOTE_DIR}/'"

ok "Code uploaded to VM"

# ─── Upload .env File with Blob Storage credentials ─────────────────────────
step 5 "Uploading .env file (with Blob Storage connection)"

# Append blob storage config to .env
ENV_WITH_BLOB=$(cat "${ENV_FILE}")
ENV_WITH_BLOB="${ENV_WITH_BLOB}

# ============================================================================
# Azure Blob Storage (for results push — populated by setup-vm.sh)
# ============================================================================
AZURE_STORAGE_CONNECTION_STRING=${STORAGE_CONN_STRING}
AZURE_STORAGE_CONTAINER=${STORAGE_CONTAINER}
AZURE_STORAGE_ACCOUNT=${STORAGE_ACCOUNT}
AZURE_STORAGE_SAS_TOKEN=${VM_SAS_TOKEN}
"

# Use base64 encoding to safely transport the .env content
ENV_B64=$(echo "${ENV_WITH_BLOB}" | base64)

az ssh vm \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${VM_NAME}" \
    -- "bash -c 'set -euo pipefail; \
        echo \"${ENV_B64}\" | base64 -d > /tmp/agentassay-env; \
        sudo mv /tmp/agentassay-env ${REMOTE_DIR}/.env; \
        sudo chown ${REMOTE_USER}:${REMOTE_USER} ${REMOTE_DIR}/.env; \
        sudo chmod 600 ${REMOTE_DIR}/.env; \
        echo \".env deployed (permissions: 600, includes Blob Storage config)\"'"

ok ".env uploaded with restricted permissions (600) + Blob Storage credentials"

# ─── Install Dependencies ───────────────────────────────────────────────────
if [[ "${SKIP_INSTALL}" == "false" ]]; then
    step 6 "Installing Python dependencies on VM"

    az ssh vm \
        --resource-group "${RESOURCE_GROUP}" \
        --name "${VM_NAME}" \
        -- "bash -s" <<REMOTE_INSTALL
set -euo pipefail

echo "=== Installing agentassay in editable mode ==="
sudo -u ${REMOTE_USER} bash -c '
    source /home/${REMOTE_USER}/.venv/bin/activate
    cd ${REMOTE_DIR}
    pip install -e ".[dev]" 2>&1 | tail -5
    pip install azure-storage-blob 2>&1 | tail -2
    echo ""
    echo "=== Key packages ==="
    pip list | grep -iE "agent|azure|blob" || true
    echo ""
    pip list --format=columns | wc -l
    echo "total packages installed"
'
REMOTE_INSTALL

    ok "Dependencies installed (including azure-storage-blob)"
else
    info "Skipping pip install (--skip-install)"
fi

# ─── Install systemd Service ────────────────────────────────────────────────
step 7 "Installing systemd service"

SERVICE_FILE="${SCRIPT_DIR}/agentassay-runner.service"

if [[ ! -f "${SERVICE_FILE}" ]]; then
    err "Service file not found: ${SERVICE_FILE}"
    exit 1
fi

SERVICE_CONTENT=$(cat "${SERVICE_FILE}")

az ssh vm \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${VM_NAME}" \
    -- "bash -s" <<REMOTE_SERVICE
set -euo pipefail

cat > /tmp/agentassay-runner.service <<'SERVICEFILE'
${SERVICE_CONTENT}
SERVICEFILE

sudo mv /tmp/agentassay-runner.service /etc/systemd/system/agentassay-runner.service
sudo systemctl daemon-reload
sudo systemctl enable agentassay-runner.service

echo "=== systemd service installed and enabled ==="
REMOTE_SERVICE

ok "systemd service installed"

# ─── Start Daemon ────────────────────────────────────────────────────────────
if [[ "${NO_START}" == "false" ]]; then
    step 8 "Starting experiment daemon"

    az ssh vm \
        --resource-group "${RESOURCE_GROUP}" \
        --name "${VM_NAME}" \
        -- "bash -s" <<'REMOTE_START'
set -euo pipefail

sudo systemctl start agentassay-runner.service

# Wait a moment, then check status
sleep 5

if sudo systemctl is-active --quiet agentassay-runner.service; then
    echo "=== Daemon is RUNNING ==="
    sudo systemctl status agentassay-runner.service --no-pager -l | head -15
else
    echo "=== WARNING: Daemon may have failed to start ==="
    sudo systemctl status agentassay-runner.service --no-pager -l
    echo ""
    echo "=== Last 20 log lines ==="
    sudo journalctl -u agentassay-runner.service --no-pager -n 20
fi
REMOTE_START

    ok "Experiment daemon started"
else
    info "Skipping daemon start (--no-start)"
    info "Start manually: az ssh vm -g ${RESOURCE_GROUP} -n ${VM_NAME} -- 'sudo systemctl start agentassay-runner.service'"
fi

# ─── Cleanup ─────────────────────────────────────────────────────────────────
rm -f "${TARBALL}"

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  Deployment Complete — Accenture Security Compliant${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo -e "${CYAN}Results flow:${NC}"
echo "  VM pushes to Blob Storage every 30 min"
echo "  Pull: ./experiments/infra/pull-results.sh"
echo ""
echo -e "${CYAN}Monitor experiments (Blob-based, no SSH):${NC}"
echo "  ./experiments/infra/monitor.sh"
echo ""
echo -e "${CYAN}Interactive shell (AAD tunnel):${NC}"
echo "  ./experiments/infra/monitor.sh --shell"
echo ""
echo -e "${CYAN}Watch live logs (AAD tunnel):${NC}"
echo "  az ssh vm -g ${RESOURCE_GROUP} -n ${VM_NAME} -- 'sudo journalctl -u agentassay-runner -f'"
echo ""
echo -e "${CYAN}Redeploy code only (no reinstall):${NC}"
echo "  ./experiments/infra/deploy.sh --skip-install"
echo ""
