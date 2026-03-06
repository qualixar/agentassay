#!/bin/bash
###############################################################################
# AgentAssay Experiment VM Setup — Accenture Security Compliant
#
# Creates an Azure VM with ZERO public IP and ZERO open inbound ports.
# Results flow via Azure Blob Storage (VM pushes, Mac pulls).
# Interactive access via `az ssh vm` (AAD tunnel, no ports needed).
#
# Security posture:
#   - NO public IP address
#   - NO inbound NSG rules (all inbound DENIED by default)
#   - Outbound: only AzureCloud:443 and Storage:443 allowed
#   - Deny-all outbound at priority 4096 (Qualys requirement)
#   - Results sync via Azure Blob Storage with time-limited SAS tokens
#
# Usage:
#   ./experiments/infra/setup-vm.sh [OPTIONS]
#
# Options:
#   --region REGION      Azure region (default: eastus2)
#   --size SIZE          VM size (default: Standard_B2ms)
#   --subscription SUB   Subscription ID (default: TAP_Atlas)
#   --help               Show this help
#
# Tears down with: ./experiments/infra/teardown.sh
###############################################################################
set -euo pipefail

# ─── Defaults ────────────────────────────────────────────────────────────────
REGION="eastus2"
VM_SIZE="Standard_B2ms"
SUBSCRIPTION="ceb60a98-12a2-43bf-bc35-abfaa0a99d69"  # TAP_Atlas
RESOURCE_GROUP="agentassay-experiments"
VM_NAME="agentassay-runner"
IMAGE="Canonical:ubuntu-24_04-lts:server:latest"
ADMIN_USER="azureuser"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
STATE_FILE="${SCRIPT_DIR}/.vm-state.json"

# Storage — name must be globally unique (3-24 lowercase alphanumeric)
STORAGE_SUFFIX=$(LC_ALL=C tr -dc 'a-z0-9' < /dev/urandom | head -c 6)
STORAGE_ACCOUNT="agentassayres${STORAGE_SUFFIX}"
STORAGE_CONTAINER="experiment-results"

# SAS token expiry: 14 days from now (covers full experiment window)
SAS_EXPIRY=$(date -u -v+14d +%Y-%m-%dT%H:%M:%SZ 2>/dev/null || date -u -d "+14 days" +%Y-%m-%dT%H:%M:%SZ)

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
    sed -n '/^# Usage:/,/^###/p' "$0" | head -n -1 | sed 's/^# //'
    echo ""
    echo "Defaults:"
    echo "  Region:       ${REGION}"
    echo "  VM Size:      ${VM_SIZE} (2 vCPU, 8 GB RAM, ~\$2/day)"
    echo "  Subscription: TAP_Atlas"
    echo "  VM Name:      ${VM_NAME}"
    echo ""
    echo "Security:"
    echo "  Public IP:    NONE"
    echo "  Inbound:      ALL DENIED (zero rules)"
    echo "  Outbound:     AzureCloud:443 + Storage:443 only"
    echo "  Results:      Azure Blob Storage (SAS token, time-limited)"
    echo "  Access:       az ssh vm (AAD tunnel, no open ports)"
    echo ""
    echo "Cost estimate: ~\$2/day VM + ~\$0.01/day storage"
    exit 0
}

# ─── Parse Args ──────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --region)       REGION="$2"; shift 2 ;;
        --size)         VM_SIZE="$2"; shift 2 ;;
        --subscription) SUBSCRIPTION="$2"; shift 2 ;;
        --help|-h)      show_help ;;
        *)              err "Unknown option: $1"; show_help ;;
    esac
done

# ─── Prerequisites ───────────────────────────────────────────────────────────
step 1 "Checking prerequisites"

if ! command -v az &>/dev/null; then
    err "Azure CLI (az) not found. Install: https://learn.microsoft.com/en-us/cli/azure/install-azure-cli"
    exit 1
fi
ok "Azure CLI found: $(az version --query '"azure-cli"' -o tsv)"

if ! command -v jq &>/dev/null; then
    err "jq not found. Install: brew install jq"
    exit 1
fi
ok "jq found"

# Check az login status
if ! az account show &>/dev/null; then
    err "Not logged in to Azure. Run: az login"
    exit 1
fi

CURRENT_ACCOUNT=$(az account show --query name -o tsv)
ok "Logged in as: ${CURRENT_ACCOUNT}"

# Set subscription
az account set --subscription "${SUBSCRIPTION}" 2>/dev/null || {
    err "Cannot set subscription ${SUBSCRIPTION}. Check access."
    exit 1
}
ok "Subscription set: $(az account show --query name -o tsv)"

# Check if ssh extension is installed
if ! az extension show --name ssh &>/dev/null 2>&1; then
    info "Installing az ssh extension..."
    az extension add --name ssh --yes
fi
ok "az ssh extension available"

# ─── Check Existing Resources ────────────────────────────────────────────────
step 2 "Checking for existing resources"

if az group show -n "${RESOURCE_GROUP}" &>/dev/null 2>&1; then
    warn "Resource group '${RESOURCE_GROUP}' already exists."
    echo ""
    read -r -p "Delete existing and recreate? [y/N]: " CONFIRM
    if [[ "${CONFIRM}" =~ ^[Yy]$ ]]; then
        info "Deleting existing resource group (this takes 2-3 minutes)..."
        az group delete -n "${RESOURCE_GROUP}" --yes --no-wait
        info "Waiting for deletion to complete..."
        az group wait --deleted -n "${RESOURCE_GROUP}" --timeout 300 2>/dev/null || true
        ok "Old resource group deleted"
    else
        err "Resource group exists. Delete it first with ./experiments/infra/teardown.sh"
        exit 1
    fi
fi

# ─── Create Resource Group ───────────────────────────────────────────────────
step 3 "Creating resource group"

az group create \
    --name "${RESOURCE_GROUP}" \
    --location "${REGION}" \
    --tags "project=agentassay" "purpose=experiments" "auto-delete=true" \
    -o none

ok "Resource group '${RESOURCE_GROUP}' created in ${REGION}"

# ─── Create Storage Account + Container ──────────────────────────────────────
step 4 "Creating Azure Blob Storage for results"

info "Storage account: ${STORAGE_ACCOUNT} (globally unique)"

az storage account create \
    --name "${STORAGE_ACCOUNT}" \
    --resource-group "${RESOURCE_GROUP}" \
    --location "${REGION}" \
    --sku Standard_LRS \
    --kind StorageV2 \
    --min-tls-version TLS1_2 \
    --allow-blob-public-access false \
    --https-only true \
    --tags "project=agentassay" "purpose=experiment-results" \
    -o none

ok "Storage account created: ${STORAGE_ACCOUNT}"

# Get storage connection string
STORAGE_CONN_STRING=$(az storage account show-connection-string \
    --name "${STORAGE_ACCOUNT}" \
    --resource-group "${RESOURCE_GROUP}" \
    --query connectionString -o tsv)

if [[ -z "${STORAGE_CONN_STRING}" ]]; then
    err "Failed to get storage connection string."
    exit 1
fi
ok "Storage connection string retrieved"

# Create container
az storage container create \
    --name "${STORAGE_CONTAINER}" \
    --account-name "${STORAGE_ACCOUNT}" \
    --connection-string "${STORAGE_CONN_STRING}" \
    --public-access off \
    -o none

ok "Container created: ${STORAGE_CONTAINER}"

# Generate VM SAS token (write-only: create, write, add, list)
VM_SAS_TOKEN=$(az storage container generate-sas \
    --name "${STORAGE_CONTAINER}" \
    --account-name "${STORAGE_ACCOUNT}" \
    --connection-string "${STORAGE_CONN_STRING}" \
    --permissions cwla \
    --expiry "${SAS_EXPIRY}" \
    --https-only \
    -o tsv)

ok "VM SAS token generated (write-only, expires: ${SAS_EXPIRY})"

# Generate Mac SAS token (read-only: read, list)
MAC_SAS_TOKEN=$(az storage container generate-sas \
    --name "${STORAGE_CONTAINER}" \
    --account-name "${STORAGE_ACCOUNT}" \
    --connection-string "${STORAGE_CONN_STRING}" \
    --permissions rl \
    --expiry "${SAS_EXPIRY}" \
    --https-only \
    -o tsv)

ok "Mac SAS token generated (read-only, expires: ${SAS_EXPIRY})"

# ─── Create VM (NO Public IP) ───────────────────────────────────────────────
step 5 "Creating VM (${VM_SIZE}, Ubuntu 24.04 LTS, NO public IP)"
info "This takes 2-3 minutes..."

VM_OUTPUT=$(az vm create \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${VM_NAME}" \
    --image "${IMAGE}" \
    --size "${VM_SIZE}" \
    --admin-username "${ADMIN_USER}" \
    --generate-ssh-keys \
    --public-ip-address "" \
    --nsg "${VM_NAME}-nsg" \
    --os-disk-size-gb 64 \
    --tags "project=agentassay" "purpose=experiments" \
    -o json)

VM_ID=$(echo "${VM_OUTPUT}" | jq -r '.id // empty')

if [[ -z "${VM_ID}" ]]; then
    err "VM creation failed. Check Azure portal."
    exit 1
fi
ok "VM created: ${VM_NAME} (NO public IP — Accenture compliant)"

# ─── Configure NSG Rules (ZERO inbound, strict outbound) ────────────────────
step 6 "Configuring NSG rules (Accenture/Qualys compliant)"

NSG_NAME="${VM_NAME}-nsg"

# Delete ANY default inbound rules that Azure may have created
for RULE_NAME in $(az network nsg rule list \
    --resource-group "${RESOURCE_GROUP}" \
    --nsg-name "${NSG_NAME}" \
    --query "[?direction=='Inbound' && access=='Allow'].name" \
    -o tsv 2>/dev/null); do
    info "Removing default inbound rule: ${RULE_NAME}"
    az network nsg rule delete \
        --resource-group "${RESOURCE_GROUP}" \
        --nsg-name "${NSG_NAME}" \
        --name "${RULE_NAME}" \
        -o none 2>/dev/null || true
done

ok "All inbound allow rules removed (default deny applies)"

# Outbound: Allow AzureCloud service tag on 443 (for az ssh vm relay + AAD)
az network nsg rule create \
    --resource-group "${RESOURCE_GROUP}" \
    --nsg-name "${NSG_NAME}" \
    --name "AllowOutbound-AzureCloud-443" \
    --priority 100 \
    --direction Outbound \
    --access Allow \
    --protocol Tcp \
    --source-address-prefixes '*' \
    --destination-address-prefixes AzureCloud \
    --destination-port-ranges 443 \
    --description "Allow outbound to Azure services (AAD, relay, management)" \
    -o none

ok "NSG outbound: AzureCloud:443 ALLOWED"

# Outbound: Allow Storage service tag on 443 (for Blob Storage push)
az network nsg rule create \
    --resource-group "${RESOURCE_GROUP}" \
    --nsg-name "${NSG_NAME}" \
    --name "AllowOutbound-Storage-443" \
    --priority 200 \
    --direction Outbound \
    --access Allow \
    --protocol Tcp \
    --source-address-prefixes '*' \
    --destination-address-prefixes Storage \
    --destination-port-ranges 443 \
    --description "Allow outbound to Azure Blob Storage for results push" \
    -o none

ok "NSG outbound: Storage:443 ALLOWED"

# Outbound: DENY ALL at priority 4096 (MANDATORY per Accenture Qualys policy)
az network nsg rule create \
    --resource-group "${RESOURCE_GROUP}" \
    --nsg-name "${NSG_NAME}" \
    --name "DenyAllOutbound" \
    --priority 4096 \
    --direction Outbound \
    --access Deny \
    --protocol '*' \
    --source-address-prefixes '*' \
    --destination-address-prefixes Internet \
    --destination-port-ranges '*' \
    --description "MANDATORY: Deny all outbound to Internet (Accenture Qualys)" \
    -o none

ok "NSG outbound: DENY ALL Internet at priority 4096 (Qualys compliant)"

# Verify: print all NSG rules
info "NSG rules summary:"
az network nsg rule list \
    --resource-group "${RESOURCE_GROUP}" \
    --nsg-name "${NSG_NAME}" \
    --query "[].{Priority:priority, Direction:direction, Access:access, Dest:destinationAddressPrefix, Port:destinationPortRange, Name:name}" \
    -o table

# ─── Wait for VM to be Ready ────────────────────────────────────────────────
step 7 "Waiting for VM to be ready"

info "Waiting for VM provisioning to complete..."
az vm wait \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${VM_NAME}" \
    --created \
    --timeout 300

ok "VM is provisioned"

# Give cloud-init a moment to finish initial boot
info "Waiting 30s for cloud-init to settle..."
sleep 30

# ─── Configure VM (via az ssh vm — AAD tunnel, no ports needed) ─────────────
step 8 "Configuring VM software (Python 3.12, system deps, Azure CLI)"

info "Running configuration via az ssh vm (AAD tunnel, no open ports)..."

az ssh vm \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${VM_NAME}" \
    -- "bash -s" <<'REMOTE_SETUP'
set -euo pipefail

echo "=== Updating system packages ==="
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get upgrade -y -qq

echo "=== Installing Python 3.12 and system dependencies ==="
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    python3.12 \
    python3.12-venv \
    python3.12-dev \
    python3-pip \
    git \
    curl \
    jq \
    tmux \
    htop \
    unzip \
    build-essential

echo "=== Installing Azure CLI (for blob storage push) ==="
curl -sL https://aka.ms/InstallAzureCLIDeb | sudo bash

echo "=== Creating agentassay user ==="
if ! id -u agentassay &>/dev/null; then
    sudo useradd -m -s /bin/bash agentassay
    sudo mkdir -p /home/agentassay/agentassay
    sudo mkdir -p /home/agentassay/.venv
    sudo chown -R agentassay:agentassay /home/agentassay/
fi

echo "=== Setting up Python venv for agentassay user ==="
sudo -u agentassay bash -c '
    python3.12 -m venv /home/agentassay/.venv
    source /home/agentassay/.venv/bin/activate
    pip install --upgrade pip setuptools wheel
    pip install azure-storage-blob
'

echo "=== Configuring system limits ==="
sudo bash -c 'cat > /etc/security/limits.d/agentassay.conf <<EOF
agentassay soft nofile 65536
agentassay hard nofile 65536
agentassay soft nproc 4096
agentassay hard nproc 4096
EOF'

echo "=== Setting up log rotation ==="
sudo bash -c 'cat > /etc/logrotate.d/agentassay <<EOF
/home/agentassay/agentassay/experiments/logs/*.log {
    daily
    rotate 7
    compress
    missingok
    notifempty
    copytruncate
}
EOF'

echo "=== VM setup complete ==="
python3.12 --version
az --version | head -1
REMOTE_SETUP

ok "VM configured: Python 3.12, Azure CLI, agentassay user, NO sshd on 443"

# ─── Auto-shutdown Schedule ──────────────────────────────────────────────────
step 9 "Setting auto-shutdown at 19:00 UTC daily"

SHUTDOWN_TIME="1900"  # 7 PM UTC
az vm auto-shutdown \
    --resource-group "${RESOURCE_GROUP}" \
    --name "${VM_NAME}" \
    --time "${SHUTDOWN_TIME}" \
    -o none 2>/dev/null || warn "Auto-shutdown not set (non-critical)"

ok "Auto-shutdown scheduled at ${SHUTDOWN_TIME} UTC daily"

# ─── Save State ──────────────────────────────────────────────────────────────
step 10 "Saving VM state"

cat > "${STATE_FILE}" <<EOF
{
    "resource_group": "${RESOURCE_GROUP}",
    "vm_name": "${VM_NAME}",
    "region": "${REGION}",
    "vm_size": "${VM_SIZE}",
    "subscription": "${SUBSCRIPTION}",
    "admin_user": "${ADMIN_USER}",
    "nsg_name": "${NSG_NAME}",
    "storage_account": "${STORAGE_ACCOUNT}",
    "storage_container": "${STORAGE_CONTAINER}",
    "storage_connection_string": "${STORAGE_CONN_STRING}",
    "vm_sas_token": "${VM_SAS_TOKEN}",
    "mac_sas_token": "${MAC_SAS_TOKEN}",
    "sas_expiry": "${SAS_EXPIRY}",
    "created_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
    "security": {
        "public_ip": "NONE",
        "inbound_rules": "ALL DENIED",
        "outbound_rules": "AzureCloud:443 + Storage:443 only",
        "deny_all_outbound_priority": 4096,
        "qualys_compliant": true
    }
}
EOF

ok "State saved to ${STATE_FILE}"

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${GREEN}  VM Setup Complete — Accenture Security Compliant${NC}"
echo -e "${GREEN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo ""
echo "  Resource Group:   ${RESOURCE_GROUP}"
echo "  VM Name:          ${VM_NAME}"
echo -e "  Public IP:        ${GREEN}NONE${NC} (Qualys compliant)"
echo -e "  Inbound Rules:    ${GREEN}ALL DENIED${NC} (zero open ports)"
echo -e "  Outbound Rules:   ${GREEN}AzureCloud + Storage only${NC}"
echo "  Region:           ${REGION}"
echo "  Size:             ${VM_SIZE}"
echo "  Storage Account:  ${STORAGE_ACCOUNT}"
echo "  Container:        ${STORAGE_CONTAINER}"
echo "  SAS Expiry:       ${SAS_EXPIRY}"
echo "  Cost:             ~\$2/day (VM) + ~\$0.01/day (storage)"
echo ""
echo -e "${CYAN}Connection (AAD tunnel — no open ports needed):${NC}"
echo "  az ssh vm -g ${RESOURCE_GROUP} -n ${VM_NAME}"
echo ""
echo -e "${CYAN}Emergency access (Azure Portal):${NC}"
echo "  Serial Console: portal.azure.com > ${VM_NAME} > Serial Console"
echo ""
echo -e "${CYAN}Results flow:${NC}"
echo "  VM pushes to Blob Storage every 30 minutes"
echo "  Mac pulls from Blob Storage: ./experiments/infra/pull-results.sh"
echo ""
echo -e "${CYAN}Next Steps:${NC}"
echo "  1. Deploy code:          ./experiments/infra/deploy.sh"
echo "  2. Monitor experiments:  ./experiments/infra/monitor.sh"
echo "  3. Pull results:         ./experiments/infra/pull-results.sh"
echo "  4. Delete everything:    ./experiments/infra/teardown.sh"
echo ""
