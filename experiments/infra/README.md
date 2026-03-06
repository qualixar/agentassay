# AgentAssay — Experiment Infrastructure (Accenture Security Compliant)

Azure VM infrastructure for running AgentAssay experiments unattended over 2-3 days.
Designed for zero Qualys violations: NO public IP, NO open inbound ports.

## Security Architecture

```
VM (NO public IP) --> pushes results to Azure Blob Storage every 30 min
Mac              <-- pulls from Blob Storage (az storage blob download)
Mac              --> interactive access via az ssh vm (AAD tunnel, no ports needed)
Emergency        --> Azure Portal Serial Console
```

**NSG Rules (Qualys Compliant):**

| Priority | Direction | Source | Dest | Port | Protocol | Action |
|----------|-----------|--------|------|------|----------|--------|
| 100 | Outbound | * | AzureCloud | 443 | TCP | Allow |
| 200 | Outbound | * | Storage | 443 | TCP | Allow |
| 4096 | Outbound | * | Internet | * | * | DENY |
| (default) | Inbound | * | * | * | * | DENY |

## Prerequisites

1. **Azure CLI** installed and logged in:
   ```bash
   brew install azure-cli
   az login
   ```

2. **az ssh extension** (auto-installed by setup script):
   ```bash
   az extension add --name ssh
   ```

3. **jq** for JSON parsing:
   ```bash
   brew install jq
   ```

4. **Subscription access**: TAP_Atlas subscription must be accessible.

5. **.env file** in project root with API keys for AI Foundry.

## Quick Start

```bash
# 1. Create VM + Blob Storage (~5 min)
./experiments/infra/setup-vm.sh

# 2. Deploy code and start experiments
./experiments/infra/deploy.sh

# 3. Monitor progress (reads from Blob Storage — no SSH)
./experiments/infra/monitor.sh

# 4. Pull results from Blob Storage to local Mac
./experiments/infra/pull-results.sh

# 5. Delete ALL Azure resources when done
./experiments/infra/teardown.sh
```

## Connectivity

**ZERO public IP. ZERO open inbound ports.** All access via AAD tunnel.

| Method | How It Works | Command |
|--------|-------------|---------|
| **az ssh vm** | Tunnels through Azure management plane (AAD auth). No ports needed. | `az ssh vm -g agentassay-experiments -n agentassay-runner` |
| **Blob Storage** | VM pushes results; Mac pulls. No SSH needed for monitoring. | `./experiments/infra/pull-results.sh` |
| **Serial Console** | Emergency access via Azure Portal. | Portal > VM > Serial Console |

## Cost Estimate

| Resource | Cost/Day | 3-Day Run |
|----------|----------|-----------|
| VM (Standard_B2ms, 2 vCPU, 8 GB) | ~$2.00 | ~$6.00 |
| OS Disk (64 GB Standard SSD) | ~$0.15 | ~$0.45 |
| Blob Storage | ~$0.01 | ~$0.03 |
| **Total** | **~$2.16** | **~$6.48** |

Auto-shutdown is configured at 19:00 UTC daily as a safety net.

## Scripts

| Script | Purpose |
|--------|---------|
| `setup-vm.sh` | Create VM (no public IP), NSG, Blob Storage, SAS tokens |
| `deploy.sh` | Upload code + .env via az ssh vm, install deps, start daemon |
| `monitor.sh` | Check status from Blob Storage (no SSH for default mode) |
| `pull-results.sh` | Download results from Blob Storage incrementally |
| `teardown.sh` | Pull final results, then delete ALL Azure resources |

## Monitoring Commands

```bash
# Status from Blob Storage (no SSH needed)
./experiments/infra/monitor.sh

# Live polling every 60s from Blob Storage
./experiments/infra/monitor.sh --follow

# Open interactive shell (AAD tunnel)
./experiments/infra/monitor.sh --shell

# View daemon logs (AAD tunnel)
./experiments/infra/monitor.sh --logs 50

# Show cost estimate
./experiments/infra/monitor.sh --cost

# Restart daemon (AAD tunnel)
./experiments/infra/monitor.sh --restart

# Stop daemon (AAD tunnel)
./experiments/infra/monitor.sh --stop
```

## Results Flow

```
VM (daemon running)
  |
  +--> writes results/*.json, logs/*.log, status.json locally
  |
  +--> every 30 min: push_results_to_blob() uploads to Azure Blob Storage
  |      (tries azure-storage-blob SDK first, falls back to az CLI)
  |      (self-healing: if push fails, daemon continues, retries next interval)
  |
Azure Blob Storage (experiment-results container)
  |
  +--> Mac pulls: ./experiments/infra/pull-results.sh
         (incremental by default, --full for everything)
         (uses read-only SAS token, no SSH needed)
```

## Troubleshooting

### "az ssh vm" hangs or fails

1. Check Azure CLI login: `az account show`
2. Re-login if expired: `az login`
3. Check ssh extension: `az extension list | grep ssh`
4. Emergency: Azure Portal Serial Console

### Daemon is not running

```bash
# Check status from Blob Storage
./experiments/infra/monitor.sh

# View detailed logs (requires AAD tunnel)
./experiments/infra/monitor.sh --logs 50

# Restart
./experiments/infra/monitor.sh --restart

# Manual debug
./experiments/infra/monitor.sh --shell
# Then on VM:
sudo journalctl -u agentassay-runner -n 100 --no-pager
```

### No results in Blob Storage

1. Daemon may not have run for 30 minutes yet (first push is after first experiment)
2. Check if daemon is running: `./experiments/infra/monitor.sh`
3. Check .env has AZURE_STORAGE_CONNECTION_STRING set on VM
4. Check NSG outbound rules allow Storage:443

### Cannot connect at all

1. VM may have stopped (auto-shutdown or Azure issue):
   ```bash
   az vm start -g agentassay-experiments -n agentassay-runner
   ```

2. Nuclear option: teardown and recreate:
   ```bash
   ./experiments/infra/teardown.sh --no-pull
   ./experiments/infra/setup-vm.sh
   ./experiments/infra/deploy.sh
   ```

## Architecture

```
Local Mac                          Azure (eastus2)
---------                          ----------------
                                   Resource Group: agentassay-experiments
                                   +----------------------------------+
                                   |                                  |
setup-vm.sh ---az CLI------------>| VM (NO public IP)                |
deploy.sh ----az ssh vm---------->|   Ubuntu 24.04, Python 3.12     |
                                   |   systemd: agentassay-runner    |
monitor.sh ---Blob Storage------->|     |                            |
pull-results.sh-Blob Storage----->|     +--> results/*.json          |
                                   |     +--> logs/*.log             |
                                   |     +--> status.json            |
                                   |     +--> push_results_to_blob() |
                                   |             |                   |
                                   | Storage Account                 |
                                   |   Container: experiment-results |
                                   |   <-------- results pushed here |
                                   +----------------------------------+
                                   NSG:
                                     Inbound:  ALL DENIED (zero rules)
                                     Outbound: AzureCloud:443, Storage:443
                                     Priority 4096: DENY ALL Internet
```

## Security

- ZERO public IP address assigned to VM
- ZERO inbound NSG rules (all inbound denied by Azure default)
- Outbound restricted to AzureCloud:443 and Storage:443 only
- Deny-all outbound at priority 4096 (Qualys mandatory requirement)
- Blob Storage: TLS 1.2 minimum, no public blob access, HTTPS only
- SAS tokens: time-limited (14 days), VM=write-only, Mac=read-only
- SSH keys generated per-VM (no shared keys)
- .env file has 600 permissions (owner-only read)
- systemd service runs as unprivileged `agentassay` user
- `ProtectSystem=strict` prevents writing outside designated paths
- Auto-shutdown as cost safety net
- Teardown deletes entire resource group (nothing left behind)
