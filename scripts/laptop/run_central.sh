#!/usr/bin/env bash
# Central server + dashboard on the laptop, reachable by the GPU edge box over the network.
#   scripts/laptop/run_central.sh                       # dashboard: http://<laptop-ip>:8000 (login: operator)
# Set your own secrets (the edge must use the same API key):
#   SNATCHGUARD_API_KEYS=my-edge-key SNATCHGUARD_DASHBOARD_PASSWORD=change-me scripts/laptop/run_central.sh
set -euo pipefail
cd "$(dirname "$0")/../.."
export SNATCHGUARD_API_KEYS="${SNATCHGUARD_API_KEYS:-dev-edge-key}"
export SNATCHGUARD_DASHBOARD_PASSWORD="${SNATCHGUARD_DASHBOARD_PASSWORD:-snatchguard}"
IP=$(hostname -I 2>/dev/null | awk '{print $1}')
echo "Dashboard:   http://${IP:-<laptop-ip>}:8000   (user: operator, password: \$SNATCHGUARD_DASHBOARD_PASSWORD)"
echo "Edge config: central.url = http://${IP:-<laptop-ip>}:8000   api_key = \$SNATCHGUARD_API_KEYS"
echo "If the edge can't connect, open the port:  sudo ufw allow 8000/tcp"
exec .venv/bin/python -m snatchguard.central --host 0.0.0.0 --port 8000
