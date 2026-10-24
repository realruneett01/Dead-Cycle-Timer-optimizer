#!/usr/bin/env bash
# DCTO installer for Linux/macOS: picks a compatible Python, builds .venv, auto-configures for this host.
#   ./install.sh [--endpoint opc.tcp://plc:4840] [--service]
set -euo pipefail
cd "$(dirname "$0")"

ENDPOINT=""; SERVICE=0
while [ $# -gt 0 ]; do
  case "$1" in
    --endpoint) ENDPOINT="$2"; shift 2 ;;
    --service) SERVICE=1; shift ;;
    *) echo "unknown option: $1" >&2; exit 2 ;;
  esac
done

# requirements.txt pins numpy<2, which has wheels only up to Python 3.12.
PY=""
for cand in python3.12 python3.11 python3; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import sys; sys.exit(0 if (3,11) <= sys.version_info[:2] <= (3,12) else 1)'; then
    PY="$cand"; break
  fi
done
if [ -z "$PY" ]; then
  echo "No Python 3.11/3.12 found." >&2
  echo "Install one, or use Docker: docker compose up -d" >&2
  exit 1
fi

"$PY" -m venv .venv
. .venv/bin/activate
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

ARGS=()
[ -n "$ENDPOINT" ] && ARGS+=(--endpoint "$ENDPOINT")
python -m deploy configure "${ARGS[@]}"
python -m deploy doctor >/dev/null

if [ "$SERVICE" = 1 ]; then
  [ "$(id -u)" = 0 ] || { echo "--service needs root (sudo ./install.sh --service)" >&2; exit 1; }
  sed "s#@INSTALL_DIR@#$(pwd)#g" deploy/dcto.service > /etc/systemd/system/dcto.service
  systemctl daemon-reload
  systemctl enable --now dcto.service
  echo "dcto.service installed and started"
else
  echo "Installed. Start with: . .venv/bin/activate && python -m deploy run"
fi
