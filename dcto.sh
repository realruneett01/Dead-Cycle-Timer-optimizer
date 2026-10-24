#!/bin/sh
# Pendrive launcher for Linux/macOS. Run as `sh dcto.sh` (FAT/exFAT sticks drop the exec bit).
#   sh dcto.sh                     guided: verify (or first-time calibrate), then run
#   sh dcto.sh --demo              rehearse the whole flow against the built-in simulator
#   sh dcto.sh --endpoint opc.tcp://192.168.0.10:4840 --scan
cd "$(dirname "$0")" || exit 1
for py in python3.12 python3.11 python3 python; do
  if command -v "$py" >/dev/null 2>&1 && "$py" -c 'import sys; sys.exit(0 if sys.version_info[:2] in ((3,11),(3,12)) else 1)'; then
    exec "$py" launch.py "$@"
  fi
done
echo "Python 3.11 or 3.12 is required but was not found on this computer." >&2
exit 1
