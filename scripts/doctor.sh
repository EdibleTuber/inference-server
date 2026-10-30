#!/usr/bin/env bash
# scripts/doctor.sh -- read-only health check for the llama manager host.
#
#   doctor.sh               run all checks (must be root)
#   doctor.sh --list-slots  print "<name> <systemd_unit> <env_file>" per slot
#                           (no root needed; checks 2 and 3 consume this list)
#
# Unit and env-file names are never derived in this script: the slot list
# comes from manager.slot_config.build_slots, fed with the deployed
# manager.env -- $DOCTOR_MGR_ENV, default /etc/llama/manager.env.
#
# doctor.sh is read-only: it inspects the host and changes nothing.
#
# Exit: 0 all checks passed (or --list-slots printed the list),
#       1 a check failed, or root was required,
#       2 bad usage.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
MGR_ENV="${DOCTOR_MGR_ENV:-/etc/llama/manager.env}"
PY="/opt/llama/manager/venv/bin/python"
FAILED=0

pass() { printf 'PASS %s %s\n' "$1" "$2"; }
fail() { printf 'FAIL %s %s\n' "$1" "$2"; FAILED=1; }

# The manager's own slot config, loaded exactly like _template_env in
# tests/test_repo_consistency.py: blank and '#' lines skipped, first '=' split.
# A missing env file means "no overrides": build_slots falls back to its
# defaults (main).
list_slots() {
  PYTHONPATH="$REPO" "$PY" - "$MGR_ENV" <<'PYEOF'
import sys
from manager.slot_config import build_slots

env = {}
try:
    with open(sys.argv[1]) as f:
        lines = f.read().splitlines()
except FileNotFoundError:
    lines = []
for line in lines:
    if line and not line.startswith("#") and "=" in line:
        k, v = line.split("=", 1)
        env[k.strip()] = v.strip()
for s in build_slots(getenv=env.get):
    print(s.name, s.systemd_unit, s.env_file)
PYEOF
}

case "${1:-}" in
  --list-slots)
    list_slots
    exit 0
    ;;
  "")
    ;;
  *)
    echo "usage: doctor.sh [--list-slots]" >&2
    exit 2
    ;;
esac

if [ "$(id -u)" -ne 0 ]; then
  echo "doctor.sh must run as root"
  exit 1
fi

SLOTS="$(list_slots)"

# 1. _llama-mgr is in group llama (reads the model files the servers open).
if id -nG _llama-mgr 2>/dev/null | grep -qw llama; then
  pass 1 "_llama-mgr is in group llama"
else
  fail 1 "_llama-mgr is not in group llama"
fi

# 2. Every slot's env file exists and is owned by _llama-mgr.
bad=""
while read -r name unit env_file; do
  if [ ! -f "$env_file" ]; then
    bad="slot $name: $env_file missing"
    break
  fi
  owner="$(stat -c %U "$env_file")"
  if [ "$owner" != "_llama-mgr" ]; then
    bad="slot $name: $env_file owned by $owner, not _llama-mgr"
    break
  fi
done <<< "$SLOTS"
if [ -z "$bad" ]; then
  pass 2 "every slot env file exists and is owned by _llama-mgr"
else
  fail 2 "$bad"
fi

# 3. _llama-mgr may restart each slot's unit.
SUDO_LIST="$(sudo -l -U _llama-mgr 2>/dev/null || true)"
bad=""
while read -r name unit env_file; do
  if ! grep -qF "systemctl restart $unit" <<< "$SUDO_LIST"; then
    bad="slot $name: 'systemctl restart $unit' not in sudo -l -U _llama-mgr"
    break
  fi
done <<< "$SLOTS"
if [ -z "$bad" ]; then
  pass 3 "_llama-mgr may restart every slot unit"
else
  fail 3 "$bad"
fi

# 4. _llama-mgr can write the log directory.
if runuser -u _llama-mgr -- test -w /var/log/llama; then
  pass 4 "_llama-mgr can write /var/log/llama"
else
  fail 4 "_llama-mgr cannot write /var/log/llama"
fi

# 5. Every repo systemd unit that is installed on the host is unmodified.
bad=""
for src in "$REPO"/systemd/*.service; do
  base="$(basename "$src")"
  [ -e "/etc/systemd/system/$base" ] || continue
  if [ "$(sha256sum "$src" | cut -c1-12)" != "$(sha256sum "/etc/systemd/system/$base" | cut -c1-12)" ]; then
    bad="/etc/systemd/system/$base differs from $REPO/systemd/$base"
    break
  fi
done
if [ -z "$bad" ]; then
  pass 5 "installed systemd units match the repo"
else
  fail 5 "$bad"
fi

# 6. No leftovers from the re->egpu rename or the batch removal. This is the
# one live file allowed to name the old unit (see the exclusion in
# tests/test_repo_consistency.py).
bad=""
for path in /etc/systemd/system/llama-server-re.service \
            /etc/systemd/system/llama-server-batch.service \
            /etc/llama/llama-server-batch.env; do
  if [ -e "$path" ]; then
    bad="$path still present"
    break
  fi
done
if [ -z "$bad" ]; then
  pass 6 "no rename/batch leftovers on the host"
else
  fail 6 "$bad"
fi

# 7. The deploy script sees no drift between this checkout and the host.
if "$REPO/scripts/deploy-manager.sh" --check >/dev/null 2>&1; then
  pass 7 "deploy-manager.sh --check reports no drift"
else
  fail 7 "deploy-manager.sh --check reports drift (run it for the details)"
fi

exit "$FAILED"
