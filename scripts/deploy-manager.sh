#!/usr/bin/env bash
# Deploy the model manager from this checkout to /opt/llama/manager, with
# provenance. For UPDATES -- scripts/setup.sh is the first-time installer.
#
#   ./scripts/deploy-manager.sh --check    # read-only, no sudo, exits 1 on drift
#   sudo ./scripts/deploy-manager.sh       # install, stamp, restart, verify
#
# Why this exists: there was no update path. setup.sh is an 8-step installer
# that creates users, a venv, sudoers entries and systemd units -- running it to
# ship a code change does far more than deploy. And README.md said
# `cp -r manager/ /opt/llama/manager/`, which was right exactly once: with the
# target already present it creates /opt/llama/manager/manager/, leaving the new
# code one level down and the manager still running the old. A silent no-op.
#
# The result was drift nobody could see: app.py in /opt was behind this repo by
# a merged commit, so the deployed 409 message lacked the "here is what IS
# loaded" text and the GPU query was still the blocking one.
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST=/opt/llama/manager
OWNER=_llama-mgr
UNIT=llama-manager
STAMP="$DEST/DEPLOYED_FROM"
# Where to verify the manager after restarting it. Derived from the DEPLOYED
# manager.env, not hardcoded: this host binds the manager to its LAN IP, so a
# hardcoded 127.0.0.1 got connection-refused every time and the script reported
# "did not report healthy within 15s" and exit 1 for deploys that had in fact
# succeeded. A verification that can never pass also can never fail
# meaningfully -- it reports the same thing whether the deploy worked or not,
# which is the opposite of what this script exists for.
# 0.0.0.0 is a bind address, not a destination, so fall back to loopback there.
MGR_ENV=/etc/llama/manager.env
MGR_HOST="$(sed -n 's/^HOST=//p' "$MGR_ENV" 2>/dev/null | tail -1)"
MGR_PORT="$(sed -n 's/^PORT=//p' "$MGR_ENV" 2>/dev/null | tail -1)"
if [ -z "$MGR_HOST" ] || [ "$MGR_HOST" = "0.0.0.0" ]; then
  MGR_HOST=127.0.0.1
fi
if [ -z "$MGR_PORT" ]; then
  MGR_PORT=11434
fi
BASE="http://${MGR_HOST}:${MGR_PORT}"
HEALTH="$BASE/health"

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

# -d "$REPO/.git" is NOT the right test: in a git worktree, .git is a file
# (a pointer into the main repo's .git/worktrees/), so a perfectly valid
# worktree checkout was rejected. Ask git itself instead -- this passes for
# both regular checkouts and worktrees, and still fails for plain copies.
if ! git -C "$REPO" rev-parse --git-dir >/dev/null 2>&1; then
  echo "ERROR: $REPO is not a git checkout. Deploying from a copy is how the"
  echo "drift this script exists to end got started."
  exit 2
fi

HEAD_SHA="$(git -C "$REPO" rev-parse --short HEAD)"
DIRTY="$(git -C "$REPO" status --porcelain -- manager | wc -l)"
echo "checkout $REPO @ $HEAD_SHA${DIRTY:+ ($DIRTY modified under manager/)}"
[ -r "$STAMP" ] && echo "deployed: $(head -1 "$STAMP")" || echo "deployed: no provenance stamp yet"
echo

drift=0
changed_reqs=0
for src in "$REPO"/manager/*.py "$REPO"/manager/requirements.txt; do
  [ -e "$src" ] || continue
  base="$(basename "$src")"
  a=$(sha256sum "$src" | cut -c1-12)
  b=$(sha256sum "$DEST/$base" 2>/dev/null | cut -c1-12)
  if [ "$a" = "$b" ]; then
    printf '  %-22s current\n' "$base"
  else
    printf '  %-22s STALE (repo %s, deployed %s)\n' "$base" "$a" "${b:-absent}"
    drift=1
    [ "$base" = "requirements.txt" ] && changed_reqs=1
  fi
done

echo
if [ "$drift" -eq 0 ]; then
  echo "Everything deployed matches $HEAD_SHA."
  exit 0
fi
if [ "$CHECK_ONLY" -eq 1 ]; then
  echo "Drift above. Re-run with sudo (no --check) to deploy."
  exit 1
fi
if [ "$(id -u)" -ne 0 ]; then
  echo "Deploying needs root. Re-run: sudo $0"
  exit 1
fi

# Only *.py and requirements.txt, explicitly. NOT `cp -r manager/`, which
# nests, and not a wildcard that could sweep in a stray file from the checkout.
install -m 644 -o "$OWNER" -g "$OWNER" "$REPO"/manager/*.py "$DEST"/
install -m 644 -o "$OWNER" -g "$OWNER" "$REPO"/manager/requirements.txt "$DEST"/
echo "  installed manager/*.py + requirements.txt"

if [ "$changed_reqs" -eq 1 ]; then
  echo "  requirements.txt changed — reinstalling dependencies"
  "$DEST/venv/bin/pip" install -r "$DEST/requirements.txt" --quiet
  chown -R "$OWNER:$OWNER" "$DEST/venv"
fi

# Stamp BEFORE the restart, so a manager that fails to come back still leaves a
# truthful record of what was put there.
printf '%s  deployed %s from %s\n' "$HEAD_SHA" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$REPO" > "$STAMP"
chmod 644 "$STAMP"; chown "$OWNER:$OWNER" "$STAMP"
echo "  stamped $STAMP"

echo "  restarting $UNIT (this briefly interrupts inference)"
systemctl restart "$UNIT"

# Verify it came back. A deploy that leaves the manager dead is worse than the
# drift it fixed, and the slots reload their own state on startup.
for i in $(seq 1 15); do
  sleep 1
  code=$(curl -s -m3 -o /dev/null -w '%{http_code}' "$HEALTH" 2>/dev/null)
  if [ "$code" = "200" ]; then
    echo "  $UNIT healthy after ${i}s"
    curl -s -m5 "$BASE/status" 2>/dev/null |
      python3 -c 'import json,sys
d=json.load(sys.stdin)
for n,s in d["slots"].items():
    print(f"    slot {n:<6} loaded={s[\"loaded_model\"]!r} healthy={s[\"healthy\"]}")' 2>/dev/null
    exit 0
  fi
done
echo "  WARNING: $UNIT did not report healthy within 15s"
echo "  journalctl -u $UNIT -n 40 --no-pager"
exit 1
