#!/usr/bin/env bash
# scripts/setup.sh
#
# Initial setup for the inference server on Ubuntu.
# Run once as root: sudo bash scripts/setup.sh
#
# Creates system users, directories, permissions, sudoers entry,
# Python virtualenv, and installs systemd services.

set -euo pipefail

if [[ $EUID -ne 0 ]]; then
    echo "ERROR: Run as root (sudo bash scripts/setup.sh)"
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

echo "=== Inference Server Setup ==="
echo "Repository: $REPO_DIR"
echo ""

# -- 1. System users --
echo "[1/8] Creating system users..."

if ! id -u _llama &>/dev/null; then
    useradd --system --shell /usr/sbin/nologin --no-create-home _llama
    echo "  Created: _llama"
else
    echo "  Exists: _llama"
fi

if ! id -u _llama-mgr &>/dev/null; then
    useradd --system --shell /usr/sbin/nologin --no-create-home _llama-mgr
    echo "  Created: _llama-mgr"
else
    echo "  Exists: _llama-mgr"
fi

# -- 2. Shared group --
echo "[2/8] Creating llama group..."

if ! getent group llama &>/dev/null; then
    groupadd llama
    echo "  Created: llama"
else
    echo "  Exists: llama"
fi

usermod -aG llama _llama
usermod -aG video _llama  # GPU access

SUDO_USER_NAME="${SUDO_USER:-}"
if [[ -n "$SUDO_USER_NAME" ]]; then
    usermod -aG llama "$SUDO_USER_NAME"
    echo "  Added $SUDO_USER_NAME to llama group (re-login to take effect)"
fi

# -- 3. Directories --
echo "[3/8] Creating directories..."

mkdir -p /opt/llama/bin /opt/llama/models /opt/llama/manager /opt/llama/data
mkdir -p /etc/llama
mkdir -p /var/log/llama

# -- 4. Permissions --
echo "[4/8] Setting permissions..."

# Models: _llama owns, llama group can write (admins add models)
chown _llama:llama /opt/llama/models
chmod 775 /opt/llama/models
echo "  /opt/llama/models → _llama:llama 775"

chown -R _llama-mgr:_llama-mgr /opt/llama/manager
chmod 755 /opt/llama/manager
echo "  /opt/llama/manager → _llama-mgr 755"

chown _llama-mgr:_llama-mgr /opt/llama/data
chmod 755 /opt/llama/data
echo "  /opt/llama/data → _llama-mgr 755"

chown root:root /opt/llama/bin
chmod 755 /opt/llama/bin

chown root:llama /var/log/llama
chmod 775 /var/log/llama

# -- 5. Config files --
echo "[5/8] Installing config files..."

# Helper: install config file without clobbering existing customizations.
# If the target exists, back it up and warn. If not, copy fresh.
install_config() {
    local src="$1" dst="$2"
    if [[ -f "$dst" ]]; then
        local backup="${dst}.bak.$(date +%Y%m%d%H%M%S)"
        cp "$dst" "$backup"
        echo "  Backed up: $dst → $backup"
        # Merge strategy: keep existing file, copy new template next to it
        cp "$src" "${dst}.new"
        echo "  New template: ${dst}.new (review and merge manually)"
    else
        cp "$src" "$dst"
        echo "  Installed: $dst"
    fi
    chown _llama-mgr:_llama-mgr "$dst"
    chmod 644 "$dst"
}

install_config "$REPO_DIR/config/llama-server.env" /etc/llama/llama-server.env
install_config "$REPO_DIR/config/manager.env" /etc/llama/manager.env
install_config "$REPO_DIR/config/collections.json" /etc/llama/collections.json

# -- 6. Sudoers --
echo "[6/8] Installing sudoers entry..."

cat > /etc/sudoers.d/llama-manager << 'EOF'
# Allow model manager to restart llama-server for model swapping.
_llama-mgr ALL=(root) NOPASSWD: /usr/bin/systemctl restart llama-server.service
EOF
chmod 440 /etc/sudoers.d/llama-manager

if ! visudo -c -f /etc/sudoers.d/llama-manager &>/dev/null; then
    echo "  ERROR: sudoers syntax error!"
    rm /etc/sudoers.d/llama-manager
    exit 1
fi
echo "  Sudoers entry installed"

# -- 7. Python virtualenv --
echo "[7/8] Setting up Python virtualenv..."

python3 -m venv /opt/llama/manager/venv

# Copy the manager package (including __init__.py)
cp "$REPO_DIR/manager/"*.py /opt/llama/manager/
cp "$REPO_DIR/manager/requirements.txt" /opt/llama/manager/

/opt/llama/manager/venv/bin/pip install -r /opt/llama/manager/requirements.txt --quiet
chown -R _llama-mgr:_llama-mgr /opt/llama/manager
echo "  Virtualenv created, dependencies installed"

# -- 8. Systemd services --
echo "[8/8] Installing systemd services..."

cp "$REPO_DIR/systemd/llama-server.service" /etc/systemd/system/
cp "$REPO_DIR/systemd/llama-manager.service" /etc/systemd/system/
cp "$REPO_DIR/systemd/llama-embeddings.service" /etc/systemd/system/
systemctl daemon-reload

# Logrotate
cp "$REPO_DIR/config/llama-logrotate" /etc/logrotate.d/llama

echo ""
echo "=== Setup Complete ==="
echo ""
echo "Next steps:"
echo "  1. Place llama-server binary at /opt/llama/bin/llama-server"
echo "  2. Download models:"
echo "     ./scripts/download-model.sh <repo> <file>          # chat model"
echo "     Download nomic-embed-text GGUF to /opt/llama/models/  # embedding model"
echo "  3. Review config files in /etc/llama/:"
echo "     - Set HOST in manager.env to your LAN IP"
echo "     - Update source_dir paths in collections.json"
echo "     - If *.new files exist, merge changes into existing configs"
echo "  4. sudo systemctl enable --now llama-server llama-manager llama-embeddings"
echo "  5. Test: curl http://<LAN_IP>:11434/health"
