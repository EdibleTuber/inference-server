# manager/slot_config.py
"""Slot definitions for the model manager.

A slot is one backing llama-server process the manager fronts. Which slots
exist, and in what order, is configuration rather than code: SLOTS names them,
and each slot reads SLOT_<NAME>_* variables.

Order is significant -- it is the routing priority that resolve_slot walks.

The legacy LLAMA_SERVER_* / BATCH_SERVER_* variables remain the source for the
'main' and 'batch' slots, so a manager.env written before slots were
configurable produces exactly the behaviour it always did. A SLOT_<NAME>_*
variable, when set, wins over the legacy name for that field.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

DEFAULT_SLOTS = ("main", "batch")

# Per-field suffix used to build the SLOT_<NAME>_<SUFFIX> variable name.
# Note env_file/systemd_unit use short suffixes (ENV/UNIT) rather than the
# field name itself -- SLOT_RE_ENV, not SLOT_RE_ENV_FILE.
_FIELD_SUFFIX = {
    "host": "HOST",
    "port": "PORT",
    "env_file": "ENV",
    "systemd_unit": "UNIT",
    "queue_limit": "QUEUE_LIMIT",
}

# Per-slot field -> (legacy env var by slot name). Keyed by the same field
# names as _FIELD_SUFFIX (and as SlotConfig itself), so _lookup can use one
# field key consistently for both the prefixed and the legacy name -- there
# is no separate "env"/"unit" key vocabulary to fall out of sync with this.
_LEGACY = {
    "host":         {"main": "LLAMA_SERVER_HOST", "batch": "BATCH_SERVER_HOST"},
    "port":         {"main": "LLAMA_SERVER_PORT", "batch": "BATCH_SERVER_PORT"},
    "env_file":     {"main": "LLAMA_SERVER_ENV",  "batch": "BATCH_SERVER_ENV"},
    "systemd_unit": {"main": "LLAMA_SERVER_UNIT", "batch": "BATCH_SERVER_UNIT"},
    "queue_limit":  {"main": "QUEUE_LIMIT",       "batch": "BATCH_QUEUE_LIMIT"},
}

_DEFAULTS = {
    "main":  {"host": "127.0.0.1", "port": 8081, "env_file": "/etc/llama/llama-server.env",
              "systemd_unit": "llama-server.service", "queue_limit": 20},
    "batch": {"host": "127.0.0.1", "port": 8083, "env_file": "/etc/llama/llama-server-batch.env",
              "systemd_unit": "llama-server-batch.service", "queue_limit": 20},
}


@dataclass(frozen=True)
class SlotConfig:
    """One slot's static configuration. Frozen: slot identity must not drift."""
    name: str
    host: str
    port: int
    env_file: str
    systemd_unit: str
    queue_limit: int
    device: str | None = None


def _lookup(getenv, name: str, field: str, default):
    """SLOT_<NAME>_<FIELD-SUFFIX> wins; then the legacy name; then the default.

    `field` is always one of the SlotConfig field names ("env_file",
    "systemd_unit", ...) and is used, unchanged, as the key into both
    _FIELD_SUFFIX (to build the prefixed variable name) and _LEGACY (to find
    the pre-existing variable name for "main"/"batch"). Using one key for
    both lookups is what keeps them from drifting apart.
    """
    suffix = _FIELD_SUFFIX[field]
    prefixed = f"SLOT_{name.upper()}_{suffix}"
    raw = getenv(prefixed)
    if raw is not None:
        return raw, prefixed
    legacy = _LEGACY.get(field, {}).get(name)
    if legacy:
        raw = getenv(legacy)
        if raw is not None:
            return raw, legacy
    return default, prefixed


def _as_int(raw, source: str):
    try:
        return int(raw)
    except (TypeError, ValueError):
        raise ValueError(
            f"Configuration error: {source}={raw!r} is not a valid integer"
        ) from None


def build_slots(getenv=os.getenv) -> tuple[SlotConfig, ...]:
    """Build the ordered slot list. getenv is injectable so tests need no monkeypatching."""
    raw_names = getenv("SLOTS") or ",".join(DEFAULT_SLOTS)
    names = [n.strip().lower() for n in raw_names.split(",") if n.strip()]
    if not names:
        raise ValueError("Configuration error: SLOTS must name at least one slot")
    if len(set(names)) != len(names):
        raise ValueError(f"Configuration error: SLOTS has duplicate names: {raw_names!r}")

    slots = []
    for name in names:
        d = _DEFAULTS.get(name, {})
        host, _ = _lookup(getenv, name, "host", d.get("host", "127.0.0.1"))
        port_raw, port_src = _lookup(getenv, name, "port", d.get("port"))
        env_file, _ = _lookup(getenv, name, "env_file", d.get("env_file"))
        unit, _ = _lookup(getenv, name, "systemd_unit", d.get("systemd_unit"))
        ql_raw, ql_src = _lookup(getenv, name, "queue_limit", d.get("queue_limit", 20))
        if port_raw is None:
            raise ValueError(f"Configuration error: {port_src} must be set for slot {name!r}")
        slots.append(SlotConfig(
            name=name,
            host=host,
            port=_as_int(port_raw, port_src),
            env_file=env_file or f"/etc/llama/llama-server-{name}.env",
            systemd_unit=unit or f"llama-server-{name}.service",
            queue_limit=_as_int(ql_raw, ql_src),
            device=getenv(f"SLOT_{name.upper()}_DEVICE"),
        ))
    return tuple(slots)
