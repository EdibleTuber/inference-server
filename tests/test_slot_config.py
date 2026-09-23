# tests/test_slot_config.py
"""Tests for manager/slot_config.py: building the ordered slot list from env."""
import pytest
from manager.slot_config import SlotConfig, build_slots


def _env(mapping):
    """Return a getenv-alike backed by a dict, so tests never touch os.environ."""
    return lambda key, default=None: mapping.get(key, default)


def test_defaults_to_main_and_batch_in_that_order():
    slots = build_slots(_env({}))
    assert [s.name for s in slots] == ["main", "batch"]


def test_main_reads_legacy_llama_server_vars():
    # systemd_unit is deliberately set to a value that differs from the
    # built-in default for "main" ("llama-server.service"). If it matched
    # the default, this assertion would pass even when the legacy-variable
    # lookup for systemd_unit was silently broken and the value seen were
    # actually just the fallback default, not the env var.
    slots = build_slots(_env({
        "LLAMA_SERVER_HOST": "10.0.0.1",
        "LLAMA_SERVER_PORT": "9001",
        "LLAMA_SERVER_ENV": "/etc/llama/main.env",
        "LLAMA_SERVER_UNIT": "custom-llama-server.service",
        "QUEUE_LIMIT": "50",
    }))
    main = slots[0]
    assert (main.host, main.port) == ("10.0.0.1", 9001)
    assert main.env_file == "/etc/llama/main.env"
    assert main.systemd_unit == "custom-llama-server.service"
    assert main.queue_limit == 50


def test_batch_reads_legacy_batch_server_vars():
    # Every value here is deliberately distinct from _DEFAULTS["batch"]
    # (port 8083, env_file ".../llama-server-batch.env", systemd_unit
    # "llama-server-batch.service", queue_limit 20). If any assertion used
    # a value that matched the default, it would keep passing even when the
    # legacy lookup for that field was broken and silently fell through to
    # the hardcoded default instead of reading the env var.
    slots = build_slots(_env({
        "BATCH_SERVER_HOST": "127.0.0.1",
        "BATCH_SERVER_PORT": "8099",
        "BATCH_SERVER_ENV": "/etc/llama/batch.env",
        "BATCH_SERVER_UNIT": "custom-llama-batch.service",
        "BATCH_QUEUE_LIMIT": "45",
    }))
    batch = next(s for s in slots if s.name == "batch")
    assert (batch.port, batch.queue_limit) == (8099, 45)
    assert batch.env_file == "/etc/llama/batch.env"
    assert batch.systemd_unit == "custom-llama-batch.service"


def test_slots_var_controls_membership_and_order():
    # SLOT_RE_PORT is required here: an unconfigured port raises (see
    # test_unconfigured_port_raises_and_names_the_variable below), and this
    # test is only about membership/order, not port validation.
    slots = build_slots(_env({"SLOTS": "main,re,batch", "SLOT_RE_PORT": "8084"}))
    assert [s.name for s in slots] == ["main", "re", "batch"]


def test_new_slot_reads_prefixed_vars():
    slots = build_slots(_env({
        "SLOTS": "main,re",
        "SLOT_RE_HOST": "127.0.0.1",
        "SLOT_RE_PORT": "8084",
        "SLOT_RE_ENV": "/etc/llama/llama-server-re.env",
        "SLOT_RE_UNIT": "llama-server-re.service",
        "SLOT_RE_QUEUE_LIMIT": "30",
        "SLOT_RE_DEVICE": "GPU-abc123",
    }))
    re_slot = slots[1]
    assert re_slot.port == 8084
    assert re_slot.env_file == "/etc/llama/llama-server-re.env"
    assert re_slot.systemd_unit == "llama-server-re.service"
    assert re_slot.queue_limit == 30
    assert re_slot.device == "GPU-abc123"


def test_prefixed_var_overrides_legacy_for_main():
    """An explicit SLOT_MAIN_PORT wins over LLAMA_SERVER_PORT."""
    slots = build_slots(_env({"LLAMA_SERVER_PORT": "8081", "SLOT_MAIN_PORT": "9999"}))
    assert slots[0].port == 9999


def test_slot_names_are_normalised_and_whitespace_tolerant():
    # SLOT_RE_PORT is required for the same reason as in
    # test_slots_var_controls_membership_and_order above.
    slots = build_slots(_env({"SLOTS": " main , RE ", "SLOT_RE_PORT": "8084"}))
    assert [s.name for s in slots] == ["main", "re"]


def test_duplicate_slot_names_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        build_slots(_env({"SLOTS": "main,main"}))


def test_empty_slots_var_is_rejected():
    with pytest.raises(ValueError, match="at least one"):
        build_slots(_env({"SLOTS": "  "}))


def test_bad_port_names_the_variable():
    with pytest.raises(ValueError, match="SLOT_RE_PORT"):
        build_slots(_env({"SLOTS": "re", "SLOT_RE_PORT": "not-a-number"}))


def test_unconfigured_port_raises_and_names_the_variable():
    """A slot with no default (not main/batch) and no configured port must
    fail loudly, naming the variable the operator needs to set, rather than
    silently picking a port that could collide with another slot."""
    with pytest.raises(ValueError, match="SLOT_RE_PORT"):
        build_slots(_env({"SLOTS": "re"}))


def test_slot_config_is_frozen():
    slots = build_slots(_env({}))
    with pytest.raises(Exception):
        slots[0].port = 1234
