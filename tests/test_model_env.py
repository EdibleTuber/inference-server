# tests/test_model_env.py
"""Phase 3 Task 2: a swap renders the profile into the slot's generated
model env file, writes it atomically, and the units load it (Q29, Q32).

The discriminating test expands the repo's real unit files the way systemd
does (tests/systemd_expand.py), so it checks what llama-server would be
exec'd with, not what the manager believes it wrote.
"""
import asyncio
import dataclasses
import logging
import os
import stat
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from tests.systemd_expand import expand_exec_start, parse_env_file

REPO = Path(__file__).resolve().parent.parent
UNIT = {"main": "llama-server.service", "egpu": "llama-server-egpu.service"}
SLOT_ENV = {"main": "llama-server.env", "egpu": "llama-server-egpu.env"}
SLOT_OWNED = {"MODEL_PATH", "SAMPLING_ARGS", "CTX_SIZE", "N_GPU_LAYERS", "MODEL_ARGS"}


# ---------------------------------------------------------------------------
# write_model_env
# ---------------------------------------------------------------------------

def test_write_model_env_writes_the_text_mode_0644(tmp_path):
    from manager.swap import write_model_env
    path = tmp_path / "main.env"
    path.write_text("OLD\n")
    path.chmod(0o600)
    write_model_env(str(path), "MODEL_ARGS=--model /m/a.gguf\n")
    assert path.read_text() == "MODEL_ARGS=--model /m/a.gguf\n"
    assert stat.S_IMODE(path.stat().st_mode) == 0o644


def test_write_model_env_creates_the_file(tmp_path):
    from manager.swap import write_model_env
    path = tmp_path / "egpu.env"
    write_model_env(str(path), "X=1\n")
    assert path.read_text() == "X=1\n"
    assert sorted(os.listdir(tmp_path)) == ["egpu.env"]


def test_write_model_env_replaces_via_a_synced_temp_file_in_the_same_dir(tmp_path, monkeypatch):
    from manager.swap import write_model_env
    path = tmp_path / "main.env"
    path.write_text("OLD\n")
    seen = {}
    real_replace, real_fsync = os.replace, os.fsync

    def spy_fsync(fd):
        seen["fsync"] = True
        return real_fsync(fd)

    def spy_replace(src, dst):
        seen["src"], seen["dst"] = os.fspath(src), os.fspath(dst)
        seen["dst_before"] = path.read_text()
        seen["src_text"] = Path(src).read_text()
        seen["synced_before_replace"] = seen.get("fsync", False)
        return real_replace(src, dst)

    monkeypatch.setattr(os, "fsync", spy_fsync)
    monkeypatch.setattr(os, "replace", spy_replace)
    write_model_env(str(path), "NEW\n")

    assert seen["dst"] == str(path)
    assert os.path.dirname(seen["src"]) == str(tmp_path)
    assert seen["src"] != seen["dst"]
    assert seen["dst_before"] == "OLD\n"        # the target is never written in place
    assert seen["src_text"] == "NEW\n"
    assert seen["synced_before_replace"] is True
    assert path.read_text() == "NEW\n"
    assert sorted(os.listdir(tmp_path)) == ["main.env"]


def test_a_failed_replace_leaves_the_old_file_and_no_temp_file(tmp_path, monkeypatch):
    from manager.swap import write_model_env
    path = tmp_path / "main.env"
    path.write_text("OLD\n")

    def boom(src, dst):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="disk full"):
        write_model_env(str(path), "NEW\n")
    assert path.read_text() == "OLD\n"
    assert sorted(os.listdir(tmp_path)) == ["main.env"]


def test_a_failed_write_leaves_the_old_file_and_no_temp_file(tmp_path, monkeypatch):
    from manager.swap import write_model_env
    path = tmp_path / "main.env"
    path.write_text("OLD\n")

    def boom(fd):
        raise OSError("io error")

    monkeypatch.setattr(os, "fsync", boom)
    with pytest.raises(OSError, match="io error"):
        write_model_env(str(path), "NEW\n")
    assert path.read_text() == "OLD\n"
    assert sorted(os.listdir(tmp_path)) == ["main.env"]


# ---------------------------------------------------------------------------
# ModelSwapper.swap_to
# ---------------------------------------------------------------------------

def _swapper(test_config, slot_name="main"):
    from manager.app import ServerState
    server = ServerState(test_config)
    return server.slots[slot_name], server.slots[slot_name].swapper


@pytest.mark.asyncio
async def test_swap_to_writes_the_file_before_it_restarts(test_config, monkeypatch):
    slot, swapper = _swapper(test_config, "egpu")
    at_restart = {}

    async def fake_restart():
        at_restart["text"] = Path(slot.model_env_file).read_text()

    monkeypatch.setattr(swapper, "restart_llama_server", fake_restart)
    monkeypatch.setattr(swapper, "wait_for_health", AsyncMock(return_value=True))
    assert await swapper.swap_to("test-model-q8", "MODEL_ARGS=--model /m/q8.gguf\n") is True
    assert at_restart["text"] == "MODEL_ARGS=--model /m/q8.gguf\n"
    assert slot.model_env_file == test_config.slots[1].model_env_file


@pytest.mark.asyncio
async def test_a_failed_write_leaves_the_old_file_and_does_not_restart(test_config, monkeypatch):
    slot, swapper = _swapper(test_config)
    Path(slot.model_env_file).write_text("MODEL_ARGS=--model /m/old.gguf\n")
    restart = AsyncMock()
    health = AsyncMock(return_value=True)
    monkeypatch.setattr(swapper, "restart_llama_server", restart)
    monkeypatch.setattr(swapper, "wait_for_health", health)

    def boom(src, dst):
        raise OSError("read-only file system")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="read-only"):
        await swapper.swap_to("test-model-q8", "MODEL_ARGS=--model /m/new.gguf\n")
    restart.assert_not_called()
    health.assert_not_called()
    assert Path(slot.model_env_file).read_text() == "MODEL_ARGS=--model /m/old.gguf\n"


@pytest.mark.asyncio
async def test_swap_to_reports_a_health_timeout(test_config, monkeypatch):
    slot, swapper = _swapper(test_config)
    monkeypatch.setattr(swapper, "restart_llama_server", AsyncMock())
    monkeypatch.setattr(swapper, "wait_for_health", AsyncMock(return_value=False))
    assert await swapper.swap_to("test-model-q4", "X=1\n") is False


# ---------------------------------------------------------------------------
# ensure_model_on_slot renders from the profile
# ---------------------------------------------------------------------------

PROFILE_TOML = """\
home = "main"

[slots.main]
ctx_size = 131072
n_gpu_layers = -1
flags = ["--jinja", "--spec-type", "draft-mtp", "--spec-draft-n-max", "1"]

[slots.main.sampling]
temp = 1.0
top_k = 20
top_p = 0.95
min_p = 0.0
presence_penalty = 0.0
frequency_penalty = 0.0
repeat_penalty = 1.0

[slots.egpu]
ctx_size = 65536
n_gpu_layers = 30
flags = ["--jinja"]

[slots.egpu.sampling]
temp = 0.6
top_k = 20
top_p = 0.95
min_p = 0.05
presence_penalty = 0.0
frequency_penalty = 0.0
repeat_penalty = 1.0
"""

MAIN_ONLY_TOML = PROFILE_TOML.split("[slots.egpu]")[0]


def _config_with(test_config, tmp_path, profiles: dict[str, str]):
    """test_config with its own profiles dir and a non-empty gguf per profile."""
    pdir = tmp_path / "own-profiles"
    pdir.mkdir()
    for name, text in profiles.items():
        (pdir / f"{name}.toml").write_text(text)
        (Path(test_config.models_dir) / f"{name}.gguf").write_bytes(b"GGUF")
    return dataclasses.replace(test_config, profiles_dir=str(pdir))


@pytest.mark.asyncio
async def test_ensure_renders_the_section_for_the_target_slot(test_config, tmp_path):
    from manager.app import ServerState
    config = _config_with(test_config, tmp_path, {"Mtp-Model-Q6": PROFILE_TOML})
    server = ServerState(config)
    swap = server.slots["egpu"].swapper.swap_to = AsyncMock(return_value=True)

    assert await server.ensure_model_on_slot("egpu", "MTP-MODEL-Q6.gguf") is True

    name, env_text = swap.await_args.args
    assert name == "Mtp-Model-Q6"
    assert env_text.splitlines()[1] == (
        f"MODEL_ARGS=--model {config.models_dir}/Mtp-Model-Q6.gguf --ctx-size 65536 "
        "--n-gpu-layers 30 --temp 0.6 --top-k 20 --top-p 0.95 --min-p 0.05 "
        "--presence-penalty 0.0 --frequency-penalty 0.0 --repeat-penalty 1.0 --jinja")
    assert server.slots["egpu"].loaded_model == "Mtp-Model-Q6"


async def _refused(server, slot_name, model_name, caplog):
    """Ask for a swap the profile can't serve, with the slot loaded, healthy
    and one request queued; return (result, swap mock, queued item)."""
    slot = server.slots[slot_name]
    slot.loaded_model, slot.healthy = "test-model-q4", True
    swap = slot.swapper.swap_to = AsyncMock(return_value=True)
    item = {"event": asyncio.Event(), "error": None}
    await slot.queue.enqueue(item)
    with caplog.at_level(logging.ERROR, logger="manager.app"):
        result = await server.ensure_model_on_slot(slot_name, model_name)
    return result, swap, item


def _assert_untouched(server, slot_name, swap, item):
    """A refusal happens before anything is stopped, so the old model is
    still serving: the slot stays healthy and its queue stays queued."""
    slot = server.slots[slot_name]
    swap.assert_not_called()
    assert (slot.loaded_model, slot.healthy) == ("test-model-q4", True)
    assert slot.queue.depth == 1
    assert (item["error"], item["event"].is_set()) == (None, False)


@pytest.mark.asyncio
async def test_a_model_without_a_profile_is_refused_and_the_slot_untouched(test_config, caplog):
    from manager.app import ServerState
    (Path(test_config.models_dir) / "no-profile-q4.gguf").write_bytes(b"GGUF")
    server = ServerState(test_config)
    result, swap, item = await _refused(server, "main", "no-profile-q4", caplog)
    assert result is False
    _assert_untouched(server, "main", swap, item)
    assert "No profile for model no-profile-q4" in caplog.text


@pytest.mark.asyncio
async def test_a_profile_without_a_section_for_the_slot_is_refused_and_the_slot_untouched(
        test_config, tmp_path, caplog):
    from manager.app import ServerState
    config = _config_with(test_config, tmp_path, {"Main-Only-Q4": MAIN_ONLY_TOML})
    server = ServerState(config)
    result, swap, item = await _refused(server, "egpu", "Main-Only-Q4", caplog)
    assert result is False
    _assert_untouched(server, "egpu", swap, item)
    assert "Model Main-Only-Q4 has no [slots.egpu] section; it may run on: main" in caplog.text


@pytest.mark.asyncio
async def test_a_failed_write_is_a_failed_swap_not_a_crash(test_config, monkeypatch):
    from manager.app import ServerState
    server = ServerState(test_config)
    swapper = server.slots["main"].swapper
    restart = AsyncMock()
    monkeypatch.setattr(swapper, "restart_llama_server", restart)

    def boom(src, dst):
        raise OSError("read-only file system")

    monkeypatch.setattr(os, "replace", boom)
    assert await server.ensure_model_on_slot("main", "test-model-q4") is False
    restart.assert_not_called()
    assert server.slots["main"].healthy is False


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

def test_build_slots_puts_each_model_env_file_under_var_lib():
    from manager.slot_config import build_slots
    env = {"SLOTS": "main,egpu", "SLOT_EGPU_PORT": "8084"}
    slots = build_slots(env.get)
    assert [s.model_env_file for s in slots] == [
        "/var/lib/llama-manager/main.env", "/var/lib/llama-manager/egpu.env"]


def test_model_env_dir_overrides_the_directory():
    from manager.slot_config import build_slots
    env = {"SLOTS": "main,egpu", "SLOT_EGPU_PORT": "8084", "MODEL_ENV_DIR": "/srv/gen"}
    slots = build_slots(env.get)
    assert [s.model_env_file for s in slots] == ["/srv/gen/main.env", "/srv/gen/egpu.env"]


def test_server_state_hands_each_slot_its_model_env_file(test_config):
    from manager.app import ServerState
    server = ServerState(test_config)
    assert {n: s.model_env_file for n, s in server.slots.items()} == {
        sc.name: sc.model_env_file for sc in test_config.slots}


def test_profiles_dir_defaults_to_etc_and_reads_profiles_dir(monkeypatch):
    from manager.config import ManagerConfig
    monkeypatch.delenv("PROFILES_DIR", raising=False)
    assert ManagerConfig.from_env().profiles_dir == "/etc/llama/models.d"
    monkeypatch.setenv("PROFILES_DIR", "/srv/profiles")
    assert ManagerConfig.from_env().profiles_dir == "/srv/profiles"


def test_the_sidecar_and_regex_paths_are_gone():
    import manager.swap as swap
    assert not hasattr(swap, "sampling_args_for_model")
    assert not hasattr(swap.ModelSwapper, "update_env_file")


# ---------------------------------------------------------------------------
# The repo's units and slot env templates (the Q42 regression)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("slot", ["main", "egpu"])
def test_slot_env_templates_carry_no_model_settings(slot):
    env = parse_env_file((REPO / "config" / SLOT_ENV[slot]).read_text())
    assert not (set(env) & SLOT_OWNED)
    assert not [k for k in env if k.startswith("LLAMA_ARG_")]


def _expand(slot: str, model_env_path: Path) -> list[str]:
    def read(path: str) -> str:
        if path == f"/var/lib/llama-manager/{slot}.env":
            return model_env_path.read_text()
        if path == f"/etc/llama/{SLOT_ENV[slot]}":
            return (REPO / "config" / SLOT_ENV[slot]).read_text()
        raise AssertionError(f"{UNIT[slot]} loads an unexpected env file: {path}")
    return expand_exec_start((REPO / "systemd" / UNIT[slot]).read_text(), read)


def _slot_prefix(slot: str) -> list[str]:
    env = parse_env_file((REPO / "config" / SLOT_ENV[slot]).read_text())
    return ["/opt/llama/bin/llama-server", "--host", env["HOST"], "--port", env["PORT"],
            "--device", "CUDA0", "--parallel", "1", "--cache-ram", "0"]


@pytest.mark.parametrize("slot", ["main", "egpu"])
@pytest.mark.asyncio
async def test_a_swap_passes_exactly_the_profiles_flags_to_llama_server(slot, test_config, tmp_path):
    """Swap an MTP model in, then a plain one: the second argv is exactly the
    slot's own flags plus the plain profile's, and nothing of the first."""
    from manager.app import ServerState
    plain = PROFILE_TOML.replace(
        'flags = ["--jinja", "--spec-type", "draft-mtp", "--spec-draft-n-max", "1"]',
        'flags = ["--n-predict", "32768"]').replace(
        'flags = ["--jinja"]', 'flags = ["--n-predict", "32768"]')
    config = _config_with(test_config, tmp_path, {"Mtp-Model-Q6": PROFILE_TOML, "Plain-Model-Q4": plain})
    server = ServerState(config)
    swapper = server.slots[slot].swapper
    swapper.restart_llama_server = AsyncMock()
    swapper.wait_for_health = AsyncMock(return_value=True)
    model_env = Path(server.slots[slot].model_env_file)

    assert await server.ensure_model_on_slot(slot, "Mtp-Model-Q6") is True
    first = _expand(slot, model_env)
    assert ("--spec-type" if slot == "main" else "--jinja") in first

    assert await server.ensure_model_on_slot(slot, "Plain-Model-Q4") is True
    ctx, ngl, temp, min_p = ("131072", "-1", "1.0", "0.0") if slot == "main" else ("65536", "30", "0.6", "0.05")
    assert _expand(slot, model_env) == _slot_prefix(slot) + [
        "--model", f"{config.models_dir}/Plain-Model-Q4.gguf",
        "--ctx-size", ctx, "--n-gpu-layers", ngl,
        "--temp", temp, "--top-k", "20", "--top-p", "0.95", "--min-p", min_p,
        "--presence-penalty", "0.0", "--frequency-penalty", "0.0", "--repeat-penalty", "1.0",
        "--n-predict", "32768",
    ]


@pytest.mark.parametrize("slot", ["main", "egpu"])
def test_the_slot_env_file_wins_over_the_model_env_file(slot, tmp_path):
    """The unit loads the model env file first, so a HOST/PORT/DEVICE that
    slipped into it can't move the server. MODEL_ARGS proves the file loaded."""
    model_env = tmp_path / f"{slot}.env"
    model_env.write_text("MODEL_ARGS=--model /m/x.gguf\nHOST=0.0.0.0\nPORT=1\nDEVICE=CUDA1\n")
    assert _expand(slot, model_env) == _slot_prefix(slot) + ["--model", "/m/x.gguf"]


@pytest.mark.parametrize("slot", ["main", "egpu"])
def test_the_slot_env_files_device_wins_over_the_unit_default(slot, tmp_path):
    """Environment=DEVICE=CUDA0 is only a default: a host whose slot env file
    names another device must get that device."""
    model_env = tmp_path / f"{slot}.env"
    model_env.write_text("MODEL_ARGS=--model /m/x.gguf\n")
    slot_env = (REPO / "config" / SLOT_ENV[slot]).read_text().replace("DEVICE=CUDA0", "DEVICE=CUDA1")
    assert "DEVICE=CUDA1" in slot_env

    def read(path: str) -> str:
        if path == f"/var/lib/llama-manager/{slot}.env":
            return model_env.read_text()
        if path == f"/etc/llama/{SLOT_ENV[slot]}":
            return slot_env
        raise AssertionError(f"{UNIT[slot]} loads an unexpected env file: {path}")
    argv = expand_exec_start((REPO / "systemd" / UNIT[slot]).read_text(), read)
    assert argv[argv.index("--device") + 1] == "CUDA1"
