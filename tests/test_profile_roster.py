# tests/test_profile_roster.py
"""Phase 3 Task 3: the profiles are the roster (Q13, Q30).

A GGUF with no profile is not a model the manager knows. /swap takes a
verified (model, slot) pair, or any pair the profile allows with
trial=true, and an untargeted /swap goes to the profile's home slot.
"""
import dataclasses
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi.testclient import TestClient

from tests.conftest import wait_for_swap

SAMPLING = """
[slots.{slot}.sampling]
temp = 1.0
top_k = 20
top_p = 0.95
min_p = 0.0
presence_penalty = 0.0
frequency_penalty = 0.0
repeat_penalty = 1.0
"""

VERIFIED = """
[slots.{slot}.verified]
date = "2026-10-31"
llama_cpp_build = 11136
llama_cpp_commit = "057494f93"
"""


def _section(slot: str, verified: bool) -> str:
    text = f"\n[slots.{slot}]\nctx_size = 8192\nn_gpu_layers = -1\n" + SAMPLING.format(slot=slot)
    return text + (VERIFIED.format(slot=slot) if verified else "")


# name -> (home, {slot: verified})
EXTRA = {
    "Home-Egpu-Q4": ("egpu", {"main": False, "egpu": True}),
    "Unverified-Q4": ("main", {"main": False}),
}


@pytest.fixture
def roster_config(test_config):
    """test_config plus: Home-Egpu-Q4 (home egpu, verified on egpu only),
    Unverified-Q4 (main only, never verified), Orphan-Q4 (a GGUF with no
    profile) and Broken-Q4 (a profile that fails validation)."""
    pdir = Path(test_config.profiles_dir)
    mdir = Path(test_config.models_dir)
    for name, (home, sections) in EXTRA.items():
        text = f'home = "{home}"\n' + "".join(_section(s, v) for s, v in sections.items())
        (pdir / f"{name}.toml").write_text(text)
        (mdir / f"{name}.gguf").write_bytes(b"GGUF")
    (mdir / "Orphan-Q4.gguf").write_bytes(b"GGUF")
    (pdir / "Broken-Q4.toml").write_text('home = "main"\n[slots.main]\nctx_size = true\n')
    (mdir / "Broken-Q4.gguf").write_bytes(b"GGUF")
    return test_config


@pytest.fixture
def roster(roster_config, monkeypatch):
    """(client, swaps): swaps records (slot, model) for every swap_to call."""
    from manager.app import create_app
    swaps = []

    async def fake_swap_to(self, model_name, env_text):
        swaps.append((self._slot.name, model_name))
        return True

    monkeypatch.setattr("manager.swap.ModelSwapper.swap_to", fake_swap_to)
    # A running lifespan: a swap is a queue item (phase 5), run by the consumer.
    with TestClient(create_app(roster_config)) as client:
        yield client, swaps


def _error(r):
    return r.json()["error"]["type"]


# ---------------------------------------------------------------------------
# ServerState
# ---------------------------------------------------------------------------

def test_server_state_uses_a_given_profile_set(test_config):
    from manager.app import ServerState
    from manager.profiles import ProfileSet
    ps = ProfileSet(profiles={}, errors={})
    missing = dataclasses.replace(test_config, profiles_dir="/nonexistent/models.d")
    server = ServerState(missing, profiles=ps)
    assert server.profiles is ps


def test_server_state_loads_profiles_and_logs_each_error(roster_config, caplog):
    from manager.app import ServerState
    with caplog.at_level("ERROR"):
        server = ServerState(roster_config)
    assert server.profiles.get("home-egpu-q4") is not None
    assert set(server.profiles.errors) == {"Broken-Q4"}
    assert any("Broken-Q4" in r.getMessage() and r.levelname == "ERROR" for r in caplog.records)


@pytest.mark.asyncio
async def test_force_swaps_a_model_that_is_already_loaded(test_config):
    from manager.app import ServerState
    server = ServerState(test_config)
    slot = server.slots["main"]
    slot.loaded_model, slot.healthy = "test-model-q4", True
    swap = slot.swapper.swap_to = AsyncMock(return_value=True)

    assert await server.ensure_model_on_slot("main", "test-model-q4") is True
    swap.assert_not_called()
    assert await server.ensure_model_on_slot("main", "test-model-q4", force=True) is True
    swap.assert_awaited_once()


# ---------------------------------------------------------------------------
# Chat requests
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("path", ["/v1/chat/completions", "/v1/messages"])
def test_a_gguf_without_a_profile_is_not_a_model(roster, path):
    # Not marked loaded on any slot: against the old GGUF roster that would
    # queue the request, and with no consumer running nothing answers it.
    client, _ = roster
    r = client.post(path, json={"model": "Orphan-Q4", "max_tokens": 8,
                                "messages": [{"role": "user", "content": "hi"}]})
    assert r.status_code == 404, r.text
    if path == "/v1/chat/completions":   # its error shape types this invalid_request_error
        assert r.json()["error"]["message"] == "Model not found: Orphan-Q4"


# ---------------------------------------------------------------------------
# POST /swap
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("trial", ["yes", 1, 0, None, []])
def test_trial_must_be_a_boolean(roster, trial):
    client, swaps = roster
    r = client.post("/swap", json={"model": "test-model-q4", "target": "main", "trial": trial})
    assert r.status_code == 400, r.text
    assert _error(r) == "invalid_trial"
    assert swaps == []


def test_a_gguf_without_a_profile_is_404_and_touches_no_slot(roster):
    client, swaps = roster
    server = client.app.state.server
    main = server.slots["main"]
    main.loaded_model, main.healthy = "test-model-q4", True
    r = client.post("/swap", json={"model": "Orphan-Q4", "target": "main"})
    assert r.status_code == 404, r.text
    assert _error(r) == "model_not_found"
    assert "add config/models/Orphan-Q4.toml" in r.json()["error"]["message"]
    assert swaps == []
    assert (main.loaded_model, main.healthy) == ("test-model-q4", True)


def test_a_profile_that_failed_validation_is_404(roster):
    client, swaps = roster
    r = client.post("/swap", json={"model": "Broken-Q4", "target": "main", "trial": True})
    assert r.status_code == 404, r.text
    assert _error(r) == "model_not_found"
    assert swaps == []


def test_the_checks_run_in_order(roster):
    client, swaps = roster
    # invalid_trial before model_not_found
    r = client.post("/swap", json={"model": "Orphan-Q4", "target": "nope", "trial": "x"})
    assert _error(r) == "invalid_trial"
    # model_not_found before invalid_target
    r = client.post("/swap", json={"model": "Orphan-Q4", "target": "nope"})
    assert (r.status_code, _error(r)) == (404, "model_not_found")
    # invalid_target before slot_not_allowed / not_verified
    r = client.post("/swap", json={"model": "Unverified-Q4", "target": "nope"})
    assert (r.status_code, _error(r)) == (400, "invalid_target")
    # slot_not_allowed before not_verified, and even with trial
    r = client.post("/swap", json={"model": "Unverified-Q4", "target": "egpu", "trial": True})
    assert (r.status_code, _error(r)) == (400, "slot_not_allowed")
    assert swaps == []


@pytest.mark.parametrize("body", [{}, {"target": None}], ids=["absent", "null"])
def test_an_untargeted_swap_goes_to_the_profiles_home(roster, body):
    client, swaps = roster
    r = client.post("/swap", json={"model": "HOME-EGPU-Q4.gguf", **body},
                    headers={"X-Request-Id": "home"})
    assert r.status_code == 202, r.text
    assert r.json()["slot"] == "egpu"
    wait_for_swap(client.app.state.server, "egpu", "home")
    assert swaps == [("egpu", "Home-Egpu-Q4")]


def test_a_slot_the_profile_has_no_section_for_is_refused(roster):
    client, swaps = roster
    r = client.post("/swap", json={"model": "Unverified-Q4", "target": "egpu"})
    assert r.status_code == 400, r.text
    assert _error(r) == "slot_not_allowed"
    msg = r.json()["error"]["message"]
    assert "[slots.egpu]" in msg and "it may run on: main" in msg
    assert swaps == []


@pytest.mark.parametrize("trial_field", [{}, {"trial": False}], ids=["absent", "false"])
def test_an_unverified_pair_is_refused_without_trial(roster, trial_field):
    client, swaps = roster
    r = client.post("/swap", json={"model": "Home-Egpu-Q4", "target": "main", **trial_field})
    assert r.status_code == 400, r.text
    assert _error(r) == "not_verified"
    assert "scripts/onboard.py Home-Egpu-Q4 --slot main" in r.json()["error"]["message"]
    assert swaps == []


def test_refusals_name_the_profile_not_the_request(roster):
    """The not_verified hint is a command to paste, and onboard opens
    config/models/<name>.toml by its exact name, so a case-variant request
    must still get the profile's own name back."""
    client, swaps = roster
    r = client.post("/swap", json={"model": "home-egpu-q4.gguf", "target": "main"})
    assert (r.status_code, _error(r)) == (400, "not_verified")
    msg = r.json()["error"]["message"]
    assert msg.startswith("model 'Home-Egpu-Q4' is not verified on slot 'main'"), msg
    assert "scripts/onboard.py Home-Egpu-Q4 --slot main" in msg
    r = client.post("/swap", json={"model": "unverified-q4", "target": "egpu"})
    assert (r.status_code, _error(r)) == (400, "slot_not_allowed")
    assert "model 'Unverified-Q4' has no [slots.egpu] section" in r.json()["error"]["message"]
    assert swaps == []


def test_an_untargeted_swap_to_an_unverified_home_is_refused(roster):
    client, swaps = roster
    r = client.post("/swap", json={"model": "Unverified-Q4"})
    assert (r.status_code, _error(r)) == (400, "not_verified")
    assert swaps == []


def test_a_trial_swap_loads_an_unverified_pair(roster):
    client, swaps = roster
    r = client.post("/swap", json={"model": "Home-Egpu-Q4", "target": "main", "trial": True},
                    headers={"X-Request-Id": "trial"})
    assert r.status_code == 202, r.text
    wait_for_swap(client.app.state.server, "main", "trial")
    assert swaps == [("main", "Home-Egpu-Q4")]
    assert client.app.state.server.slots["main"].loaded_model == "Home-Egpu-Q4"


def test_a_trial_swap_restarts_even_when_the_model_is_loaded(roster):
    """Review Focus 5: onboarding the two live models must really reload them."""
    client, swaps = roster
    main = client.app.state.server.slots["main"]
    main.loaded_model, main.healthy = "test-model-q4", True
    r = client.post("/swap", json={"model": "test-model-q4", "target": "main", "trial": True},
                    headers={"X-Request-Id": "reload"})
    assert r.status_code == 202, r.text
    assert r.json()["status"] == "swapping"
    wait_for_swap(client.app.state.server, "main", "reload")
    assert swaps == [("main", "test-model-q4")]


def test_a_verified_swap_of_the_loaded_model_is_still_200(roster):
    client, swaps = roster
    main = client.app.state.server.slots["main"]
    main.loaded_model, main.healthy = "test-model-q4", True
    r = client.post("/swap", json={"model": "test-model-q4", "target": "main", "trial": False})
    assert r.status_code == 200, r.text
    assert swaps == []


# ---------------------------------------------------------------------------
# GET /v1/models and GET /status
# ---------------------------------------------------------------------------

def test_models_lists_only_onboarded_profiles(roster):
    client, _ = roster
    client.app.state.server.slots["egpu"].loaded_model = "home-egpu-q4.gguf"
    data = client.get("/v1/models").json()["data"]
    assert [m["id"] for m in data] == ["Home-Egpu-Q4", "test-model-q4", "test-model-q8"]
    home = data[0]
    assert set(home) == {"id", "object", "created", "owned_by",
                         "home_slot", "verified_slots", "loaded_on"}
    assert (home["object"], home["owned_by"]) == ("model", "local")
    assert home["home_slot"] == "egpu"
    assert home["verified_slots"] == ["egpu"]
    assert home["loaded_on"] == ["egpu"]
    assert data[1]["verified_slots"] == ["main", "egpu"]
    assert data[1]["loaded_on"] == []


def test_status_reports_home_models_and_profile_errors(roster):
    client, _ = roster
    body = client.get("/status").json()
    assert body["slots"]["main"]["models"] == ["Unverified-Q4", "test-model-q4", "test-model-q8"]
    assert body["slots"]["egpu"]["models"] == ["Home-Egpu-Q4"]
    assert set(body["profile_errors"]) == {"Broken-Q4"}
    assert "ctx_size" in body["profile_errors"]["Broken-Q4"]


def test_a_slot_with_no_home_models_reports_an_empty_list(three_slot_config):
    from manager.app import create_app
    # The lifespan runs: /status asks each slot's consumer to probe (phase 5).
    with TestClient(create_app(three_slot_config)) as client:
        body = client.get("/status").json()
    assert body["slots"]["aux"]["models"] == []
    assert body["profile_errors"] == {}


# ---------------------------------------------------------------------------
# What Task 3 removes
# ---------------------------------------------------------------------------

def test_leftover_slot_models_vars_are_ignored():
    """manager.env keeps SLOT_*_MODELS until cutover step 7; a model listed on
    two slots must not stop the manager starting in the meantime."""
    from manager.slot_config import SlotConfig, build_slots
    env = {"SLOTS": "main,egpu", "SLOT_EGPU_PORT": "8084",
           "SLOT_MAIN_MODELS": "Foo-Q4", "SLOT_EGPU_MODELS": "foo-q4.gguf"}
    slots = build_slots(env.get)
    assert [s.name for s in slots] == ["main", "egpu"]
    assert "models" not in {f.name for f in dataclasses.fields(SlotConfig)}


def test_the_gguf_roster_and_home_map_are_gone():
    import manager.routing as routing
    from manager.app import ServerState
    assert not hasattr(routing, "home_slot")
    assert not hasattr(ServerState, "model_path")
    assert not hasattr(ServerState, "list_models")
