# tests/test_onboard.py
"""Phase 3 Task 4: scripts/onboard.py verifies a (model, slot) pair with a
trial load and records it in the repo profile (Q25).

The manager is a fake behind httpx.MockTransport. Its /status replays what
the real one shows after POST /swap: the swap runs as a background task, so
the first polls can still show the slot exactly as it was (the old model
healthy, not swapping, the old last_swap_utc), then swapping, then the
outcome. Onboarding a model that is already loaded is the main use (Task 7),
and there the stale polls look exactly like success.
"""
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import httpx
import pytest

REPO = Path(__file__).resolve().parent.parent

from manager.profiles import load_profile


def _load_onboard():
    spec = importlib.util.spec_from_file_location("onboard", REPO / "scripts" / "onboard.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


onboard_mod = _load_onboard()

SAMPLING = ("temp = 1.0\ntop_k = 20\ntop_p = 0.95\nmin_p = 0.0\n"
            "presence_penalty = 0.0\nfrequency_penalty = 0.0\nrepeat_penalty = 1.0\n")


def _section(slot: str, verified: bool = False) -> str:
    text = (f"\n[slots.{slot}]\nctx_size = 4096\nn_gpu_layers = -1\nflags = [\"--jinja\"]\n"
            f"\n[slots.{slot}.sampling]\n{SAMPLING}")
    if verified:
        text += (f"\n[slots.{slot}.verified]\ndate = \"2026-10-01\"\nllama_cpp_build = 11000\n"
                 "llama_cpp_commit = \"abc1234\"\n")
    return text


NEW = "New-Model-Q4"
PROFILE = 'home = "main"\n' + _section("main") + _section("egpu")


class FakeManager:
    """A manager whose /status replays a background swap.

    stale: polls after a POST /swap that still show the slot unchanged.
    outcome: "ok" (lands), "fail" (shows swapping, then unhealthy),
    "fast-fail" (unhealthy, never seen swapping), "never" (swapping forever).
    """

    def __init__(self, slots, *, stale=0, outcome="ok", back_outcome="ok",
                 chat_status=200, chat_choices=None):
        self.slots = {n: dict(s) for n, s in slots.items()}
        for s in self.slots.values():
            s.setdefault("healthy", True)
            s.setdefault("swapping", False)
            s.setdefault("last_swap_utc", "2026-10-30T00:00:00.000001+00:00")
        self.stale, self.outcome, self.back_outcome = stale, outcome, back_outcome
        self.chat_status = chat_status
        self.chat_choices = [{"message": {"content": "ready"}}] if chat_choices is None else chat_choices
        self.swaps: list[dict] = []
        self.chats: list[dict] = []
        self.status_calls = 0
        self.polls_after_settle = 0      # /status calls once a swap has failed for good
        self._pending = None             # [slot, model, outcome, phase-list]
        self._stamp = 0
        self.landed = set()              # slots whose last swap has landed

    def _advance(self):
        if self._pending is None:
            return
        slot, model, outcome, phases = self._pending
        s = self.slots[slot]
        phase = phases.pop(0) if phases else "settled"
        if phase == "stale":
            return
        if phase == "swapping":
            s.update(swapping=True, healthy=False)
            return
        if outcome == "ok":
            self._stamp += 1
            s.update(swapping=False, healthy=True, loaded_model=model,
                     last_swap_utc=f"2026-10-31T00:00:00.{self._stamp:06d}+00:00")
            self.landed.add(slot)
            self._pending = None
        elif outcome == "never":
            s.update(swapping=True, healthy=False)
            phases.append("final")
        else:
            s.update(swapping=False, healthy=False)
            self.polls_after_settle += 1

    def handler(self, request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if request.method == "GET" and path == "/status":
            self.status_calls += 1
            self._advance()
            return httpx.Response(200, json={"slots": {n: dict(s) for n, s in self.slots.items()}})
        if request.method == "POST" and path == "/swap":
            body = json.loads(request.content)
            self.swaps.append(body)
            outcome = self.outcome if len(self.swaps) == 1 else self.back_outcome
            phases = ["stale"] * self.stale
            if outcome != "fast-fail":
                phases.append("swapping")
            phases.append("final")
            self._pending = [body["target"], body["model"], outcome, phases]
            self.landed.discard(body["target"])
            return httpx.Response(202, json={"slot": body["target"], "model": body["model"],
                                             "status": "swapping"})
        if request.method == "POST" and path == "/v1/chat/completions":
            body = json.loads(request.content)
            self.chats.append(body)
            # llama-server is mid-restart until the swap lands.
            if self._pending is not None or not self.landed:
                return httpx.Response(503, json={"error": {"type": "slot_unavailable"}})
            return httpx.Response(self.chat_status, json={"choices": self.chat_choices})
        return httpx.Response(404, json={"error": {"type": "not_found"}})


@pytest.fixture
def env(tmp_path):
    repo = tmp_path / "config-models"
    deployed = tmp_path / "models.d"
    models = tmp_path / "models"
    for d in (repo, deployed, models):
        d.mkdir()
    (models / f"{NEW}.gguf").write_bytes(b"GGUF")

    def put(text: str, name: str = NEW, deploy: bool = True):
        (repo / f"{name}.toml").write_text(text)
        if deploy:
            (deployed / f"{name}.toml").write_text(text)
    put(PROFILE)
    return {"repo": repo, "deployed": deployed, "models": models, "put": put}


def _run(env, fake, model=NEW, slot="main", *, timeout=5.0, poll=0):
    client = httpx.Client(transport=httpx.MockTransport(fake.handler), base_url="http://manager")
    return onboard_mod.onboard(
        model, slot, client=client, repo_profiles=str(env["repo"]),
        deployed_profiles=str(env["deployed"]), models_dir=str(env["models"]),
        llama_version=lambda: (11136, "057494f93"), today=lambda: "2026-10-31",
        timeout=timeout, poll=poll)


def _two_slots(main_model="Old-Model-Q4", egpu_model="Egpu-Model-Q4"):
    return {"main": {"loaded_model": main_model}, "egpu": {"loaded_model": egpu_model}}


VERIFIED_BLOCK = ('\n[slots.main.verified]\ndate = "2026-10-31"\nllama_cpp_build = 11136\n'
                  'llama_cpp_commit = "057494f93"\n')


# --- success -----------------------------------------------------------------

def test_a_passing_trial_appends_the_verified_block_and_swaps_back(env):
    fake = FakeManager(_two_slots())
    assert _run(env, fake) == 0
    assert (env["repo"] / f"{NEW}.toml").read_text() == PROFILE + VERIFIED_BLOCK
    profile = load_profile(str(env["repo"] / f"{NEW}.toml"), ["main", "egpu"], str(env["models"]))
    assert profile.verified_slots() == ("main",)
    assert fake.swaps == [{"model": NEW, "target": "main", "trial": True},
                          {"model": "Old-Model-Q4", "target": "main", "trial": True}]
    assert fake.chats == [{"model": NEW, "max_tokens": 256, "messages": [
        {"role": "user", "content": "Reply with the single word: ready"}]}]
    assert fake.slots["main"]["loaded_model"] == "Old-Model-Q4"


def test_the_deployed_copy_is_not_written(env):
    fake = FakeManager(_two_slots())
    assert _run(env, fake) == 0
    assert (env["deployed"] / f"{NEW}.toml").read_text() == PROFILE


def test_the_slot_defaults_to_the_profiles_home(env):
    env["put"]('home = "egpu"\n' + _section("main") + _section("egpu"))
    fake = FakeManager(_two_slots())
    assert _run(env, fake, slot=None) == 0
    assert fake.swaps[0] == {"model": NEW, "target": "egpu", "trial": True}
    assert "[slots.egpu.verified]" in (env["repo"] / f"{NEW}.toml").read_text()


@pytest.mark.parametrize("previous", [None, NEW, NEW.lower() + ".gguf"])
def test_no_swap_back_when_the_slot_was_empty_or_held_the_model(env, previous):
    fake = FakeManager(_two_slots(main_model=previous), stale=0)
    assert _run(env, fake) == 0
    assert fake.swaps == [{"model": NEW, "target": "main", "trial": True}]


@pytest.mark.parametrize("stale", [1, 3])
def test_onboarding_the_loaded_model_waits_for_the_trial_to_land(env, stale):
    """The slot already holds the model, so stale polls show it healthy and
    loaded. Only a changed last_swap_utc proves the trial reload happened;
    the fake's chat answers 503 until it has."""
    fake = FakeManager(_two_slots(main_model=NEW), stale=stale)
    assert _run(env, fake) == 0
    assert len(fake.chats) == 1
    assert "[slots.main.verified]" in (env["repo"] / f"{NEW}.toml").read_text()


def test_stale_polls_before_the_swap_back_are_waited_out_too(env):
    fake = FakeManager(_two_slots(main_model="Old-Model-Q4"), stale=2)
    assert _run(env, fake) == 0
    assert fake.slots["main"]["loaded_model"] == "Old-Model-Q4"
    assert fake.slots["main"]["healthy"] is True


def test_a_slot_that_starts_unhealthy_is_not_mistaken_for_a_failed_trial(env):
    """A slot that was already down shows unhealthy and not swapping in the
    stale polls; that is not this trial failing."""
    slots = _two_slots(main_model=None)
    slots["main"]["healthy"] = False
    fake = FakeManager(slots, stale=2)
    assert _run(env, fake) == 0


def test_a_non_home_slot_is_verified_under_its_own_section(env):
    fake = FakeManager(_two_slots())
    assert _run(env, fake, slot="egpu") == 0
    assert (env["repo"] / f"{NEW}.toml").read_text() == PROFILE + VERIFIED_BLOCK.replace(
        "[slots.main.verified]", "[slots.egpu.verified]")
    assert [s["target"] for s in fake.swaps] == ["egpu", "egpu"]


def test_the_poll_loop_sleeps_poll_seconds_between_polls(env, monkeypatch):
    """Against a real manager a loop without the sleep is a tight CPU spin."""
    slept = []
    monkeypatch.setattr(onboard_mod.time, "sleep", slept.append)
    fake = FakeManager(_two_slots(), stale=2)
    assert _run(env, fake, poll=0.5) == 0
    assert slept and set(slept) == {0.5}


# --- trial failures: exit 1, nothing written, swap back still runs -------------

@pytest.mark.parametrize("outcome", ["fail", "fast-fail"])
def test_a_failed_trial_swap_exits_1_promptly_and_still_swaps_back(env, outcome):
    fake = FakeManager(_two_slots(), outcome=outcome)
    assert _run(env, fake, timeout=3.0) == 1
    assert (env["repo"] / f"{NEW}.toml").read_text() == PROFILE
    assert fake.chats == []
    assert fake.swaps[1] == {"model": "Old-Model-Q4", "target": "main", "trial": True}
    # A failure is recognised when /status shows it, not by running out the clock.
    assert fake.polls_after_settle <= 2


def test_a_trial_that_never_lands_times_out(env):
    fake = FakeManager(_two_slots(), outcome="never")
    assert _run(env, fake, timeout=0.2) == 1
    assert (env["repo"] / f"{NEW}.toml").read_text() == PROFILE


@pytest.mark.parametrize("status,choices", [(500, None), (200, [])])
def test_a_failed_chat_exits_1_and_still_swaps_back(env, status, choices):
    fake = FakeManager(_two_slots(), chat_status=status, chat_choices=choices)
    assert _run(env, fake) == 1
    assert (env["repo"] / f"{NEW}.toml").read_text() == PROFILE
    assert fake.swaps[1]["model"] == "Old-Model-Q4"
    assert fake.slots["main"]["loaded_model"] == "Old-Model-Q4"


@pytest.mark.parametrize("back", ["fail", "fast-fail"])
def test_a_failed_swap_back_exits_1_and_writes_nothing(env, back):
    fake = FakeManager(_two_slots(), back_outcome=back)
    assert _run(env, fake, timeout=3.0) == 1
    assert (env["repo"] / f"{NEW}.toml").read_text() == PROFILE


def test_a_refused_swap_exits_1_without_polling(env):
    fake = FakeManager(_two_slots())
    real = fake.handler

    def refuse(request):
        if request.url.path == "/swap":
            fake.swaps.append(json.loads(request.content))
            return httpx.Response(400, json={"error": {"type": "slot_not_allowed"}})
        return real(request)
    fake.handler = refuse
    log = []
    counted = fake.handler

    def logged(request):
        log.append(request.url.path)
        return counted(request)
    fake.handler = logged
    assert _run(env, fake) == 1
    # After each refused POST: at most the next swap's one pre-read, never a poll.
    after = log[log.index("/swap"):]
    assert after in (["/swap"], ["/swap", "/status", "/swap"]), log
    assert fake.chats == []
    assert (env["repo"] / f"{NEW}.toml").read_text() == PROFILE


# --- refusals before any swap ------------------------------------------------

def _assert_no_swap(env, fake, original=PROFILE):
    assert fake.swaps == [] and fake.chats == []
    assert (env["repo"] / f"{NEW}.toml").read_text() == original


@pytest.mark.parametrize("text", [
    'home = "main"\n' + _section("main").replace("ctx_size = 4096\n", ""),   # missing key
    'home = "main"\n' + _section("aux"),                                      # not a manager slot
    "not toml [",
])
def test_an_invalid_repo_profile_exits_2(env, text):
    env["put"](text)
    fake = FakeManager(_two_slots())
    assert _run(env, fake) == 2
    _assert_no_swap(env, fake, text)


def test_a_missing_repo_profile_exits_2(env):
    (env["repo"] / f"{NEW}.toml").unlink()
    fake = FakeManager(_two_slots())
    assert _run(env, fake) == 2
    assert fake.swaps == []


def test_slot_names_come_from_the_manager(env):
    """A profile with an [slots.aux] section is valid only if the manager has aux."""
    text = 'home = "main"\n' + _section("main") + _section("aux")
    env["put"](text)
    slots = _two_slots()
    slots["aux"] = {"loaded_model": None}
    assert _run(env, FakeManager(slots)) == 0


@pytest.mark.parametrize("slot", ["aux", "egpu"])
def test_a_slot_the_profile_cannot_run_on_exits_2(env, slot, capsys):
    """aux is not a manager slot; egpu is, but this profile has no section for it."""
    text = 'home = "main"\n' + _section("main")
    env["put"](text)
    fake = FakeManager(_two_slots())
    assert _run(env, fake, slot=slot) == 2
    _assert_no_swap(env, fake, text)


def test_an_already_verified_slot_exits_3(env, capsys):
    text = 'home = "main"\n' + _section("main", verified=True) + _section("egpu")
    env["put"](text)
    fake = FakeManager(_two_slots())
    assert _run(env, fake) == 3
    assert "[slots.main.verified]" in capsys.readouterr().out
    _assert_no_swap(env, fake, text)


def test_another_slots_verification_does_not_block(env):
    text = 'home = "main"\n' + _section("main") + _section("egpu", verified=True)
    env["put"](text)
    fake = FakeManager(_two_slots())
    assert _run(env, fake) == 0


@pytest.mark.parametrize("deployed", ["differs", "missing"])
def test_a_repo_profile_that_is_not_deployed_exits_4(env, deployed, capsys):
    if deployed == "missing":
        (env["deployed"] / f"{NEW}.toml").unlink()
    else:
        (env["deployed"] / f"{NEW}.toml").write_text(PROFILE + "\n# edited on the host\n")
    fake = FakeManager(_two_slots())
    assert _run(env, fake) == 4
    assert "deploy" in capsys.readouterr().out.lower()
    _assert_no_swap(env, fake)


def test_the_model_loaded_on_another_slot_exits_5(env, capsys):
    """Chat goes to the first configured slot holding the model, so with the
    model also on main, a chat 'verifying' egpu would be answered by main."""
    fake = FakeManager(_two_slots(main_model=NEW.lower()))
    assert _run(env, fake, slot="egpu") == 5
    assert "main" in capsys.readouterr().out
    _assert_no_swap(env, fake)


# --- the script itself ---------------------------------------------------------

def test_the_script_runs_from_anywhere_without_pythonpath(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    r = subprocess.run([sys.executable, str(REPO / "scripts" / "onboard.py"), "--help"],
                       cwd=tmp_path, env=env, capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    for opt in ("--slot", "--manager"):
        assert opt in r.stdout


def test_parse_llama_version():
    # Verbatim from /opt/llama/bin/llama-server --version on the host, 2026-10-31.
    out = ("0.00.000.606 I srv  llama_server: initializing ...\n"
           "version: 0.4.1-dev (build 11136, commit 057494f93)\n"
           "built with GNU 13.3.0 for Linux x86_64\n")
    assert onboard_mod.parse_llama_version(out) == (11136, "057494f93")


@pytest.mark.parametrize("text,url", [
    ("HOST=192.168.1.14\nPORT=11434\n", "http://192.168.1.14:11434"),   # this host today
    ("# comment\nHOST=0.0.0.0\nPORT=9000\n", "http://127.0.0.1:9000"),  # all interfaces
    ("PORT=11434\n", "http://127.0.0.1:11434"),
    ("", "http://127.0.0.1:11434"),
])
def test_the_default_manager_url_comes_from_manager_env(text, url):
    """The manager binds HOST from manager.env; on this host that is the LAN
    address, and 127.0.0.1:11434 refuses connections."""
    assert onboard_mod.manager_url_from_env(text) == url



def test_the_llama_version_is_read_from_stderr(monkeypatch):
    """llama-server prints --version to stderr only (checked on the host,
    2026-11-01: stdout is empty). _llama_version is main()'s probe."""
    def fake_run(cmd, **kwargs):
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr=(
            "0.00.000.533 I srv  llama_server: initializing ...\n"
            "version: 0.4.1-dev (build 11136, commit 057494f93)\n"))
    monkeypatch.setattr(onboard_mod.subprocess, "run", fake_run)
    assert onboard_mod._llama_version() == (11136, "057494f93")
