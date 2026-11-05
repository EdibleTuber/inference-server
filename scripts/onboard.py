#!/usr/bin/env python3
# scripts/onboard.py
"""Verify that a model runs on a slot, and record the pass in the repo profile.

onboard trial-loads the model on the target slot (POST /swap with "trial":
true), sends one short chat through the manager, swaps the slot back to the
model it held, and only if all of that succeeds appends a
[slots.<s>.verified] block to the REPO profile. The deployed copy is never
touched: the operator commits the profile and deploys (Q25).

Each POST /swap carries its own X-Request-Id, and the swap is decided only by
the last_swap entry in /status that has that id: outcome "ok" has landed it;
any other outcome (failed, fell_back) fails it at once, printing the outcome
and its message. The id is what decides it, not the slot's health or
last_swap_utc: /swap answers 202 before its background task takes the slot's
lock, so the first polls can still show the slot exactly as it was --
including an older swap's last_swap, which the id check ignores -- and a
fallback that puts the slot back on its old model shows it healthy, which
used to look like success. A non-202 from /swap fails at once, with no
polling.

Exit codes:
  0  verified: the repo profile carries the [slots.<s>.verified] block;
     commit it and deploy
  1  the trial, the verification chat, or the swap back failed, or the
     manager answered 401 (export MANAGER_TOKEN); nothing was written
  2  the repo profile is missing or invalid, the manager has no such slot,
     or the profile has no section for the slot
  3  the target slot is already verified (delete [slots.<s>.verified] to
     re-verify)
  4  the deployed profile differs from the repo copy (or is missing);
     deploy first
  5  the model is already loaded on another slot: the chat would be
     answered there, not on the target
"""
from __future__ import annotations

import argparse
import hashlib
import os
import re
import subprocess
import sys
import time
import uuid
from collections.abc import Callable, Sequence
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import httpx

from manager.names import same_model
from manager.profiles import ProfileError, load_profile

class _TokenRequired(Exception):
    """The manager answered 401: it requires a token this run did not send."""


MANAGER_ENV = "/etc/llama/manager.env"
DEPLOYED_PROFILES = "/etc/llama/models.d"
MODELS_DIR = "/opt/llama/models"
LLAMA_SERVER = "/opt/llama/bin/llama-server"

_VERSION_RE = re.compile(r"build (\d+), commit (\w+)")


def parse_llama_version(text: str) -> tuple[int, str]:
    """The (build, commit) pair out of `llama-server --version` output.

    The binary prints a log line before the version line
    (`version: 0.4.1-dev (build 11136, commit 057494f93)`), so the pattern is
    searched, not matched at the start of the text.
    """
    match = _VERSION_RE.search(text)
    if match is None:
        raise ValueError(f"no 'build <n>, commit <c>' line in: {text!r}")
    return int(match.group(1)), match.group(2)


def manager_url_from_env(text: str) -> str:
    """http://<HOST>:<PORT> from the body of /etc/llama/manager.env.

    HOST absent or 0.0.0.0 means 127.0.0.1; PORT absent means 11434. On this
    host the manager binds the LAN address (192.168.1.14), so a hardcoded
    127.0.0.1 default would refuse every connection.
    """
    host, port = "", ""
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if key == "HOST" and value:
            host = value
        elif key == "PORT" and value:
            port = value
    if host in ("", "0.0.0.0"):
        host = "127.0.0.1"
    if not port:
        port = "11434"
    return f"http://{host}:{port}"


def onboard(
    model: str,
    slot: str | None,
    *,
    client: httpx.Client,
    repo_profiles: str,
    deployed_profiles: str,
    models_dir: str,
    llama_version: Callable[[], tuple[int, str]],
    today: Callable[[], str],
    timeout: float = 900,
    poll: float = 2,
) -> int:
    """Verify (model, slot) with a trial load; see the module docstring.

    Returns the process exit code. Only the repo profile is ever written.
    """
    # -- what the manager has, before anything else (exit 2 needs its slots) --
    slots = client.get("/status").json()["slots"]
    slot_names = list(slots)

    # -- exit 2: the repo profile must exist and be valid ----------------------
    profile_path = Path(repo_profiles) / f"{model}.toml"
    if not profile_path.is_file():
        print(f"no repo profile for '{model}': {profile_path} is missing")
        return 2
    try:
        profile = load_profile(str(profile_path), slot_names, models_dir)
    except ProfileError as exc:
        print(f"the repo profile is invalid: {exc}")
        return 2

    if slot is None:
        slot = profile.home
    if slot not in slots:
        print(f"the manager has no slot '{slot}'; it has: {', '.join(slot_names)}")
        return 2
    section = profile.section(slot)
    if section is None:
        print(f"model '{profile.name}' has no [slots.{slot}] section; "
              f"it may run on: {', '.join(profile.sections)}")
        return 2

    # -- exit 3: the target slot is already verified ----------------------------
    if section.verified is not None:
        v = section.verified
        print(f"slot '{slot}' is already verified ({v.date}, build {v.llama_cpp_build}, "
              f"commit {v.llama_cpp_commit}); delete [slots.{slot}.verified] to re-verify")
        return 3

    # -- exit 4: the manager must load the exact file being verified ------------
    repo_digest = hashlib.sha256(profile_path.read_bytes()).digest()
    deployed_path = Path(deployed_profiles) / f"{model}.toml"
    deployed_bytes = deployed_path.read_bytes() if deployed_path.is_file() else None
    if deployed_bytes is None or hashlib.sha256(deployed_bytes).digest() != repo_digest:
        where = "is missing" if deployed_bytes is None else "differs from the repo copy"
        print(f"the deployed profile {deployed_path} {where}; deploy first: the manager "
              f"must be loading the exact file that is being verified")
        return 4

    # -- exit 5: the model is already loaded elsewhere ---------------------------
    # Chat goes to the first configured slot holding the model, so a chat
    # "verifying" the target would be answered there and prove nothing.
    for name in slot_names:
        if name != slot and same_model(model, slots[name].get("loaded_model")):
            print(f"model '{model}' is already loaded on slot '{name}': a chat for it "
                  f"would be answered there, not on '{slot}'. Swap it off '{name}' "
                  f"first, or onboard '{name}'")
            return 5

    # -- the trial ---------------------------------------------------------------
    previous = slots[slot].get("loaded_model")

    def refused_detail(response: httpx.Response) -> str:
        try:
            body = response.json()
        except ValueError:
            body = {}
        error = body.get("error") if isinstance(body, dict) else None
        parts = [str(error[key]) for key in ("type", "message")
                 if isinstance(error, dict) and error.get(key)]
        return f" (HTTP {response.status_code}" + (f", {' '.join(parts)})" if parts else ")")

    def slot_status() -> dict:
        return client.get("/status").json()["slots"][slot]

    def swap_and_wait(swap_model: str) -> bool:
        """POST /swap (trial) and poll /status until this swap's own
        last_swap entry decides it: outcome "ok" has landed it; any other
        outcome (failed, fell_back) fails it at once."""
        rid = uuid.uuid4().hex
        response = client.post(
            "/swap",
            json={"model": swap_model, "target": slot, "trial": True},
            headers={"X-Request-Id": rid})
        if response.status_code != 202:
            print(f"swap of '{swap_model}' onto '{slot}' was refused{refused_detail(response)}")
            return False
        deadline = time.monotonic() + timeout
        while True:
            last_swap = slot_status().get("last_swap")
            if isinstance(last_swap, dict) and last_swap.get("id") == rid:
                outcome = last_swap.get("outcome")
                if outcome == "ok":
                    return True
                message = last_swap.get("message")
                print(f"swap of '{swap_model}' onto '{slot}' failed: {outcome}"
                      + (f": {message}" if message else ""))
                return False
            if time.monotonic() >= deadline:
                print(f"timed out after {timeout:g}s waiting for '{swap_model}' to land "
                      f"on '{slot}'")
                return False
            time.sleep(poll)

    def trial_chat() -> bool:
        body = {"model": model, "max_tokens": 256,
                "messages": [{"role": "user", "content": "Reply with the single word: ready"}]}
        response = client.post("/v1/chat/completions", json=body)
        if response.status_code != 200:
            print(f"the verification chat failed{refused_detail(response)}")
            return False
        try:
            choices = response.json().get("choices") or []
        except ValueError:
            choices = []
        if not choices:
            print("the verification chat returned no choices")
            return False
        return True

    print(f"trial-swapping '{model}' onto '{slot}' ...")
    landed = swap_and_wait(model)
    chat_ok = False
    if landed:
        print("trial landed; sending the verification chat")
        chat_ok = trial_chat()

    back_ok = True
    if previous and not same_model(previous, model):
        print(f"swapping '{slot}' back to '{previous}' ...")
        back_ok = swap_and_wait(previous)

    if not (landed and chat_ok and back_ok):
        print("onboarding failed; the profile was not modified")
        return 1

    try:
        build, commit = llama_version()
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print(f"could not determine the llama.cpp build: {exc}")
        return 1
    with profile_path.open("a", encoding="utf-8") as f:
        f.write(f"\n[slots.{slot}.verified]\n"
                f'date = "{today()}"\n'
                f"llama_cpp_build = {build}\n"
                f'llama_cpp_commit = "{commit}"\n')
    try:
        reloaded = load_profile(str(profile_path), slot_names, models_dir)
    except ProfileError as exc:
        print(f"internal error: the profile is invalid after writing the verified "
              f"block: {exc}")
        return 1
    if slot not in reloaded.verified_slots():
        print(f"internal error: slot '{slot}' is not in verified_slots() after the write")
        return 1
    print(f"OK: '{profile.name}' is verified on '{slot}' (build {build}, commit {commit}); "
          f"commit the profile and deploy")
    return 0


def _llama_version() -> tuple[int, str]:
    # --version prints a log line before the version line; search both streams.
    result = subprocess.run([LLAMA_SERVER, "--version"], capture_output=True,
                            text=True, timeout=30)
    return parse_llama_version((result.stdout or "") + (result.stderr or ""))


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="onboard.py",
        description="Verify that a model runs on a slot: trial-load it, send one "
                    "short chat, swap the slot back, and record the pass in the repo "
                    "profile, which you then commit and deploy.")
    parser.add_argument("model",
                        help="the model to verify: its profile's file name without .toml, exactly")
    parser.add_argument("--slot", default=None,
                        help="slot to verify on (default: the profile's home slot)")
    parser.add_argument("--manager", default=None,
                        help="manager base URL (default: HOST and PORT from "
                             "/etc/llama/manager.env)")
    args = parser.parse_args(argv)

    manager_url = args.manager
    if manager_url is None:
        try:
            text = Path(MANAGER_ENV).read_text(encoding="utf-8")
        except OSError:
            text = ""
        manager_url = manager_url_from_env(text)

    # The token comes from the environment, never an argument (a command
    # line is visible in ps). Set, every call carries it as a Bearer header;
    # unset, no header at all.
    token = os.environ.get("MANAGER_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else None

    # A 401 must be one line, not a traceback: the first /status call would
    # otherwise raise KeyError on a body with no "slots".
    def refuse_401(response: httpx.Response) -> None:
        if response.status_code == 401:
            raise _TokenRequired()

    # A cold load and the first chat can take minutes, so per-request timeouts
    # are generous; the swap deadline is enforced by the poll loop, not by them.
    client = httpx.Client(base_url=manager_url, timeout=600.0, headers=headers,
                          event_hooks={"response": [refuse_401]})
    try:
        return onboard(
            args.model, args.slot,
            client=client,
            repo_profiles=str(REPO_ROOT / "config" / "models"),
            deployed_profiles=DEPLOYED_PROFILES,
            models_dir=MODELS_DIR,
            llama_version=_llama_version,
            today=date.today().isoformat,
        )
    except _TokenRequired:
        print("the manager answered 401: it requires a token — export "
              "MANAGER_TOKEN (the manager's MANAGER_TOKEN) and retry")
        return 1
    except httpx.HTTPError as exc:
        print(f"cannot talk to the manager at {manager_url}: {exc}")
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
