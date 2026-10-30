# Phase 1: Hygiene and Cuts Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the suite hermetic and deploys trustworthy, cut the batch slot and `/v1/embeddings`, require JSON on POST, and rename the `re` slot to `egpu`. These are the ground every later phase stands on.

**Architecture:** Most tasks are mechanical edits to existing modules (`manager/config.py`, `manager/slot_config.py`, `manager/app.py`), the two shell scripts, and the test fixtures. The slot code is already generic over slot names (`slot_config.py:155-159` derives `/etc/llama/llama-server-<name>.env` and `llama-server-<name>.service`), so the rename needs no manager code change: only templates, units, docs, tests and host files. Task 10 is a host cutover that the owner runs by hand with `!` commands, because it needs root.

**Tech Stack:** Python 3 / FastAPI / httpx / pytest + pytest-asyncio, bash, systemd.

**Spec:** `docs/design-review-2026-10-29.md`. This phase covers Q6, Q8, Q24 (the `/v1/embeddings` removal), Q35/Q39, Q36, Q37 and Q41. Read the spec's Housekeeping section and phase table before starting.

**Status of code in this plan:** the code blocks are sketches of tests and interfaces written without running them. Verify them against the real files. Deviating where the codebase disagrees is correct, not a failure to follow the plan. Say so in the task report.

## Global Constraints

- Run the suite with `PYTHONPATH=. /opt/llama/manager/venv/bin/python -m pytest tests/ -q -p no:cacheprovider` from the repo root. It must be green at the end of every task.
- **Tests must never reach the live ports** 8080, 8081, 8082, 8083, 8084 or 11434. After Task 1, a guard enforces this.
- Work in a worktree on branch `phase1/hygiene-and-cuts`, never on `main`.
- Stage explicit paths. Never use `git add -A` or `git add .`.
- One commit per task, using the message given in the task, ending with the attribution lines the session provides.
- Do not touch `/etc`, `/opt`, `/var` or any systemd unit on the host. Only Task 10 does, and the owner runs it.
- No claim in a doc or comment may carry a count that will go stale (test counts, line counts). Name the command that reports it instead.

## Implementer Rules

- **Do one task, then stop.** Report what you changed, the test output, and any place you departed from the plan and why. Don't start the next task.
- **Edit tests only where the task allows.** Each task's **Files** list names the existing tests it may change (for example "every test referencing batch" in Task 2, or "tests that POST raw bodies without a Content-Type" in Task 5). Don't weaken or delete an assertion to make a test pass outside that list. Stop and report instead.
- **Stop on any surprise:** a new test that passes before the change, or an unrelated test that fails after it.
- **Never touch the host or remote:** no `sudo`, `systemctl`, `deploy-manager.sh` (other than `--check`), `git push`, or requests to the live ports. Task 10 belongs to the owner.
- **Stage explicit paths only,** and commit with the message the task gives.

## Review Focus

1. **A client sending `application/json; charset=utf-8`**, or `Application/JSON`, must be accepted by the JSON gate (Task 5). Media-type parameters and case are legal.
2. **A test that posts a raw body with no Content-Type** (`content=b"..."` in TestClient) will start getting 415 after Task 5. Fix each such test to send the header, unless it's the test proving the 415.
3. **The deploy script under `set -e`**: a `grep` or `[ ]` that legitimately returns nonzero (no match, empty stamp file) must not abort the deploy. Guard each one with `|| true` or an `if` (Task 6).
4. **The test-suite run inside `deploy-manager.sh`** happens under sudo. It must not leave root-owned `__pycache__` or `.pytest_cache` files in the checkout (Task 6).
5. **An unset `SLOTS` after the batch removal** must still yield a working single `main` slot (Task 2). A host `manager.env` that still has `BATCH_*` keys must be ignored, not rejected.

Each line has a test in its owning task. Line 4 is checked by a verification step, because it needs a real sudo run.

---

### Task 1: Hermetic test suite

The suite is green today, but `tests/conftest.py:40-79` points `test_config` at the live ports 8081, 8082 and 8083. Only the per-file `no_network` guard in `tests/test_consumer.py:33-42` protects one file. This task closes finding Q13.

**Files:**
- Create: `tests/live_ports.py` (the shared names; see below for why they don't live in `conftest.py`)
- Modify: `tests/conftest.py` (add the guard; change the `test_config` ports)
- Create: `tests/test_hermetic.py`
- Modify: any test that hardcodes a live port in a URL or assertion (find them with `grep -n -E '808[0-4]|11434' tests/*.py`)

**Interfaces:**
- Produces: `tests/live_ports.py` defines `LIVE_PORTS: frozenset[int] = frozenset({8080, 8081, 8082, 8083, 8084, 11434})` and `class LivePortBlocked(ConnectionRefusedError)`. They go in their own module because `tests/` has no `__init__.py`: pytest loads `conftest.py` as a top-level module named `conftest`, so a test doing `from tests.conftest import ...` would get a *second* copy, with a different `LivePortBlocked` class, and `pytest.raises` would never match. Both `conftest.py` and the tests import from `tests.live_ports` (the suite runs with `PYTHONPATH=.`).
- Produces: `tests/conftest.py` gets an autouse fixture `block_live_ports` that monkeypatches `socket.socket.connect` and `socket.socket.connect_ex`. Any connect to a port in `LIVE_PORTS`, on any host, raises `LivePortBlocked`. All other connects pass through. The real-socket test `tests/test_stream_disconnect_real.py` uses an ephemeral port and must keep working.
- Produces: `test_config` ports become `18081` (main), `18082` (embeddings), `18083` (second slot).

- [ ] **Step 1: Write the failing tests** in `tests/test_hermetic.py`

```python
import dataclasses
import socket
import pytest
from tests.live_ports import LIVE_PORTS, LivePortBlocked

def test_connect_to_a_live_port_is_blocked_by_the_guard():
    # 8083 is the decommissioned batch port: nothing listens there, so
    # before the guard exists this raises a plain ConnectionRefusedError,
    # not LivePortBlocked, and the test fails without touching a live service.
    s = socket.socket()
    try:
        with pytest.raises(LivePortBlocked):
            s.connect(("127.0.0.1", 8083))
    finally:
        s.close()

def test_test_config_uses_no_live_port(test_config):
    ports = {test_config.llama_server_port, test_config.embeddings_port}
    ports |= {s.port for s in test_config.slots}
    assert not ports & LIVE_PORTS
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PYTHONPATH=. /opt/llama/manager/venv/bin/python -m pytest tests/test_hermetic.py -v -p no:cacheprovider`
Expected: FAIL with a `ModuleNotFoundError` for `tests.live_ports`. To see each test fail for its own reason, create `tests/live_ports.py` first and run again. The guard test then fails because a plain `ConnectionRefusedError` isn't `LivePortBlocked`, and the config test fails on the 8081 overlap.

- [ ] **Step 3: Create `tests/live_ports.py`, add the autouse `block_live_ports` fixture to `tests/conftest.py` (importing from `tests.live_ports`), and move the `test_config` ports**

Use `monkeypatch.setattr(socket.socket, "connect", ...)`, the same technique as `tests/test_consumer.py:33-42`. Address tuples are `(host, port, ...)`. Leave `AF_UNIX` alone. Change every port in `test_config` and its `SlotConfig`s. Leave `test_consumer.py`'s own stricter `no_network` fixture in place.

- [ ] **Step 4: Run the full suite**

Run: the global suite command.
Expected: all tests pass. Fix each test that hardcoded a live port so it uses `test_config`'s port. Don't add ports to an allowlist.

- [ ] **Step 5: Commit**

```bash
git add tests/live_ports.py tests/conftest.py tests/test_hermetic.py <each test file touched>
git commit -m "test: block live ports suite-wide; move test_config off them"
```

---

### Task 2: Remove the batch slot

The spec (Q6) says the batch slot is gone for good. Today it lives on in `manager/config.py` (the six `batch_*` fields at lines 73-78, `batch_server_url` at 94-96, `from_env` at 116-121), in `manager/slot_config.py` (`DEFAULT_SLOTS` at 26, the batch entries in `_LEGACY` at 44-48 and `_DEFAULTS` at 54-55, and the module docstring at 10-13), and in the second slot of `tests/conftest.py`.

**Files:**
- Modify: `manager/config.py`, `manager/slot_config.py`, `manager/app.py` (comments at 190-192 and 242-243 that describe batch only as history may stay; remove any live reference)
- Modify: `tests/conftest.py`. Delete `tmp_batch_env_file` and the `batch_*` kwargs. Rename the second `SlotConfig` to `name="egpu"` with `env_file` from a new `tmp_egpu_env_file` fixture and `systemd_unit="llama-server-egpu.service"`.
- Modify: every test referencing batch (`grep -ln -i batch tests/*.py`). Tests of the multi-slot behaviour keep that behaviour and use `egpu` as the second slot. Tests that exist only for batch-specific legacy aliases are deleted.
- Modify: `config/manager.env` (delete the `BATCH_*` block), `README.md` (remove batch mentions)
- Test: `tests/test_slot_config.py`, `tests/test_config.py`

**Interfaces:**
- Produces: `slot_config.DEFAULT_SLOTS == ("main",)`. `_LEGACY` and `_DEFAULTS` have only `"main"` keys. `ManagerConfig` has no field or property whose name starts with `batch_`.
- Produces: the shared second test slot is named `egpu`. Task 3 relies on this.

- [ ] **Step 1: Write the failing tests**

```python
# tests/test_slot_config.py
def test_unset_slots_is_main_only():
    slots = build_slots(getenv={}.get)
    assert [s.name for s in slots] == ["main"]

def test_leftover_batch_vars_are_ignored_not_rejected():
    env = {"BATCH_SERVER_PORT": "8083", "BATCH_SERVER_UNIT": "llama-server-batch.service"}
    slots = build_slots(getenv=env.get)
    assert [s.name for s in slots] == ["main"]

# tests/test_config.py
def test_manager_config_has_no_batch_fields():
    names = {f.name for f in dataclasses.fields(ManagerConfig)}
    names |= {n for n in dir(ManagerConfig) if not n.startswith("_")}
    assert not any(n.startswith("batch_") for n in names)
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PYTHONPATH=. /opt/llama/manager/venv/bin/python -m pytest tests/test_slot_config.py tests/test_config.py -v -p no:cacheprovider`
Expected: FAIL. The unset-`SLOTS` case yields `["main", "batch"]`, and `batch_server_host` is still a field.

- [ ] **Step 3: Remove the batch code and config and update the fixtures**

`ManagerConfig.from_env` stops reading `BATCH_*`. Any `ManagerConfig(...)` constructed in a test drops the `batch_*` keyword arguments.

- [ ] **Step 4: Run the full suite**

Run: the global suite command.
Expected: all pass. `grep -rn -i 'batch' manager/ config/manager.env tests/` shows only `embed_batch` (`manager/embeddings.py:31`, unrelated) and the historical comments in `app.py`.

- [ ] **Step 5: Commit**

```bash
git add manager/config.py manager/slot_config.py manager/app.py config/manager.env README.md tests/<files>
git commit -m "refactor: remove the decommissioned batch slot"
```

---

### Task 3: Rename the `re` slot to `egpu` in the repo

The spec is Q41. No manager code names `re`, but these files do: `systemd/llama-server-re.service`, `config/llama-server-re.env`, the `SLOT_RE_*` keys in `config/manager.env`, the comment at `manager/slot_config.py:30`, `README.md`, and tests. Find them all with `grep -rn -E 'SLOT_RE_|llama-server-re|\bre slot|"re"' --exclude-dir=docs .` and handle each one.

**Files:**
- Rename: `git mv systemd/llama-server-re.service systemd/llama-server-egpu.service` and `git mv config/llama-server-re.env config/llama-server-egpu.env`, then fix the self-references inside each (paths, log files `llama-server-egpu.{log,err}`, descriptions, install comments)
- Modify: `config/manager.env` (`SLOTS=main,egpu`, `SLOT_EGPU_*`), `manager/slot_config.py:30` (example `SLOT_EGPU_ENV`), `README.md`
- Modify: `scripts/setup.sh:118-121`. The sudoers heredoc gains `_llama-mgr ALL=(root) NOPASSWD: /usr/bin/systemctl restart llama-server-egpu.service`. The live host has a hand-added `re` line that setup never wrote, which is why `doctor.sh` (Task 8) checks for it.
- Modify: tests that use `"re"` as a slot name
- Create: `tests/test_repo_consistency.py`

**Interfaces:**
- Consumes: Task 2's `egpu` second test slot.
- Produces: the repo's template `config/manager.env` builds slots whose unit and env files exist in the repo. Task 8's `doctor.sh` relies on the `llama-server-<slot>` naming.

- [ ] **Step 1: Write the failing tests** in `tests/test_repo_consistency.py`

```python
import os, re, subprocess
from pathlib import Path
from manager.slot_config import build_slots

REPO = Path(__file__).resolve().parent.parent

def _template_env() -> dict[str, str]:
    env = {}
    for line in (REPO / "config/manager.env").read_text().splitlines():
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            env[k.strip()] = v.strip()
    return env

def test_every_template_slot_has_its_unit_and_env_file_in_the_repo():
    for s in build_slots(getenv=_template_env().get):
        assert (REPO / "systemd" / s.systemd_unit).is_file(), s.systemd_unit
        assert (REPO / "config" / os.path.basename(s.env_file)).is_file(), s.env_file

def test_no_live_file_names_the_old_re_slot():
    # Historical records under docs/ keep their wording; everything else must not.
    out = subprocess.run(
        ["git", "-C", str(REPO), "grep", "-n", "-E", r"SLOT_RE_|llama-server-re\b",
         "--", ".", ":!docs/"],
        capture_output=True, text=True,
    )
    assert out.stdout == "", out.stdout
```

- [ ] **Step 2: Run them to verify they fail**

Run: `PYTHONPATH=. /opt/llama/manager/venv/bin/python -m pytest tests/test_repo_consistency.py -v -p no:cacheprovider`
Expected: `test_no_live_file_names_the_old_re_slot` FAILS, listing the `SLOT_RE_` lines. The first test passes before the rename too, because the `re` files exist. It's a guard for the rename, not proof of it, which is acceptable.

- [ ] **Step 3: Do the renames and edits listed under Files**

- [ ] **Step 4: Run the full suite**

Run: the global suite command.
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add -u systemd/ config/ && git add systemd/llama-server-egpu.service config/llama-server-egpu.env manager/slot_config.py scripts/setup.sh README.md tests/test_repo_consistency.py tests/<files>
git commit -m "refactor: rename the re slot to egpu"
```

---

### Task 4: Remove `/v1/embeddings`

The spec is Q24: nothing calls it, and collections call `:8082` directly through `manager/embeddings.py`. The route is `manager/app.py:1104-1127`.

**Files:**
- Modify: `manager/app.py` (delete the route and its banner comment), `README.md` (the architecture box at line 19, the endpoint table, any example)
- Modify: tests of the proxy (`grep -n 'v1/embeddings' tests/*.py`). Delete the proxy tests. Keep the `EmbeddingsClient` tests in `tests/test_embeddings.py`, which test the client the indexer uses.
- Test: `tests/test_endpoints.py`

**Interfaces:** none. `ManagerConfig.embeddings_url` stays, because the indexer uses it.

- [ ] **Step 1: Write the failing test**

```python
def test_embeddings_proxy_is_gone(client):   # use the module's existing TestClient fixture
    r = client.post("/v1/embeddings", json={"input": "x"})
    assert r.status_code == 404
```

- [ ] **Step 2: Run it to verify it fails**

Run: `PYTHONPATH=. /opt/llama/manager/venv/bin/python -m pytest tests/test_endpoints.py -k embeddings_proxy_is_gone -v -p no:cacheprovider`
Expected: FAIL. The route exists and tries the (blocked or refused) embeddings port, so the result isn't 404.

- [ ] **Step 3: Delete the route**

- [ ] **Step 4: Run the full suite**

Expected: all pass. `grep -n '/v1/embeddings' manager/ README.md` finds nothing in `manager/`.

- [ ] **Step 5: Commit**

```bash
git add manager/app.py README.md tests/<files>
git commit -m "refactor: remove the unused /v1/embeddings proxy"
```

---

### Task 5: Require `application/json` on every POST

The spec is Q8, which closes X1. **Scope note:** the spec's wording names `/swap` and `/collections/*/reindex`. X1's own text also names "queue GPU work", which a cross-site `text/plain` POST to `/v1/chat/completions` can do today. So this task gates **every POST**. Every known client sends JSON: agent_core uses httpx `json=` (including `json={}` for reindex at `agent_core/retrieval.py:81-87`), and opencode uses `@ai-sdk/openai-compatible`. The owner approves this widening when they approve the plan.

**Files:**
- Modify: `manager/app.py`. Add one HTTP middleware registered in the app factory next to the routes.
- Modify: tests that POST raw bodies without a Content-Type (Review Focus 2)
- Test: `tests/test_endpoints.py`

**Interfaces:**
- Produces: for `request.method == "POST"`, when the media type of `Content-Type` (the part before `;`, stripped, compared case-insensitively) isn't `application/json`, including when the header is missing, the middleware answers **415** with body `{"error": {"type": "unsupported_media_type", "message": "Content-Type must be application/json"}}`. The handler never runs. GET and other methods aren't affected.

- [ ] **Step 1: Write the failing tests**

```python
def test_text_plain_swap_is_refused_and_never_swaps(client, server):  # use the module's fixtures
    r = client.post("/swap", content=b'{"model":"test-model-q8"}',
                    headers={"content-type": "text/plain"})
    assert r.status_code == 415
    assert r.json()["error"]["type"] == "unsupported_media_type"
    # assert the swap path was not entered: no swap task was scheduled /
    # ensure_model_on_slot not called (use the module's existing swap mock)

def test_text_plain_chat_is_refused_and_never_queued(client, server):
    r = client.post("/v1/chat/completions", content=b'{"model":"test-model-q4","messages":[]}',
                    headers={"content-type": "text/plain"})
    assert r.status_code == 415
    assert all(s.queue.depth == 0 for s in server.slots.values())  # use the real queue-depth accessor

def test_missing_content_type_on_reindex_is_refused(client):
    r = client.post("/collections/skills/reindex", content=b"")
    assert r.status_code == 415

@pytest.mark.parametrize("ct", ["application/json; charset=utf-8", "Application/JSON"])
def test_json_media_type_with_params_or_case_is_accepted(client, ct):
    r = client.post("/swap", content=b'{"model":"no-such-model"}', headers={"content-type": ct})
    assert r.status_code != 415
```

- [ ] **Step 2: Run them to verify they fail**

Expected: the three refusal tests FAIL (today the requests return 202, 200 or 503, not 415). The parametrized acceptance test passes before and after. It guards Review Focus 1.

- [ ] **Step 3: Add the middleware** with `@app.middleware("http")`

- [ ] **Step 4: Run the full suite**

Expected: all pass, after updating the tests that posted raw bodies without a header.

- [ ] **Step 5: Commit**

```bash
git add manager/app.py tests/<files>
git commit -m "fix(security): require application/json on POST (X1)"
```

---

### Task 6: Harden `deploy-manager.sh`

The spec is Q36, findings S4–S6. Today the script runs `set -uo pipefail` without `-e` (`scripts/deploy-manager.sh:18`). It never checks `install` (90-91). It reports, but never refuses, a dirty `manager/`. It waits 15s for health (113). It runs no tests.

**Files:**
- Modify: `scripts/deploy-manager.sh`
- Create: `tests/test_deploy_script.py`

**Interfaces:**
- Produces: these env overrides, for tests: `DEPLOY_DEST` (default `/opt/llama/manager`), `DEPLOY_MGR_ENV` (default `/etc/llama/manager.env`), `HEALTH_WAIT` (default `60`, in seconds).
- Produces: exit codes. `2` means not a git checkout (as today). `3` means the tracked tree is dirty and this isn't `--check`. `4` means the test suite failed. `1` means other failures (as today).
- Produces: the order of checks. The git-checkout check, then the dirty check, then the root check, then the drift check. The dirty check runs before the root check so it can be tested without sudo.

- [ ] **Step 1: Write the failing test** in `tests/test_deploy_script.py`

```python
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

def test_deploy_refuses_a_dirty_tree_before_asking_for_root(tmp_path):
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(REPO), str(clone)], check=True)
    (clone / "manager" / "app.py").write_text("# dirty\n", encoding="utf-8")
    r = subprocess.run(
        ["bash", str(clone / "scripts/deploy-manager.sh")],
        capture_output=True, text=True,
        env={"PATH": "/usr/bin:/bin", "DEPLOY_DEST": str(tmp_path / "dest"),
             "DEPLOY_MGR_ENV": str(tmp_path / "none.env")},
    )
    assert r.returncode == 3, r.stdout + r.stderr
    assert "uncommitted" in r.stdout
```

- [ ] **Step 2: Run it to verify it fails**

Expected: FAIL with exit code 1, from today's "Deploying needs root" path.

- [ ] **Step 3: Change the script**

The changes:
- Use `set -euo pipefail`, and guard every command whose nonzero exit is legitimate (Review Focus 3).
- Add the env overrides.
- The dirty check is `git -C "$REPO" status --porcelain --untracked-files=no`. If it's non-empty and this isn't `--check`, print `refusing: uncommitted changes in the checkout` and exit 3. `--check` still reports.
- Fix the `${DIRTY:+…}` message, which always prints because `0` is non-empty.
- After the root check and before `install`, run the suite as the invoking user: `runuser -u "${SUDO_USER:-root}" -- env PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$REPO" "$DEST/venv/bin/python" -m pytest "$REPO/tests" -q -p no:cacheprovider`. Exit 4 on failure.
- `set -e` covers each `install`.
- Wait `HEALTH_WAIT` seconds instead of 15.

- [ ] **Step 4: Verify**

Run: the global suite command (all pass), then `bash -n scripts/deploy-manager.sh`, then `./scripts/deploy-manager.sh --check` against the real host.
Expected: `--check` exits 1 with drift listed (the branch differs from what's deployed) and no shell errors.

- [ ] **Step 5: Commit**

```bash
git add scripts/deploy-manager.sh tests/test_deploy_script.py
git commit -m "fix(deploy): set -e, refuse a dirty tree, run the suite, wait longer for health"
```

---

### Task 7: Lockfile from the deployed venv

The spec is Q36, finding S7. `manager/requirements.txt` has only lower bounds, so a fresh venv gets versions nobody has tested.

**Files:**
- Create: `manager/requirements.lock`, from `/opt/llama/manager/venv/bin/pip freeze`, with a header comment naming the command that regenerates it
- Modify: `scripts/deploy-manager.sh`. Sha-check `requirements.lock` alongside `requirements.txt`. When the lock changed, install it with `pip install -r requirements.lock`.
- Modify: `scripts/setup.sh:138-140`. Copy and install `requirements.lock`, not `requirements.txt`.
- Test: `tests/test_repo_consistency.py`

**Interfaces:** none new.

- [ ] **Step 1: Write the failing test**

```python
def test_every_top_level_requirement_is_pinned_in_the_lock():
    def name(line): return re.split(r"[<>=\[ ]", line.strip(), maxsplit=1)[0].lower().replace("_", "-")
    reqs = [l for l in (REPO / "manager/requirements.txt").read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    lock = {name(l): l for l in (REPO / "manager/requirements.lock").read_text().splitlines()
            if l.strip() and not l.startswith("#")}
    for r in reqs:
        assert name(r) in lock and "==" in lock[name(r)], r
```

- [ ] **Step 2: Run it to verify it fails**

Expected: FAIL with `FileNotFoundError` for `requirements.lock`.

- [ ] **Step 3: Generate the lock and update both scripts**

- [ ] **Step 4: Run the full suite and `bash -n` on both scripts**

Expected: all pass, and no syntax errors.

- [ ] **Step 5: Commit**

```bash
git add manager/requirements.lock scripts/deploy-manager.sh scripts/setup.sh tests/test_repo_consistency.py
git commit -m "build: pin dependencies with a lockfile from the deployed venv"
```

---

### Task 8: `scripts/doctor.sh`

The spec is Q36, finding Q12. S8 and S9 were host misconfigurations that a check like this would have shown immediately.

**Files:**
- Create: `scripts/doctor.sh`

**Interfaces:**
- Consumes: the slot naming `llama-server-<slot>.service` and `/etc/llama/llama-server-<slot>.env`, unless `SLOT_<NAME>_UNIT`/`_ENV` is set in `/etc/llama/manager.env` (`DOCTOR_MGR_ENV` overrides that path).
- Produces: a script that must run as root and is read-only. It prints one `PASS`/`FAIL <reason>` line per check and exits 1 if any check fails. The checks:
  1. `_llama-mgr` is in group `llama`.
  2. Each slot's env file in `SLOTS` exists and is owned by `_llama-mgr`.
  3. `sudo -l -U _llama-mgr` lists `/usr/bin/systemctl restart <unit>` for each slot unit.
  4. `runuser -u _llama-mgr -- test -w /var/log/llama`.
  5. Each `systemd/*.service` in the repo whose unit exists in `/etc/systemd/system/` has the same sha256.
  6. `/etc/systemd/system/llama-server-re.service`, `llama-server-batch.service` and `/etc/llama/llama-server-batch.env` are absent. These are leftovers from the rename and the batch removal.
  7. `scripts/deploy-manager.sh --check` exits 0.

- [ ] **Step 1: Write the script**

- [ ] **Step 2: Verify the syntax**

Run: `bash -n scripts/doctor.sh && ./scripts/doctor.sh; echo "exit=$?"`
Expected: no syntax error. As non-root it prints `doctor.sh must run as root` and exits 1. The real run is in Task 10. Before the cutover it's expected to FAIL checks 6 and 7, which is how the owner can see that it detects them.

- [ ] **Step 3: Commit**

```bash
git add scripts/doctor.sh
git commit -m "feat(ops): add doctor.sh host check"
```

---

### Task 9: Docs and repo housekeeping

The spec is Q37.

**Files:**
- Delete: `ARCHITECTURE.md` (`git rm`). Nothing outside `docs/` links to it (`grep -rn ARCHITECTURE.md --exclude-dir=docs .`).
- Modify: `docs/audit/findings.md`, Status section (lines 3-10):
  - I1 and I2 are fixed (PR #16, `fb55b33`, verified live 2026-10-29).
  - S8 and S9 are confirmed fixed on the host (checked 2026-10-30).
  - D3's unit-norm assumption is checked (a sample `:8082` vector has norm 1.0000).
  - X1, S4–S7, W1, D5 and Q12–Q13 are fixed by this phase.
  - Everything still open now points to `docs/design-review-2026-10-29.md` and its phase.
- Modify: `docs/design-review-brief.md`, the doc-trust table. `ARCHITECTURE.md` row: "deleted in phase 1".
- Host-side, done by the implementer in the worktree's parent repo, not committed:
  - `rm config/manager.env.orig` (untracked and stale: `HOST=0.0.0.0`, `PORT=11343`).
  - `git worktree remove /mnt/secondary/inference-server-wt-nslot && git branch -d feat/manager-n-slot`. Before running it, confirm `git log --oneline main..feat/manager-n-slot` is empty. If it isn't, stop and report.

- [ ] **Step 1: Make the edits**
- [ ] **Step 2: Verify**

Run: the global suite (all pass), `git worktree list` (one entry), `ls config/manager.env.orig` (no such file).

- [ ] **Step 3: Commit**

```bash
git add docs/audit/findings.md docs/design-review-brief.md && git rm -q ARCHITECTURE.md
git commit -m "docs: drop ARCHITECTURE.md; bring findings status up to date"
```

---

### Task 10: Host cutover (the owner runs this, after the PR merges)

Not for the implementer. The owner runs each block with `!` from a clean `main` checkout that has the merged phase. Every step can be undone by reversing it until the GGUF deletion, which is last and needs a separate yes.

- [ ] **Step 1: Deploy the manager and confirm it's healthy**

```
! cd /mnt/secondary/inference-server && git switch main && git pull --ff-only && sudo ./scripts/deploy-manager.sh
```

Expected: the suite runs and passes, the files install, and the manager is healthy, with `slot main` and `slot re` both healthy. The code is generic over slot names, so the old `SLOTS=main,re` keeps working until Step 3.

- [ ] **Step 2: Stage the new sudoers file and validate it before installing it**

```
! sudo cp /etc/sudoers.d/llama-manager /root/llama-manager.sudoers.bak && sed 's/llama-server-re\.service/llama-server-egpu.service/' /etc/sudoers.d/llama-manager | sudo tee /tmp/llama-manager.new >/dev/null && sudo visudo -cf /tmp/llama-manager.new && sudo install -m 440 -o root -g root /tmp/llama-manager.new /etc/sudoers.d/llama-manager && sudo cat /etc/sudoers.d/llama-manager
```

Expected: `parsed OK`, and the output lists both `llama-server.service` and `llama-server-egpu.service`.

- [ ] **Step 3: Move the egpu slot's unit, env file and logs, and update `manager.env`**

```
! sudo systemctl disable --now llama-server-re.service && sudo mv /etc/llama/llama-server-re.env /etc/llama/llama-server-egpu.env && sudo install -m 644 systemd/llama-server-egpu.service /etc/systemd/system/ && sudo rm /etc/systemd/system/llama-server-re.service && for f in log err; do [ -e /var/log/llama/llama-server-re.$f ] && sudo mv /var/log/llama/llama-server-re.$f /var/log/llama/llama-server-egpu.$f; done; sudo cp /etc/llama/manager.env /root/manager.env.bak && sudo sed -i -e 's/^SLOTS=main,re$/SLOTS=main,egpu/' -e 's/^SLOT_RE_/SLOT_EGPU_/' -e 's#llama-server-re\.#llama-server-egpu.#g' -e '/^BATCH_/d' /etc/llama/manager.env && grep -E '^(SLOTS|SLOT_EGPU_)' /etc/llama/manager.env
```

Expected: `SLOTS=main,egpu`, and every `SLOT_EGPU_*` key points at `-egpu` paths. The ownership of the moved env file stays `_llama-mgr`.

- [ ] **Step 4: Start the renamed unit and restart the manager**

```
! sudo systemctl daemon-reload && sudo systemctl enable --now llama-server-egpu.service && sudo systemctl restart llama-manager && sleep 5 && curl -s http://192.168.1.14:11434/status | python3 -m json.tool | grep -E '"(egpu|main|loaded_model|healthy)"'
```

Expected: slots `main` and `egpu`, both with their models and `healthy: true`. `egpu` may report unhealthy for a minute while the 19 GB model loads. Re-run the `curl` until it reports healthy.

- [ ] **Step 5: Remove the batch leftovers and run the doctor**

```
! sudo rm -f /etc/llama/llama-server-batch.env /etc/systemd/system/llama-server-batch.service && sudo systemctl daemon-reload && sudo ./scripts/doctor.sh
```

Expected: every check `PASS`, exit 0.

- [ ] **Step 6: Delete the dropped GGUFs (Q35, Q39). Ask for a separate yes before running this.**

```
! cd /opt/llama/models && sudo rm -v gemma-3-4b-it-q4_k_m.gguf Qwen3.5-35B-A3B-Q4_K_M.gguf Qwen3.5-35B-A3B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf gemma-4-E4B-it-Q4_K_M.gguf Devstral-Small-2-24B-Instruct-2512-Q5_K_M.gguf Qwen3-Coder-30B-A3B-Instruct-UD-Q4_K_XL.gguf && sudo rm -rv CYBER-FROST-3.8 && ls
```

Expected: these remain: `Qwen3.6-35B-A3B-UD-IQ4_NL_XL.{gguf,args}`, `Qwen3.8-27B-UD-Q6_K.{gguf,args}`, `gemma-4-26b-a4b-it-q4_k_m.gguf`, `nomic-embed-text-v1.5.Q8_0.gguf`, and the `Qwen3.8-Flash-Next/` directory. The owner decided on 2026-10-30 that this directory stays.

**Rollback before Step 6:** restore `/root/llama-manager.sudoers.bak` and `/root/manager.env.bak`, reverse the moves in Step 3, and redeploy the previous commit with `deploy-manager.sh` from a checkout at that commit.
