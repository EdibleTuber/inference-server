# Manager N-Slot Generalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the model manager front an arbitrary number of inference slots, so a third slot (the Tesla P40 returning as an eGPU) can be added by configuration rather than by code.

**Architecture:** Replace the hardcoded `main`/`batch` pair with an ordered collection of slot definitions built from environment variables. `app.py` constructs `self.slots` from that collection; `routing.py` iterates it in configured order; `/swap` validates targets against it. Legacy `LLAMA_SERVER_*` and `BATCH_SERVER_*` variable names keep working unchanged, so a deployed `manager.env` does not have to change in the same step as the code. Two independent defects are fixed alongside: single-GPU reporting in `gpu.py`, and `/status` reporting stale startup state.

**Tech Stack:** Python 3.12, FastAPI, httpx, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-09-23-dual-gpu-three-slot-design.md`

## Global Constraints

- Slot **order is significant** — it is routing priority. Carry it in an ordered structure (`tuple`/`list`), never rely on dict insertion order by accident.
- Existing env-var names (`LLAMA_SERVER_HOST/PORT/ENV/UNIT`, `QUEUE_LIMIT`, `BATCH_SERVER_HOST/PORT/ENV/UNIT`, `BATCH_QUEUE_LIMIT`) must keep working with unchanged semantics. A deployed `manager.env` that has never heard of this change must produce exactly today's two-slot behaviour.
- Model-name comparison semantics are unchanged — keep using `manager.names.same_model` / `display_name` per `docs/superpowers/specs/2026-06-29-model-name-normalization-design.md`.
- Tests assert **relationships**, not literals, wherever possible. `len(reported_slots) == len(configured_slots)` survives adding a slot; `len(...) == 2` does not.
- Every regression test must be **verified failing against the pre-fix code** before the fix lands. A test that passes before the fix proves nothing.
- Run the suite with `/opt/llama/manager/venv/bin/python -m pytest` from the repo root.
  The repo's own `.venv` has **no pytest installed** — only the deployed manager venv does.
  Baseline before any change: **150 passed**.

**On the code in this plan:** the blocks below are sketches written away from the files. Verify each against the real source before committing it; where a sketch and the codebase disagree, the codebase wins and deviating from this plan is correct.

---

## File Structure

| File | Responsibility | Change |
|---|---|---|
| `manager/slot_config.py` | `SlotConfig` value type + building the ordered slot list from env | **create** |
| `manager/config.py` | `ManagerConfig` gains `slots`; legacy fields retained | modify |
| `manager/app.py:58-81` | build `self.slots` from `config.slots` | modify |
| `manager/app.py:744-748` | `/swap` target validation against configured slots | modify |
| `manager/app.py:404-414` | `/status` reports fresh slot state | modify |
| `manager/routing.py:23-28` | iterate configured slots instead of naming two | modify |
| `manager/gpu.py:29-58` | report every GPU, not `lines[1]` | modify |
| `tests/test_slot_config.py` | unit tests for slot enumeration | **create** |
| `tests/test_routing.py` | add three-slot cases | modify |
| `tests/test_gpu.py` | multi-GPU parsing cases | modify |
| `tests/test_config.py` | slot-list construction and legacy compatibility | modify |
| `tests/test_endpoints.py` | `/status` freshness, `/swap` target validation | modify |
| `tests/conftest.py` | fixture gains a third slot where needed | modify |

---

### Task 1: `SlotConfig` and slot enumeration from environment

**Files:**
- Create: `manager/slot_config.py`
- Test: `tests/test_slot_config.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `SlotConfig` (frozen dataclass with fields `name: str`, `host: str`, `port: int`, `env_file: str`, `systemd_unit: str`, `queue_limit: int`, `device: str | None`), and `build_slots(getenv) -> tuple[SlotConfig, ...]`. Task 2 calls `build_slots`; Tasks 2–4 read these field names.

The naming scheme: `SLOTS` is a comma-separated ordered list of slot names, defaulting to `main,batch`. Each slot reads `SLOT_<NAME>_HOST` etc., falling back to the legacy variable for `main` and `batch` so existing deployments are unaffected. Adding a slot later is then a `manager.env` edit, not a code change.

- [ ] **Step 1: Write the failing tests**

```python
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
    slots = build_slots(_env({
        "LLAMA_SERVER_HOST": "10.0.0.1",
        "LLAMA_SERVER_PORT": "9001",
        "LLAMA_SERVER_ENV": "/etc/llama/main.env",
        "LLAMA_SERVER_UNIT": "llama-server.service",
        "QUEUE_LIMIT": "50",
    }))
    main = slots[0]
    assert (main.host, main.port) == ("10.0.0.1", 9001)
    assert main.env_file == "/etc/llama/main.env"
    assert main.systemd_unit == "llama-server.service"
    assert main.queue_limit == 50


def test_batch_reads_legacy_batch_server_vars():
    slots = build_slots(_env({
        "BATCH_SERVER_HOST": "127.0.0.1",
        "BATCH_SERVER_PORT": "8083",
        "BATCH_SERVER_UNIT": "llama-server-batch.service",
        "BATCH_QUEUE_LIMIT": "20",
    }))
    batch = next(s for s in slots if s.name == "batch")
    assert (batch.port, batch.queue_limit) == (8083, 20)
    assert batch.systemd_unit == "llama-server-batch.service"


def test_slots_var_controls_membership_and_order():
    slots = build_slots(_env({"SLOTS": "main,re,batch"}))
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
    slots = build_slots(_env({"SLOTS": " main , RE "}))
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


def test_slot_config_is_frozen():
    slots = build_slots(_env({}))
    with pytest.raises(Exception):
        slots[0].port = 1234
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `/opt/llama/manager/venv/bin/python -m pytest tests/test_slot_config.py -v`
Expected: collection error — `ModuleNotFoundError: No module named 'manager.slot_config'`.

- [ ] **Step 3: Write the implementation**

```python
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

# Per-slot field -> (legacy env var by slot name, default value).
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
    """SLOT_<NAME>_<FIELD> wins; then the legacy name; then the default."""
    prefixed = f"SLOT_{name.upper()}_{field.upper()}"
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
        env_file, _ = _lookup(getenv, name, "env", d.get("env_file"))
        unit, _ = _lookup(getenv, name, "unit", d.get("systemd_unit"))
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
```

Note: `_lookup` uses field key `"env"`/`"unit"` for the prefixed form but `_LEGACY` keys on `"env_file"`/`"systemd_unit"`. Reconcile these when implementing — pick one key set and use it in both, rather than transcribing this mismatch.

- [ ] **Step 4: Run the tests to verify they pass**

Run: `/opt/llama/manager/venv/bin/python -m pytest tests/test_slot_config.py -v`
Expected: all PASS.

- [ ] **Step 5: Run the whole suite to confirm nothing else moved**

Run: `/opt/llama/manager/venv/bin/python -m pytest -q`
Expected: no new failures versus the baseline captured before starting.

- [ ] **Step 6: Commit**

```bash
git add manager/slot_config.py tests/test_slot_config.py
git commit -m "feat(manager): slot definitions from configuration, not code

SLOTS names the slots and fixes their routing order; each slot reads
SLOT_<NAME>_* with the legacy LLAMA_SERVER_*/BATCH_SERVER_* names still
honoured for main and batch, so a deployed manager.env behaves exactly as
before."
```

---

### Task 2: `ManagerConfig` exposes the slot list

**Files:**
- Modify: `manager/config.py`
- Test: `tests/test_config.py`

**Interfaces:**
- Consumes: `build_slots`, `SlotConfig` from Task 1.
- Produces: `ManagerConfig.slots: tuple[SlotConfig, ...]`. Task 3 reads it.

Existing `ManagerConfig` fields stay exactly as they are. `tests/conftest.py` constructs `ManagerConfig` by keyword with every field, so removing fields would break the whole suite for no benefit. `slots` is added with a default so existing construction sites keep working.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_config.py
from manager.slot_config import SlotConfig


def test_from_env_populates_slots_in_order(monkeypatch):
    monkeypatch.setenv("SLOTS", "main,re,batch")
    monkeypatch.setenv("SLOT_RE_PORT", "8084")
    from manager.config import ManagerConfig
    cfg = ManagerConfig.from_env()
    assert [s.name for s in cfg.slots] == ["main", "re", "batch"]


def test_from_env_defaults_to_the_legacy_two_slots(monkeypatch):
    monkeypatch.delenv("SLOTS", raising=False)
    from manager.config import ManagerConfig
    cfg = ManagerConfig.from_env()
    assert [s.name for s in cfg.slots] == ["main", "batch"]


def test_legacy_ports_flow_into_slots(monkeypatch):
    monkeypatch.delenv("SLOTS", raising=False)
    monkeypatch.setenv("LLAMA_SERVER_PORT", "8081")
    monkeypatch.setenv("BATCH_SERVER_PORT", "8083")
    from manager.config import ManagerConfig
    cfg = ManagerConfig.from_env()
    by_name = {s.name: s for s in cfg.slots}
    assert by_name["main"].port == 8081
    assert by_name["batch"].port == 8083


def test_slots_default_is_empty_for_direct_construction(test_config):
    """Existing construction sites that pass no slots still work."""
    assert isinstance(test_config.slots, tuple)
```

- [ ] **Step 2: Run to verify failure**

Run: `/opt/llama/manager/venv/bin/python -m pytest tests/test_config.py -v`
Expected: FAIL — `AttributeError: 'ManagerConfig' object has no attribute 'slots'`.

- [ ] **Step 3: Implement**

In `manager/config.py`, import `SlotConfig` and `build_slots`, add the field to the dataclass after the existing ones (a default is required because earlier fields have none):

```python
    slots: tuple[SlotConfig, ...] = ()
```

and in `from_env`, pass `slots=build_slots()`. Add `slots` to the class docstring's Attributes list alongside the others.

- [ ] **Step 4: Run to verify pass**

Run: `/opt/llama/manager/venv/bin/python -m pytest tests/test_config.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add manager/config.py tests/test_config.py
git commit -m "feat(manager): ManagerConfig carries the ordered slot list"
```

---

### Task 3: `app.py` builds slots from configuration

**Files:**
- Modify: `manager/app.py:58-81`
- Test: `tests/test_endpoints.py`, `tests/conftest.py`

**Interfaces:**
- Consumes: `ManagerConfig.slots` from Task 2.
- Produces: `server.slots` keyed by configured slot name, built in configured order. Tasks 4–6 rely on this.

The literal dict at `app.py:59-77` and the two `ModelSwapper` assignments at `app.py:80-81` become a loop. `ModelSwapper(config, slot=...)` currently reads the slot's env file and unit off `config`; check how it resolves those and make it take them from the `SlotState` if it does not already — a swapper that still reads `config.batch_server_env` will write the wrong file for a third slot.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_endpoints.py
def test_server_builds_every_configured_slot(test_config):
    """Slot construction follows configuration, including a third slot."""
    from dataclasses import replace
    from manager.slot_config import SlotConfig
    from manager.app import ModelManagerServer

    cfg = replace(test_config, slots=(
        SlotConfig("main", "127.0.0.1", 8081, "/tmp/main.env", "main.service", 20),
        SlotConfig("batch", "127.0.0.1", 8083, "/tmp/batch.env", "batch.service", 20),
        SlotConfig("re", "127.0.0.1", 8084, "/tmp/re.env", "re.service", 30),
    ))
    server = ModelManagerServer(cfg)

    assert list(server.slots) == ["main", "batch", "re"]
    assert server.slots["re"].port == 8084
    assert server.slots["re"].queue.max_size == 30
    assert server.slots["re"].env_file == "/tmp/re.env"
    assert server.slots["re"].systemd_unit == "re.service"
    assert all(s.swapper is not None for s in server.slots.values())


def test_slot_count_follows_configuration(test_config):
    """A relationship, not a literal: adding a slot must not need a test edit."""
    from manager.app import ModelManagerServer
    server = ModelManagerServer(test_config)
    assert len(server.slots) == len(test_config.slots)
```

Update the `test_config` fixture in `tests/conftest.py` to populate `slots=` with `main` and `batch` matching the ports and temp env files it already sets, so `ModelManagerServer(test_config)` has something to build from.

- [ ] **Step 2: Run to verify failure**

Run: `/opt/llama/manager/venv/bin/python -m pytest tests/test_endpoints.py -k slot -v`
Expected: FAIL — only `main` and `batch` exist; `KeyError: 're'`.

- [ ] **Step 3: Implement**

Replace the literal construction with a loop over `config.slots`:

```python
        self.slots: dict[str, SlotState] = {}
        for sc in config.slots:
            self.slots[sc.name] = SlotState(
                name=sc.name,
                host=sc.host,
                port=sc.port,
                env_file=sc.env_file,
                systemd_unit=sc.systemd_unit,
                queue=RequestQueue(max_size=sc.queue_limit),
            )
        # Attach a swapper per slot (SlotState does not reference ModelSwapper).
        for slot in self.slots.values():
            slot.swapper = ModelSwapper(config, slot=slot)
```

`dict` preserves insertion order, and insertion follows `config.slots`, so `server.slots` is ordered. Task 4 depends on that; do not reorder it elsewhere.

- [ ] **Step 4: Run to verify pass**

Run: `/opt/llama/manager/venv/bin/python -m pytest tests/test_endpoints.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add manager/app.py tests/conftest.py tests/test_endpoints.py
git commit -m "feat(manager): build slots from configuration rather than literals"
```

---

### Task 4: Routing iterates configured slots

**Files:**
- Modify: `manager/routing.py:23-28`
- Test: `tests/test_routing.py`

**Interfaces:**
- Consumes: the ordered `slots` mapping from Task 3.
- Produces: `resolve_slot(model, slots)` unchanged in signature and return type.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_routing.py
def test_resolves_a_third_slot():
    slots = {
        "main": _slot("main", "coding-model"),
        "batch": _slot("batch", "small-model"),
        "re": _slot("re", "gemma-4-26b-a4b-it-q4_k_m"),
    }
    assert resolve_slot("gemma-4-26b-a4b-it-q4_k_m", slots) == "re"


def test_first_configured_slot_wins_when_duplicated():
    """Priority is iteration order, so the earliest configured slot wins."""
    slots = {"re": _slot("re", "shared"), "main": _slot("main", "shared")}
    assert resolve_slot("shared", slots) == "re"


def test_unknown_model_across_three_slots_returns_none():
    slots = {
        "main": _slot("main", "a"), "batch": _slot("batch", "b"), "re": _slot("re", "c"),
    }
    assert resolve_slot("d", slots) is None


def test_third_slot_comparison_is_normalised():
    slots = {"main": _slot("main", None), "re": _slot("re", "Gemma-4-26B")}
    assert resolve_slot("/models/gemma-4-26b.gguf", slots) == "re"
```

- [ ] **Step 2: Run to verify failure**

Run: `/opt/llama/manager/venv/bin/python -m pytest tests/test_routing.py -v`
Expected: `test_resolves_a_third_slot` FAILS returning `None` — the current body only inspects `main` and `batch`. `test_first_configured_slot_wins_when_duplicated` FAILS returning `"main"`.

- [ ] **Step 3: Implement**

```python
def resolve_slot(model: str, slots: dict[str, SlotState]) -> Optional[str]:
    """Return the slot that should handle the request, or None if unloaded.

    Walks the slots in iteration order, which the caller sets from
    configuration: the first configured slot holding the model wins. Returning
    None means the model is loaded nowhere and the caller answers 409 --
    implicit swaps happen only via POST /swap. Comparison is
    case/suffix/path-insensitive and None-safe.
    """
    for name, slot in slots.items():
        if slot is not None and same_model(model, slot.loaded_model):
            return name
    return None
```

- [ ] **Step 4: Run to verify pass**

Run: `/opt/llama/manager/venv/bin/python -m pytest tests/test_routing.py -v`
Expected: all PASS, including the pre-existing `test_resolve_model_on_both_prefers_main` (its dict lists `main` first, so order preserves the old result).

- [ ] **Step 5: Commit**

```bash
git add manager/routing.py tests/test_routing.py
git commit -m "feat(manager): route over configured slots in priority order"
```

---

### Task 5: `/swap` validates targets against configured slots

**Files:**
- Modify: `manager/app.py:726-767`
- Test: `tests/test_endpoints.py`

**Interfaces:**
- Consumes: `server.slots` from Task 3.
- Produces: no new symbols; the endpoint's 400 message now names the configured slots.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_endpoints.py
def test_swap_accepts_a_configured_third_slot(client_with_three_slots):
    """A configured slot name is a valid /swap target."""
    resp = client_with_three_slots.post(
        "/swap", json={"model": "test-model-q4", "target": "re"})
    assert resp.status_code != 400, resp.text


def test_swap_rejects_an_unconfigured_target(client_with_three_slots):
    resp = client_with_three_slots.post(
        "/swap", json={"model": "test-model-q4", "target": "nope"})
    assert resp.status_code == 400
    body = resp.json()["error"]["message"]
    for name in ("main", "batch", "re"):
        assert name in body, f"400 message should name configured slot {name!r}: {body}"
```

Add a `client_with_three_slots` fixture to `tests/conftest.py` mirroring the existing client fixture but with the three-slot config from Task 3, and with the swap machinery mocked the way the existing `/swap` tests in `tests/test_swap.py` do it — follow that file's existing mocking approach rather than inventing a new one.

- [ ] **Step 2: Run to verify failure**

Run: `/opt/llama/manager/venv/bin/python -m pytest tests/test_endpoints.py -k swap -v`
Expected: `test_swap_accepts_a_configured_third_slot` FAILS with 400 — `target` is checked against the literal tuple `("main", "batch")`.

- [ ] **Step 3: Implement**

Replace the hardcoded membership test at `app.py:744-748`:

```python
        target = body.get("target", "main")
        if target not in server.slots:
            valid = ", ".join(repr(n) for n in server.slots)
            return JSONResponse(
                status_code=400,
                content={"error": {"type": "invalid_target",
                                   "message": f"'target' must be one of: {valid}"}},
            )
```

Match the surrounding code's existing error-response idiom — check how neighbouring handlers build their 4xx bodies and follow that, rather than copying this sketch verbatim.

- [ ] **Step 4: Run to verify pass**

Run: `/opt/llama/manager/venv/bin/python -m pytest tests/test_endpoints.py tests/test_swap.py -v`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add manager/app.py tests/conftest.py tests/test_endpoints.py
git commit -m "feat(manager): /swap validates target against configured slots"
```

---

### Task 6: `gpu.py` reports every GPU

**Files:**
- Modify: `manager/gpu.py:29-58`
- Test: `tests/test_gpu.py`

**Interfaces:**
- Consumes: nothing from earlier tasks.
- Produces: `get_gpu_info()` returns `{"gpus": [ {index, name, vram_total_mb, vram_used_mb}, ... ]}` **plus** the existing top-level `name`/`vram_total_mb`/`vram_used_mb` keys mirroring the first GPU, so any existing `/status` consumer keeps working. Task 7 does not depend on this.

`manager/gpu.py:49` reads `lines[1]` only. With two cards that silently reports the first and hides the second — the exact opposite of what a dual-GPU deployment needs from its status page.

- [ ] **Step 1: Write the failing tests**

```python
# append to tests/test_gpu.py
from unittest.mock import patch
from manager.gpu import get_gpu_info

_TWO_GPU_CSV = (
    "name, memory.total [MiB], memory.used [MiB]\n"
    "Tesla PG500-216, 32768 MiB, 19039 MiB\n"
    "Tesla P40, 24576 MiB, 20710 MiB\n"
)


def _run(stdout):
    class R:
        pass
    r = R()
    r.stdout = stdout
    return r


def test_reports_every_gpu():
    with patch("manager.gpu.subprocess.run", return_value=_run(_TWO_GPU_CSV)):
        info = get_gpu_info()
    assert len(info["gpus"]) == 2
    assert info["gpus"][1]["name"] == "Tesla P40"
    assert info["gpus"][1]["vram_total_mb"] == 24576
    assert info["gpus"][1]["vram_used_mb"] == 20710


def test_gpus_carry_their_index():
    with patch("manager.gpu.subprocess.run", return_value=_run(_TWO_GPU_CSV)):
        info = get_gpu_info()
    assert [g["index"] for g in info["gpus"]] == [0, 1]


def test_top_level_keys_mirror_the_first_gpu():
    """Backwards compatibility for any consumer of the single-GPU shape."""
    with patch("manager.gpu.subprocess.run", return_value=_run(_TWO_GPU_CSV)):
        info = get_gpu_info()
    assert info["name"] == info["gpus"][0]["name"] == "Tesla PG500-216"
    assert info["vram_total_mb"] == info["gpus"][0]["vram_total_mb"]


def test_single_gpu_still_reports_one_entry():
    csv = "name, memory.total [MiB], memory.used [MiB]\nTesla PG500-216, 32768 MiB, 19039 MiB\n"
    with patch("manager.gpu.subprocess.run", return_value=_run(csv)):
        info = get_gpu_info()
    assert len(info["gpus"]) == 1
    assert info["name"] == "Tesla PG500-216"


def test_nvidia_smi_failure_returns_empty_gpu_list():
    with patch("manager.gpu.subprocess.run", side_effect=FileNotFoundError):
        info = get_gpu_info()
    assert info["gpus"] == []
    assert info["name"] == "unknown"
```

- [ ] **Step 2: Run to verify failure**

Run: `/opt/llama/manager/venv/bin/python -m pytest tests/test_gpu.py -v`
Expected: FAIL — `KeyError: 'gpus'`.

- [ ] **Step 3: Implement**

Parse every data row rather than `lines[1]`, keeping the existing never-raise contract and the `_unknown_gpu()` fallback. Add `index` from enumeration order. Keep the existing top-level keys populated from the first GPU, or from `_unknown_gpu()` when the list is empty.

Keep the `--query-gpu` field list as it is. Consider adding `uuid` here only if Task 7 turns out to want it; do not add it speculatively.

- [ ] **Step 4: Run to verify pass**

Run: `/opt/llama/manager/venv/bin/python -m pytest tests/test_gpu.py -v`
Expected: all PASS.

- [ ] **Step 5: Check no consumer broke**

Run: `cd /mnt/secondary/inference-server && grep -rn "vram_total_mb\|vram_used_mb\|\[.gpu.\]" --include=*.py --include=*.md . | grep -v tests/`
Expected: review each hit; the mirrored top-level keys should cover them. Resolve the third open question in the spec here — record what you found.

- [ ] **Step 6: Commit**

```bash
git add manager/gpu.py tests/test_gpu.py
git commit -m "fix(manager): report every GPU, not just the first

gpu.py parsed only lines[1] of nvidia-smi output, so a second card was
silently invisible. Top-level keys still mirror GPU 0 for existing consumers."
```

---

### Task 7: `/status` reports fresh slot state

**Files:**
- Modify: `manager/app.py:404-414`
- Test: `tests/test_endpoints.py`

**Interfaces:**
- Consumes: `server.reprobe_all_slots()` (`manager/app.py:131`), which already exists.
- Produces: no new symbols.

This is spec finding F8, confirmed on the live server: `/status` reported both slots `loaded_model: null, healthy: false` while both backends were serving, and a single chat request corrected it. The startup probe (`app.py:336-347`) runs once; `reprobe_all_slots` is reachable only from the chat 409 path (`app.py:478`); `/status` never probes.

The spec leaves open whether to re-probe inside `/status` or to run a periodic background probe. **Re-probe inside `/status`** is the choice here: it is small, it has no new lifecycle to get wrong, and `reprobe_all_slots` already skips any slot mid-swap so it cannot stall behind a model load. A periodic probe is the better long-term answer if more passive readers appear; note that in the commit message rather than building it speculatively.

- [ ] **Step 1: Write the failing test**

```python
# append to tests/test_endpoints.py
def test_status_reprobes_before_reporting(test_config):
    """F8: /status must not serve stale startup state.

    Fails against the pre-fix code, which reports whatever the startup probe
    left behind until some chat request happens to trigger a reprobe.
    """
    from unittest.mock import AsyncMock, patch
    from fastapi.testclient import TestClient
    from manager.app import create_app

    with patch("manager.app.ModelManagerServer.reprobe_all_slots",
               new_callable=AsyncMock) as reprobe:
        app = create_app(test_config)
        with TestClient(app) as client:
            reprobe.reset_mock()          # ignore any startup-time calls
            resp = client.get("/status")
    assert resp.status_code == 200
    reprobe.assert_awaited_once()


def test_status_still_reports_every_slot(test_config):
    from fastapi.testclient import TestClient
    from manager.app import create_app
    app = create_app(test_config)
    with TestClient(app) as client:
        body = client.get("/status").json()
    assert len(body["slots"]) == len(test_config.slots)
```

Check `create_app`'s real signature before using it — `manager/app.py:390` builds the `FastAPI` instance and the manager unit invokes it via `--factory`, so it may take no arguments and read config itself. Adapt the fixture to however the existing endpoint tests build their client.

- [ ] **Step 2: Run to verify failure**

Run: `/opt/llama/manager/venv/bin/python -m pytest tests/test_endpoints.py -k status -v`
Expected: `test_status_reprobes_before_reporting` FAILS — `reprobe.assert_awaited_once()` raises because `/status` never calls it. **Confirm this failure before writing the fix**; a passing test here would mean the test is not exercising the bug.

- [ ] **Step 3: Implement**

```python
    @app.get("/status")
    async def status():
        # Refresh before reporting. The startup probe runs once, and until this
        # was added the only other reprobe was on the chat 409 path -- so a
        # backend that became ready later showed here as a total outage for the
        # life of the process, while requests to it succeeded. Slots mid-swap
        # are skipped inside reprobe_all_slots, so this cannot stall behind a
        # model load.
        await server.reprobe_all_slots()
        gpu = await get_gpu_info_async()
        return {
            "slots": {
                name: slot.to_status_dict()
                for name, slot in server.slots.items()
            },
            "gpu": gpu,
            "uptime_seconds": int(time.time() - _start_time),
        }
```

- [ ] **Step 4: Run to verify pass**

Run: `/opt/llama/manager/venv/bin/python -m pytest tests/test_endpoints.py -v`
Expected: all PASS.

- [ ] **Step 5: Verify against the live server**

Restart the manager and check `/status` *before* sending any chat request:

```bash
sudo systemctl restart llama-manager
curl -s http://192.168.1.14:11434/status | python3 -m json.tool
```

Expected: slots report their real `loaded_model` and `healthy: true` with no chat request first. This is the behaviour the live experiment in spec F8 showed was missing — an API returning 200 is not the fix; the reported state being correct is.

- [ ] **Step 6: Commit**

```bash
git add manager/app.py tests/test_endpoints.py
git commit -m "fix(manager): /status reprobes instead of serving startup state

reprobe_all_slots existed but was reachable only from the chat 409 path, so
/status reported a total outage while both backends served correctly. A
periodic background probe would serve every future passive reader and is the
better answer if more appear; this is the smaller fix for the one reader we
have."
```

---

### Task 8: Pin each slot to a physical GPU by UUID

**Files:**
- Modify: `systemd/llama-server.service`, `config/llama-server.env`
- Create: `systemd/llama-server-re.service`, `config/llama-server-re.env`
- Modify: `README.md` (slot/device table), `config/manager.env`

**Interfaces:**
- Consumes: `SlotConfig.device` from Task 1.
- Produces: no Python symbols. This task's deliverable is configuration plus a verification procedure.

**This task is specified, not coded, deliberately.** Its failure mode is silent: bind the wrong card and the model loads, serves, and looks entirely healthy while sitting on the wrong GPU. Unexecuted code written into a plan is the wrong tool for that, so the requirements and the checks are below and the implementer writes the units against the real files and the real hardware.

**The invariant:** restarting any slot's service, in any order, with both cards present, must never place a model on another slot's card.

**Why ordinals cannot carry it.** `--device CUDA0` names an enumeration position, not a card. CUDA's default device order is not guaranteed to match PCI bus order, and the eGPU arrives on a bus that did not exist when the current pin was written. `config/llama-server.env:10` still says "CUDA0 is the Tesla P40" — it is now the V100, which is exactly this bug already having happened once, harmlessly, because only one card was present.

**Requirements:**

1. Each GPU-backed unit sets `CUDA_VISIBLE_DEVICES` to its card's **UUID** (`nvidia-smi --query-gpu=uuid --format=csv,noheader` gives `GPU-<uuid>`), so the process sees exactly one card and `--device CUDA0` inside it is unambiguous.
2. Ordering matters in the unit file for the same reason it did in `961a93d`: systemd applies `Environment=` and `EnvironmentFile=` in file order, later assignments winning. Keep any unit-level default **before** `EnvironmentFile=`, and say so in a comment, as `systemd/llama-server.service` already does for `DEVICE`.
3. Correct the stale comments at `config/llama-server.env:10,17,27` that describe CUDA0 as a Tesla P40 with 24GB.
4. The `re` slot's env file mirrors the main slot's structure. Its `CTX_SIZE` comes from the measured `llama_kv_cache` line per spec F4/F5 — not from the 21.8 GiB projection.
5. `config/manager.env` gains `SLOTS=main,batch,re` and the `SLOT_RE_*` block. Until the eGPU is installed, leave `re` out of `SLOTS` so the manager does not front a slot with no backend.

**Verification — the invariant needs an actual test, not an assumption:**

- [ ] Record each card's UUID and its PCI bus id.
- [ ] Start both GPU services; for each, confirm from its own log which card it got. `ggml_cuda_init` prints the device name and, with `CUDA_VISIBLE_DEVICES` set correctly, must report exactly one device.
- [ ] Restart them in the reverse order and confirm each still has the same card. Order-dependence is the failure this is guarding.
- [ ] Reboot the host and confirm again, since enumeration can differ across a cold boot.
- [ ] Physically confirm the mapping rather than trusting the name: with both cards present the names differ (Tesla PG500-216 vs Tesla P40), so the log line is sufficient evidence here. Record it.

- [ ] **Commit**

```bash
git add systemd/ config/ README.md
git commit -m "feat(ops): pin each slot to a GPU by UUID, not by ordinal

CUDA0 names an enumeration position, not a card, and the position moves when
the eGPU appears. CUDA_VISIBLE_DEVICES=GPU-<uuid> gives each service exactly
one card. Also corrects env comments that still described CUDA0 as a P40 with
24GB -- it has been the V100 since the swap."
```

---

## Self-Review

**Spec coverage.** Phase 3 of the spec maps to tasks as follows: slot configuration → Tasks 1–3; routing → Task 4; `/swap` guard surface → Task 5; GPU reporting → Task 6; F8 → Task 7; device pinning invariant → Task 8. Spec open question 3 (single-GPU `/status` consumers) is resolved in Task 6 Step 5; open question 4 (re-probe vs periodic) is decided in Task 7.

**Not covered here, by design.** Spec F9 — `agent_core` auto-swapping on a 409 — is a policy question about which slot a request may evict. It needs a decision before it needs code, and it belongs with PARE and the coding agent's configuration rather than in this refactor. Carry it forward. Phases 1, 2 and 4 are ops work with their own sequence in the spec and are not part of this plan.

**Placeholders.** None: every code step carries real code or an explicit instruction to follow an existing pattern in a named file.

**Type consistency.** `SlotConfig` field names (`name`, `host`, `port`, `env_file`, `systemd_unit`, `queue_limit`, `device`) are used identically in Tasks 1, 2, 3 and 8. `build_slots` is referenced with the same signature in Tasks 1 and 2. One deliberate inconsistency is flagged inline in Task 1 Step 3 — `_lookup`'s field keys versus `_LEGACY`'s — with instructions to reconcile rather than transcribe.
