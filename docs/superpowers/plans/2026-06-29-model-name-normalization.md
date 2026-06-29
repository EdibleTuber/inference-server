# Model-Name Normalization + Device-Pin + 409 Policy Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make model-name handling consistent (store a clean display form, compare case/suffix/path-insensitively), persist the main slot's `--device CUDA0` pin fail-loud, replace the implicit unknown-model→main swap with an explicit 409, and gate the post-5xx reprobe on the swap lock.

**Architecture:** A new pure module `manager/names.py` provides the single normalization vocabulary (`display_name`, `match_key`, `same_model`). Every writer of `SlotState.loaded_model` stores the display form; every routing/guard comparison goes through `same_model`; file resolution matches case-folded keys to real on-disk paths. The chat path returns 409 for a not-loaded model instead of silently restarting main; main now boots pre-loaded via its env (like batch). Systemd/env changes persist the device pin.

**Tech Stack:** Python 3.12, FastAPI, httpx, pytest / pytest-asyncio. Backend processes are llama.cpp `llama-server` under systemd.

## Global Constraints

- Model names are assumed **ASCII**; `casefold()` is used for case-insensitivity (locale-independent). Add a one-line comment stating the ASCII assumption in `names.py`.
- On-disk GGUF files must use a **lowercase `.gguf`** extension (the directory glob is case-sensitive). Document this in `model_path`.
- The main device pin is **fail-loud**: an explicit `--device` with **no** silent fallback, and **no `MemoryMax`** on the main unit.
- Reported model identity is the **clean display form**: basename, no `.gguf`, original on-disk case. Comparisons are case-folded.
- `SlotState.loaded_model` is `Optional[str]` and is legitimately `None`; **every** comparison must treat `None` as "never matches" (handled centrally by `same_model`).
- Follow existing repo patterns: modules under `manager/`, tests as `tests/test_<module>.py` using the `test_config`/`client` fixtures in `tests/conftest.py`.
- End every commit message with the repo's trailers:
  `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>` and the `Claude-Session:` line (see existing commits). Step commit blocks below show only the subject for brevity.
- Spec: `docs/superpowers/specs/2026-06-29-model-name-normalization-design.md`.

---

### Task 1: `manager/names.py` — normalization helpers

**Files:**
- Create: `manager/names.py`
- Test: `tests/test_names.py`

**Interfaces:**
- Produces:
  - `display_name(raw: str | None) -> str | None` — basename, drop one trailing `.gguf` (case-insensitive), original case; `None` for falsy/empty result.
  - `match_key(name: str | None) -> str | None` — `display_name` then `.casefold()`; `None` if display is `None`.
  - `same_model(requested: str | None, loaded: str | None) -> bool` — `True` iff both have equal non-`None` `match_key`s.

- [ ] **Step 1: Write the failing test**

Create `tests/test_names.py`:

```python
"""Tests for manager/names.py: pure model-name normalization helpers."""
from manager.names import display_name, match_key, same_model


def test_display_name_plain_stem():
    assert display_name("gemma-4-E4B-it-Q4_K_M") == "gemma-4-E4B-it-Q4_K_M"


def test_display_name_strips_gguf():
    assert display_name("gemma-4-E4B-it-Q4_K_M.gguf") == "gemma-4-E4B-it-Q4_K_M"


def test_display_name_strips_gguf_case_insensitive():
    assert display_name("Model.GGUF") == "Model"


def test_display_name_strips_directory():
    assert display_name("/mnt/secondary/llama-models/foo-q4.gguf") == "foo-q4"


def test_display_name_strips_only_one_extension():
    # Path.stem semantics: only the final .gguf is removed.
    assert display_name("x.gguf.gguf") == "x.gguf"


def test_display_name_empty_and_none_are_none():
    assert display_name("") is None
    assert display_name(None) is None
    assert display_name("foo.gguf/") is None  # trailing slash -> empty basename


def test_match_key_folds_case_and_suffix_and_path():
    assert match_key("/p/Gemma-4-E4B.GGUF") == "gemma-4-e4b"
    assert match_key(None) is None
    assert match_key("") is None


def test_same_model_tolerant():
    assert same_model("gemma-4-e4b-it-q4_k_m", "gemma-4-E4B-it-Q4_K_M") is True
    assert same_model("/p/gemma-4-E4B-it-Q4_K_M.gguf", "gemma-4-e4b-it-q4_k_m") is True


def test_same_model_none_never_matches():
    assert same_model("anything", None) is False
    assert same_model(None, None) is False
    assert same_model("", "real-model") is False
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_names.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'manager.names'`.

- [ ] **Step 3: Write minimal implementation**

Create `manager/names.py`:

```python
"""
Model-name normalization for the dual-slot manager.

One vocabulary, used everywhere a model name is stored or compared:
  - display_name: the clean form we STORE and REPORT (basename, no .gguf, original case)
  - match_key:    the case-folded key we COMPARE on
  - same_model:   the single comparison primitive (None-safe)

Model filenames are assumed ASCII; casefold() is locale-independent.
"""
from __future__ import annotations


def display_name(raw: str | None) -> str | None:
    """Storage/report form: basename, drop one trailing .gguf, keep original case.

    Returns None for falsy/empty input so the empty->None invariant lives in ONE
    place (e.g. a trailing slash like 'foo.gguf/' yields an empty basename).
    """
    if not raw:
        return None
    name = raw.rsplit("/", 1)[-1]            # strip any directory component
    if name.lower().endswith(".gguf"):       # case-insensitive: handles .GGUF
        name = name[: -len(".gguf")]
    return name or None


def match_key(name: str | None) -> str | None:
    """Tolerant comparison key: display form, case-folded. None-total."""
    d = display_name(name)
    return d.casefold() if d is not None else None


def same_model(requested: str | None, loaded: str | None) -> bool:
    """The single routing/guard comparison. None loaded_model never matches."""
    lk = match_key(loaded)
    return lk is not None and lk == match_key(requested)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_names.py -v`
Expected: PASS (all 9 tests).

- [ ] **Step 5: Commit**

```bash
git add manager/names.py tests/test_names.py
git commit -m "feat(manager): add model-name normalization helpers"
```

---

### Task 2: Writers store the display form (`probe`, `mark_swapped` caller, short-circuit)

**Files:**
- Modify: `manager/slots.py:85-92` (`probe`)
- Modify: `manager/app.py:108-135` (`ensure_model_on_slot` — short-circuit + the `mark_swapped` call)
- Test: `tests/test_slots.py`, `tests/test_endpoints.py`

**Interfaces:**
- Consumes: `display_name`, `same_model` from `manager.names` (Task 1).
- Produces: after a swap, `slot.loaded_model` holds `display_name(<resolved real path>)` (authoritative on-disk stem); the short-circuit skips a swap when `same_model(request, loaded)`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_slots.py`:

```python
@pytest.mark.asyncio
async def test_probe_normalizes_full_path_id():
    """If llama-server reports a full path, probe stores the clean stem."""
    slot = _make_slot()
    client = MagicMock()
    mock_response = MagicMock(status_code=200)
    mock_response.json.return_value = {
        "data": [{"id": "/mnt/secondary/llama-models/gemma-4-26b-a4b-it-q4_k_m.gguf"}],
    }
    client.get = AsyncMock(return_value=mock_response)

    await slot.probe(client)
    assert slot.loaded_model == "gemma-4-26b-a4b-it-q4_k_m"
    assert slot.healthy is True
```

Add to `tests/test_endpoints.py`:

```python
@pytest.mark.asyncio
async def test_mark_swapped_stores_ondisk_stem_not_request_casing(test_config):
    """A swap requested with odd casing/suffix is stored as the real on-disk stem."""
    from manager.app import ServerState
    server = ServerState(test_config)
    server.slots["main"].swapper.swap_to = AsyncMock(return_value=True)

    ok = await server.ensure_model_on_slot("main", "TEST-MODEL-Q4.gguf")
    assert ok is True
    # tmp_models_dir has 'test-model-q4.gguf' (lowercase) -> canonical stem stored.
    assert server.slots["main"].loaded_model == "test-model-q4"


@pytest.mark.asyncio
async def test_ensure_model_skips_swap_on_case_variant(test_config):
    """Already-loaded model requested with different case must NOT re-swap."""
    from manager.app import ServerState
    server = ServerState(test_config)
    server.slots["main"].healthy = True
    server.slots["main"].loaded_model = "test-model-q4"
    server.slots["main"].swapper.swap_to = AsyncMock(return_value=True)

    result = await server.ensure_model_on_slot("main", "TEST-MODEL-Q4")
    assert result is True
    server.slots["main"].swapper.swap_to.assert_not_called()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_slots.py::test_probe_normalizes_full_path_id tests/test_endpoints.py::test_mark_swapped_stores_ondisk_stem_not_request_casing tests/test_endpoints.py::test_ensure_model_skips_swap_on_case_variant -v`
Expected: FAIL — path id stored verbatim minus `.gguf` (still a path); `loaded_model` is `"TEST-MODEL-Q4.gguf"` raw; case-variant short-circuit re-swaps.

- [ ] **Step 3: Implement**

In `manager/slots.py`, add the import near the top (with the other `manager` imports):

```python
from manager.names import display_name
```

Replace the normalization block in `probe` (currently `slots.py:85-92`):

```python
        raw_id = entries[0].get("id") or ""
        # Normalize to the clean display form (basename, no .gguf, original case).
        # display_name also collapses a full path to its stem and "" -> None.
        self.loaded_model = display_name(raw_id)
        self.healthy = bool(self.loaded_model)
```

In `manager/app.py`, add to the existing `from manager.names import ...` (create it if absent, near the other `manager` imports):

```python
from manager.names import display_name, same_model
```

In `ensure_model_on_slot` (`app.py:108-135`), change the short-circuit (line 116) and the `mark_swapped` call (line 129):

```python
            if slot.healthy and same_model(model_name, slot.loaded_model):
                return True
```

```python
            success = await slot.swapper.swap_to(path)
            if success:
                slot.mark_swapped(display_name(path))   # store the real on-disk stem
                return True
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_slots.py tests/test_endpoints.py -v`
Expected: PASS, including the three new tests and all pre-existing slot/endpoint tests (e.g. `test_probe_strips_gguf_suffix`, `test_ensure_model_on_slot_skips_swap_if_already_loaded`).

- [ ] **Step 5: Commit**

```bash
git add manager/slots.py manager/app.py tests/test_slots.py tests/test_endpoints.py
git commit -m "feat(manager): store canonical display name on probe and swap"
```

---

### Task 3: Case-folded file resolution in `model_path`

**Files:**
- Modify: `manager/app.py:92-96` (`model_path`)
- Test: `tests/test_endpoints.py`

**Interfaces:**
- Consumes: `display_name`, `match_key` from `manager.names`.
- Produces: `model_path(name)` returns the real on-disk path for an exact stem, else a case-folded match; `None` if unresolved. Warns on a `match_key` collision.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_endpoints.py` (top-level, near the other endpoint tests):

```python
from pathlib import Path as _Path  # if not already imported at module top


def test_model_path_exact_and_suffix(test_config):
    from manager.app import ServerState
    server = ServerState(test_config)
    assert server.model_path("test-model-q4").endswith("test-model-q4.gguf")
    assert server.model_path("test-model-q4.gguf").endswith("test-model-q4.gguf")


def test_model_path_case_insensitive(test_config):
    from manager.app import ServerState
    server = ServerState(test_config)
    assert server.model_path("TEST-MODEL-Q4").endswith("test-model-q4.gguf")


def test_model_path_unknown_is_none(test_config):
    from manager.app import ServerState
    server = ServerState(test_config)
    assert server.model_path("nope") is None
    assert server.model_path("") is None


def test_model_path_collision_is_deterministic_and_warns(test_config, caplog):
    from manager.app import ServerState
    d = test_config.models_dir
    _Path(d, "Dup-Model.gguf").touch()
    _Path(d, "dup-model.gguf").touch()
    server = ServerState(test_config)
    # Exact-case match wins, no warning needed.
    assert server.model_path("Dup-Model").endswith("Dup-Model.gguf")
    assert server.model_path("dup-model").endswith("dup-model.gguf")
    # A folded-only variant resolves to the sorted-first file ('D' < 'd') and warns.
    with caplog.at_level("WARNING"):
        resolved = server.model_path("DUP-MODEL")
    assert resolved.endswith("Dup-Model.gguf")
    assert any("collision" in r.message.lower() for r in caplog.records)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_endpoints.py -k model_path -v`
Expected: FAIL — `test_model_path_case_insensitive` returns `None` (old exact `f"{name}.gguf"` check), collision test has no warning.

- [ ] **Step 3: Implement**

Replace `model_path` in `manager/app.py:92-96`:

```python
    def model_path(self, model_name: str) -> str | None:
        """Resolve a model name to its file path in MODELS_DIR.

        Case- and .gguf-suffix-insensitive. An exact-case stem match wins;
        otherwise a case-folded (match_key) match is used. NOTE: the glob is
        case-sensitive, so on-disk files MUST use a lowercase '.gguf' extension.
        """
        requested = display_name(model_name)
        if requested is None:
            return None
        models_dir = Path(self._config.models_dir)
        if not models_dir.exists():
            return None
        by_key: dict[str, Path] = {}
        for p in sorted(models_dir.glob("*.gguf")):
            if p.stem == requested:                 # exact-case match wins
                return str(p)
            key = match_key(p.stem)
            if key in by_key:
                logger.warning(
                    "model_path: case-collision on %r; keeping %s, ignoring %s",
                    key, by_key[key].name, p.name,
                )
                continue
            by_key[key] = p
        match = by_key.get(match_key(model_name))
        return str(match) if match is not None else None
```

Ensure `manager/app.py` imports `match_key` (extend the Task 2 import):

```python
from manager.names import display_name, match_key, same_model
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_endpoints.py -k model_path -v`
Expected: PASS (4 tests). Also run `pytest tests/test_endpoints.py -v` — `test_chat_completions_unknown_model` still 404s (`nonexistent-model` resolves to nothing).

- [ ] **Step 5: Commit**

```bash
git add manager/app.py tests/test_endpoints.py
git commit -m "feat(manager): case-folded model file resolution with exact-match priority"
```

---

### Task 4: `resolve_slot` → Optional + `same_model`; chat path returns 409 for not-loaded

**Files:**
- Modify: `manager/routing.py:1-26`
- Modify: `manager/app.py:418-437` (chat routing + unhealthy guard)
- Test: `tests/test_routing.py` (update 2 tests, add tolerance tests), `tests/test_endpoints.py` (409 + None-safety)

**Interfaces:**
- Consumes: `same_model` from `manager.names`.
- Produces: `resolve_slot(model, slots) -> Optional[str]` — slot name if the model is loaded somewhere, else `None`. The chat handler maps `None` → HTTP 409 `{"error": {"type": "model_not_loaded", ...}}`.

- [ ] **Step 1: Update the broken routing tests and add tolerance/None tests**

In `tests/test_routing.py`, **replace** `test_resolve_model_on_neither_returns_main` and `test_resolve_empty_model_returns_main` with:

```python
def test_resolve_model_on_neither_returns_none():
    slots = {"main": _slot("main", "model-A"), "batch": _slot("batch", "model-B")}
    assert resolve_slot("unknown-C", slots) is None


def test_resolve_empty_model_returns_none():
    slots = {"main": _slot("main", None), "batch": _slot("batch", None)}
    assert resolve_slot("", slots) is None


def test_resolve_is_case_and_suffix_and_path_insensitive():
    slots = {"main": _slot("main", "gemma-4-E4B-it-Q4_K_M"), "batch": _slot("batch", None)}
    assert resolve_slot("gemma-4-e4b-it-q4_k_m", slots) == "main"
    assert resolve_slot("GEMMA-4-E4B-IT-Q4_K_M.gguf", slots) == "main"
    assert resolve_slot("/mnt/models/gemma-4-E4B-it-Q4_K_M.gguf", slots) == "main"


def test_resolve_none_loaded_model_is_safe():
    """A slot with loaded_model=None must not crash the comparison."""
    slots = {"main": _slot("main", None), "batch": _slot("batch", None)}
    assert resolve_slot("anything", slots) is None
```

Add 409 + None-safety tests to `tests/test_endpoints.py`:

```python
def test_chat_completions_409_when_not_loaded(client):
    """A model that exists on disk but is loaded on neither slot -> 409 (no implicit swap)."""
    app = client.app
    server = app.state.server
    server.slots["main"].loaded_model = "test-model-q4"
    server.slots["main"].healthy = True
    server.slots["batch"].loaded_model = None
    server.slots["batch"].healthy = False

    r = client.post("/v1/chat/completions", json={
        "model": "test-model-q8",  # on disk, not loaded anywhere
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 409
    assert r.json()["error"]["type"] == "model_not_loaded"


def test_chat_completions_cold_start_no_crash(client):
    """Both slots unloaded (loaded_model=None, unhealthy): request must not 500."""
    app = client.app
    server = app.state.server
    server.slots["main"].loaded_model = None
    server.slots["main"].healthy = False
    server.slots["batch"].loaded_model = None
    server.slots["batch"].healthy = False

    r = client.post("/v1/chat/completions", json={
        "model": "test-model-q4",  # exists on disk, not loaded
        "messages": [{"role": "user", "content": "hi"}],
    })
    assert r.status_code == 409  # not a 500
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_routing.py tests/test_endpoints.py -k "resolve or 409 or cold_start" -v`
Expected: FAIL — `resolve_slot` still returns `"main"` for unknown/empty; chat path enqueues instead of 409.

- [ ] **Step 3: Implement**

Replace `manager/routing.py` entirely:

```python
"""
Pure routing decisions for the dual-slot model manager.

Given a requested model name and current per-slot state, return which slot
should handle the request, or None if the model is loaded on neither slot.
Pure function: no I/O, no side effects.
"""
from typing import Optional

from manager.names import same_model
from manager.slots import SlotState


def resolve_slot(model: str, slots: dict[str, SlotState]) -> Optional[str]:
    """Return the slot that should handle the request, or None if unloaded.

    Rules:
      1. If the model is loaded on 'main', return 'main'.
      2. Else if loaded on 'batch', return 'batch'.
      3. Else return None. The caller returns 409; implicit swaps happen only
         via POST /swap. Comparison is case/suffix/path-insensitive and None-safe.
    """
    main = slots.get("main")
    if main is not None and same_model(model, main.loaded_model):
        return "main"
    batch = slots.get("batch")
    if batch is not None and same_model(model, batch.loaded_model):
        return "batch"
    return None
```

In `manager/app.py`, replace the routing block at `app.py:418-437`:

```python
        slot_name = resolve_slot(model_name, server.slots)
        if slot_name is None:
            # Loaded on neither slot. Do NOT implicitly restart main; tell the
            # caller to load it explicitly. (Implicit swaps live only on POST /swap.)
            return JSONResponse(
                {"error": {
                    "type": "model_not_loaded",
                    "message": f"model {model_name} not loaded on any slot; use POST /swap",
                }},
                status_code=409,
            )
        slot = server.slots[slot_name]

        if not slot.healthy and same_model(model_name, slot.loaded_model):
            # Model IS loaded on this slot but the slot is unhealthy: typed 503.
            return JSONResponse(
                {"error": {
                    "type": f"{slot_name}_unavailable",
                    "message": f"{slot_name} slot not ready",
                }},
                status_code=503,
                headers={"Retry-After": "5"},
            )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_routing.py tests/test_endpoints.py -v`
Expected: PASS. Confirm pre-existing chat tests still pass: `test_chat_completions_routes_batch_model` (q8 on batch → batch), `test_chat_completions_503_on_batch_unhealthy` (503), `test_chat_completions_routes_main_for_unknown` (q4 on main → main; note: this test's name is now stale — the model it sends is loaded on main).

- [ ] **Step 5: Rename the now-stale test for clarity, then commit**

In `tests/test_endpoints.py` rename `test_chat_completions_routes_main_for_unknown` → `test_chat_completions_routes_main_when_loaded_on_main` and update its docstring to `"""Model loaded on main routes to main."""`.

```bash
git add manager/routing.py manager/app.py tests/test_routing.py tests/test_endpoints.py
git commit -m "feat(manager): 409 for not-loaded model; None-safe folded routing"
```

---

### Task 5: Gate the post-5xx reprobe on `swap_lock`

**Files:**
- Modify: `manager/app.py:182-187` (`_reprobe_for`)
- Test: `tests/test_endpoints.py`

**Interfaces:**
- Produces: `_reprobe_for(slot)` acquires `slot.swap_lock` before probing, so a reprobe serializes after any in-flight/queued swap and cannot overwrite a fresh `mark_swapped`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_endpoints.py`:

```python
@pytest.mark.asyncio
async def test_reprobe_waits_for_swap_lock(test_config):
    """A post-5xx reprobe must not run while a swap holds the slot lock."""
    from manager.app import ServerState, _reprobe_for
    server = ServerState(test_config)
    slot = server.slots["main"]
    slot.reconcile_on_error = AsyncMock()

    await slot.swap_lock.acquire()          # simulate an in-flight swap
    task = asyncio.create_task(_reprobe_for(slot))
    await asyncio.sleep(0.01)
    slot.reconcile_on_error.assert_not_called()   # blocked on the lock
    slot.swap_lock.release()
    await task
    slot.reconcile_on_error.assert_awaited_once()  # ran after release
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_endpoints.py::test_reprobe_waits_for_swap_lock -v`
Expected: FAIL — `reconcile_on_error` is called immediately (no lock), so `assert_not_called` fails.

- [ ] **Step 3: Implement**

Replace `_reprobe_for` in `manager/app.py:182-187`:

```python
async def _reprobe_for(slot) -> None:
    """Fire-and-forget re-probe used after a backend 5xx.

    Acquires the slot's swap_lock first, so the reprobe serializes AFTER any
    in-flight/queued swap and can never clobber a fresh mark_swapped. Between
    the two writers of loaded_model, the swap is authoritative.
    """
    async with slot.swap_lock:
        async with httpx.AsyncClient() as probe_client:
            await slot.reconcile_on_error(probe_client)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_endpoints.py::test_reprobe_waits_for_swap_lock -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add manager/app.py tests/test_endpoints.py
git commit -m "fix(manager): gate post-5xx reprobe on swap_lock"
```

---

### Task 6: `POST /swap` echoes the authoritative loaded name

**Files:**
- Modify: `manager/app.py:686-694` (`/swap` handler return)
- Test: `tests/test_endpoints.py`

**Interfaces:**
- Produces: `/swap` 200 body `model` field equals the slot's post-swap `loaded_model` (canonical on-disk stem), matching `/status`.

- [ ] **Step 1: Write the failing test**

Add to `tests/test_endpoints.py`:

```python
def test_swap_echo_is_canonical(client, monkeypatch):
    """A swap requested with odd casing/suffix echoes the canonical on-disk stem."""
    app = client.app
    server = app.state.server

    async def fake_swap(self, model):
        return True
    monkeypatch.setattr("manager.swap.ModelSwapper.swap_to", fake_swap)

    r = client.post("/swap", json={"model": "TEST-MODEL-Q4.gguf", "target": "main"})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "test-model-q4"                       # canonical
    assert body["model"] == server.slots["main"].loaded_model     # agrees with /status
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_endpoints.py::test_swap_echo_is_canonical -v`
Expected: FAIL — handler echoes the raw `"TEST-MODEL-Q4.gguf"`.

- [ ] **Step 3: Implement**

Replace the final return of the `/swap` handler (`manager/app.py:694`):

```python
        return {"slot": target, "model": server.slots[target].loaded_model, "status": "ok"}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_endpoints.py -k swap -v`
Expected: PASS, including the pre-existing `test_swap_valid_main` (its assertion `body["model"] == "test-model-q4"` still holds — the request used exact case, and the canonical stem equals it).

- [ ] **Step 5: Commit**

```bash
git add manager/app.py tests/test_endpoints.py
git commit -m "feat(manager): /swap echoes canonical loaded model name"
```

---

### Task 7: Persist the main device pin + preload main + docs

**Files:**
- Modify: `systemd/llama-server.service:38-43` (ExecStart + add `Environment=`)
- Modify: `config/llama-server.env`
- Modify: `README.md:249`
- Modify: `manager/names.py` (no-alias invariant note) — comment only

**Interfaces:** none (config/docs). No automated test; includes a documented **host verification** step.

- [ ] **Step 1: Add the device flag and a unit-level default to the main systemd unit**

In `systemd/llama-server.service`, change `ExecStart` (lines 38-43) to add the device line, and add an `Environment=` default in `[Service]` (so a redeploy that doesn't touch the live env still pins the device — `setup.sh` does not overwrite an existing env file):

```ini
# In [Service], near the EnvironmentFile line:
Environment=DEVICE=CUDA0

ExecStart=/opt/llama/bin/llama-server \
    --host ${HOST} \
    --port ${PORT} \
    --model ${MODEL_PATH} \
    --device ${DEVICE} \
    --n-gpu-layers ${N_GPU_LAYERS} \
    --ctx-size ${CTX_SIZE}
```

Update the comment block above `ExecStart` to document `--device` (fail-loud; no silent iGPU fallback).

- [ ] **Step 2: Add DEVICE and a default MODEL_PATH to the committed env template**

In `config/llama-server.env`, set:

```bash
# Path to the GGUF model file to load on startup. Preloaded like the batch slot
# so main boots with a model and the device pin is exercised at startup.
MODEL_PATH=/mnt/secondary/llama-models/gemma-4-26b-a4b-it-q4_k_m.gguf

# CUDA device selector. CUDA0 is the Tesla P40. Explicit + fail-loud: the
# service errors if CUDA0 is absent rather than silently falling back to the
# iGPU (which has no dedicated VRAM and OOMs the host with a large model).
DEVICE=CUDA0
```

Keep `N_GPU_LAYERS=-1`, `HOST`, `PORT` as-is.

- [ ] **Step 3: Update README matching contract**

In `README.md`, replace the line at `README.md:249`:

> The manager uses exact filename matching — the model name in a request must exactly match the filename without `.gguf`.

with:

```markdown
The manager matches model names **case-insensitively** and ignores a `.gguf`
suffix or any leading path; an exact-case filename match wins when present.
The authoritative model identity for clients is `slots.<slot>.loaded_model`
in `GET /status` (the canonical on-disk stem). On-disk GGUF files must use a
lowercase `.gguf` extension. A request for a model loaded on neither slot
returns **409**; load it first via `POST /swap`.
```

- [ ] **Step 4: Add the no-alias invariant note to `names.py`**

Append to the module docstring in `manager/names.py`:

```
Correctness assumes llama-server reports the model by its file path/stem (no
--alias), so probe() and a swap agree on the same physical file. If --alias is
ever added it must equal the GGUF stem, or probe() must resolve it back.
```

- [ ] **Step 5: Run the full suite (no behavior regressions from config/docs)**

Run: `pytest -q`
Expected: PASS (config/doc changes do not affect tests; the `tmp_env_file` fixture is independent of the committed template).

- [ ] **Step 6: Commit**

```bash
git add systemd/llama-server.service config/llama-server.env README.md manager/names.py
git commit -m "feat(ops): persist main --device CUDA0 pin, preload main, update matching docs"
```

- [ ] **Step 7: HOST VERIFICATION (manual, before deploy — record result in the runbook)**

These run on the remote Ubuntu host, not in CI:

1. Confirm the device selector: `/opt/llama/bin/llama-server --list-devices` shows `CUDA0` = the Tesla P40.
2. **Prove fail-loud on empty device** (the realistic failure mode): `/opt/llama/bin/llama-server --device '' --model <any.gguf>` must exit non-zero rather than selecting a default device. If it does NOT fail loud, do not deploy until `DEVICE` is guaranteed set (the `Environment=DEVICE=CUDA0` default covers this) — a silent empty-device → iGPU fallback would reproduce the original host OOM.
3. After deploy: `grep ^DEVICE= /etc/llama/llama-server.env` (or `systemctl show -p Environment llama-server`) resolves `CUDA0`; diff the live `ExecStart` against the repo unit so no prior hardcoded `--device` is lost.

---

## Self-Review

**Spec coverage:**
- §1 `names.py` (display_name/match_key/same_model) → Task 1. ✓
- §2 writers store display form (probe, mark_swapped via caller) → Task 2. ✓
- §3 comparisons via same_model (resolve_slot, app.py:116, app.py:421) + model_path folded resolution → Tasks 2 (116), 3 (model_path), 4 (resolve_slot, 421). ✓
- §4 409 policy + preload-via-env first boot → Task 4 (409), Task 7 (preload). ✓
- §5 reprobe race → Task 5. ✓
- §6 device pin + Environment= default + host verification → Task 7. ✓
- §7 docs: README, /swap echo, no-alias note → Task 6 (echo), Task 7 (README, no-alias). ✓
- Testing section items → covered across Tasks 1-6 test steps. ✓

**Placeholder scan:** No TBD/TODO/"handle edge cases"; every code step shows complete code. ✓

**Type consistency:** `display_name`/`match_key`/`same_model` signatures defined in Task 1 are used identically in Tasks 2-4. `resolve_slot` returns `Optional[str]` (Task 4) and the chat handler consumes `None` (same task). `model_path` returns `str | None` throughout. ✓

## Execution Handoff

(Choose after review — see below.)
