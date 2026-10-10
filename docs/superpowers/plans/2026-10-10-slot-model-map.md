# Operator-defined Model → Slot Map Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** An untargeted `POST /swap` loads the model onto the slot the operator assigned it via `SLOT_<NAME>_MODELS`, or refuses with 400 — never silently onto `main`.

**Architecture:** `SlotConfig` gains a `models` tuple parsed from `SLOT_<NAME>_MODELS`, validated at startup so each model has at most one home. A pure `routing.home_slot()` looks a model up in that map; the `/swap` handler uses it when `target` is absent or null. `/status` reports each slot's home list. Targeted swaps and the chat path are untouched.

**Tech Stack:** Python 3.12, FastAPI, pytest. No new dependencies.

**Spec:** `docs/superpowers/specs/2026-10-10-slot-model-map-design.md`

> **Code in this plan is a sketch to be verified against the real files.** The work is mechanical and single-file per task, which is why code is included, but if the source disagrees with a snippet (a line number moved, a name differs), follow the source and say so in your report. Deviating from a snippet that is wrong is correct.

## Global Constraints

- Run tests with `/opt/llama/manager/venv/bin/python -m pytest` from the repo root. The repo's `.venv` has no pytest.
- No new dependencies.
- Model-name comparison goes through `manager/names.py` only (`same_model`, `match_key`). Never compare model names with `==` or `.lower()`.
- New env variable name: exactly `SLOT_<NAME>_MODELS` (`<NAME>` = slot name uppercased), comma-separated.
- New error type string: exactly `no_home_slot`. New `/status` key: exactly `models`.
- Do not touch `config/manager.env.orig`: it is an untracked file that predates this work. Stage explicit paths only; never `git add -A`.
- Work on a branch off `main` (e.g. `feat/slot-model-map`), not on `docs/readme-sync-three-slots`.
- Every new test that claims to discriminate the change must be seen failing before the implementation step. Tests marked **(pin)** guard unchanged behaviour and are expected to pass before and after. Say so in the report rather than forcing them red.

## Review Focus

1. **Case / `.gguf` variants between the map and the request.** `SLOT_RE_MODELS=TEST-MODEL-Q8.gguf` with a request for `test-model-q8` must resolve to `re`. Test: Task 2 `test_untargeted_swap_matches_home_case_and_suffix_insensitively`.
2. **The same model listed on two slots, differing only by case or suffix.** The manager must refuse to start and name both variables, not pick one. Test: Task 1 `test_model_with_two_home_slots_is_rejected_naming_both_vars`.
3. **Sloppy list syntax** (`"a, b,"`, spaces, trailing comma). Tolerated; no empty-string model. Test: Task 1 `test_slot_models_parsed_and_whitespace_tolerant`.
4. **`"target": null` from a JSON client that serialises None.** Treated as untargeted, not as `invalid_target`. Test: Task 2 `test_untargeted_swap_with_null_target_uses_the_map`.
5. **A mapped model whose GGUF does not exist.** 404 `model_not_found` from the home slot's path, not 400 and not a swap attempt. Test: Task 2 `test_untargeted_swap_of_mapped_but_missing_model_is_404` (pin).

Known, pre-existing, **out of scope**: a non-string `model` (e.g. a list) reaches `names.display_name` and 500s. That happens today through `server.model_path`; this change does not make it worse. Do not fix it here.

---

### Task 1: Parse `SLOT_<NAME>_MODELS` into `SlotConfig.models`

**Files:**
- Modify: `manager/slot_config.py` (module docstring, `SlotConfig`, `build_slots`, new helpers)
- Test: `tests/test_slot_config.py`

**Interfaces:**
- Consumes: `manager.names.match_key(name: str | None) -> str | None`
- Produces: `SlotConfig.models: tuple[str, ...]` (default `()`), entries stripped, as written by the operator (not normalised). `build_slots()` raises `ValueError` whose message contains both `SLOT_<A>_MODELS` and `SLOT_<B>_MODELS` when one model is listed on two slots.

- [ ] **Step 1: Write the failing tests** — append to `tests/test_slot_config.py` (it already defines `_env(mapping)`):

```python
def test_slot_models_default_to_empty():
    slots = build_slots(_env({}))
    assert all(s.models == () for s in slots)


def test_slot_models_parsed_and_whitespace_tolerant():
    slots = build_slots(_env({
        "SLOTS": "main,re",
        "SLOT_RE_PORT": "8084",
        "SLOT_MAIN_MODELS": " model-a , model-b,",
        "SLOT_RE_MODELS": "model-c",
    }))
    by_name = {s.name: s for s in slots}
    assert by_name["main"].models == ("model-a", "model-b")
    assert by_name["re"].models == ("model-c",)


def test_model_with_two_home_slots_is_rejected_naming_both_vars():
    # Differs only by case and .gguf suffix: names.match_key treats these as
    # one model, so this is a conflict, not two models.
    with pytest.raises(ValueError) as exc:
        build_slots(_env({
            "SLOT_MAIN_MODELS": "Foo-Q4",
            "SLOT_BATCH_MODELS": "foo-q4.gguf",
        }))
    assert "SLOT_MAIN_MODELS" in str(exc.value)
    assert "SLOT_BATCH_MODELS" in str(exc.value)


def test_model_repeated_within_one_slot_is_allowed():
    slots = build_slots(_env({"SLOT_MAIN_MODELS": "model-a,MODEL-A.gguf"}))
    assert slots[0].models == ("model-a", "MODEL-A.gguf")
```

- [ ] **Step 2: Run them to verify they fail**

Run: `/opt/llama/manager/venv/bin/python -m pytest tests/test_slot_config.py -v`
Expected: the four new tests FAIL (`AttributeError: 'SlotConfig' object has no attribute 'models'`, and no `ValueError` raised for the conflict test). Existing tests pass.

- [ ] **Step 3: Implement** in `manager/slot_config.py`:

Add the import under `from dataclasses import dataclass`:

```python
from manager.names import match_key
```

Add the field to `SlotConfig`, after `device`:

```python
    device: str | None = None
    models: tuple[str, ...] = ()   # models whose home is this slot (SLOT_<NAME>_MODELS)
```

Add helpers above `build_slots`:

```python
def _parse_models(raw: str | None) -> tuple[str, ...]:
    """SLOT_<NAME>_MODELS -> tuple of names. Whitespace and empty entries dropped."""
    if not raw:
        return ()
    return tuple(m.strip() for m in raw.split(",") if m.strip())


def _check_one_home_per_model(slots) -> None:
    """A model may call at most one slot home; compared via names.match_key."""
    home: dict[str, str] = {}
    for s in slots:
        for m in s.models:
            key = match_key(m)
            other = home.get(key)
            if other is not None and other != s.name:
                raise ValueError(
                    f"Configuration error: model {m!r} is listed in both "
                    f"SLOT_{other.upper()}_MODELS and SLOT_{s.name.upper()}_MODELS; "
                    f"a model has one home slot"
                )
            home[key] = s.name
```

In `build_slots`, pass the new field in the `SlotConfig(...)` call, after `device=...`:

```python
            models=_parse_models(getenv(f"SLOT_{name.upper()}_MODELS")),
```

and replace the final `return tuple(slots)` with:

```python
    _check_one_home_per_model(slots)
    return tuple(slots)
```

Extend the module docstring with one paragraph:

```
SLOT_<NAME>_MODELS lists the models whose home is that slot: an untargeted
POST /swap loads a model onto its home slot (see routing.home_slot). It has no
legacy alias, and a model listed on two slots is a startup error.
```

- [ ] **Step 4: Run the file, then the whole suite**

Run: `/opt/llama/manager/venv/bin/python -m pytest tests/test_slot_config.py -v`, then `/opt/llama/manager/venv/bin/python -m pytest -q`
Expected: all PASS.

- [ ] **Step 5: Commit**

```bash
git add manager/slot_config.py tests/test_slot_config.py
git commit -m "feat(config): SLOT_<NAME>_MODELS assigns each model a home slot"
```

---

### Task 2: Untargeted `/swap` resolves the home slot or refuses

**Files:**
- Modify: `manager/routing.py` (add `home_slot`)
- Modify: `manager/app.py` — the `swap_slot` handler (currently `@app.post("/swap")` near line 967, target resolution at ~990-996) and its docstring
- Test: `tests/test_routing.py`, `tests/test_endpoints.py`

**Interfaces:**
- Consumes: `SlotConfig.models` (Task 1); `manager.names.same_model`; `config` (the `ManagerConfig` in `create_app`'s scope, already closed over by the handlers)
- Produces: `manager.routing.home_slot(model: str, slot_configs: Iterable[SlotConfig]) -> Optional[str]`. `/swap` 400 body `{"error": {"type": "no_home_slot", "message": str}}`.

- [ ] **Step 1: Write the failing unit tests** — append to `tests/test_routing.py`:

```python
from manager.routing import home_slot
from manager.slot_config import SlotConfig


def _sc(name, *models):
    return SlotConfig(name=name, host="127.0.0.1", port=1, env_file="/tmp/env",
                      systemd_unit="u.service", queue_limit=1, models=tuple(models))


def test_home_slot_finds_the_configured_slot():
    scs = [_sc("main", "model-a"), _sc("re", "model-b")]
    assert home_slot("model-b", scs) == "re"


def test_home_slot_is_case_and_suffix_insensitive():
    scs = [_sc("main"), _sc("re", "Model-B.gguf")]
    assert home_slot("model-b", scs) == "re"


def test_home_slot_none_for_unmapped_model():
    scs = [_sc("main", "model-a"), _sc("re")]
    assert home_slot("model-z", scs) is None
```

- [ ] **Step 2: Write the failing endpoint tests** in `tests/test_endpoints.py`. **Delete** `test_swap_default_target_is_main` and `test_swap_default_target_is_first_configured_slot_not_the_literal_main`: they assert the "untargeted means first slot" behaviour this task removes. Add in their place:

```python
def _with_homes(config, **homes):
    """Copy of config with SLOT_<NAME>_MODELS-equivalent home lists set."""
    import dataclasses
    slots = tuple(
        dataclasses.replace(s, models=tuple(homes.get(s.name, ())))
        for s in config.slots
    )
    return dataclasses.replace(config, slots=slots)


def _homes_client(config, monkeypatch, **homes):
    """Client with home lists set; records the name of each slot a swap hit."""
    from manager.app import create_app
    swapped = []

    async def fake_swap(self, model_path):
        swapped.append(self._slot.name)
        return True
    monkeypatch.setattr("manager.swap.ModelSwapper.swap_to", fake_swap)
    return TestClient(create_app(_with_homes(config, **homes))), swapped


def test_untargeted_swap_goes_to_the_models_home_slot(three_slot_config, monkeypatch):
    client, swapped = _homes_client(three_slot_config, monkeypatch, re=["test-model-q8"])
    r = client.post("/swap", json={"model": "test-model-q8"})
    assert r.status_code == 200, r.text
    assert r.json()["slot"] == "re"
    assert swapped == ["re"]


def test_untargeted_swap_matches_home_case_and_suffix_insensitively(
        three_slot_config, monkeypatch):
    client, swapped = _homes_client(
        three_slot_config, monkeypatch, re=["TEST-MODEL-Q8.gguf"])
    r = client.post("/swap", json={"model": "test-model-q8"})
    assert r.status_code == 200, r.text
    assert swapped == ["re"]


def test_untargeted_swap_with_null_target_uses_the_map(three_slot_config, monkeypatch):
    client, swapped = _homes_client(three_slot_config, monkeypatch, re=["test-model-q8"])
    r = client.post("/swap", json={"model": "test-model-q8", "target": None})
    assert r.status_code == 200, r.text
    assert swapped == ["re"]


def test_untargeted_swap_of_unmapped_model_is_refused_and_touches_no_slot(
        three_slot_config, monkeypatch):
    # test-model-q4 exists on disk but no slot calls it home.
    client, swapped = _homes_client(three_slot_config, monkeypatch, re=["test-model-q8"])
    r = client.post("/swap", json={"model": "test-model-q4"})
    assert r.status_code == 400, r.text
    body = r.json()
    assert body["error"]["type"] == "no_home_slot"
    assert "SLOT_<NAME>_MODELS" in body["error"]["message"]
    assert swapped == []


def test_untargeted_swap_of_mapped_but_missing_model_is_404(three_slot_config, monkeypatch):
    # (pin) 404 before and after: proves the map lookup runs before, and does
    # not replace, the model-file check.
    client, swapped = _homes_client(three_slot_config, monkeypatch, re=["does-not-exist"])
    r = client.post("/swap", json={"model": "does-not-exist"})
    assert r.status_code == 404, r.text
    assert r.json()["error"]["type"] == "model_not_found"
    assert swapped == []


def test_targeted_swap_ignores_the_map(three_slot_config, monkeypatch):
    # (pin) An explicit target is an operator override, even against the map.
    client, swapped = _homes_client(three_slot_config, monkeypatch, re=["test-model-q4"])
    r = client.post("/swap", json={"model": "test-model-q4", "target": "main"})
    assert r.status_code == 200, r.text
    assert swapped == ["main"]
```

`self._slot` is the attribute `ModelSwapper` uses for its slot (it logs `self._slot.name` in `swap_to`, `manager/swap.py:94`). If it differs in the source, use whatever `ModelSwapper` actually stores.

- [ ] **Step 3: Run them to verify they fail**

Run: `/opt/llama/manager/venv/bin/python -m pytest tests/test_routing.py tests/test_endpoints.py -v -k "home or untargeted or targeted_swap_ignores"`
Expected: the routing tests FAIL (`ImportError: cannot import name 'home_slot'`). In the endpoint tests, the first four FAIL: the untargeted ones land on `main`, the null-target one gets 400 `invalid_target`, and the unmapped one gets 200. The two **(pin)** tests PASS.

- [ ] **Step 4: Implement `home_slot`** in `manager/routing.py`. Update the imports:

```python
from typing import Iterable, Optional

from manager.names import same_model
from manager.slot_config import SlotConfig
from manager.slots import SlotState
```

Append:

```python
def home_slot(model: str, slot_configs: Iterable[SlotConfig]) -> Optional[str]:
    """Return the slot whose SLOT_<NAME>_MODELS lists `model`, or None.

    Used only to place an untargeted POST /swap. build_slots guarantees a model
    has at most one home, so the first match is the only match. Comparison is
    names.same_model: case/suffix/path-insensitive.
    """
    for sc in slot_configs:
        if any(same_model(model, m) for m in sc.models):
            return sc.name
    return None
```

- [ ] **Step 5: Use it in the `/swap` handler** in `manager/app.py`. Add `home_slot` to the existing `from manager.routing import ...` line. Replace the target-resolution block (the `target = body.get("target", next(iter(server.slots)))` line and the `invalid_target` check under it) with:

```python
        valid = ", ".join(repr(n) for n in server.slots)
        target = body.get("target")
        if target is None:
            # Untargeted: the operator's map decides, never "the first slot".
            # agent_core clients send exactly this after a model_not_loaded
            # 409; guessing a slot here is what used to evict main.
            target = home_slot(model_name, config.slots)
            if target is None:
                return JSONResponse(
                    {"error": {"type": "no_home_slot",
                               "message": (
                                   f"model '{model_name}' has no home slot; pass "
                                   f"'target' (one of: {valid}) or list it in "
                                   f"SLOT_<NAME>_MODELS")}},
                    status_code=400,
                )
        elif not isinstance(target, str) or target not in server.slots:
            return JSONResponse(
                {"error": {"type": "invalid_target",
                           "message": f"'target' must be one of: {valid}"}},
                status_code=400,
            )
```

Keep the existing `isinstance` guard ahead of the `in` test. `test_swap_non_string_target_returns_structured_400_not_500` depends on it, because an unhashable target would otherwise raise inside the dict membership test.

Update the handler docstring's body line to:

```
        Body: {"model": str, "target": <slot name> (optional)}.
        Without a target (absent or null) the model goes to its home slot from
        SLOT_<NAME>_MODELS; a model with no home slot is refused with 400.
```

- [ ] **Step 6: Run the targeted tests, then the whole suite**

Run: `/opt/llama/manager/venv/bin/python -m pytest tests/test_routing.py tests/test_endpoints.py -v`, then `/opt/llama/manager/venv/bin/python -m pytest -q`
Expected: all PASS.

- [ ] **Step 7: Commit**

```bash
git add manager/routing.py manager/app.py tests/test_routing.py tests/test_endpoints.py
git commit -m "feat(swap): untargeted /swap goes to the model's home slot or is refused"
```

---

### Task 3: `/status` reports home lists; docs and config template

**Files:**
- Modify: `manager/app.py` — the `/status` handler's `"slots"` dict comprehension (near line 664)
- Modify: `README.md` (lines ~178, ~187, ~387, ~400, and the `/status` example and field list near ~215-245 / ~416)
- Modify: `config/manager.env` (template; slot section, near lines 60-83)
- Test: `tests/test_endpoints.py`

**Interfaces:**
- Consumes: `SlotConfig.models` (Task 1)
- Produces: each `/status` slot object gains `"models": list[str]`, the configured home list as written.

- [ ] **Step 1: Write the failing test, and update the existing shape test.** In `tests/test_endpoints.py`, add `"models"` to the expected key set in `test_status_endpoint_has_slots` (near line 40). Then add:

```python
def test_status_reports_each_slots_home_models(three_slot_config):
    from manager.app import create_app
    client = TestClient(create_app(_with_homes(three_slot_config, re=["test-model-q8"])))
    slots = client.get("/status").json()["slots"]
    assert slots["re"]["models"] == ["test-model-q8"]
    assert slots["main"]["models"] == []
```

(`_with_homes` was added in Task 2.)

- [ ] **Step 2: Run them to verify they fail**

Run: `/opt/llama/manager/venv/bin/python -m pytest tests/test_endpoints.py -v -k status`
Expected: `test_status_endpoint_has_slots` and `test_status_reports_each_slots_home_models` FAIL (`KeyError: 'models'` / key set mismatch).

- [ ] **Step 3: Implement.** In `create_app`, right after `server = ServerState(config)`, add:

```python
    # Each slot's SLOT_<NAME>_MODELS list, for /status. Static: config is frozen.
    home_models = {sc.name: list(sc.models) for sc in config.slots}
```

In the `/status` handler, replace the `name: slot.to_status_dict()` entry with:

```python
                name: {**slot.to_status_dict(), "models": home_models.get(name, [])}
```

`SlotState.to_status_dict` is deliberately left alone: the home list is configuration, not slot state, and `tests/test_slots.py::test_to_status_dict_shape` stays valid.

- [ ] **Step 4: Run the suite**

Run: `/opt/llama/manager/venv/bin/python -m pytest -q`
Expected: all PASS.

- [ ] **Step 5: Update `config/manager.env` (template).** After the `SLOTS=main,batch,re` line, add:

```
# Home slot per model (manager/slot_config.py, routing.home_slot). An untargeted
# POST /swap -- which agent_core >= v1.11 clients send after a model_not_loaded
# 409 -- loads the model onto its home slot. A model listed nowhere is refused
# with 400 rather than loaded onto main. A model may have only one home; listing
# it on two slots stops the manager at startup. A swap that names a 'target' is
# unaffected. Keep main's list short: anything listed there can evict main.
SLOT_MAIN_MODELS=Qwen3.8-27B-UD-Q6_K
SLOT_RE_MODELS=Qwen3.6-35B-A3B-UD-IQ4_NL_XL
```

- [ ] **Step 6: Update `README.md`.**
  - Line ~178: replace "naming the slot as `target` (default: the first slot in `SLOTS`)" with "naming the slot as `target`. Without a `target`, the model goes to its home slot from `SLOT_<NAME>_MODELS`, and a model with no home slot is refused with 400."
  - Line ~187 (errors): add "400 `no_home_slot` for an untargeted swap of a model no slot lists in `SLOT_<NAME>_MODELS`".
  - Line ~387 (config table): add `_MODELS` to the per-slot list. Default: _(unset)_. Meaning: "comma-separated models whose home is this slot (used by untargeted `/swap`); a model may appear in only one slot's list".
  - Line ~400 (endpoint table): replace "`target` defaults to the first slot in `SLOTS`" with "without `target`, the model's home slot from `SLOT_<NAME>_MODELS`, else 400".
  - `/status` example JSON (~215-245): add a `"models": [...]` line to each slot object, matching the template values. Field list (~416): add a bullet: "`models` — the slot's `SLOT_<NAME>_MODELS` list as configured (empty if unset)."
  - Run `grep -n "first slot in" README.md ARCHITECTURE.md`. Expected: no hits that describe `/swap`'s default.

- [ ] **Step 7: Commit**

```bash
git add manager/app.py tests/test_endpoints.py config/manager.env README.md
git commit -m "feat(status): report each slot's home models; document SLOT_<NAME>_MODELS"
```

---

### Task 4: Remove the batch slot from the repo's template and docs

The batch slot was decommissioned on the live host on 2026-10-10: `SLOTS=main,re` in `/etc/llama/manager.env`, and `llama-server-batch` is stopped and disabled. It had served 2 chat requests since 2026-09-27, and removing it freed ~5.2 GiB of iGPU GTT system RAM. The repo still describes three slots. This task makes it describe the host as deployed.

**Out of scope:** the code's legacy batch support. That is `BATCH_*` in `slot_config._LEGACY`/`_DEFAULTS`, `DEFAULT_SLOTS = ("main", "batch")`, and the `batch_*` fields in `ManagerConfig`. Changing those changes behaviour for a config with `SLOTS` unset, which is a code change with its own tests. Leave them alone.

**Files:**
- Modify: `config/manager.env` (delete the `# Batch inference slot` block, lines ~51-58; `SLOTS=main,re`; fix the slot-order comment that mentions the `main,batch` default only if it now reads wrong)
- Modify: `README.md` (every batch mention; current hits at lines 25, 29, 43, 80, 229, 364-366, 377-378, 386, 539, 546, 558, 582, 593-602, 682, 691, 727)
- Delete: `systemd/llama-server-batch.service`, `config/llama-server-batch.env`

- [ ] **Step 1: Template.** Edit `config/manager.env` as listed above. Check: `grep -n -i batch config/manager.env` returns nothing.
- [ ] **Step 2: README.** For each hit, rewrite it for a two-slot host (`main` on the V100, `re` on the P40). Drop the batch column from the architecture diagram, the `"batch"` object from the `/status` example, and the `llama-server-batch.env` section and `BATCH_*` table rows. Remove `llama-server-batch` from the systemctl command lists and the file trees. Retitle "Adding the batch and re slots" to "Adding the re slot", with the commands shown using `-re` names. Keep the sentence that the same steps add any further slot. In the `SLOTS` row, keep the code default `main,batch` (it is still the code default) and change the template value to `main,re`. Check: `grep -n -i batch README.md` hits only the `SLOTS` default and the `BATCH_*` legacy-variable mention, if one is kept to explain the default. Read each remaining hit.
- [ ] **Step 3: Delete the files.** `git rm systemd/llama-server-batch.service config/llama-server-batch.env`. Check: `grep -rn "llama-server-batch" --include=*.md --include=*.sh --include=*.env --include=*.service . | grep -v docs/superpowers` returns nothing. Historical plans and specs under `docs/superpowers` keep their batch references.
- [ ] **Step 4: Suite.** `/opt/llama/manager/venv/bin/python -m pytest -q`. Expected: all PASS. No code changed, but `scripts/setup.sh` or the tests may reference the deleted files, so grep `scripts/` and `tests/` for `llama-server-batch` and fix any hit.
- [ ] **Step 5: Commit**

```bash
git add config/manager.env README.md
git commit -m "chore(config,docs): remove the decommissioned batch slot from template and docs"
```

(The `git rm` in Step 3 already staged the deletions.)

---

## Deployment (operator, after merge — not an implementer task)

Needs sudo. Do it in this order. Code that reads `SLOT_*_MODELS` with no lines set refuses every untargeted swap, which is safe. The reverse order is harmless too, because old code ignores unknown variables.

1. Add the two `SLOT_*_MODELS` lines (Task 3 Step 5 values, or whatever the operator chooses) to `/etc/llama/manager.env`. Read it back with `grep _MODELS /etc/llama/manager.env`.
2. `./scripts/deploy-manager.sh --check` (read-only), then `sudo ./scripts/deploy-manager.sh`. Confirm it reports the merged commit.
3. Verify against the live manager:
   - `curl -s http://192.168.1.14:11434/status | python3 -m json.tool | grep -A3 '"models"'` shows the configured lists.
   - Refusal: `curl -s -w '\n%{http_code}\n' -X POST http://192.168.1.14:11434/swap -H 'content-type: application/json' -d '{"model":"Devstral-Small-2-24B-Instruct-2512-Q5_K_M"}'` returns 400 `no_home_slot`, and `/status` still shows main on Qwen3.8.
   - Home resolution without a reload: an untargeted swap of `Qwen3.6-35B-A3B-UD-IQ4_NL_XL` returns 200 `{"slot": "re", ...}` immediately. `ensure_model_on_slot` short-circuits because it is already loaded, so no restart happens.
   - Do **not** test a targeted swap on the live box. It would evict a slot.
