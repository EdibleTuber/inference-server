# Model-name normalization + main device-pin + explicit-swap policy

**Date:** 2026-06-29
**Status:** Design approved, ready for implementation plan
**Origin:** Follow-up to `2026-06-29-main-slot-implicit-swap-routing-fix.md` (the host-RAM OOM
investigation). This spec is the *implementation* design for the normalization (Fix B part 1),
the device-pin persistence (Fix A), and — pulled in after a design-review panel — the
explicit-swap policy (Fix B part 2). Pressure-tested by a 5-lens adversarial review on
2026-06-29; every must-fix from that panel is folded in below.

## Problem recap

The manager fronts two llama.cpp slots — **main** (P40/CUDA0, the 26B chat model) and
**batch** (iGPU/Vulkan0, the small Gemma E4B model). Three defects compound:

1. **Inconsistent model-name handling.** `loaded_model` has two writers with two different
   conventions: `probe()` strips `.gguf` to a stem (`slots.py:85-91`); `mark_swapped()` stores
   the client's raw request string verbatim (`slots.py:101-105`, fed `body.get("model")` at
   `app.py:211,129`). There is **no periodic probe** — `probe()` runs only at startup
   (`app.py:285`) and once after a backend 5xx (`reconcile_on_error`). So a swap-stamped name
   (including `.gguf` and arbitrary case) sticks indefinitely. This is why the gemma4 main model
   shows `.gguf` in `/status` while the batch model does not: main's name was last set by a
   swap, batch's by the startup probe.

2. **Brittle routing.** `resolve_slot` (`routing.py:11-26`) compares with raw string equality,
   so it breaks on `.gguf` suffix, on case (the batch stem is mixed-case `gemma-4-E4B-it-Q4_K_M`),
   and on path. A near-miss silently mis-routes.

3. **Implicit main swaps + no device pin = host OOM.** Rule 3 ("unknown model → main") makes any
   unrecognized name restart the 26B; and the main systemd unit has no `--device`, so a relaunch
   can place the 26B on the iGPU (no dedicated VRAM → ~29 GB into host RAM → OOM-killer). The live
   process was hand-pinned `--device CUDA0`, but a swap is `systemctl restart` and `ExecStart`
   comes from the committed unit, so the hand-pin is discarded on the next restart.

## Decisions (locked during brainstorming)

- **Scope:** normalization + device-pin persistence + explicit-swap (4xx) policy + a pre-existing
  reprobe race that this spec touches anyway.
- **Reported names:** store/report a **clean display form** (basename, no `.gguf`, original case),
  and **compare case-folded**. `/v1/models` keeps natural-looking names; the spurious `.gguf`
  disappears.
- **Device pin:** `--device CUDA0`, **fail-loud** (no silent iGPU fallback). **No `MemoryMax`** on
  main — on the P40 the weights live in VRAM (not RSS) and the file mmap is reclaimable page cache,
  so a host-RAM cap mostly guards a case the device pin already closes and a too-low cap could
  OOM a healthy load.
- **Unknown model → 409**, not an implicit main swap (pulled into scope so case-folding cannot
  widen the set of inputs that silently restart the 26B).
- **Reprobe race:** fix now by gating the post-5xx reprobe on `swap_lock`.

## Design

### 1. `manager/names.py` — pure normalization helpers

No I/O, unit-testable in isolation. Three functions:

```python
def display_name(raw: str | None) -> str | None:
    """Storage/report form: basename, drop one .gguf suffix, keep original case.
    Returns None for falsy/empty input so the empty->None invariant lives in ONE place
    (a trailing slash like 'foo.gguf/' yields '' from rsplit and must not leak)."""
    if not raw:
        return None
    name = raw.rsplit("/", 1)[-1]          # strip any directory component
    if name.lower().endswith(".gguf"):     # case-insensitive: handles .GGUF
        name = name[: -len(".gguf")]
    return name or None

def match_key(name: str | None) -> str | None:
    """Tolerant comparison key: display form, case-folded. None-total.
    Model filenames are assumed ASCII; casefold is locale-independent."""
    d = display_name(name)
    return d.casefold() if d is not None else None

def same_model(requested: str | None, loaded: str | None) -> bool:
    """The single comparison primitive used at every routing/guard site.
    None loaded_model never matches — closes the None-deref crash class in one place."""
    lk = match_key(loaded)
    return lk is not None and lk == match_key(requested)
```

`display_name` vs `match_key` is the standard store-display / compare-folded split; it is
load-bearing here because the batch model has a genuinely mixed-case on-disk stem we want to
report verbatim while matching case-insensitively. `same_model` exists specifically so the
`None`-guard is written once, not copy-pasted across four comparison sites.

### 2. Writers store the clean display form

- **`slots.py:85-91` (`probe`):** `self.loaded_model = display_name(raw_id)`. Also fixes the case
  where llama-server returns a full *path* (today's code only strips `.gguf`, leaving a path).
- **`slots.py:101-105` (`mark_swapped`):** signature unchanged (`mark_swapped(model: str)`); it
  still stores what it is given. The **caller** (`ensure_model_on_slot`, `app.py:129`) passes
  `display_name(path)` using the resolved real on-disk path, so the stored/reported name is the
  authoritative on-disk stem regardless of how a client cased its request. No path/basename logic
  leaks into `SlotState`.

### 3. Comparisons and file resolution go through the helpers

- **`routing.py` (`resolve_slot`):** change return type to `Optional[str]`. Use `same_model(...)`
  for the main and batch checks. **Return `None`** when the model is loaded on neither slot
  (this is what the 409 policy keys off — rule 3's silent "→ main" is removed).
- **`app.py:116` (`ensure_model_on_slot` short-circuit):** `if slot.healthy and same_model(model_name, slot.loaded_model)`.
- **`app.py:421` (unhealthy guard):** use `same_model(...)`. Note: after the 409 policy, if
  `resolve_slot` returned a real slot then the model is loaded on it by construction, so this guard
  effectively becomes "slot unhealthy → 503"; the `same_model` check stays as defensive belt.
- **`app.py:94-101` (`model_path`):** replace the blind `f"{name}.gguf"` lookup with a folded
  resolution that prefers an exact match:
  - Iterate `sorted(models_dir.glob("*.gguf"))` (deterministic order).
  - First pass: if a stem equals the request exactly, return that path.
  - Else map by `match_key` using `dict.setdefault` (first-sorted wins) and return the match;
    **log a warning on any `match_key` collision** (two stems differing only by case) so a
    pathological setup is never silent.
  - Document the **lowercase-`.gguf`-extension assumption**: the glob is case-sensitive, so a file
    literally named `foo.GGUF` will not be found even though `display_name` would normalize it.
- **`app.py:102` (`_models_on_disk`):** already emits `.stem` (clean, original case); left as-is.

### 4. Explicit-swap (409) policy — removing the implicit main swap

- **`app.py` chat handler (~`app.py:418-437`):** after the 404 existence check, call
  `resolve_slot`. If it returns `None`, return **409** with a typed error
  (`{"type": "model_not_loaded", "message": "model <X> not loaded on any slot; use POST /swap"}`).
  No enqueue, no swap. Implicit swaps are now reachable **only** through `POST /swap`
  (`app.py:649-694`, which backs the PAL/PARE `/model` command).
- Rewrite the now-stale comment at `app.py:425-429` (it described the rule-3 fallthrough that no
  longer exists).
- **First-boot / bootstrap (preload via env, mirroring batch):** the batch slot is not auto-loaded
  at runtime — it ships a concrete `MODEL_PATH` in `config/llama-server-batch.env:9`, so systemd
  launches it pre-loaded and the startup probe records it. Do the same for main: give
  `config/llama-server.env` a default `MODEL_PATH` pointing at the 26B (today's committed template
  ships it empty). Then main boots already loaded, the startup probe sets `loaded_model`, and **no
  bootstrap `POST /swap` is required** — the 409 only ever fires for a genuinely not-loaded model.
  This also removes the empty-`--model` first-boot crash-loop and means the device pin is exercised
  at boot (a bad pin fails loud immediately rather than at the first swap). Note `install_config()`
  won't overwrite a live env, so an existing host keeps its current `MODEL_PATH`; this changes only
  the committed template and fresh installs.
- **Shared-infra note:** this is a routing-policy change affecting **both** PAL and PARE. Before
  shipping, confirm neither relies on the legacy auto-load-on-first-request behavior.

### 5. Reprobe-race fix

`probe()` writes `loaded_model`/`healthy` outside `swap_lock`, and the post-5xx reprobe is
`create_task`'d (`app.py:241,251`) so it runs concurrently with the next swap. It can (a) re-mark a
flaky slot healthy so a needed swap is skipped and the request is routed back to the failing
backend, or (b) land during the restart window and mark a freshly-swapped slot unhealthy → spurious
503.

**Fix:** `_reprobe_for` acquires `slot.swap_lock` before probing, so the reprobe serializes after
any in-flight/queued swap and can never overwrite a fresh `mark_swapped`. Document that, between the
two writers, the swap is authoritative.

### 6. Device-pin persistence (fail-loud)

- **`systemd/llama-server.service`:** add `--device ${DEVICE}` to `ExecStart`, **and** add
  `Environment=DEVICE=CUDA0` to the `[Service]` section. The unit is copied + `daemon-reload`ed on
  every redeploy, so the `Environment=` default always reaches the box; a live `EnvironmentFile`
  value still overrides it (EnvironmentFile is read after Environment=). This closes the redeploy
  gap: `setup.sh`'s `install_config()` refuses to overwrite an existing live env
  (`scripts/setup.sh:94-111`, manual `.new` merge only), so relying on `config/llama-server.env`
  alone would leave the new `ExecStart` expanding an unset `${DEVICE}` → `--device ""` during the
  very deploy of this fix.
- **`config/llama-server.env`:** add `DEVICE=CUDA0`, and set a default `MODEL_PATH` to the 26B
  (parity with the batch env; see §4 — this is what lets main boot pre-loaded and exercises the pin
  at startup).
- **`CUDA0` is correct and stable:** with a single NVIDIA GPU the CUDA enumeration is deterministic
  across reboots, unlike Vulkan indices (Vulkan0=iGPU, Vulkan1=P40). `swap.py` only rewrites
  `MODEL_PATH`, so `DEVICE` and `CTX_SIZE` persist across swaps.
- **Pre-merge verification gate (must do, not assume):** on the live host, confirm
  `/opt/llama/bin/llama-server --device ''` exits non-zero rather than falling through to default
  device selection. "Fail-loud" is proven for a *misspelled* device name; the empty-arg case is the
  realistic failure mode and must be verified, because with no `MemoryMax` on main a silent
  empty-device → iGPU fallback would reproduce the original host OOM. Record the result in the
  runbook.
- **Deploy runbook check:** before activating the new unit, verify
  `grep ^DEVICE= /etc/llama/llama-server.env` (or the unit `Environment=`) resolves to `CUDA0`, and
  diff the live `ExecStart` against the repo unit so any existing hardcoded `--device` is not
  silently lost. With main now preloaded via a default `MODEL_PATH` (§4), the device pin is
  exercised at the first boot, so a bad/empty `DEVICE` fails loud immediately rather than waiting
  for the first swap.

### 7. Documentation and contract updates

- **`README.md:249`:** the "exact filename matching" statement is now false — update to
  case-insensitive and `.gguf`-suffix-insensitive matching, and document that `/status.loaded_model`
  is the authoritative model identity (canonical on-disk stem).
- **`POST /swap` echo (`app.py:694`):** currently returns the raw request string while
  `/status.loaded_model` now returns the canonical on-disk stem — a swap-then-verify client would
  see a case mismatch. Make the echo authoritative (return the slot's post-swap `loaded_model`, or
  `display_name(path)`). Update the pinned assertion at `tests/test_endpoints.py:263`.
- **No-`--alias` invariant:** correctness assumes llama-server's `/v1/models` id derives from the
  model path (so `display_name(probe_id) == display_name(swap_path)`). Neither unit sets `--alias`
  today. Document this in `names.py` / the probe comment; if an alias is ever added it must equal the
  GGUF stem, or `probe` must resolve it back to the file.

## Testing

Unit (`names.py`):
- `display_name`/`match_key`/`same_model` over: stem, `.gguf`, `.GGUF`, full path, double
  suffix `x.gguf.gguf`, trailing slash → `None`, empty/`None` input → `None`.

Integration (`tests/test_endpoints.py` + manager tests):
- **None-safety:** a not-yet-healthy slot (`loaded_model=None`, `healthy=False`) receiving a chat
  request does **not** crash (regression test for the panel's blocker).
- **Routing tolerance:** case/suffix/path variants of a loaded model route without a swap.
- **File resolution:** case-variant request resolves to the real file; two case-differing files
  select deterministically (sorted-first) and emit the collision warning; exact-case match wins over
  the folded fallback.
- **Folded same-model:** a probe-vs-swap name difference no longer triggers a redundant swap.
- **409 policy:** a valid-on-disk but not-loaded model returns 409 (not an implicit main swap); an
  unknown-on-disk model still 404s; `POST /swap` still loads it.
- **Reprobe race:** a reprobe fired concurrently with `ensure_model_on_slot` leaves the post-swap
  state equal to `mark_swapped`'s, not the probe's.
- **`/swap` echo:** the `/swap` response `model` equals `/status.loaded_model` after the swap.

## Out of scope (tracked for follow-up)

- **Sudoers gap** (`scripts/setup.sh:120`): NOPASSWD restart is granted for `llama-server.service`
  only, not the batch unit — a batch swap via `sudo systemctl restart` would hit a password prompt.
  Latent batch-swap bug, unrelated to this change.
- **Repo vs live config drift:** committed `config/llama-server.env` is a template; live values
  (e.g. `CTX_SIZE`) differ and are the sole lever since the manager never sets `CTX_SIZE`. Worth
  documenting the live values in version control.

## File reference index

- `manager/names.py` — new module (`display_name`, `match_key`, `same_model`)
- `manager/slots.py:85-91` (`probe`), `:101-105` (`mark_swapped`) — writers store display form
- `manager/routing.py:11-26` (`resolve_slot`) — `Optional[str]`, `same_model`, `None` when unloaded
- `manager/app.py:94-101` (`model_path`) — folded resolution, exact-match-first, collision warning
- `manager/app.py:116` (short-circuit), `:418-437` (chat routing + 409 + unhealthy guard)
- `manager/app.py:127-129` — pass `display_name(path)` into `mark_swapped`
- `manager/app.py:182-187,241,251` (`_reprobe_for`) — acquire `swap_lock` before probing
- `manager/app.py:649-694` (`POST /swap`) — authoritative echo
- `systemd/llama-server.service` — `--device ${DEVICE}` + `Environment=DEVICE=CUDA0`
- `config/llama-server.env` — `DEVICE=CUDA0` + default `MODEL_PATH` (preload main, mirroring batch)
- `scripts/setup.sh:94-111,147,150` — env install (manual-merge), unit copy + daemon-reload
- `README.md:249` — matching contract update
- `tests/test_endpoints.py:263` — `/swap` echo assertion update
