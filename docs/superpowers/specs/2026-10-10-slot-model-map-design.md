# Operator-defined model → slot map — design

**Status:** agreed 2026-10-10. Implementation plan:
`docs/superpowers/plans/2026-10-10-slot-model-map.md`.

## Problem

`POST /swap` without a `target` loads the model onto the first slot in `SLOTS`
(`manager/app.py:990`, `target = body.get("target", next(iter(server.slots)))`).
With `SLOTS=main,batch,re` that is always `main`.

`agent_core` >= v1.11.0 sends exactly that request — `POST /swap {"model": m}`, no
target, by design (`agent_core/inference.py`, `_request_model_swap`) — whenever a
completion comes back `409 model_not_loaded`, then retries once. So any agent_core
client whose configured model is not loaded evicts whatever is on `main`. The swap
also persists across reboots, because `ModelSwapper.swap_to` rewrites the slot's
`MODEL_PATH` before restarting it (`manager/swap.py:95`).

Today only PARE (agent_core v1.12.0) is exposed, and only stays safe because
`PARE_MODEL` matches what is loaded on `re`. PAL (agent_core v1.4.0) has no auto-swap
and surfaced the 409 instead, which is how this was found.

## Goal

The operator decides which slot each model lives on. Untargeted swaps use that
decision, so agent_core clients heal onto the right slot; a model with no assigned
slot is refused rather than guessed.

## Design

1. **Config.** New per-slot variable `SLOT_<NAME>_MODELS`: comma-separated model
   names, whitespace and empty entries ignored. Read for every slot in `SLOTS`,
   including `main` and `batch` (no legacy alias). Absent means "no models call this
   slot home". Stored on `SlotConfig.models: tuple[str, ...]`, default `()`.
2. **One home per model.** If the same model (compared with `names.match_key`, so case
   and `.gguf` suffix are ignored) appears in two slots' lists, the manager refuses to
   start, naming both variables. Repeating a model within one slot's list is harmless
   and allowed.
3. **Untargeted `/swap`** (`target` absent **or** JSON `null`): resolve the slot by
   looking the model up in the map (`names.same_model`). Found → swap there, exactly as
   a targeted swap to that slot would. Not found → **400**
   `{"error": {"type": "no_home_slot", "message": ...}}`, no slot touched. The message
   names the remedy: pass `target`, or list the model in `SLOT_<NAME>_MODELS`.
4. **Targeted `/swap`** is unchanged, map or no map. Naming a slot is an operator
   override; it is not validated against the map.
5. **Chat endpoints are unchanged.** They still never swap; an unloaded model is still
   a 409 (`manager/app.py:447-458`).
6. **`/status`** gains a `models` list per slot (the configured home list, as written),
   so the operator can confirm the deployed map without reading `/etc`.

Error ordering in `/swap` keeps today's shape: missing `model` (400) → target
resolution (400 `invalid_target` / `no_home_slot`) → model file exists (404) →
swap (503 on failure).

## Effect on clients

- agent_core >= v1.11 (PARE): a mapped model heals onto its home slot. An unmapped
  model gets 400; agent_core then raises the original 409 (its `_request_model_swap`
  returns False on any non-200), so the client degrades to a clear error.
- PAL's `/model` always sends `target` (`pal/agent.py:686`) — unaffected.
- Anyone relying on "untargeted means main" now gets 400 for models not mapped to
  main. Intended.

## Non-goals

- No implicit swaps on the chat path.
- No checking targeted swaps against the map.
- No single-slot special case: with one slot configured, an untargeted swap still
  needs that slot's `_MODELS` list. One rule, no exceptions.
- No "already loaded elsewhere" short-circuit for untargeted swaps. A client only
  sends one after a 409, and the 409 path has already reprobed every slot
  (`reprobe_all_slots`, `manager/app.py:441-445`).

## Known trade-off

A stale client config still causes a swap, now onto that model's home slot. Mapping
few models to `main` keeps `main` stable; unmapped models are always refused.
Single user today, so slot contention between clients is not a design concern.
